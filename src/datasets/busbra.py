"""BUSBRA manifest, split, and dataset adapters used by training."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.datasets.types import CaseSample
from src.engine.descriptors import ROI_DESCRIPTOR_FEATURES, extract_roi_descriptors
from src.preprocess.io import read_image, read_mask
from src.preprocess.roi import crop_to_mask_bbox
from src.preprocess.transforms import prepare_classifier_input, prepare_mask_target
from src.utils.runtime import optional_import, timestamp_now


torch = optional_import("torch")
torch_utils_data = optional_import("torch.utils.data")
DatasetBase = torch_utils_data.Dataset if torch_utils_data is not None else object


LABEL_TO_INDEX = {"benign": 0, "malignant": 1}


def _mask_path_for_sample(root: Path, sample_id: str) -> Path:
    """Infer the BUSBRA mask path from a BUS image sample id."""
    return root / "Masks" / sample_id.replace("bus_", "mask_").replace(".png", "") \
        if sample_id.endswith(".png") else root / "Masks" / sample_id.replace("bus_", "mask_")


def load_busbra_manifest(root: str | Path) -> pd.DataFrame:
    """Load BUSBRA CSV metadata into the shared manifest schema."""
    root_path = Path(root)
    csv_path = root_path / "bus_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"BUSBRA metadata file not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        sample_id = str(row.ID)
        image_path = root_path / "Images" / f"{sample_id}.png"
        mask_path = root_path / "Masks" / f"{sample_id.replace('bus_', 'mask_')}.png"
        sample = CaseSample(
            sample_id=sample_id,
            case_id=str(row.Case),
            dataset_name="BUSBRA",
            image_path=str(image_path),
            mask_path=str(mask_path) if mask_path.exists() else None,
            pathology_label=str(row.Pathology).lower(),
            view_side=str(row.Side).lower() if pd.notna(row.Side) else None,
        )
        payload = asdict(sample)
        payload["bbox"] = getattr(row, "BBOX", None)
        payload["birads"] = getattr(row, "BIRADS", None)
        records.append(payload)
    return pd.DataFrame.from_records(records)


def _build_group_splitter(n_splits: int, seed: int):
    """Prefer StratifiedGroupKFold, falling back to GroupKFold when unavailable."""
    model_selection = optional_import("sklearn.model_selection")
    if model_selection is not None and hasattr(model_selection, "StratifiedGroupKFold"):
        return model_selection.StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
    return GroupKFold(n_splits=n_splits)


def generate_busbra_split_assignments(
    manifest: pd.DataFrame, *, n_splits: int = 5, seed: int = 42
) -> pd.DataFrame:
    """Generate case-level BUSBRA fold assignments with pathology stratification when possible."""
    splitter = _build_group_splitter(n_splits=n_splits, seed=seed)
    y = manifest["pathology_label"].map(LABEL_TO_INDEX).to_numpy()
    groups = manifest["case_id"].to_numpy()
    rows: list[dict[str, Any]] = []
    split_iterator = splitter.split(manifest, y, groups)
    for fold_index, (train_idx, val_idx) in enumerate(split_iterator, start=1):
        for stage, indices in (("train", train_idx), ("val", val_idx)):
            subset = manifest.iloc[indices]
            for item in subset.itertuples(index=False):
                rows.append(
                    {
                        "sample_id": item.sample_id,
                        "case_id": item.case_id,
                        "pathology_label": item.pathology_label,
                        "fold_id": fold_index,
                        "stage": stage,
                        "generated_at": timestamp_now(),
                        "generator_signature": f"generate_busbra_split_assignments:{n_splits}:{seed}",
                    }
                )
    return pd.DataFrame.from_records(rows)


class BUSBRAClassificationDataset(DatasetBase):
    """Torch dataset wrapper for BUSBRA classification samples."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        image_size: int = 224,
        transform: Callable[[np.ndarray], Any] | None = None,
    ) -> None:
        """Store the manifest and preprocessing settings for classification batches."""
        self.manifest = manifest.reset_index(drop=True)
        self.image_size = image_size
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of manifest rows."""
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one BUSBRA image and return classifier input, label, and identifiers."""
        row = self.manifest.iloc[index]
        image = read_image(row["image_path"], grayscale=True)
        data = (
            self.transform(image)
            if self.transform is not None
            else prepare_classifier_input(image, self.image_size)
        )
        return {
            "image": data,
            "label": LABEL_TO_INDEX[row["pathology_label"]],
            "sample_id": row["sample_id"],
            "case_id": row["case_id"],
        }


class BUSBRAClassificationDualViewDataset(DatasetBase):
    """Torch dataset wrapper that returns full-image and lesion-ROI classifier views."""

    descriptor_features = ROI_DESCRIPTOR_FEATURES

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        image_size: int = 224,
        transform: Callable[[np.ndarray], Any] | None = None,
        roi_transform: Callable[[np.ndarray], Any] | None = None,
        mask_threshold: float = 0.4,
        margin_ratio: float = 0.35,
        min_area_ratio: float = 0.08,
        max_area_ratio: float = 0.75,
        largest_component: bool = True,
    ) -> None:
        """Store manifest and ROI extraction settings for dual-view batches."""
        self.manifest = manifest.reset_index(drop=True)
        self.image_size = image_size
        self.transform = transform
        self.roi_transform = roi_transform or transform
        self.mask_threshold = float(mask_threshold)
        self.margin_ratio = float(margin_ratio)
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.largest_component = bool(largest_component)

    def __len__(self) -> int:
        """Return the number of manifest rows."""
        return len(self.manifest)

    def _prepare_view(self, image: np.ndarray, transform: Callable[[np.ndarray], Any] | None):
        """Apply a configured classifier transform or default preprocessing."""
        if transform is not None:
            return transform(image)
        return prepare_classifier_input(image, self.image_size)

    def _roi_image_and_descriptors(self, image: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, dict[str, float]]:
        """Return a mask-cropped ROI image and stable numeric ROI descriptors."""
        descriptors = extract_roi_descriptors(
            image,
            mask,
            threshold=self.mask_threshold,
            margin_ratio=self.margin_ratio,
            min_area_ratio=self.min_area_ratio,
            largest_component=self.largest_component,
        )
        roi_valid = bool(descriptors.get("roi_valid", 0.0) >= 0.5)
        roi_area = float(descriptors.get("roi_area_ratio", 1.0))
        if roi_valid and self.min_area_ratio <= roi_area <= self.max_area_ratio:
            roi_image = crop_to_mask_bbox(
                image,
                mask,
                threshold=self.mask_threshold,
                margin_ratio=self.margin_ratio,
                min_area_ratio=self.min_area_ratio,
                largest_component=self.largest_component,
            )
            return roi_image, descriptors
        descriptors = dict(descriptors)
        descriptors["roi_valid"] = 0.0
        descriptors["roi_area_ratio"] = 1.0
        return image.copy(), descriptors

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one BUSBRA sample and return full, ROI, descriptors, and identifiers."""
        row = self.manifest.iloc[index]
        image = read_image(row["image_path"], grayscale=True)
        mask = read_mask(row.get("roi_mask_path", row.get("mask_path")))
        roi_image, descriptors = self._roi_image_and_descriptors(image, mask)
        descriptor_values = np.asarray(
            [float(descriptors.get(name, 0.0)) for name in self.descriptor_features],
            dtype=np.float32,
        )
        if torch is not None:
            descriptor_values = torch.from_numpy(descriptor_values)
            roi_valid = torch.tensor(float(descriptors.get("roi_valid", 0.0)), dtype=torch.float32)
        else:
            roi_valid = np.asarray(float(descriptors.get("roi_valid", 0.0)), dtype=np.float32)
        return {
            "image_full": self._prepare_view(image, self.transform),
            "image_roi": self._prepare_view(roi_image, self.roi_transform),
            "roi_descriptor": descriptor_values,
            "roi_valid": roi_valid,
            "label": LABEL_TO_INDEX[row["pathology_label"]],
            "sample_id": row["sample_id"],
            "case_id": row["case_id"],
        }


class BUSBRASegmentationDataset(DatasetBase):
    """Torch dataset wrapper for BUSBRA segmentation masks."""

    def __init__(self, manifest: pd.DataFrame, *, image_size: int = 256) -> None:
        """Store the manifest and target size for segmentation batches."""
        self.manifest = manifest.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self) -> int:
        """Return the number of manifest rows."""
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one BUSBRA image/mask pair, substituting an empty mask when absent."""
        row = self.manifest.iloc[index]
        image = read_image(row["image_path"], grayscale=True)
        mask = read_mask(row["mask_path"])
        if mask is None:
            mask = np.zeros_like(image, dtype=np.uint8)
        return {
            "image": prepare_classifier_input(image, self.image_size),
            "mask": prepare_mask_target(mask, self.image_size),
            "sample_id": row["sample_id"],
        }
