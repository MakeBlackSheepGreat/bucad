"""BUSBRA manifest, split, and dataset adapters used by training."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.datasets.types import CaseSample
from src.preprocess.io import read_image, read_mask
from src.preprocess.transforms import prepare_classifier_input, prepare_mask_target
from src.utils.runtime import optional_import, timestamp_now


torch = optional_import("torch")
torch_utils_data = optional_import("torch.utils.data")
DatasetBase = torch_utils_data.Dataset if torch_utils_data is not None else object


LABEL_TO_INDEX = {"benign": 0, "malignant": 1}


def _mask_path_for_sample(root: Path, sample_id: str) -> Path:
    return root / "Masks" / sample_id.replace("bus_", "mask_").replace(".png", "") \
        if sample_id.endswith(".png") else root / "Masks" / sample_id.replace("bus_", "mask_")


def load_busbra_manifest(root: str | Path) -> pd.DataFrame:
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
    model_selection = optional_import("sklearn.model_selection")
    if model_selection is not None and hasattr(model_selection, "StratifiedGroupKFold"):
        return model_selection.StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
    return GroupKFold(n_splits=n_splits)


def generate_busbra_split_assignments(
    manifest: pd.DataFrame, *, n_splits: int = 5, seed: int = 42
) -> pd.DataFrame:
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
    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        image_size: int = 224,
        transform: Callable[[np.ndarray], Any] | None = None,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.image_size = image_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
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


class BUSBRASegmentationDataset(DatasetBase):
    def __init__(self, manifest: pd.DataFrame, *, image_size: int = 256) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
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
