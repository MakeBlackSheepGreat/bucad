"""BUSI manifest and dataset adapters used by external evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.preprocess.io import read_image, read_mask
from src.preprocess.transforms import prepare_classifier_input, prepare_mask_target
from src.utils.runtime import optional_import


torch_utils_data = optional_import("torch.utils.data")
DatasetBase = torch_utils_data.Dataset if torch_utils_data is not None else object

LABEL_TO_INDEX = {"benign": 0, "malignant": 1, "normal": 0}


def _pick_mask_for_image(image_path: Path) -> Path | None:
    candidates = sorted(image_path.parent.glob(f"{image_path.stem}_mask*.png"))
    return candidates[0] if candidates else None


def load_busi_manifest(root: str | Path, *, include_normal: bool = False) -> pd.DataFrame:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"BUSI root not found: {root_path}")

    records: list[dict[str, Any]] = []
    for label_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        pathology_label = label_dir.name.lower()
        if pathology_label == "normal" and not include_normal:
            continue
        for image_path in sorted(label_dir.glob("*.png")):
            if "_mask" in image_path.stem:
                continue
            mask_path = _pick_mask_for_image(image_path)
            sample_id = image_path.stem
            records.append(
                {
                    "sample_id": sample_id,
                    "case_id": sample_id,
                    "dataset_name": "BUSI",
                    "image_path": str(image_path),
                    "mask_path": str(mask_path) if mask_path is not None else None,
                    "pathology_label": pathology_label,
                    "view_side": None,
                    "quality_flag": "valid",
                }
            )
    return pd.DataFrame.from_records(records)


class BUSIDataset(DatasetBase):
    """Torch dataset wrapper for BUSI classification and optional masks."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        image_size: int = 224,
        transform: Callable[[np.ndarray], Any] | None = None,
        segmentation: bool = False,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.image_size = image_size
        self.transform = transform
        self.segmentation = segmentation

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        image = read_image(row["image_path"], grayscale=True)
        item = {
            "image": self.transform(image)
            if self.transform is not None
            else prepare_classifier_input(image, self.image_size),
            "label": LABEL_TO_INDEX.get(row["pathology_label"], 0),
            "sample_id": row["sample_id"],
        }
        if self.segmentation:
            mask = read_mask(row["mask_path"])
            if mask is None:
                mask = np.zeros_like(image, dtype=np.uint8)
            item["mask"] = prepare_mask_target(mask, self.image_size)
        return item
