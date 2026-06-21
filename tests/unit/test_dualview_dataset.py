"""Unit tests for BUSBRA dual-view classification dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.datasets import busbra
from src.datasets.busbra import BUSBRAClassificationDualViewDataset
from src.preprocess.io import save_image


@pytest.mark.skipif(busbra.torch is None, reason="torch is not installed")
def test_dualview_dataset_returns_full_roi_and_descriptors(tmp_path) -> None:
    """Verify dual-view dataset emits full view, ROI view, and descriptor vector."""
    image = np.full((32, 32), 32, dtype=np.uint8)
    image[10:22, 10:22] = 220
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 10:22] = 255
    image_path = save_image(tmp_path / "bus_1.png", image)
    mask_path = save_image(tmp_path / "mask_1.png", mask)
    manifest = pd.DataFrame(
        [
            {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "pathology_label": "malignant",
                "sample_id": "bus_1",
                "case_id": "case_1",
            }
        ]
    )

    dataset = BUSBRAClassificationDualViewDataset(manifest, image_size=16)
    sample = dataset[0]

    assert tuple(sample["image_full"].shape) == (3, 16, 16)
    assert tuple(sample["image_roi"].shape) == (3, 16, 16)
    assert tuple(sample["roi_descriptor"].shape) == (len(dataset.descriptor_features),)
    assert sample["roi_valid"].item() == 1.0
    assert sample["label"] == 1


@pytest.mark.skipif(busbra.torch is None, reason="torch is not installed")
def test_dualview_dataset_falls_back_when_mask_missing_or_area_out_of_range(tmp_path) -> None:
    """Verify invalid ROI masks fall back to the full-image view."""
    image = np.zeros((24, 24), dtype=np.uint8)
    image[6:18, 6:18] = 180
    image_path = save_image(tmp_path / "bus_2.png", image)
    manifest = pd.DataFrame(
        [
            {
                "image_path": str(image_path),
                "mask_path": None,
                "pathology_label": "benign",
                "sample_id": "bus_2",
                "case_id": "case_2",
            }
        ]
    )

    dataset = BUSBRAClassificationDualViewDataset(manifest, image_size=12)
    sample = dataset[0]

    assert sample["roi_valid"].item() == 0.0
    assert sample["roi_descriptor"][0].item() == 0.0
    assert busbra.torch.allclose(sample["image_full"], sample["image_roi"])


@pytest.mark.skipif(busbra.torch is None, reason="torch is not installed")
def test_dualview_dataset_prefers_roi_mask_path_override(tmp_path) -> None:
    """Verify predicted ROI mask paths can override ground-truth mask paths."""
    image = np.full((32, 32), 20, dtype=np.uint8)
    image[4:12, 4:12] = 220
    image[20:28, 20:28] = 180
    gt_mask = np.zeros((32, 32), dtype=np.uint8)
    gt_mask[4:12, 4:12] = 255
    roi_mask = np.zeros((32, 32), dtype=np.uint8)
    roi_mask[20:28, 20:28] = 255
    image_path = save_image(tmp_path / "bus_3.png", image)
    gt_mask_path = save_image(tmp_path / "mask_gt.png", gt_mask)
    roi_mask_path = save_image(tmp_path / "mask_pred.png", roi_mask)
    manifest = pd.DataFrame(
        [
            {
                "image_path": str(image_path),
                "mask_path": str(gt_mask_path),
                "roi_mask_path": str(roi_mask_path),
                "pathology_label": "malignant",
                "sample_id": "bus_3",
                "case_id": "case_3",
            }
        ]
    )

    dataset = BUSBRAClassificationDualViewDataset(manifest, image_size=16, margin_ratio=0.0, min_area_ratio=0.01)
    sample = dataset[0]

    assert sample["roi_valid"].item() == 1.0
    assert sample["image_roi"].mean().item() > sample["image_full"].mean().item()
