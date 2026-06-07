"""Unit tests for paper guided segmentation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.engine import segmentation_losses
from src.engine import train_seg
from src.engine.train_seg import _atomic_torch_save
from src.models import segmenter
from src.models.segmenter import create_segmenter
from src.utils.metrics import boundary_f1_score, hd95_score, iou_score


@pytest.mark.skipif(segmenter.torch is None, reason="torch is not installed")
def test_cenet_lite_outputs_mask_and_auxiliary_boundary() -> None:
    model = create_segmenter(
        architecture="cenet_lite",
        in_channels=3,
        classes=1,
        base_channels=8,
        use_dseb=True,
        use_cfam=True,
        use_nonlocal=False,
        boundary_head=True,
    )
    batch = segmenter.torch.zeros((2, 3, 64, 64), dtype=segmenter.torch.float32)

    outputs = model.forward_with_aux(batch)

    assert tuple(outputs["mask"].shape) == (2, 1, 64, 64)
    assert tuple(outputs["boundary"].shape) == (2, 1, 64, 64)
    assert outputs["prototype_features"].shape[0] == 2


@pytest.mark.skipif(segmenter.torch is None, reason="torch is not installed")
def test_unet_segmenter_falls_back_to_tiny_model_when_smp_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(segmenter, "smp", None)

    model = create_segmenter(architecture="unet", in_channels=3, classes=1)

    assert isinstance(model, segmenter.TinySegmentationNet)


@pytest.mark.skipif(segmentation_losses.torch is None, reason="torch is not installed")
def test_segmentation_loss_combines_boundary_pal_and_prototype_terms() -> None:
    torch = segmentation_losses.torch
    outputs = {
        "mask": torch.zeros((2, 1, 16, 16), dtype=torch.float32, requires_grad=True),
        "boundary": torch.zeros((2, 1, 16, 16), dtype=torch.float32, requires_grad=True),
        "prototype_features": torch.randn((2, 4, 16, 16), dtype=torch.float32, requires_grad=True),
    }
    targets = torch.zeros((2, 1, 16, 16), dtype=torch.float32)
    targets[:, :, 4:12, 4:12] = 1.0

    loss, components = segmentation_losses.segmentation_loss(
        outputs,
        targets,
        {
            "name": "bce_dice_boundary_pal_prototype",
            "bce_weight": 1.0,
            "dice_weight": 1.0,
            "boundary_weight": 0.25,
            "pal_weight": 0.1,
            "foreground_prototype_weight": 0.1,
            "edge_prototype_weight": 0.05,
        },
    )

    assert loss.requires_grad
    assert {"bce", "dice", "boundary", "pal", "foreground_prototype", "edge_prototype"} <= set(components)
    assert float(loss.detach()) > 0.0


def test_segmentation_metrics_cover_overlap_boundary_and_hd95() -> None:
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:20, 8:20] = 1.0
    shifted = np.zeros_like(mask)
    shifted[9:21, 8:20] = 1.0

    assert iou_score(mask, mask) == pytest.approx(1.0)
    assert boundary_f1_score(mask, mask) == pytest.approx(1.0)
    assert 0.0 < boundary_f1_score(shifted, mask) <= 1.0
    assert hd95_score(mask, mask) == pytest.approx(0.0)
    assert math.isfinite(hd95_score(shifted, mask))


@pytest.mark.skipif(segmentation_losses.torch is None, reason="torch is not installed")
def test_segmentation_checkpoint_save_is_atomic(tmp_path) -> None:
    destination = tmp_path / "segmenter.pt"

    _atomic_torch_save({"value": 1}, destination)
    _atomic_torch_save({"value": 2}, destination)

    assert destination.exists()
    assert not list(tmp_path.glob(".segmenter.pt.*.tmp"))


def test_run_segmentation_training_smoke_uses_prepared_run(tmp_path, monkeypatch) -> None:
    if train_seg.torch is None:
        return
    torch = train_seg.torch
    manifest = train_seg.pd.DataFrame(
        [
            {"sample_id": "a", "case_id": "c1"},
            {"sample_id": "b", "case_id": "c2"},
            {"sample_id": "c", "case_id": "c3"},
            {"sample_id": "d", "case_id": "c4"},
        ]
    )

    class _Dataset(torch.utils.data.Dataset):
        def __init__(self, frame, **_kwargs) -> None:
            self.frame = frame.reset_index(drop=True)

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, index: int):
            value = float(index % 2)
            return {
                "image": torch.full((3, 8, 8), value, dtype=torch.float32),
                "mask": torch.full((1, 8, 8), value, dtype=torch.float32),
            }

    class _TinySegmenter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 1, kernel_size=1)

        def forward(self, images):
            return self.conv(images)

    class _Paths:
        project_root = tmp_path
        busbra_root = tmp_path / "unused"
        checkpoints_root = tmp_path / "checkpoints"
        reports_root = tmp_path / "reports"

    config_path = tmp_path / "config.yml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    monkeypatch.setattr(
        train_seg,
        "load_project_config",
        lambda _path: (
            {
                "device": "cpu",
                "training": {
                    "epochs": 1,
                    "max_train_batches": 1,
                    "max_val_batches": 1,
                },
                "output": {},
                "model": {},
                "loss": {"name": "bce"},
            },
            _Paths(),
        ),
    )
    monkeypatch.setattr(train_seg, "load_busbra_manifest", lambda _root: manifest)
    monkeypatch.setattr(
        train_seg,
        "_prepare_fold_manifests",
        lambda **_kwargs: (manifest.iloc[:2], manifest.iloc[2:]),
    )
    monkeypatch.setattr(train_seg, "BUSBRASegmentationDataset", _Dataset)
    monkeypatch.setattr(train_seg, "_build_segmenter", lambda _model_cfg, **_kwargs: _TinySegmenter())

    report = train_seg.run_segmentation_training(config_path, fold=1)

    assert report["fold"] == 1
    assert report["train_size"] == 2
    assert report["val_size"] == 2
    assert report["max_train_batches"] == 1
    assert report["max_val_batches"] == 1
    assert (tmp_path / "checkpoints" / "segmenter_fold1.pt").exists()
