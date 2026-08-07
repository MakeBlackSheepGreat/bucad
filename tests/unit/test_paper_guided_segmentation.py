"""Unit tests for paper guided segmentation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.engine import segmentation_losses
from src.engine import train_seg
from src.engine.train_seg import _atomic_torch_save
from src.models import segmenter
from src.models.lesionext import LesioNeXtClassifier
from src.models.segmenter import create_segmenter
from src.utils.metrics import boundary_f1_score, hd95_score, iou_score


@pytest.mark.skipif(segmenter.torch is None, reason="torch is not installed")
def test_cenet_lite_outputs_mask_and_auxiliary_boundary() -> None:
    """Verify cenet lite outputs mask and auxiliary boundary."""
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
def test_lesionext_outputs_joint_tasks_and_reliability() -> None:
    """Verify LesioNeXt exposes mask, uncertainty, routing, and class outputs."""
    model = create_segmenter(
        architecture="lesionext",
        in_channels=3,
        classes=1,
        num_classes=2,
        widths=(8, 16, 24, 32),
        embedding_dim=16,
        dropout=0.0,
    )
    batch = segmenter.torch.randn((2, 3, 64, 64), dtype=segmenter.torch.float32)
    outputs = model.forward_with_aux(batch)

    assert tuple(outputs["mask"].shape) == (2, 1, 64, 64)
    assert tuple(outputs["boundary"].shape) == (2, 1, 64, 64)
    assert tuple(outputs["uncertainty"].shape) == (2, 1, 64, 64)
    assert tuple(outputs["class_logits"].shape) == (2, 2)
    assert tuple(outputs["roi_weight"].shape) == (2, 1)
    assert bool(((outputs["roi_weight"] >= 0) & (outputs["roi_weight"] <= 1)).all())


@pytest.mark.skipif(segmenter.torch is None, reason="torch is not installed")
def test_lesionext_classifier_wrapper_returns_logits() -> None:
    """Verify the classifier factory wrapper preserves the classifier contract."""
    model = LesioNeXtClassifier(
        in_channels=3,
        num_classes=2,
        widths=(8, 16, 24, 32),
        embedding_dim=16,
        dropout=0.0,
    )
    batch = segmenter.torch.randn((2, 3, 64, 64), dtype=segmenter.torch.float32)
    logits = model(batch)
    assert tuple(logits.shape) == (2, 2)


@pytest.mark.skipif(segmenter.torch is None, reason="torch is not installed")
def test_unet_segmenter_falls_back_to_tiny_model_when_smp_is_missing(monkeypatch) -> None:
    """Verify unet segmenter falls back to tiny model when smp is missing."""
    monkeypatch.setattr(segmenter, "smp", None)

    model = create_segmenter(architecture="unet", in_channels=3, classes=1)

    assert isinstance(model, segmenter.TinySegmentationNet)


@pytest.mark.skipif(segmentation_losses.torch is None, reason="torch is not installed")
def test_segmentation_loss_combines_boundary_pal_and_prototype_terms() -> None:
    """Verify segmentation loss combines boundary pal and prototype terms."""
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


@pytest.mark.skipif(segmentation_losses.torch is None, reason="torch is not installed")
def test_segmentation_loss_calibrates_lesionext_uncertainty_proxy() -> None:
    """Verify the uncertainty proxy contributes a differentiable loss term."""
    torch = segmentation_losses.torch
    outputs = {
        "mask": torch.zeros((1, 1, 16, 16), requires_grad=True),
        "uncertainty": torch.zeros((1, 1, 16, 16), requires_grad=True),
    }
    targets = torch.zeros((1, 1, 16, 16))
    targets[:, :, 4:12, 4:12] = 1.0
    loss, components = segmentation_losses.segmentation_loss(
        outputs,
        targets,
        {"name": "bce_dice_uncertainty", "uncertainty_weight": 0.1},
    )
    assert "uncertainty" in components
    assert loss.requires_grad


@pytest.mark.skipif(train_seg.torch is None, reason="torch is not installed")
def test_lesionext_joint_training_adds_classification_loss() -> None:
    """Verify the LesioNeXt training loop jointly optimizes class logits."""
    torch = train_seg.torch

    class _JointDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int):
            image = torch.full((3, 16, 16), float(index), dtype=torch.float32)
            mask = torch.zeros((1, 16, 16), dtype=torch.float32)
            mask[:, 4:12, 4:12] = 1.0
            return {"image": image, "mask": mask, "label": index}

    class _JointModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mask = torch.nn.Conv2d(3, 1, kernel_size=1)
            self.classifier = torch.nn.Linear(3, 2)

        def forward_with_aux(self, images):
            return {
                "mask": self.mask(images),
                "class_logits": self.classifier(images.mean(dim=(2, 3))),
            }

    model = _JointModel()
    loader = torch.utils.data.DataLoader(_JointDataset(), batch_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mean_loss, components = train_seg._train_segmentation_epoch(
        model,
        loader,
        optimizer,
        device="cpu",
        loss_cfg={"name": "bce", "classification_weight": 1.0},
        max_train_batches=None,
    )

    assert mean_loss > 0.0
    assert "classification" in components


@pytest.mark.skipif(train_seg.torch is None, reason="torch is not installed")
def test_lesionext_evaluation_reports_diagnostic_metrics() -> None:
    """Verify joint validation includes AUC and confusion-derived metrics."""
    torch = train_seg.torch

    class _Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int):
            values = (-2.0, 1.0, -1.0, 2.0)
            return {
                "image": torch.full((3, 8, 8), values[index], dtype=torch.float32),
                "mask": torch.zeros((1, 8, 8), dtype=torch.float32),
                "label": index % 2,
            }

    class _Model(torch.nn.Module):
        def forward_with_aux(self, images):
            values = images[:, 0, 0, 0]
            return {
                "mask": torch.zeros((len(images), 1, 8, 8), dtype=torch.float32),
                "class_logits": torch.stack([-values, values], dim=1),
            }

    loader = torch.utils.data.DataLoader(_Dataset(), batch_size=2)
    metrics = train_seg._evaluate_model(_Model(), loader, "cpu")

    assert metrics["classification"]["auc"] == pytest.approx(1.0)
    assert metrics["classification"]["accuracy"] == pytest.approx(1.0)
    assert metrics["classification"]["confusion"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_segmentation_metrics_cover_overlap_boundary_and_hd95() -> None:
    """Verify segmentation metrics cover overlap boundary and hd95."""
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
    """Verify segmentation checkpoint save is atomic."""
    destination = tmp_path / "segmenter.pt"

    _atomic_torch_save({"value": 1}, destination)
    _atomic_torch_save({"value": 2}, destination)

    assert destination.exists()
    assert not list(tmp_path.glob(".segmenter.pt.*.tmp"))


def test_run_segmentation_training_smoke_uses_prepared_run(tmp_path, monkeypatch) -> None:
    """Verify run segmentation training smoke uses prepared run."""
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
        """Represent Dataset for this module."""
        def __init__(self, frame, **_kwargs) -> None:
            """Initialize this lightweight test helper."""
            self.frame = frame.reset_index(drop=True)

        def __len__(self) -> int:
            """Return the number of samples in this helper dataset."""
            return len(self.frame)

        def __getitem__(self, index: int):
            """Return one sample from this helper dataset."""
            value = float(index % 2)
            return {
                "image": torch.full((3, 8, 8), value, dtype=torch.float32),
                "mask": torch.full((1, 8, 8), value, dtype=torch.float32),
            }

    class _TinySegmenter(torch.nn.Module):
        """Represent TinySegmenter for this module."""
        def __init__(self) -> None:
            """Initialize this lightweight test helper."""
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 1, kernel_size=1)

        def forward(self, images):
            """Run forward."""
            return self.conv(images)

    class _Paths:
        """Represent Paths for this module."""
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
