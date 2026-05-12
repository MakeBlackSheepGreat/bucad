from __future__ import annotations

import math

import numpy as np
import pytest

from src.engine import segmentation_losses
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
