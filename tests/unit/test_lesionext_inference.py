"""Unit tests for the unified LesioNeXt inference path."""

from __future__ import annotations

import numpy as np
import pytest

from src.engine.inference import BreastUltrasoundInferenceService
from src.models.lesionext import LesioNeXt
from src.utils.runtime import optional_import


torch = optional_import("torch")


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_lesionext_inference_uses_one_joint_checkpoint(tmp_path) -> None:
    """Verify diagnosis uses unified class logits and cached lesion masks."""
    checkpoint = tmp_path / "lesionext.pt"
    model = LesioNeXt(
        in_channels=3,
        classes=1,
        num_classes=2,
        widths=(8, 16, 24, 32),
        embedding_dim=16,
        dropout=0.0,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "architecture": "lesionext",
                "in_channels": 3,
                "classes": 1,
                "num_classes": 2,
                "widths": (8, 16, 24, 32),
                "embedding_dim": 16,
                "dropout": 0.0,
            },
        },
        checkpoint,
    )
    service = BreastUltrasoundInferenceService(
        {
            "model_family": "lesionext",
            "lesionext_checkpoint": str(checkpoint),
            "lesionext_device": "cpu",
            "lesionext_image_size": 64,
            "lesionext_model": {
                "architecture": "lesionext",
                "in_channels": 3,
                "classes": 1,
                "num_classes": 2,
                "widths": (8, 16, 24, 32),
                "embedding_dim": 16,
                "dropout": 0.0,
            },
            "default_threshold": 0.5,
            "gradcam_enabled": False,
        }
    )
    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))

    response = service.diagnose(image, need_segmentation=True, need_explanation=False)

    assert response.status == "completed"
    assert response.result is not None
    assert response.lesion_overlay_view is not None
    assert response.metadata["model_identifier"] == "LesioNeXt-BUS"
    assert 0.0 <= response.metadata["roi_reliability"] <= 1.0
