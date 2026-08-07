"""Tests for the compact LesioNeXt Block Attention Residual classifier."""

from __future__ import annotations

import pytest

from src.models.classifier import create_classifier
from src.utils.runtime import optional_import


torch = optional_import("torch")
timm = optional_import("timm")


@pytest.mark.skipif(torch is None or timm is None, reason="torch and timm are required")
def test_block_attnres_zero_gate_preserves_convnext_stage_output() -> None:
    """Verify zero initialized residual gates preserve the pretrained backbone path."""
    model = create_classifier("lesionext_attnres_tiny", pretrained=False)
    model.eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        features = model.backbone.stem(image)
        features = model.backbone.stages[0](features)
        features = model.backbone.stages[1](features)
        expected = model.backbone.stages[2](features.clone())
        actual = model.attnres["3"](model.backbone.stages[2], features.clone())
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(torch is None or timm is None, reason="torch and timm are required")
def test_block_attnres_classifier_reports_depth_attention() -> None:
    """Verify classifier output and per-stage depth attention diagnostics are available."""
    model = create_classifier("lesionext_attnres_tiny", pretrained=False)
    logits = model(torch.randn(2, 3, 64, 64))
    summary = model.attention_summary()

    assert logits.shape == (2, 2)
    assert len(summary["stage_3"]) == 9
    assert len(summary["stage_4"]) == 3


@pytest.mark.skipif(torch is None or timm is None, reason="torch and timm are required")
def test_block_boundary_injection_reduces_attention_calls() -> None:
    """Verify block-boundary mode keeps the backbone path while reducing attention calls."""
    model = create_classifier(
        "lesionext_attnres_tiny",
        pretrained=False,
        attnres_block_size=2,
        attnres_inject_every=2,
    )
    logits = model(torch.randn(1, 3, 64, 64))
    summary = model.attention_summary()

    assert logits.shape == (1, 2)
    assert len(summary["stage_3"]) == 5
    assert len(summary["stage_4"]) == 2
