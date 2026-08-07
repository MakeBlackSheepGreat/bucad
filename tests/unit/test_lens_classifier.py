"""Unit tests for LesioNeXt-LENS evidence interpolation modes."""

from __future__ import annotations

import pytest

from src.models.classifier import create_classifier
from src.utils.runtime import optional_import


torch = optional_import("torch")
timm = optional_import("timm")


@pytest.mark.skipif(torch is None or timm is None, reason="torch and timm are required")
def test_lens_static_gate_returns_one_shared_alpha() -> None:
    """Verify v1-compatible mode retains one alpha value across a batch."""
    model = create_classifier(
        "lesionext_lens_tiny",
        pretrained=False,
        evidence_alpha_init=0.12,
        evidence_gate_mode="static",
    )
    logits = model(torch.randn(2, 3, 64, 64))

    alpha = model.last_evidence_alpha
    assert logits.shape == (2, 2)
    assert alpha.shape == (2,)
    assert torch.allclose(alpha[0], alpha[1])
    assert 0.0 < float(alpha[0]) < 1.0


@pytest.mark.skipif(torch is None or timm is None, reason="torch and timm are required")
def test_lens_disabled_gate_returns_exact_zero_alpha() -> None:
    """Verify the evidence-only control removes local interpolation exactly."""
    model = create_classifier(
        "lesionext_lens_tiny",
        pretrained=False,
        evidence_gate_mode="disabled",
    )
    logits = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 2)
    assert torch.count_nonzero(model.last_evidence_alpha) == 0


@pytest.mark.skipif(torch is None or timm is None, reason="torch and timm are required")
def test_lens_local_auxiliary_logits_exist_only_during_training() -> None:
    """Verify label-aware local logits remain a training-only auxiliary output."""
    model = create_classifier(
        "lesionext_lens_tiny",
        pretrained=False,
        evidence_gate_mode="disabled",
        local_evidence_auxiliary=True,
    )
    model.train()
    _ = model(torch.randn(2, 3, 64, 64))
    assert model.last_local_evidence_logits.shape == (2, 2)

    model.eval()
    _ = model(torch.randn(2, 3, 64, 64))
    assert model.last_local_evidence_logits is None


@pytest.mark.skipif(torch is None or timm is None, reason="torch and timm are required")
def test_lens_confidence_gate_exposes_per_image_diagnostics() -> None:
    """Verify confidence mode emits bounded per-image alpha and diagnostic values."""
    model = create_classifier(
        "lesionext_lens_tiny",
        pretrained=False,
        evidence_alpha_init=0.12,
        evidence_gate_mode="confidence",
        evidence_gate_gain_init=2.0,
    )
    with torch.no_grad():
        model.evidence_head.weight.normal_(mean=0.0, std=0.05)
    logits = model(torch.randn(2, 3, 64, 64))

    alpha = model.last_evidence_alpha
    concentration = model.last_evidence_concentration
    agreement = model.last_evidence_agreement
    assert logits.shape == (2, 2)
    assert alpha.shape == concentration.shape == agreement.shape == (2,)
    assert torch.all((alpha > 0.0) & (alpha < 1.0))
    assert torch.all((concentration >= 0.0) & (concentration <= 1.0))
    assert torch.all((agreement >= 0.0) & (agreement <= 1.0))
