"""Unit tests for classifier models."""

from __future__ import annotations

import pytest

from src.models import classifier
from src.models.classifier import create_classifier, load_classifier, resolve_gradcam_target_layer


@pytest.mark.skipif(classifier.torch is None, reason="torch is not installed")
def test_basic_cnn_classifier_outputs_two_logits() -> None:
    """Verify basic cnn classifier outputs two logits."""
    model = create_classifier("basic_cnn", pretrained=False, in_chans=3, num_classes=2)
    batch = classifier.torch.zeros((2, 3, 32, 32), dtype=classifier.torch.float32)

    logits = model(batch)

    assert tuple(logits.shape) == (2, 2)


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
def test_efficientnetv2_s_classifier_can_be_constructed_without_pretrained_weights() -> None:
    """Verify efficientnetv2 s classifier can be constructed without pretrained weights."""
    model = create_classifier("tf_efficientnetv2_s", pretrained=False, in_chans=3, num_classes=2)

    assert model is not None


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
def test_timm_classifier_accepts_regularization_kwargs() -> None:
    """Verify timm classifier accepts regularization kwargs."""
    model = create_classifier(
        "convnext_tiny",
        pretrained=False,
        in_chans=3,
        num_classes=2,
        drop_path_rate=0.1,
    )

    assert model is not None


@pytest.mark.skipif(classifier.torch is None, reason="torch is not installed")
def test_dualview_convnext_tiny_outputs_two_logits_with_descriptors(monkeypatch) -> None:
    """Verify dual-view ConvNeXt alias accepts full, ROI, and descriptor tensors."""
    torch = classifier.torch

    class _Backbone(torch.nn.Module):
        """Small backbone stand-in for timm ConvNeXt."""

        num_features = 4
        pretrained_cfg = {"input_size": (3, 8, 8)}

        def forward(self, images):
            """Return deterministic four-channel embeddings."""
            return images.mean(dim=(2, 3))[:, :1].repeat(1, self.num_features)

    monkeypatch.setattr(
        classifier.timm,
        "create_model",
        lambda *_args, **_kwargs: _Backbone(),
    )
    model = create_classifier(
        "roi_dualview_convnext_tiny",
        pretrained=False,
        descriptor_dim=14,
        fusion_hidden_dim=8,
    )
    image_full = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
    image_roi = torch.ones((2, 3, 8, 8), dtype=torch.float32)
    roi_descriptor = torch.zeros((2, 14), dtype=torch.float32)

    logits = model(
        image_full=image_full,
        image_roi=image_roi,
        roi_descriptor=roi_descriptor,
    )

    assert tuple(logits.shape) == (2, 2)
    assert model.input_mode == "dual_view_roi"


