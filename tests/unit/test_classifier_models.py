"""Unit tests for classifier models."""

from __future__ import annotations

import pytest

from src.models import classifier
from src.models.classifier import create_classifier


@pytest.mark.skipif(classifier.torch is None, reason="torch is not installed")
def test_basic_cnn_classifier_outputs_two_logits() -> None:
    model = create_classifier("basic_cnn", pretrained=False, in_chans=3, num_classes=2)
    batch = classifier.torch.zeros((2, 3, 32, 32), dtype=classifier.torch.float32)

    logits = model(batch)

    assert tuple(logits.shape) == (2, 2)


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
def test_efficientnetv2_s_classifier_can_be_constructed_without_pretrained_weights() -> None:
    model = create_classifier("tf_efficientnetv2_s", pretrained=False, in_chans=3, num_classes=2)

    assert model is not None


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
def test_timm_classifier_accepts_regularization_kwargs() -> None:
    model = create_classifier(
        "convnext_tiny",
        pretrained=False,
        in_chans=3,
        num_classes=2,
        drop_path_rate=0.1,
    )

    assert model is not None
