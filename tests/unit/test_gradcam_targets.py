"""Unit tests for gradcam targets."""

from __future__ import annotations

import pytest

from src.models import classifier
from src.models.classifier import resolve_gradcam_target_layer


pytestmark = pytest.mark.skipif(classifier.nn is None, reason="torch is not installed")


def test_resolves_resnet_style_layer4_target() -> None:
    class Model:
        def __init__(self) -> None:
            self.layer4 = classifier.nn.Sequential(
                classifier.nn.Conv2d(3, 4, 1),
                classifier.nn.Conv2d(4, 5, 1),
            )

    model = Model()

    assert resolve_gradcam_target_layer(model) is model.layer4[-1]


def test_resolves_efficientnet_style_blocks_target() -> None:
    class Model:
        def __init__(self) -> None:
            self.blocks = classifier.nn.Sequential(
                classifier.nn.Conv2d(3, 4, 1),
                classifier.nn.Conv2d(4, 5, 1),
            )

    model = Model()

    assert resolve_gradcam_target_layer(model) is model.blocks[-1]


def test_resolves_convnext_style_stages_target() -> None:
    class Model:
        def __init__(self) -> None:
            self.stages = classifier.nn.Sequential(
                classifier.nn.Conv2d(3, 4, 1),
                classifier.nn.Conv2d(4, 5, 1),
            )

    model = Model()

    assert resolve_gradcam_target_layer(model) is model.stages[-1]


def test_resolves_conv_head_target() -> None:
    class Model:
        def __init__(self) -> None:
            self.conv_head = classifier.nn.Conv2d(3, 4, 1)

    model = Model()

    assert resolve_gradcam_target_layer(model) is model.conv_head
