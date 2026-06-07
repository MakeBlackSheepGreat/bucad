"""Unit tests for inference tta."""

from __future__ import annotations

import numpy as np

from src.engine.inference import BreastUltrasoundInferenceService


def test_classifier_tta_variants_override_legacy_horizontal_flip() -> None:
    """Verify classifier tta variants override legacy horizontal flip."""
    service = BreastUltrasoundInferenceService(
        {
            "classifier_tta_horizontal_flip": True,
            "classifier_tta_variants": [
                {"name": "identity", "crop_pct": 0.95},
                {"name": "hflip", "crop_pct": 0.90},
            ],
        }
    )

    variants = service._classifier_tta_variants()

    assert variants == [
        {"name": "identity", "crop_pct": 0.95},
        {"name": "hflip", "crop_pct": 0.90},
    ]


def test_classifier_tta_horizontal_flip_variant_flips_image() -> None:
    """Verify classifier tta horizontal flip variant flips image."""
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    service = BreastUltrasoundInferenceService({})

    flipped = service._apply_classifier_tta_variant(image, {"name": "hflip"})

    assert np.array_equal(flipped, np.fliplr(image))
    assert flipped.flags["C_CONTIGUOUS"]


def test_classifier_tta_rotate_suffix_changes_image_shape_safely() -> None:
    """Verify classifier tta rotate suffix changes image shape safely."""
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)
    service = BreastUltrasoundInferenceService({})

    rotated = service._apply_classifier_tta_variant(image, {"name": "rotate_m5"})

    assert rotated.shape == image.shape


def test_classifier_tta_variants_support_member_override() -> None:
    """Verify classifier tta variants support member override."""
    service = BreastUltrasoundInferenceService(
        {
            "classifier_tta_horizontal_flip": True,
        }
    )
    member = {
        "tta_variants": [
            {"name": "identity", "crop_pct": 0.90},
            {"name": "hflip", "crop_pct": 1.00},
        ],
    }

    variants = service._classifier_tta_variants(member)

    assert variants == [
        {"name": "identity", "crop_pct": 0.90},
        {"name": "hflip", "crop_pct": 1.00},
    ]


def test_classifier_preprocess_kwargs_support_member_override() -> None:
    """Verify classifier preprocess kwargs support member override."""
    service = BreastUltrasoundInferenceService(
        {
            "classifier_apply_clahe": False,
            "classifier_interpolation": "area",
            "classifier_crop_pct": 1.0,
        }
    )
    member = {
        "apply_clahe": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "interpolation": "bicubic",
        "crop_pct": 0.95,
    }

    kwargs = service._classifier_preprocess_kwargs({"crop_pct": 0.90}, member)

    assert kwargs == {
        "apply_clahe_enabled": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "interpolation": "bicubic",
        "crop_pct": 0.90,
    }
