"""Smoke tests for single image inference."""

from __future__ import annotations

import numpy as np

from src.engine.inference import BreastUltrasoundInferenceService


def test_single_image_inference_returns_probabilities() -> None:
    """Verify single image inference returns probabilities."""
    image = np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)
    service = BreastUltrasoundInferenceService(
        {"default_threshold": 0.5, "borderline_margin": 0.05},
        classifier_predictor=lambda _: (0.2, 0.8),
    )

    response = service.diagnose(image, input_filename="sample.png", need_segmentation=False, need_explanation=False)

    assert response.status == "completed"
    assert response.result is not None
    total = response.result.benign_probability + response.result.malignant_probability
    assert abs(total - 1.0) < 1e-6
    assert response.result.final_label == "malignant"


def test_single_image_inference_handles_borderline_case() -> None:
    """Verify single image inference handles borderline case."""
    image = np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)
    service = BreastUltrasoundInferenceService(
        {"default_threshold": 0.5, "borderline_margin": 0.05},
        classifier_predictor=lambda _: (0.49, 0.51),
    )

    response = service.diagnose(image, input_filename="sample.png", need_segmentation=False, need_explanation=False)

    assert response.result is not None
    assert response.result.confidence_band == "borderline"
    assert "Manual review" in response.result.recommendation_text


def test_single_image_inference_handles_invalid_and_low_quality_inputs() -> None:
    """Verify single image inference handles invalid and low quality inputs."""
    service = BreastUltrasoundInferenceService({}, classifier_predictor=lambda _: (0.6, 0.4))

    invalid = service.diagnose(np.zeros((8, 8), dtype=np.uint8), input_filename="tiny.png")
    low_quality = service.diagnose(np.full((128, 128), 12, dtype=np.uint8), input_filename="flat.png")

    assert invalid.status == "invalid_input"
    assert low_quality.status == "quality_blocked"
