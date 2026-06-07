"""Smoke tests for visual evidence."""

from __future__ import annotations

import numpy as np

from src.engine.inference import BreastUltrasoundInferenceService


def test_visual_evidence_outputs_are_attached_when_available() -> None:
    image = np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)
    service = BreastUltrasoundInferenceService(
        {},
        classifier_predictor=lambda _: (0.4, 0.6),
        segmenter_predictor=lambda img: np.where(img > 128, 1.0, 0.0),
        explanation_generator=lambda img: img.astype(np.float32) / 255.0,
    )

    response = service.diagnose(image, need_segmentation=True, need_explanation=True)

    assert response.status == "completed"
    assert response.lesion_overlay_view is not None
    assert response.explanation_view is not None


def test_visual_evidence_reports_missing_outputs_in_place() -> None:
    image = np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)
    service = BreastUltrasoundInferenceService(
        {},
        classifier_predictor=lambda _: (0.6, 0.4),
    )

    response = service.diagnose(image, need_segmentation=True, need_explanation=True)

    assert response.status == "partial"
    assert response.lesion_visualization_missing_reason is not None
    assert response.explanation_missing_reason is not None
