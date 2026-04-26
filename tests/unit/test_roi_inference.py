from __future__ import annotations

import math

import numpy as np

from src.engine.inference import BreastUltrasoundInferenceService


def test_roi_stacker_applies_probability_features() -> None:
    service = BreastUltrasoundInferenceService({})

    probability = service._apply_roi_stacker(
        full_probability=0.2,
        roi_probability=0.7,
        stacker={
            "feature_mode": "probability",
            "scaler_mean": [0.0, 0.0],
            "scaler_scale": [1.0, 1.0],
            "coef": [0.0, 1.0],
            "intercept": 0.0,
        },
    )

    assert math.isclose(probability, 1.0 / (1.0 + math.exp(-0.7)), rel_tol=1e-6)


def test_roi_enhanced_classification_uses_second_pass_roi_probability() -> None:
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 8:24] = 1.0
    service = BreastUltrasoundInferenceService(
        {
            "roi_enhancement": {
                "enabled": True,
                "margin_ratio": 0.0,
                "mask_threshold": 0.5,
                "fallback_to_full": False,
                "stacker": {
                    "feature_mode": "probability",
                    "scaler_mean": [0.0, 0.0],
                    "scaler_scale": [1.0, 1.0],
                    "coef": [0.0, 1.0],
                    "intercept": 0.0,
                },
            }
        },
        segmenter_predictor=lambda _: mask,
    )
    calls: list[tuple[int, int]] = []

    def predictor(
        image: np.ndarray,
        *,
        member_weight_overrides: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        calls.append((tuple(image.shape[:2]), member_weight_overrides))
        return (0.8, 0.2) if len(calls) == 1 else (0.3, 0.7)

    service._predict_classifier_ensemble_on_image = predictor  # type: ignore[method-assign]

    benign_probability, malignant_probability = service._predict_classification(
        np.full((32, 32), 128, dtype=np.uint8)
    )

    expected = 1.0 / (1.0 + math.exp(-0.7))
    assert math.isclose(malignant_probability, expected, rel_tol=1e-6)
    assert math.isclose(benign_probability, 1.0 - expected, rel_tol=1e-6)
    assert calls == [((32, 32), None), ((16, 16), None)]


def test_roi_enhanced_classification_can_override_roi_member_weights() -> None:
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 8:24] = 1.0
    service = BreastUltrasoundInferenceService(
        {
            "roi_enhancement": {
                "enabled": True,
                "margin_ratio": 0.0,
                "mask_threshold": 0.5,
                "fallback_to_full": False,
                "classifier_weight_overrides": {
                    "convnext_tiny": 0.54,
                    "tf_efficientnetv2_s": 0.46,
                },
                "stacker": {
                    "feature_mode": "probability",
                    "scaler_mean": [0.0, 0.0],
                    "scaler_scale": [1.0, 1.0],
                    "coef": [0.0, 1.0],
                    "intercept": 0.0,
                },
            }
        },
        segmenter_predictor=lambda _: mask,
    )
    calls: list[dict[str, float] | None] = []

    def predictor(
        image: np.ndarray,
        *,
        member_weight_overrides: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        calls.append(member_weight_overrides)
        return (0.8, 0.2) if len(calls) == 1 else (0.3, 0.7)

    service._predict_classifier_ensemble_on_image = predictor  # type: ignore[method-assign]

    service._predict_classification(np.full((32, 32), 128, dtype=np.uint8))

    assert calls == [
        None,
        {"convnext_tiny": 0.54, "tf_efficientnetv2_s": 0.46},
    ]


def test_roi_quality_gate_falls_back_to_full_probability() -> None:
    mask = np.ones((32, 32), dtype=np.float32)
    service = BreastUltrasoundInferenceService(
        {
            "roi_enhancement": {
                "enabled": True,
                "margin_ratio": 0.0,
                "mask_threshold": 0.5,
                "quality_gate": {
                    "enabled": True,
                    "max_area_ratio": 0.75,
                    "fallback_to_full": True,
                },
                "stacker": {
                    "feature_mode": "probability",
                    "scaler_mean": [0.0, 0.0],
                    "scaler_scale": [1.0, 1.0],
                    "coef": [0.0, 1.0],
                    "intercept": 0.0,
                },
            }
        },
        segmenter_predictor=lambda _: mask,
    )
    calls = 0

    def predictor(
        image: np.ndarray,
        *,
        member_weight_overrides: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        nonlocal calls
        calls += 1
        return 0.8, 0.2

    service._predict_classifier_ensemble_on_image = predictor  # type: ignore[method-assign]

    benign_probability, malignant_probability = service._predict_classification(
        np.full((32, 32), 128, dtype=np.uint8)
    )

    assert calls == 1
    assert math.isclose(benign_probability, 0.8, rel_tol=1e-6)
    assert math.isclose(malignant_probability, 0.2, rel_tol=1e-6)


def test_roi_stack_probability_can_blend_with_full_probability() -> None:
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 8:24] = 1.0
    service = BreastUltrasoundInferenceService(
        {
            "roi_enhancement": {
                "enabled": True,
                "margin_ratio": 0.0,
                "mask_threshold": 0.5,
                "roi_stack_blend_weight": 0.5,
                "stacker": {
                    "feature_mode": "probability",
                    "scaler_mean": [0.0, 0.0],
                    "scaler_scale": [1.0, 1.0],
                    "coef": [0.0, 1.0],
                    "intercept": 0.0,
                },
            }
        },
        segmenter_predictor=lambda _: mask,
    )
    calls = 0

    def predictor(
        image: np.ndarray,
        *,
        member_weight_overrides: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        nonlocal calls
        calls += 1
        return (0.8, 0.2) if calls == 1 else (0.3, 0.7)

    service._predict_classifier_ensemble_on_image = predictor  # type: ignore[method-assign]

    _, malignant_probability = service._predict_classification(
        np.full((32, 32), 128, dtype=np.uint8)
    )

    stacked_probability = 1.0 / (1.0 + math.exp(-0.7))
    expected = 0.5 * stacked_probability + 0.5 * 0.2
    assert math.isclose(malignant_probability, expected, rel_tol=1e-6)
