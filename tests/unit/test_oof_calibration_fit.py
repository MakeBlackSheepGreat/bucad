"""Tests for BUSBRA-only nested logit calibration."""

import json

import numpy as np

from scripts.fit_oof_calibration import apply_calibrator, fit_calibrator, nested_oof_calibration


def test_identity_calibrator_preserves_probabilities() -> None:
    """Verify an identity calibration leaves probabilities unchanged."""
    probabilities = np.asarray([0.01, 0.2, 0.5, 0.9], dtype=float)
    calibrated = apply_calibrator(probabilities, {"temperature": 1.0, "bias": 0.0})
    np.testing.assert_allclose(calibrated, probabilities, atol=1e-7)


def test_bias_calibrator_is_bounded() -> None:
    """Verify fitted bias values stay within the configured bounds."""
    probabilities = np.asarray([0.01, 0.02, 0.03, 0.1], dtype=float)
    labels = np.asarray([0, 0, 1, 1], dtype=int)
    calibrator = fit_calibrator(probabilities, labels, mode="bias")
    assert -2.0 <= calibrator["bias"] <= 2.0
    assert calibrator["temperature"] == 1.0


def test_nested_calibration_returns_one_calibrator_per_fold(tmp_path) -> None:
    """Verify nested fitting produces one calibrator for every validation fold."""
    rows = []
    for fold in (1, 2, 3, 4, 5):
        rows.extend(
            [
                {"pathology_label": "benign", "malignant_probability": 0.1, "fold_id": fold},
                {"pathology_label": "malignant", "malignant_probability": 0.8, "fold_id": fold},
            ]
        )
    path = tmp_path / "oof.json"
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    frame, calibrators = nested_oof_calibration(path, mode="bias")
    assert len(frame) == 10
    assert len(calibrators) == 5
    assert set(frame["calibrated_probability"].between(0.0, 1.0)) == {True}
