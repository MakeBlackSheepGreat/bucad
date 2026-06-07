"""Unit tests for results."""

from __future__ import annotations

from src.utils.results import build_diagnostic_result


def test_build_diagnostic_result_normalizes_probabilities() -> None:
    result = build_diagnostic_result(2.0, 6.0, threshold=0.5)
    assert abs(result.benign_probability + result.malignant_probability - 1.0) < 1e-6
    assert result.final_label == "malignant"


def test_build_diagnostic_result_marks_borderline_cases() -> None:
    result = build_diagnostic_result(0.49, 0.51, threshold=0.5, borderline_margin=0.05)
    assert result.confidence_band == "borderline"
