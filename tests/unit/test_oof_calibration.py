"""Unit tests for frozen BUSBRA OOF calibration analysis."""

from __future__ import annotations

import json

from scripts.analyze_oof_calibration import analyze_oof_report


def test_calibration_analysis_reports_fixed_and_youden_metrics(tmp_path) -> None:
    """Verify the analysis uses only OOF rows and writes bounded calibration statistics."""
    report_path = tmp_path / "oof.json"
    report_path.write_text(
        json.dumps(
            {
                "model_name": "lesionext_lens_tiny",
                "rows": [
                    {"pathology_label": "benign", "malignant_probability": 0.10},
                    {"pathology_label": "benign", "malignant_probability": 0.45},
                    {"pathology_label": "malignant", "malignant_probability": 0.55},
                    {"pathology_label": "malignant", "malignant_probability": 0.90},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = analyze_oof_report(report_path, bins=4)

    assert result["fixed_threshold"] == 0.50
    assert result["sample_count"] == 4
    assert 0.0 <= result["expected_calibration_error"] <= 1.0
    assert 0.0 <= result["brier_score"] <= 1.0
    assert len(result["reliability_bins"]) == 4
    assert result["youden_threshold_metrics"]["threshold"] != 0.50
