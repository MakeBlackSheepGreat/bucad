"""Unit tests for SonoGloReNet fold hard-sample weighting."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_sonoglore_fold_hard_sample_weights.py"
SPEC = importlib.util.spec_from_file_location("build_sonoglore_fold_hard_sample_weights", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_prediction_rows_assigns_sparse_weights() -> None:
    """Verify sparse mild weights for FP, FN, and borderline samples."""
    frame = pd.DataFrame(
        [
            {"sample_id": "a", "case_id": "1", "pathology_label": "benign"},
            {"sample_id": "b", "case_id": "2", "pathology_label": "malignant"},
            {"sample_id": "c", "case_id": "3", "pathology_label": "benign"},
            {"sample_id": "d", "case_id": "4", "pathology_label": "malignant"},
        ]
    )
    probabilities = [0.92, 0.08, 0.48, 0.88]

    rows, summary = MODULE._prediction_rows(
        val_manifest=frame,
        malignant_probabilities=probabilities,
        decision_threshold=0.5,
        benign_fp_threshold=0.85,
        malignant_fn_threshold=0.15,
        borderline_low=0.45,
        borderline_high=0.60,
        benign_fp_weight=1.30,
        malignant_fn_weight=1.25,
        borderline_weight=1.10,
        extreme_bonus=0.15,
    )

    weights = {row["sample_id"]: row["sample_weight"] for row in rows}
    mechanisms = {row["sample_id"]: row["mechanism"] for row in rows}

    assert weights["a"] == 1.45
    assert mechanisms["a"] == "benign_fp_extreme"
    assert weights["b"] == 1.4
    assert mechanisms["b"] == "malignant_fn_extreme"
    assert weights["c"] == 1.1
    assert mechanisms["c"] == "borderline_correct"
    assert weights["d"] == 1.0
    assert summary["weighted_count"] == 3
    assert summary["fp_count"] == 1
    assert summary["fn_count"] == 1
    assert summary["borderline_count"] == 1


def test_prediction_rows_fn_only_skips_fp_and_borderline_weights() -> None:
    """Verify fn_only mode only weights malignant false negatives."""
    frame = pd.DataFrame(
        [
            {"sample_id": "a", "case_id": "1", "pathology_label": "benign"},
            {"sample_id": "b", "case_id": "2", "pathology_label": "malignant"},
            {"sample_id": "c", "case_id": "3", "pathology_label": "benign"},
            {"sample_id": "d", "case_id": "4", "pathology_label": "malignant"},
        ]
    )
    probabilities = [0.92, 0.08, 0.48, 0.88]

    rows, summary = MODULE._prediction_rows(
        val_manifest=frame,
        malignant_probabilities=probabilities,
        weight_mode="fn_only",
        decision_threshold=0.5,
        benign_fp_threshold=0.85,
        malignant_fn_threshold=0.15,
        borderline_low=0.45,
        borderline_high=0.60,
        benign_fp_weight=1.30,
        malignant_fn_weight=1.25,
        borderline_weight=1.10,
        extreme_bonus=0.15,
    )

    weights = {row["sample_id"]: row["sample_weight"] for row in rows}
    mechanisms = {row["sample_id"]: row["mechanism"] for row in rows}

    assert weights["a"] == 1.0
    assert mechanisms["a"] == ""
    assert weights["b"] == 1.4
    assert mechanisms["b"] == "malignant_fn_extreme"
    assert weights["c"] == 1.0
    assert mechanisms["c"] == ""
    assert weights["d"] == 1.0
    assert summary["weighted_count"] == 1
    assert summary["fp_count"] == 1
    assert summary["fn_count"] == 1
    assert summary["borderline_count"] == 1
    assert summary["weight_mode"] == "fn_only"


def test_prediction_rows_fn_extreme_only_weights_only_extreme_fn() -> None:
    """Verify fn_extreme_only mode ignores mild FN cases."""
    frame = pd.DataFrame(
        [
            {"sample_id": "a", "case_id": "1", "pathology_label": "malignant"},
            {"sample_id": "b", "case_id": "2", "pathology_label": "malignant"},
            {"sample_id": "c", "case_id": "3", "pathology_label": "benign"},
        ]
    )
    probabilities = [0.10, 0.30, 0.70]

    rows, summary = MODULE._prediction_rows(
        val_manifest=frame,
        malignant_probabilities=probabilities,
        weight_mode="fn_extreme_only",
        decision_threshold=0.5,
        benign_fp_threshold=0.85,
        malignant_fn_threshold=0.15,
        borderline_low=0.45,
        borderline_high=0.60,
        benign_fp_weight=1.30,
        malignant_fn_weight=1.25,
        borderline_weight=1.10,
        extreme_bonus=0.15,
    )

    weights = {row["sample_id"]: row["sample_weight"] for row in rows}
    mechanisms = {row["sample_id"]: row["mechanism"] for row in rows}

    assert weights["a"] == 1.4
    assert mechanisms["a"] == "malignant_fn_extreme"
    assert weights["b"] == 1.0
    assert mechanisms["b"] == ""
    assert weights["c"] == 1.0
    assert mechanisms["c"] == ""
    assert summary["weighted_count"] == 1
    assert summary["fn_count"] == 2
    assert summary["extreme_fn_count"] == 1
    assert summary["weight_mode"] == "fn_extreme_only"
