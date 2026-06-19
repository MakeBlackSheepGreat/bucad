"""Unit tests for BUSBRA-wide SonoGloReNet OOF sample weighting."""

from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_sonoglore_oof_sample_weights.py"
SPEC = importlib.util.spec_from_file_location("build_sonoglore_oof_sample_weights", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_load_oof_frame_normalizes_and_validates_rows(tmp_path: Path) -> None:
    """Verify OOF rows are normalized into a dataframe with expected types."""
    payload = {
        "rows": [
            {
                "sample_id": "a",
                "case_id": 1,
                "pathology_label": "Benign",
                "malignant_probability": 0.2,
                "fold_id": 1,
            },
            {
                "sample_id": "b",
                "case_id": 2,
                "pathology_label": "Malignant",
                "malignant_probability": 0.1,
                "fold_id": 2,
            },
        ]
    }
    path = tmp_path / "oof.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    frame = MODULE._load_oof_frame(path)

    assert list(frame["sample_id"]) == ["a", "b"]
    assert list(frame["case_id"]) == ["1", "2"]
    assert list(frame["pathology_label"]) == ["benign", "malignant"]
    assert float(frame.loc[1, "malignant_probability"]) == 0.1


def test_run_builds_full_weight_csv_from_oof_rows(tmp_path: Path) -> None:
    """Verify OOF rows can be converted into a training-ready weight CSV."""
    payload = {
        "rows": [
            {
                "sample_id": "a",
                "case_id": "1",
                "pathology_label": "benign",
                "malignant_probability": 0.9,
                "fold_id": 1,
            },
            {
                "sample_id": "b",
                "case_id": "2",
                "pathology_label": "malignant",
                "malignant_probability": 0.1,
                "fold_id": 2,
            },
            {
                "sample_id": "c",
                "case_id": "3",
                "pathology_label": "malignant",
                "malignant_probability": 0.4,
                "fold_id": 3,
            },
        ]
    }
    report_path = tmp_path / "oof.json"
    output_path = tmp_path / "weights.csv"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = MODULE.run(
        Namespace(
            oof_report=str(report_path),
            output=str(output_path),
            decision_threshold=0.5,
            benign_fp_threshold=0.85,
            malignant_fn_threshold=0.15,
            borderline_low=0.4,
            borderline_high=0.6,
            borderline_weight=1.1,
            benign_fp_weight=1.3,
            malignant_fn_weight=1.2,
            extreme_bonus=0.1,
            weight_mode="fn_extreme_only",
        )
    )

    frame = pd.read_csv(output_path, encoding="utf-8-sig")

    assert summary["sample_count"] == 3
    assert summary["weighted_count"] == 1
    assert summary["fn_count"] == 2
    assert frame.loc[frame["sample_id"] == "a", "sample_weight"].item() == 1.0
    assert frame.loc[frame["sample_id"] == "b", "sample_weight"].item() == 1.3
    assert frame.loc[frame["sample_id"] == "c", "sample_weight"].item() == 1.0
