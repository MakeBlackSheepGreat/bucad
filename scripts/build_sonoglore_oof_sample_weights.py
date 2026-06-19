"""Build BUSBRA-wide hard-sample weights from SonoGloReNet OOF predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_sonoglore_fold_hard_sample_weights import _prediction_rows


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for OOF-driven sample weight generation."""
    parser = argparse.ArgumentParser(
        description="Build BUSBRA-wide sample weights from a SonoGloReNet OOF evaluation JSON."
    )
    parser.add_argument(
        "--oof-report",
        required=True,
        help="OOF evaluation JSON path produced by scripts/evaluate_classifier_oof.py.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination CSV path with per-sample weights.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Decision threshold used to define FP and FN samples.",
    )
    parser.add_argument(
        "--benign-fp-threshold",
        type=float,
        default=0.85,
        help="Benign samples above this malignant probability count as extreme FP.",
    )
    parser.add_argument(
        "--malignant-fn-threshold",
        type=float,
        default=0.15,
        help="Malignant samples below this malignant probability count as extreme FN.",
    )
    parser.add_argument(
        "--borderline-low",
        type=float,
        default=0.40,
        help="Lower bound of the borderline band.",
    )
    parser.add_argument(
        "--borderline-high",
        type=float,
        default=0.60,
        help="Upper bound of the borderline band.",
    )
    parser.add_argument(
        "--borderline-weight",
        type=float,
        default=1.10,
        help="Mild weight assigned to borderline-but-correct samples when weight_mode=all.",
    )
    parser.add_argument(
        "--benign-fp-weight",
        type=float,
        default=1.30,
        help="Weight assigned to benign false positives when weight_mode=all.",
    )
    parser.add_argument(
        "--malignant-fn-weight",
        type=float,
        default=1.25,
        help="Base weight assigned to malignant false negatives.",
    )
    parser.add_argument(
        "--extreme-bonus",
        type=float,
        default=0.15,
        help="Additional weight added to extreme-confidence errors.",
    )
    parser.add_argument(
        "--weight-mode",
        choices=("all", "fn_only", "fn_extreme_only"),
        default="fn_only",
        help="Weighting strategy shared with build_sonoglore_fold_hard_sample_weights.py.",
    )
    return parser


def _load_oof_frame(path: str | Path) -> pd.DataFrame:
    """Load OOF evaluation rows into a normalized dataframe."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"OOF report does not contain prediction rows: {path}")
    frame = pd.DataFrame(rows)
    required_columns = {"sample_id", "case_id", "pathology_label", "malignant_probability"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"OOF rows missing required columns {sorted(missing)}: {path}")
    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["case_id"] = frame["case_id"].astype(str)
    frame["pathology_label"] = frame["pathology_label"].astype(str).str.lower()
    frame["malignant_probability"] = frame["malignant_probability"].astype(float)
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(), "sample_id"].tolist()[:5]
        raise ValueError(f"OOF rows contain duplicated sample_id values, for example: {duplicates}")
    return frame


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Generate BUSBRA-wide sample weights from one OOF report."""
    frame = _load_oof_frame(args.oof_report)
    manifest = frame[["sample_id", "case_id", "pathology_label"]].reset_index(drop=True)
    probabilities = frame["malignant_probability"].astype(float).tolist()
    rows, summary = _prediction_rows(
        val_manifest=manifest,
        malignant_probabilities=probabilities,
        weight_mode=str(args.weight_mode),
        decision_threshold=float(args.decision_threshold),
        benign_fp_threshold=float(args.benign_fp_threshold),
        malignant_fn_threshold=float(args.malignant_fn_threshold),
        borderline_low=float(args.borderline_low),
        borderline_high=float(args.borderline_high),
        benign_fp_weight=float(args.benign_fp_weight),
        malignant_fn_weight=float(args.malignant_fn_weight),
        borderline_weight=float(args.borderline_weight),
        extreme_bonus=float(args.extreme_bonus),
    )
    if "fold_id" in frame.columns:
        fold_lookup = dict(zip(frame["sample_id"].astype(str), frame["fold_id"].tolist()))
        for row in rows:
            row["fold_id"] = fold_lookup.get(str(row["sample_id"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "case_id",
        "pathology_label",
        "malignant_probability",
        "predicted_label",
        "sample_weight",
        "error_type",
        "mechanism",
    ]
    if "fold_id" in frame.columns:
        fieldnames.append("fold_id")
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "output": str(output),
        "oof_report": str(args.oof_report),
        "sample_count": int(len(rows)),
        "weighted_count": int(summary["weighted_count"]),
        "fp_count": int(summary["fp_count"]),
        "fn_count": int(summary["fn_count"]),
        "borderline_count": int(summary["borderline_count"]),
        "extreme_fp_count": int(summary["extreme_fp_count"]),
        "extreme_fn_count": int(summary["extreme_fn_count"]),
        "weight_mode": str(summary["weight_mode"]),
        "decision_threshold": float(summary["decision_threshold"]),
        "benign_fp_threshold": float(summary["benign_fp_threshold"]),
        "malignant_fn_threshold": float(summary["malignant_fn_threshold"]),
        "borderline_low": float(summary["borderline_low"]),
        "borderline_high": float(summary["borderline_high"]),
        "metrics": summary["metrics"],
    }


def main() -> None:
    """Parse CLI arguments and run the script entry point."""
    args = build_parser().parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
