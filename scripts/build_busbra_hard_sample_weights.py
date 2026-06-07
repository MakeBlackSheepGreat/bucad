"""Utility script for busbra hard sample weights workflows."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.busbra import load_busbra_manifest
from src.utils.config import load_project_config


BALANCED_FN_WEIGHTS = {
    "full_image_already_benign": 3.0,
    "full_image_error_area_gate_fallback": 2.5,
    "roi_stacker_pulled_to_benign": 2.5,
}
BALANCED_FP_WEIGHTS = {
    "roi_stacker_pushed_to_malignant": 2.0,
    "full_image_already_malignant": 2.2,
    "full_image_error_area_gate_fallback": 2.2,
}
FN_MILD_FN_WEIGHTS = {
    "full_image_already_benign": 1.8,
    "full_image_error_area_gate_fallback": 1.5,
    "roi_stacker_pulled_to_benign": 1.5,
}
FN_MILD_FP_WEIGHTS = {
    "roi_stacker_pushed_to_malignant": 1.15,
    "full_image_already_malignant": 1.2,
    "full_image_error_area_gate_fallback": 1.2,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="Build BUSBRA hard-sample weights from internal OOF error analysis."
    )
    parser.add_argument("--config", default="configs/inference/demo.yml")
    parser.add_argument(
        "--error-csv",
        default="artifacts/reports/busbra_oof_demo_error_analysis.csv",
    )
    parser.add_argument(
        "--output",
        default="artifacts/reports/busbra_oof_hard_sample_weights.csv",
    )
    parser.add_argument(
        "--profile",
        choices=["balanced", "fn_mild"],
        default="balanced",
        help="Weight profile. balanced targets all OOF errors; fn_mild lightly emphasizes false negatives.",
    )
    return parser


def _profile_weights(profile: str) -> tuple[dict[str, float], dict[str, float]]:
    """Return hard-sample weights for the selected profile."""
    if profile == "fn_mild":
        return FN_MILD_FN_WEIGHTS, FN_MILD_FP_WEIGHTS
    return BALANCED_FN_WEIGHTS, BALANCED_FP_WEIGHTS


def _load_error_weights(path: str | Path, *, profile: str) -> dict[str, dict[str, Any]]:
    """Load error weights."""
    fn_weights, fp_weights = _profile_weights(profile)
    weights: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = str(row["sample_id"])
            error_type = str(row["error_type"])
            mechanism = str(row["mechanism"])
            if error_type == "FN":
                weight = fn_weights.get(mechanism, 1.5 if profile == "fn_mild" else 2.5)
            elif error_type == "FP":
                weight = fp_weights.get(mechanism, 1.15 if profile == "fn_mild" else 2.0)
            else:
                weight = 1.0
            weights[sample_id] = {
                "sample_weight": float(weight),
                "error_type": error_type,
                "mechanism": mechanism,
                "final_probability": row.get("final_probability", ""),
                "confidence_band": row.get("confidence_band", ""),
            }
    return weights


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the experiment workflow and return the generated summary."""
    _, paths = load_project_config(args.config)
    manifest = load_busbra_manifest(paths.busbra_root)
    hard_weights = _load_error_weights(args.error_csv, profile=str(args.profile))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"weighted": 0, "fn": 0, "fp": 0}
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "sample_id",
            "case_id",
            "pathology_label",
            "sample_weight",
            "error_type",
            "mechanism",
            "confidence_band",
            "final_probability",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest.itertuples(index=False):
            extra = hard_weights.get(str(row.sample_id), {})
            weight = float(extra.get("sample_weight", 1.0))
            if weight > 1.0:
                counts["weighted"] += 1
            if extra.get("error_type") == "FN":
                counts["fn"] += 1
            if extra.get("error_type") == "FP":
                counts["fp"] += 1
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "case_id": row.case_id,
                    "pathology_label": row.pathology_label,
                    "sample_weight": weight,
                    "error_type": extra.get("error_type", "correct_or_unweighted"),
                    "mechanism": extra.get("mechanism", ""),
                    "confidence_band": extra.get("confidence_band", ""),
                    "final_probability": extra.get("final_probability", ""),
                }
            )
    return {
        "output": str(output),
        "profile": str(args.profile),
        "sample_count": int(len(manifest)),
        **counts,
    }


def main() -> None:
    """Parse CLI arguments and run the script entry point."""
    args = build_parser().parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
