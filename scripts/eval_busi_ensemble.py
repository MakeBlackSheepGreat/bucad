"""Utility script for eval busi ensemble workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.busi import load_busi_manifest
from src.models.classifier import classifier_probabilities, load_classifier
from src.preprocess.io import read_image
from src.preprocess.transforms import prepare_classifier_input
from src.utils.config import load_project_config
from src.utils.metrics import best_threshold_by_youden, classification_metrics, threshold_sweep
from src.utils.reporting import write_json_report
from src.utils.runtime import optional_import


torch = optional_import("torch")


def _parse_member(value: str) -> dict[str, Any]:
    """Parse a model checkpoint member specification."""
    parts = value.split(":", 2)
    if len(parts) == 2:
        model_name, checkpoint = parts
        weight = 1.0
    elif len(parts) == 3:
        model_name, checkpoint, weight_text = parts
        weight = float(weight_text)
    else:
        raise argparse.ArgumentTypeError(
            "Members must use model:checkpoint or model:checkpoint:weight."
        )
    return {"model_name": model_name, "checkpoint": checkpoint, "weight": weight}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for this script."""
    parser = argparse.ArgumentParser(description="Evaluate a heterogeneous BUSI ensemble.")
    parser.add_argument("--config", required=True, help="Path to inference config YAML.")
    parser.add_argument(
        "--member",
        action="append",
        required=True,
        type=_parse_member,
        help="Ensemble member as model_name:checkpoint[:weight].",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    return parser


def _load_members(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load ensemble member checkpoints on CPU."""
    if torch is None:
        raise RuntimeError("Torch is required for ensemble evaluation.")
    members = []
    for entry in entries:
        model = load_classifier(
            {
                "name": entry["model_name"],
                "pretrained": False,
                "in_chans": 3,
                "num_classes": 2,
            },
            checkpoint_path=entry["checkpoint"],
            map_location="cpu",
        )
        model.eval()
        members.append({**entry, "model": model})
    return members


def evaluate_ensemble(config_path: str | Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate a heterogeneous classifier ensemble on BUSI."""
    config, paths = load_project_config(config_path)
    runtime = config.get("runtime", {})
    members = _load_members(entries)
    manifest = load_busi_manifest(paths.busi_root, include_normal=False)
    image_size = int(runtime.get("classifier_image_size", 224))
    apply_clahe = bool(runtime.get("classifier_apply_clahe", False))
    use_tta = bool(runtime.get("classifier_tta_horizontal_flip", False))

    y_true: list[int] = []
    malignant_probabilities: list[float] = []
    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        image = read_image(row.image_path, grayscale=True)
        input_tensor = prepare_classifier_input(
            image,
            image_size,
            apply_clahe_enabled=apply_clahe,
        )
        input_tensors = [input_tensor]
        if use_tta:
            input_tensors.append(input_tensor.flip(dims=[2]))
        weighted_probs = []
        weights = []
        for member in members:
            for tensor in input_tensors:
                weighted_probs.append(
                    classifier_probabilities(member["model"], tensor, device="cpu")[0]
                    * float(member["weight"])
                )
                weights.append(float(member["weight"]))
        probs = np.sum(np.asarray(weighted_probs, dtype=np.float32), axis=0) / float(sum(weights))
        malignant_probability = float(probs[1])
        label = 1 if row.pathology_label == "malignant" else 0
        y_true.append(label)
        malignant_probabilities.append(malignant_probability)
        rows.append(
            {
                "sample_id": row.sample_id,
                "pathology_label": row.pathology_label,
                "malignant_probability": malignant_probability,
            }
        )

    metrics = classification_metrics(y_true, malignant_probabilities)
    threshold_rows = threshold_sweep(y_true, malignant_probabilities)
    best_threshold = best_threshold_by_youden(y_true, malignant_probabilities)
    return {
        "sample_count": len(rows),
        "members": [
            {
                "model_name": member["model_name"],
                "checkpoint": member["checkpoint"],
                "weight": member["weight"],
            }
            for member in members
        ],
        "metrics": metrics,
        "threshold_analysis": {
            "best_by_youden": best_threshold,
            "rows": threshold_rows,
        },
        "rows": rows,
    }


def main() -> int:
    """Parse CLI arguments and run the script entry point."""
    args = build_parser().parse_args()
    report = evaluate_ensemble(args.config, args.member)
    if args.output:
        write_json_report(args.output, report)
    print(report["metrics"])
    print({"best_threshold_by_youden": report["threshold_analysis"]["best_by_youden"]["threshold"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
