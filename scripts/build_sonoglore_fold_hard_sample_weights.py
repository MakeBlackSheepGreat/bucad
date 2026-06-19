"""Build fold-specific BUSBRA hard-sample weights for one classifier checkpoint."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.train_cls import (
    _collect_validation_probabilities,
    _prepare_classifier_training_run,
)
from src.utils.config import load_project_config
from src.utils.metrics import classification_metrics
from src.utils.runtime import require_dependency, seed_everything, select_device


torch = None


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for fold-specific hard-sample weight generation."""
    parser = argparse.ArgumentParser(
        description="Build one BUSBRA validation-fold hard-sample weight CSV from a trained classifier checkpoint."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Classifier training config used to reconstruct the validation fold.",
    )
    parser.add_argument("--fold", type=int, default=1, help="Fold id used to define the validation split.")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint path to evaluate. Defaults to the checkpoint name implied by the config and fold.",
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
        help="Benign samples above this malignant probability receive the strongest FP weight.",
    )
    parser.add_argument(
        "--malignant-fn-threshold",
        type=float,
        default=0.15,
        help="Malignant samples below this malignant probability receive the strongest FN weight.",
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
        help="Mild weight assigned to borderline-but-correct samples.",
    )
    parser.add_argument(
        "--benign-fp-weight",
        type=float,
        default=1.30,
        help="Weight assigned to benign false positives.",
    )
    parser.add_argument(
        "--malignant-fn-weight",
        type=float,
        default=1.25,
        help="Weight assigned to malignant false negatives.",
    )
    parser.add_argument(
        "--extreme-bonus",
        type=float,
        default=0.15,
        help="Additional weight added to extreme-confidence errors beyond the strong-threshold cutoffs.",
    )
    parser.add_argument(
        "--weight-mode",
        choices=("all", "fn_only", "fn_extreme_only"),
        default="all",
        help=(
            "Weighting strategy. "
            "'all' preserves the original FP/FN/borderline behavior, "
            "'fn_only' weights only malignant false negatives, "
            "'fn_extreme_only' weights only extreme-confidence malignant false negatives."
        ),
    )
    return parser


def _resolve_checkpoint_path(project_root: Path, config: dict[str, Any], fold: int, checkpoint: str | Path | None) -> Path:
    """Resolve the checkpoint path implied by either CLI input or the config output section."""
    if checkpoint:
        return Path(checkpoint)
    checkpoint_name = str(config["output"]["checkpoint_name"]).format(fold=fold)
    return project_root / "artifacts" / "checkpoints" / checkpoint_name


def _prediction_rows(
    *,
    val_manifest: pd.DataFrame,
    malignant_probabilities: list[float],
    weight_mode: str = "all",
    decision_threshold: float,
    benign_fp_threshold: float,
    malignant_fn_threshold: float,
    borderline_low: float,
    borderline_high: float,
    benign_fp_weight: float,
    malignant_fn_weight: float,
    borderline_weight: float,
    extreme_bonus: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build per-sample weight rows and a compact summary from validation predictions."""
    valid_weight_modes = {"all", "fn_only", "fn_extreme_only"}
    if weight_mode not in valid_weight_modes:
        raise ValueError(f"Unsupported weight_mode: {weight_mode!r}")
    rows: list[dict[str, Any]] = []
    counts = {
        "weighted_count": 0,
        "fp_count": 0,
        "fn_count": 0,
        "borderline_count": 0,
        "extreme_fp_count": 0,
        "extreme_fn_count": 0,
    }
    for item, malignant_probability in zip(val_manifest.itertuples(index=False), malignant_probabilities):
        label = str(item.pathology_label).lower()
        probability = float(malignant_probability)
        predicted_label = "malignant" if probability >= decision_threshold else "benign"
        weight = 1.0
        error_type = "correct_or_unweighted"
        mechanism = ""
        is_benign_fp = label == "benign" and predicted_label == "malignant"
        is_malignant_fn = label == "malignant" and predicted_label == "benign"
        is_borderline = borderline_low <= probability <= borderline_high
        if is_benign_fp:
            error_type = "FP"
            counts["fp_count"] += 1
            if probability >= benign_fp_threshold:
                counts["extreme_fp_count"] += 1
            if weight_mode == "all":
                mechanism = "benign_fp"
                weight = float(benign_fp_weight)
                if probability >= benign_fp_threshold:
                    weight += float(extreme_bonus)
                    mechanism = "benign_fp_extreme"
        elif is_malignant_fn:
            error_type = "FN"
            counts["fn_count"] += 1
            is_extreme_fn = probability <= malignant_fn_threshold
            if weight_mode in {"all", "fn_only"}:
                mechanism = "malignant_fn"
                weight = float(malignant_fn_weight)
            if is_extreme_fn:
                counts["extreme_fn_count"] += 1
                if weight_mode in {"all", "fn_only"}:
                    mechanism = "malignant_fn_extreme"
                    weight += float(extreme_bonus)
                elif weight_mode == "fn_extreme_only":
                    mechanism = "malignant_fn_extreme"
                    weight = float(malignant_fn_weight) + float(extreme_bonus)
            elif weight_mode == "fn_extreme_only":
                mechanism = ""
                weight = 1.0
        elif is_borderline:
            error_type = "borderline"
            counts["borderline_count"] += 1
            if weight_mode == "all":
                mechanism = "borderline_correct"
                weight = float(borderline_weight)
        if weight > 1.0:
            counts["weighted_count"] += 1
        rows.append(
            {
                "sample_id": str(item.sample_id),
                "case_id": str(item.case_id),
                "pathology_label": label,
                "malignant_probability": probability,
                "predicted_label": predicted_label,
                "sample_weight": round(weight, 6),
                "error_type": error_type,
                "mechanism": mechanism,
            }
        )
    metrics = classification_metrics(
        [1 if str(row.pathology_label).lower() == "malignant" else 0 for row in val_manifest.itertuples(index=False)],
        malignant_probabilities,
        threshold=float(decision_threshold),
    )
    return rows, {
        **counts,
        "metrics": metrics,
        "weight_mode": weight_mode,
        "decision_threshold": float(decision_threshold),
        "benign_fp_threshold": float(benign_fp_threshold),
        "malignant_fn_threshold": float(malignant_fn_threshold),
        "borderline_low": float(borderline_low),
        "borderline_high": float(borderline_high),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Generate one fold-specific hard-sample CSV and return a summary payload."""
    global torch
    torch = __import__("torch")
    require_dependency("torch", torch)

    config, paths = load_project_config(args.config)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = select_device(str(config.get("device", "auto")))
    prepared = _prepare_classifier_training_run(
        config=config,
        paths=paths,
        fold=int(args.fold),
        seed=seed,
        device=device,
        epochs_override=None,
    )
    checkpoint_path = _resolve_checkpoint_path(
        paths.project_root,
        config,
        int(args.fold),
        args.checkpoint,
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state.get("state_dict", state.get("model", state))
    prepared.model.load_state_dict(state_dict, strict=False)
    prepared.model = prepared.model.to(device)
    y_true, malignant_probabilities = _collect_validation_probabilities(
        prepared.model,
        prepared.val_loader,
        device,
    )
    rows, summary = _prediction_rows(
        val_manifest=prepared.val_manifest,
        malignant_probabilities=malignant_probabilities,
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
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "case_id",
                "pathology_label",
                "malignant_probability",
                "predicted_label",
                "sample_weight",
                "error_type",
                "mechanism",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return {
        "output": str(output),
        "checkpoint": str(checkpoint_path),
        "fold": int(args.fold),
        "validation_size": int(len(prepared.val_manifest)),
        "weighted_count": int(summary["weighted_count"]),
        "fp_count": int(summary["fp_count"]),
        "fn_count": int(summary["fn_count"]),
        "borderline_count": int(summary["borderline_count"]),
        "extreme_fp_count": int(summary["extreme_fp_count"]),
        "extreme_fn_count": int(summary["extreme_fn_count"]),
        "metrics": summary["metrics"],
        "weight_mode": summary["weight_mode"],
        "decision_threshold": summary["decision_threshold"],
        "benign_fp_threshold": summary["benign_fp_threshold"],
        "malignant_fn_threshold": summary["malignant_fn_threshold"],
        "borderline_low": summary["borderline_low"],
        "borderline_high": summary["borderline_high"],
    }


def main() -> None:
    """Run the CLI entry point."""
    args = build_parser().parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
