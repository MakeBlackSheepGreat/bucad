"""Utility script for seed diversity oof protocol workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import confusion_matrix, roc_auc_score

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_oof_stacking import (  # noqa: E402
    ModelView,
    _apply_tta,
    _load_or_create_splits,
    _load_view_model,
    _predict_view_for_manifest,
    _resolve_project_path,
    _val_manifest_for_fold,
)
from scripts.run_roi_area_gate_oof_protocol import _roi_images_and_areas  # noqa: E402
from src.datasets.busbra import load_busbra_manifest  # noqa: E402
from src.models.classifier import classifier_probabilities  # noqa: E402
from src.preprocess.transforms import prepare_classifier_input  # noqa: E402
from src.utils.config import load_project_config  # noqa: E402
from src.utils.reporting import write_json_report, write_markdown_report  # noqa: E402
from src.utils.runtime import require_dependency, select_device  # noqa: E402
from src.utils.runtime import optional_import  # noqa: E402


torch = optional_import("torch")

EFF_WEIGHT = 0.427
CONV_WEIGHT = 0.573
BASE_CONV_VIEW = "conv_crop_sweep"
SEED123_VIEW = ModelView(
    name="conv_seed123_crop_sweep",
    model_name="convnext_tiny",
    checkpoint_template="./artifacts/checkpoints/convnext_tiny_timm_recipe_seed123_fold{fold}.pt",
    image_size=224,
    apply_clahe=True,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    interpolation="bicubic",
    crop_pct=0.95,
    tta_variants=(
        {"name": "identity", "crop_pct": 0.90},
        {"name": "hflip", "crop_pct": 0.90},
        {"name": "identity", "crop_pct": 0.95},
        {"name": "hflip", "crop_pct": 0.95},
        {"name": "identity", "crop_pct": 1.00},
        {"name": "hflip", "crop_pct": 1.00},
    ),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="Select a ConvNeXt-Tiny seed-diversity candidate using BUSBRA OOF only."
    )
    parser.add_argument("--config", default="configs/classifier/convnext_tiny_timm_recipe.yml")
    parser.add_argument("--runtime-config", default="configs/inference/demo.yml")
    parser.add_argument("--full-oof-cache", default="artifacts/reports/oof_two_model_predictions.json")
    parser.add_argument(
        "--roi-oof-cache",
        default="artifacts/reports/roi_oof_lcc_mask04_protocol_predictions.json",
    )
    parser.add_argument(
        "--seed123-full-cache",
        default="artifacts/reports/oof_convnext_tiny_seed123_predictions.json",
    )
    parser.add_argument(
        "--seed123-roi-cache",
        default="artifacts/reports/roi_oof_convnext_tiny_seed123_lcc_mask04_predictions.json",
    )
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--margin-ratio", type=float, default=0.35)
    parser.add_argument("--mask-threshold", type=float, default=0.4)
    parser.add_argument("--min-sensitivity", type=float, default=0.84)
    parser.add_argument("--max-auc-drop", type=float, default=0.002)
    parser.add_argument("--output", default="artifacts/reports/seed_diversity_oof_protocol.json")
    parser.add_argument("--markdown", default="artifacts/reports/seed_diversity_oof_protocol.md")
    parser.add_argument(
        "--candidate-config",
        default="configs/inference/demo_seed_diversity_oof_candidate.yml",
    )
    return parser


def _load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON report and validate its top-level object."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities with the sigmoid transform."""
    return 1.0 / (1.0 + np.exp(-values))


def _logit(probabilities: np.ndarray) -> np.ndarray:
    """Convert probabilities to clipped logits."""
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    """Calculate classification metrics for a probability vector."""
    predictions = (probabilities >= float(threshold)).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else 0.0
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    npv = float(tn / (tn + fn)) if tn + fn else 0.0
    accuracy = float((tp + tn) / (tp + tn + fp + fn)) if tp + tn + fp + fn else 0.0
    f1_score = (
        float(2.0 * precision * sensitivity / (precision + sensitivity))
        if precision + sensitivity
        else 0.0
    )
    return {
        "auc": float(roc_auc_score(y_true, probabilities)),
        "threshold": float(threshold),
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1_score": f1_score,
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "youden_j": float(sensitivity + specificity - 1.0),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "fnr": float(fn / (fn + tp)) if fn + tp else 0.0,
        "sample_count": int(len(y_true)),
        "positive_count": int(np.asarray(y_true).sum()),
        "negative_count": int(len(y_true) - np.asarray(y_true).sum()),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _best_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    min_sensitivity: float,
) -> dict[str, Any]:
    """Find the threshold with the best Youden score."""
    rows = [
        _metrics(y_true, probabilities, float(threshold))
        for threshold in np.round(np.arange(0.1, 0.9001, 0.01), 2)
    ]
    feasible = [row for row in rows if row["sensitivity"] >= min_sensitivity] or rows
    return max(
        feasible,
        key=lambda row: (
            row["f1_score"],
            row["precision"],
            row["specificity"],
            row["auc"],
            row["sensitivity"],
        ),
    )


def _array_from_rows(rows: list[dict[str, Any]]) -> np.ndarray:
    """Return a NumPy probability vector from report rows."""
    return np.asarray([float(row["malignant_probability"]) for row in rows], dtype=np.float64)


def _validate_same_order(*row_sets: list[dict[str, Any]]) -> list[str]:
    """Validate same order."""
    reference = [str(row["sample_id"]) for row in row_sets[0]]
    for rows in row_sets[1:]:
        current = [str(row["sample_id"]) for row in rows]
        if current != reference:
            raise ValueError("OOF sample order mismatch.")
    return reference


def _load_labels(rows: list[dict[str, Any]]) -> np.ndarray:
    """Load labels."""
    return np.asarray(
        [1 if str(row["pathology_label"]).lower() == "malignant" else 0 for row in rows],
        dtype=np.int32,
    )


def _load_or_generate_seed123_full(args: argparse.Namespace) -> dict[str, Any]:
    """Load or generate seed123 full."""
    path = Path(args.seed123_full_cache)
    if path.exists():
        return _load_json(path)
    config, paths = load_project_config(args.config)
    manifest = load_busbra_manifest(paths.busbra_root)
    split_path = Path(
        config.get("training", {}).get(
            "split_path",
            paths.reports_root / "busbra_5fold_splits.csv",
        )
    )
    if not split_path.is_absolute():
        split_path = (paths.project_root / split_path).resolve()
    assignments = _load_or_create_splits(
        manifest,
        split_path,
        fold_count=int(args.fold_count),
        seed=int(config.get("seed", 42)),
    )
    resolved_device = select_device(str(args.device))
    rows: list[dict[str, Any]] = []
    for fold in range(1, int(args.fold_count) + 1):
        val_manifest = _val_manifest_for_fold(manifest, assignments, fold)
        rows.extend(
            _predict_view_for_manifest(
                SEED123_VIEW,
                val_manifest,
                fold=fold,
                project_root=paths.project_root,
                device=resolved_device,
                batch_size=int(args.batch_size),
            )
        )
    report = {
        "source": "BUSBRA OOF full-image ConvNeXt-Tiny seed123 crop-sweep predictions",
        "fold_count": int(args.fold_count),
        "sample_count": len(rows),
        "views": {SEED123_VIEW.name: sorted(rows, key=lambda row: str(row["sample_id"]))},
    }
    write_json_report(path, report)
    return report


def _predict_model_view_on_images(
    view: ModelView,
    images: list[np.ndarray],
    *,
    fold: int,
    project_root: Path,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Predict model view on images."""
    require_dependency("torch", torch)
    checkpoint_path = _resolve_project_path(
        project_root,
        view.checkpoint_template.format(fold=fold),
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {view.name} fold {fold}: {checkpoint_path}")
    model = _load_view_model(view, checkpoint_path)
    model.eval()
    probabilities: list[float] = []
    for start in range(0, len(images), batch_size):
        subset = images[start:start + batch_size]
        tensors = []
        for image in subset:
            for variant in view.tta_variants:
                tensors.append(
                    prepare_classifier_input(
                        _apply_tta(image, variant),
                        view.image_size,
                        apply_clahe_enabled=view.apply_clahe,
                        mean=view.mean,
                        std=view.std,
                        interpolation=view.interpolation,
                        crop_pct=float(variant.get("crop_pct", view.crop_pct)),
                    )
                )
        batch = torch.stack(tensors)
        probs = classifier_probabilities(model, batch, device=device)
        probs = probs.reshape(len(subset), len(view.tta_variants), 2).mean(axis=1)
        probabilities.extend(float(value) for value in probs[:, 1])
    return np.asarray(probabilities, dtype=np.float64)


def _load_or_generate_seed123_roi(args: argparse.Namespace) -> dict[str, Any]:
    """Load or generate seed123 roi."""
    path = Path(args.seed123_roi_cache)
    if path.exists():
        return _load_json(path)
    config, paths = load_project_config(args.config)
    manifest = load_busbra_manifest(paths.busbra_root)
    split_path = Path(
        config.get("training", {}).get(
            "split_path",
            paths.reports_root / "busbra_5fold_splits.csv",
        )
    )
    if not split_path.is_absolute():
        split_path = (paths.project_root / split_path).resolve()
    assignments = _load_or_create_splits(
        manifest,
        split_path,
        fold_count=int(args.fold_count),
        seed=int(config.get("seed", 42)),
    )
    resolved_device = select_device(str(args.device))
    rows: list[dict[str, Any]] = []
    area_by_sample: dict[str, float] = {}
    for fold in range(1, int(args.fold_count) + 1):
        val_manifest = _val_manifest_for_fold(manifest, assignments, fold)
        roi_images, area_ratios = _roi_images_and_areas(
            val_manifest,
            margin_ratio=float(args.margin_ratio),
            mask_threshold=float(args.mask_threshold),
        )
        probabilities = _predict_model_view_on_images(
            SEED123_VIEW,
            roi_images,
            fold=fold,
            project_root=paths.project_root,
            device=resolved_device,
            batch_size=int(args.batch_size),
        )
        for row, probability, area_ratio in zip(
            val_manifest.itertuples(index=False),
            probabilities,
            area_ratios,
        ):
            rows.append(
                {
                    "sample_id": row.sample_id,
                    "case_id": row.case_id,
                    "fold_id": int(fold),
                    "pathology_label": row.pathology_label,
                    "malignant_probability": float(probability),
                }
            )
            area_by_sample[str(row.sample_id)] = float(area_ratio)
    rows = sorted(rows, key=lambda row: str(row["sample_id"]))
    report = {
        "source": "BUSBRA GT-mask ROI ConvNeXt-Tiny seed123 crop-sweep OOF predictions",
        "fold_count": int(args.fold_count),
        "sample_count": len(rows),
        "margin_ratio": float(args.margin_ratio),
        "mask_threshold": float(args.mask_threshold),
        "largest_component": True,
        "views": {SEED123_VIEW.name: rows},
        "roi_area_ratios": [float(area_by_sample[str(row["sample_id"])]) for row in rows],
    }
    write_json_report(path, report)
    return report


def _runtime_stack(
    full_probabilities: np.ndarray,
    roi_probabilities: np.ndarray,
    runtime_config: dict[str, Any],
) -> np.ndarray:
    """Apply runtime stacker settings to paired probabilities."""
    stacker = runtime_config["runtime"]["roi_enhancement"]["stacker"]
    matrix = np.vstack([_logit(full_probabilities), _logit(roi_probabilities)]).T
    scaled = (
        matrix - np.asarray(stacker["scaler_mean"], dtype=np.float64)
    ) / np.asarray(stacker["scaler_scale"], dtype=np.float64)
    return _sigmoid(
        scaled @ np.asarray(stacker["coef"], dtype=np.float64)
        + float(stacker["intercept"])
    )


def _candidate_probability(
    *,
    eff_full: np.ndarray,
    conv42_full: np.ndarray,
    conv123_full: np.ndarray,
    eff_roi: np.ndarray,
    conv42_roi: np.ndarray,
    conv123_roi: np.ndarray,
    area_ratios: np.ndarray,
    runtime_config: dict[str, Any],
    seed123_share: float,
    roi_stack_blend_weight: float,
) -> np.ndarray:
    """Return candidate probability."""
    conv_full = (1.0 - seed123_share) * conv42_full + seed123_share * conv123_full
    conv_roi = (1.0 - seed123_share) * conv42_roi + seed123_share * conv123_roi
    full_probability = EFF_WEIGHT * eff_full + CONV_WEIGHT * conv_full
    roi_probability = EFF_WEIGHT * eff_roi + CONV_WEIGHT * conv_roi
    stack_probability = _runtime_stack(full_probability, roi_probability, runtime_config)
    blend_weight = float(np.clip(roi_stack_blend_weight, 0.0, 1.0))
    roi_probability = blend_weight * stack_probability + (1.0 - blend_weight) * full_probability

    gate = runtime_config["runtime"]["roi_enhancement"].get("quality_gate", {})
    min_area = float(gate.get("min_area_ratio", 0.08))
    max_area = float(gate.get("max_area_ratio", 0.75))
    fallback = (area_ratios < min_area) | (area_ratios > max_area)
    return np.where(fallback, full_probability, roi_probability)


def _scan_candidates(
    *,
    y_true: np.ndarray,
    eff_full: np.ndarray,
    conv42_full: np.ndarray,
    conv123_full: np.ndarray,
    eff_roi: np.ndarray,
    conv42_roi: np.ndarray,
    conv123_roi: np.ndarray,
    area_ratios: np.ndarray,
    runtime_config: dict[str, Any],
    baseline: dict[str, Any],
    min_sensitivity: float,
    max_auc_drop: float,
) -> list[dict[str, Any]]:
    """Scan candidates."""
    results: list[dict[str, Any]] = []
    for seed123_share in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.67, 1.0):
        for blend_weight in (0.75, 0.85, 0.95, 1.0):
            probabilities = _candidate_probability(
                eff_full=eff_full,
                conv42_full=conv42_full,
                conv123_full=conv123_full,
                eff_roi=eff_roi,
                conv42_roi=conv42_roi,
                conv123_roi=conv123_roi,
                area_ratios=area_ratios,
                runtime_config=runtime_config,
                seed123_share=float(seed123_share),
                roi_stack_blend_weight=float(blend_weight),
            )
            metrics = _best_threshold(
                y_true,
                probabilities,
                min_sensitivity=min_sensitivity,
            )
            accepted = (
                metrics["auc"] >= baseline["auc"] - max_auc_drop
                and metrics["f1_score"] > baseline["f1_score"]
                and metrics["precision"] >= baseline["precision"]
            )
            results.append(
                {
                    "seed123_share": float(seed123_share),
                    "roi_stack_blend_weight": float(blend_weight),
                    "accepted_by_oof_protocol": bool(accepted),
                    "metrics": metrics,
                }
            )
    return sorted(
        results,
        key=lambda row: (
            row["accepted_by_oof_protocol"],
            row["metrics"]["auc"],
            row["metrics"]["f1_score"],
            row["metrics"]["precision"],
            row["metrics"]["sensitivity"],
        ),
        reverse=True,
    )


def _with_sample_counts(metric: dict[str, Any]) -> dict[str, Any]:
    """Return with sample counts."""
    confusion = metric["confusion"]
    enriched = dict(metric)
    enriched["sample_count"] = int(confusion["tn"] + confusion["fp"] + confusion["fn"] + confusion["tp"])
    enriched["positive_count"] = int(confusion["tp"] + confusion["fn"])
    enriched["negative_count"] = int(confusion["tn"] + confusion["fp"])
    return enriched


def _write_candidate_config(
    *,
    runtime_config_path: str | Path,
    destination: str | Path,
    selected: dict[str, Any],
) -> None:
    """Write candidate config."""
    config = yaml.safe_load(Path(runtime_config_path).read_text(encoding="utf-8"))
    runtime = config["runtime"]
    share = float(selected["seed123_share"])
    members: list[dict[str, Any]] = []
    for member in runtime["classifier_members"]:
        copied = dict(member)
        if copied.get("model") == "convnext_tiny":
            copied["weight"] = float(CONV_WEIGHT * (1.0 - share))
        members.append(copied)
    conv_template = None
    for member in runtime["classifier_members"]:
        if member.get("model") == "convnext_tiny":
            conv_template = dict(member)
            break
    if conv_template is None:
        raise ValueError("Runtime config does not contain a ConvNeXt-Tiny member.")
    for fold in range(1, 6):
        copied = dict(conv_template)
        copied["checkpoint"] = f"./artifacts/checkpoints/convnext_tiny_timm_recipe_seed123_fold{fold}.pt"
        copied["weight"] = float(CONV_WEIGHT * share)
        members.append(copied)
    runtime["classifier_members"] = members
    runtime["default_threshold"] = float(selected["metrics"]["threshold"])
    runtime["ensemble_display_name"] = "ConvNeXt-Tiny seed diversity + EfficientNetV2-S + ROI Area Gate"
    runtime["primary_classifier_model"] = "ConvNeXt-Tiny Seed Diversity"
    runtime["roi_enhancement"]["roi_stack_blend_weight"] = float(
        selected["roi_stack_blend_weight"]
    )
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _format_metric(metric: dict[str, Any], key: str) -> str:
    """Format one metric value for report tables."""
    return f"{float(metric[key]):.4f}"


def _confusion_text(metric: dict[str, Any]) -> str:
    """Format confusion-matrix counts for report tables."""
    c = metric["confusion"]
    return f"TN {c['tn']} / FP {c['fp']} / FN {c['fn']} / TP {c['tp']}"


def _metrics_row(name: str, metric: dict[str, Any]) -> str:
    """Format one metric row for a Markdown report."""
    return (
        "| "
        + " | ".join(
            [
                name,
                str(metric["sample_count"]),
                str(metric["positive_count"]),
                str(metric["negative_count"]),
                _format_metric(metric, "auc"),
                f"{metric['threshold']:.3f}",
                _format_metric(metric, "accuracy"),
                _format_metric(metric, "sensitivity"),
                _format_metric(metric, "specificity"),
                _format_metric(metric, "precision"),
                _format_metric(metric, "f1_score"),
                _format_metric(metric, "balanced_accuracy"),
                _format_metric(metric, "youden_j"),
                _confusion_text(metric),
            ]
        )
        + " |"
    )


def build_markdown(report: dict[str, Any]) -> list[str]:
    """Build the Markdown report body for this experiment."""
    selected = report["selected_candidate"]
    lines = [
        "# ConvNeXt-Tiny Seed Diversity OOF Protocol",
        "",
        "## Boundary",
        "",
        "- Candidate training and selection used BUSBRA training/OOF data only.",
        "- BUSI is not read by this protocol.",
        "- The search is intentionally narrow: split the existing ConvNeXt-Tiny family weight between seed42 and seed123, while preserving the EfficientNetV2-S weight and current ROI area gate.",
        "",
        "## Selected Candidate",
        "",
        f"- seed123_share: `{selected['seed123_share']:.2f}`",
        f"- roi_stack_blend_weight: `{selected['roi_stack_blend_weight']:.2f}`",
        f"- threshold: `{selected['metrics']['threshold']:.3f}`",
        "",
        "## OOF Metrics",
        "",
        "| Scheme | Samples | Pos | Neg | AUC | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1 | Balanced Acc | Youden J | Confusion |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        _metrics_row("Current demo OOF", report["baseline_oof_metrics"]),
        _metrics_row("Current demo OOF best threshold", report["baseline_oof_best_threshold_metrics"]),
        _metrics_row("Seed diversity selected OOF", selected["metrics"]),
        "",
        "## Top OOF Candidates",
        "",
        "| Rank | seed123_share | ROI blend | Accepted | AUC | Threshold | Sensitivity | Specificity | Precision | F1 |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(report["top_candidates"], start=1):
        metric = row["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"{row['seed123_share']:.2f}",
                    f"{row['roi_stack_blend_weight']:.2f}",
                    str(row["accepted_by_oof_protocol"]),
                    _format_metric(metric, "auc"),
                    f"{metric['threshold']:.3f}",
                    _format_metric(metric, "sensitivity"),
                    _format_metric(metric, "specificity"),
                    _format_metric(metric, "precision"),
                    _format_metric(metric, "f1_score"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Candidate config: `{report['candidate_config']}`",
            "- Run BUSI external validation only after treating this config as frozen.",
        ]
    )
    return lines


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the experiment workflow and return the generated summary."""
    runtime_config = yaml.safe_load(Path(args.runtime_config).read_text(encoding="utf-8"))
    full_oof = _load_json(args.full_oof_cache)
    roi_oof = _load_json(args.roi_oof_cache)
    seed_full = _load_or_generate_seed123_full(args)
    seed_roi = _load_or_generate_seed123_roi(args)

    eff_full_rows = full_oof["views"]["eff_identity"]
    conv42_full_rows = full_oof["views"][BASE_CONV_VIEW]
    conv123_full_rows = seed_full["views"][SEED123_VIEW.name]
    eff_roi_rows = roi_oof["views"]["eff_identity"]
    conv42_roi_rows = roi_oof["views"][BASE_CONV_VIEW]
    conv123_roi_rows = seed_roi["views"][SEED123_VIEW.name]
    _validate_same_order(
        eff_full_rows,
        conv42_full_rows,
        conv123_full_rows,
        eff_roi_rows,
        conv42_roi_rows,
        conv123_roi_rows,
    )
    y_true = _load_labels(eff_full_rows)
    area_ratios = np.asarray(roi_oof["roi_area_ratios"], dtype=np.float64)
    if len(area_ratios) != len(y_true):
        raise ValueError("ROI area ratio length mismatch.")

    eff_full = _array_from_rows(eff_full_rows)
    conv42_full = _array_from_rows(conv42_full_rows)
    conv123_full = _array_from_rows(conv123_full_rows)
    eff_roi = _array_from_rows(eff_roi_rows)
    conv42_roi = _array_from_rows(conv42_roi_rows)
    conv123_roi = _array_from_rows(conv123_roi_rows)

    baseline_probability = _candidate_probability(
        eff_full=eff_full,
        conv42_full=conv42_full,
        conv123_full=conv123_full,
        eff_roi=eff_roi,
        conv42_roi=conv42_roi,
        conv123_roi=conv123_roi,
        area_ratios=area_ratios,
        runtime_config=runtime_config,
        seed123_share=0.0,
        roi_stack_blend_weight=float(
            runtime_config["runtime"]["roi_enhancement"].get("roi_stack_blend_weight", 1.0)
        ),
    )
    baseline_default = _metrics(
        y_true,
        baseline_probability,
        threshold=float(runtime_config["runtime"].get("default_threshold", 0.5)),
    )
    baseline_best = _best_threshold(
        y_true,
        baseline_probability,
        min_sensitivity=float(args.min_sensitivity),
    )

    candidates = _scan_candidates(
        y_true=y_true,
        eff_full=eff_full,
        conv42_full=conv42_full,
        conv123_full=conv123_full,
        eff_roi=eff_roi,
        conv42_roi=conv42_roi,
        conv123_roi=conv123_roi,
        area_ratios=area_ratios,
        runtime_config=runtime_config,
        baseline=baseline_best,
        min_sensitivity=float(args.min_sensitivity),
        max_auc_drop=float(args.max_auc_drop),
    )
    selected = next((row for row in candidates if row["accepted_by_oof_protocol"]), candidates[0])
    _write_candidate_config(
        runtime_config_path=args.runtime_config,
        destination=args.candidate_config,
        selected=selected,
    )
    report = {
        "method": "ConvNeXt-Tiny seed diversity selected on BUSBRA OOF only",
        "data_boundary": {
            "selection": "BUSBRA OOF only",
            "external_review": "Run BUSI separately after candidate config is frozen.",
        },
        "baseline_oof_metrics": _with_sample_counts(baseline_default),
        "baseline_oof_best_threshold_metrics": _with_sample_counts(baseline_best),
        "selected_candidate": selected,
        "top_candidates": candidates[:10],
        "candidate_config": str(args.candidate_config),
    }
    write_json_report(args.output, report)
    write_markdown_report(args.markdown, build_markdown(report))
    return report


def main() -> int:
    """Parse CLI arguments and run the script entry point."""
    args = build_parser().parse_args()
    report = run(args)
    selected = report["selected_candidate"]
    print(
        {
            "candidate_config": report["candidate_config"],
            "seed123_share": selected["seed123_share"],
            "roi_stack_blend_weight": selected["roi_stack_blend_weight"],
            "oof_auc": selected["metrics"]["auc"],
            "oof_threshold": selected["metrics"]["threshold"],
            "oof_sensitivity": selected["metrics"]["sensitivity"],
            "oof_specificity": selected["metrics"]["specificity"],
            "oof_precision": selected["metrics"]["precision"],
            "oof_f1": selected["metrics"]["f1_score"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
