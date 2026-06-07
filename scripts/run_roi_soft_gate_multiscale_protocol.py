"""Utility script for roi soft gate multiscale protocol workflows."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import confusion_matrix, roc_auc_score

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_roi_area_gate_oof_protocol import _generate_roi_oof_cache  # noqa: E402
from scripts.run_roi_oof_experiment import _predict_segmenter_mask, _predict_view_on_images  # noqa: E402
from src.datasets.busi import load_busi_manifest  # noqa: E402
from src.models.segmenter import load_segmenter  # noqa: E402
from src.preprocess.io import read_image  # noqa: E402
from src.preprocess.roi import crop_to_mask_bbox  # noqa: E402
from src.utils.config import load_project_config  # noqa: E402
from src.utils.reporting import write_json_report, write_markdown_report  # noqa: E402
from src.utils.runtime import select_device  # noqa: E402


PAIR_VIEWS = ("eff_identity", "conv_crop_sweep")
PAIR_WEIGHTS = {"eff_identity": 0.427, "conv_crop_sweep": 0.573}
MARGIN_PRESETS = (0.20, 0.35, 0.50)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="Select ROI soft gate and multi-scale ROI crop parameters from BUSBRA OOF."
    )
    parser.add_argument("--config", default="configs/classifier/convnext_tiny_timm_recipe.yml")
    parser.add_argument("--runtime-config", default="configs/inference/demo.yml")
    parser.add_argument("--full-oof-cache", default="artifacts/reports/oof_two_model_predictions.json")
    parser.add_argument(
        "--area-cache",
        default="artifacts/reports/roi_precision_f1_area_cache.json",
    )
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mask-threshold", type=float, default=0.4)
    parser.add_argument("--segmenter-checkpoint", default="artifacts/checkpoints/segmenter_fold1.pt")
    parser.add_argument("--min-sensitivity", type=float, default=0.84)
    parser.add_argument("--max-auc-drop", type=float, default=0.002)
    parser.add_argument("--output", default="artifacts/reports/roi_soft_gate_multiscale_protocol.json")
    parser.add_argument("--markdown", default="artifacts/reports/roi_soft_gate_multiscale_protocol.md")
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


def _normalise_metric(metric: dict[str, Any]) -> dict[str, Any]:
    """Normalize a metric value for comparison scoring."""
    confusion = metric["confusion"]
    sample_count = int(confusion["tn"] + confusion["fp"] + confusion["fn"] + confusion["tp"])
    positive_count = int(confusion["tp"] + confusion["fn"])
    negative_count = int(confusion["tn"] + confusion["fp"])
    enriched = dict(metric)
    enriched["sample_count"] = sample_count
    enriched["positive_count"] = positive_count
    enriched["negative_count"] = negative_count
    enriched["npv"] = float(confusion["tn"] / (confusion["tn"] + confusion["fn"])) if confusion["tn"] + confusion["fn"] else 0.0
    enriched["balanced_accuracy"] = float((metric["sensitivity"] + metric["specificity"]) / 2.0)
    enriched["youden_j"] = float(metric["sensitivity"] + metric["specificity"] - 1.0)
    enriched["fpr"] = float(confusion["fp"] / (confusion["fp"] + confusion["tn"])) if confusion["fp"] + confusion["tn"] else 0.0
    enriched["fnr"] = float(confusion["fn"] / (confusion["fn"] + confusion["tp"])) if confusion["fn"] + confusion["tp"] else 0.0
    return enriched


def _validate_order(views: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Validate order."""
    sample_ids = [str(row["sample_id"]) for row in views[PAIR_VIEWS[0]]]
    for view_name in PAIR_VIEWS[1:]:
        current = [str(row["sample_id"]) for row in views[view_name]]
        if current != sample_ids:
            raise ValueError(f"Sample order mismatch for {view_name}.")
    return sample_ids


def _view_array(views: dict[str, list[dict[str, Any]]], view_name: str) -> np.ndarray:
    """Return a named prediction view as a NumPy vector."""
    return np.asarray(
        [float(row["malignant_probability"]) for row in views[view_name]],
        dtype=np.float64,
    )


def _pair_blend(views: dict[str, list[dict[str, Any]]]) -> np.ndarray:
    """Blend paired probability vectors with configured weights."""
    _validate_order(views)
    return (
        PAIR_WEIGHTS["eff_identity"] * _view_array(views, "eff_identity")
        + PAIR_WEIGHTS["conv_crop_sweep"] * _view_array(views, "conv_crop_sweep")
    )


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


def _roi_oof_cache_path(margin_ratio: float) -> Path:
    """Process ROI oof cache path."""
    return Path(
        f"artifacts/reports/roi_oof_lcc_mask04_margin{int(round(margin_ratio * 100)):03d}_predictions.json"
    )


def _busi_roi_cache_path(margin_ratio: float) -> Path:
    """Return the BUSI ROI cache path for one margin setting."""
    return Path(
        f"artifacts/reports/busi_roi_lcc_mask04_margin{int(round(margin_ratio * 100)):03d}_predictions.json"
    )


def _load_or_generate_roi_oof_for_margin(
    *,
    args: argparse.Namespace,
    margin_ratio: float,
) -> dict[str, Any]:
    """Load or generate roi oof for margin."""
    if abs(margin_ratio - 0.35) < 1e-9:
        canonical = Path("artifacts/reports/roi_oof_lcc_mask04_protocol_predictions.json")
        if canonical.exists():
            return _load_json(canonical)
    path = _roi_oof_cache_path(margin_ratio)
    if path.exists():
        return _load_json(path)
    local_args = argparse.Namespace(
        config=args.config,
        fold_count=args.fold_count,
        batch_size=args.batch_size,
        device=args.device,
        margin_ratio=margin_ratio,
        mask_threshold=args.mask_threshold,
    )
    report = _generate_roi_oof_cache(local_args)
    write_json_report(path, report)
    return report


def _load_busi_full_views() -> tuple[list[dict[str, str]], np.ndarray, dict[str, list[dict[str, Any]]]]:
    """Load busi full views."""
    reports = {
        "eff_identity": "artifacts/reports/busi_efficientnetv2_s_5fold_identity.json",
        "conv_crop_sweep": "artifacts/reports/busi_convnext_tiny_tta_crop_sweep.json",
    }
    reference: list[dict[str, str]] | None = None
    y_true: np.ndarray | None = None
    views: dict[str, list[dict[str, Any]]] = {}
    for view_name, path in reports.items():
        data = _load_json(path)
        rows = data["rows"]
        current = [
            {"sample_id": str(row["sample_id"]), "pathology_label": str(row["pathology_label"])}
            for row in rows
        ]
        if reference is None:
            reference = current
            y_true = np.asarray(
                [1 if row["pathology_label"] == "malignant" else 0 for row in rows],
                dtype=np.int32,
            )
        elif current != reference:
            raise ValueError(f"BUSI full report row order mismatch: {path}")
        views[view_name] = [
            {
                "sample_id": row["sample_id"],
                "pathology_label": row["pathology_label"],
                "malignant_probability": float(row["malignant_probability"]),
            }
            for row in rows
        ]
    if reference is None or y_true is None:
        raise ValueError("No BUSI full reports loaded.")
    return reference, y_true, views


def _load_or_build_busi_masks(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Load or build busi masks."""
    _, paths = load_project_config(args.config)
    manifest = load_busi_manifest(paths.busi_root, include_normal=False)
    segmenter_checkpoint = Path(args.segmenter_checkpoint)
    if not segmenter_checkpoint.exists():
        raise FileNotFoundError(f"Segmenter checkpoint not found: {segmenter_checkpoint}")
    segmenter = load_segmenter(
        {
            "architecture": "unet",
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "in_channels": 3,
            "classes": 1,
        },
        checkpoint_path=segmenter_checkpoint,
        map_location="cpu",
    )
    segmenter.eval()
    reference: list[dict[str, str]] = []
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    y_true: list[int] = []
    for row in manifest.itertuples(index=False):
        image = read_image(row.image_path, grayscale=True)
        mask = _predict_segmenter_mask(segmenter, image)
        reference.append(
            {
                "sample_id": str(row.sample_id),
                "pathology_label": str(row.pathology_label),
            }
        )
        images.append(image)
        masks.append(mask)
        y_true.append(1 if row.pathology_label == "malignant" else 0)
    return reference, images, masks, np.asarray(y_true, dtype=np.int32)


def _load_or_generate_busi_roi_for_margin(
    *,
    args: argparse.Namespace,
    margin_ratio: float,
    reference: list[dict[str, str]],
    images: list[np.ndarray],
    masks: list[np.ndarray],
    y_true: np.ndarray,
) -> dict[str, Any]:
    """Load or generate busi roi for margin."""
    path = _busi_roi_cache_path(margin_ratio)
    if path.exists():
        return _load_json(path)
    _, paths = load_project_config(args.config)
    roi_images = [
        crop_to_mask_bbox(
            image,
            mask,
            threshold=float(args.mask_threshold),
            margin_ratio=float(margin_ratio),
            largest_component=True,
        )
        for image, mask in zip(images, masks)
    ]
    resolved_device = select_device(str(args.device))
    views: dict[str, list[dict[str, Any]]] = {}
    for view_name in PAIR_VIEWS:
        fold_probabilities: list[np.ndarray] = []
        for fold in range(1, 6):
            fold_probabilities.append(
                _predict_view_on_images(
                    view_name,
                    roi_images,
                    fold=fold,
                    project_root=paths.project_root,
                    device=resolved_device,
                    batch_size=int(args.batch_size),
                )
            )
        mean_probabilities = np.vstack(fold_probabilities).mean(axis=0)
        views[view_name] = [
            {
                "sample_id": row["sample_id"],
                "pathology_label": row["pathology_label"],
                "malignant_probability": float(probability),
            }
            for row, probability in zip(reference, mean_probabilities)
        ]
    report = {
        "source": "BUSI segmenter ROI pair-view probabilities with LCC mask threshold 0.40",
        "sample_count": len(reference),
        "margin_ratio": float(margin_ratio),
        "mask_threshold": float(args.mask_threshold),
        "largest_component": True,
        "views": views,
        "y_true": [int(value) for value in y_true],
    }
    write_json_report(path, report)
    return report


def _margin_weight_candidates(margins: tuple[float, ...]) -> list[tuple[float, ...]]:
    """Generate soft-gate margin and weight candidates."""
    if len(margins) == 1:
        return [(1.0,)]
    if len(margins) == 2:
        return [(0.5, 0.5), (0.35, 0.65), (0.65, 0.35)]
    return [
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        (0.25, 0.50, 0.25),
        (0.50, 0.30, 0.20),
        (0.20, 0.30, 0.50),
    ]


def _candidate_margin_sets() -> list[tuple[float, ...]]:
    """Generate candidate margin sets for soft-gate search."""
    margins = list(MARGIN_PRESETS)
    sets: list[tuple[float, ...]] = []
    for size in (1, 2, 3):
        for combo in itertools.combinations(margins, size):
            sets.append(tuple(combo))
    return sets


def _soft_gate_weight(
    area_ratios: np.ndarray,
    *,
    min_area_ratio: float,
    max_area_ratio: float,
    ramp_width: float,
    max_weight: float,
) -> np.ndarray:
    """Calculate the soft ROI gate weight for one area ratio."""
    if ramp_width <= 0:
        low = (area_ratios >= min_area_ratio).astype(np.float64)
        high = (area_ratios <= max_area_ratio).astype(np.float64)
    else:
        low = np.clip((area_ratios - min_area_ratio) / ramp_width, 0.0, 1.0)
        high = np.clip((max_area_ratio - area_ratios) / ramp_width, 0.0, 1.0)
    return float(max_weight) * low * high


def _weighted_average(probabilities: dict[float, np.ndarray], margins: tuple[float, ...], weights: tuple[float, ...]) -> np.ndarray:
    """Return the weighted average of probability vectors."""
    output = np.zeros_like(next(iter(probabilities.values())), dtype=np.float64)
    total_weight = 0.0
    for margin, weight in zip(margins, weights):
        output += float(weight) * probabilities[float(margin)]
        total_weight += float(weight)
    return output / max(total_weight, 1e-6)


def _search_candidates(
    *,
    y_true: np.ndarray,
    full_probabilities: np.ndarray,
    roi_probabilities_by_margin: dict[float, np.ndarray],
    area_ratios: np.ndarray,
    runtime_config: dict[str, Any],
    baseline: dict[str, Any],
    min_sensitivity: float,
    max_auc_drop: float,
) -> list[dict[str, Any]]:
    """Search candidates."""
    results: list[dict[str, Any]] = []
    min_values = [0.0, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
    max_values = [0.65, 0.75, 0.85, 0.95, 1.01]
    ramp_values = [0.03, 0.05, 0.08, 0.12, 0.20]
    max_weights = [0.50, 0.65, 0.80, 1.00]
    for margins in _candidate_margin_sets():
        for margin_weights in _margin_weight_candidates(margins):
            roi_probability = _weighted_average(roi_probabilities_by_margin, margins, margin_weights)
            stack_probability = _runtime_stack(full_probabilities, roi_probability, runtime_config)
            for min_area in min_values:
                for max_area in max_values:
                    if min_area >= max_area:
                        continue
                    for ramp_width in ramp_values:
                        for max_weight in max_weights:
                            gate_weight = _soft_gate_weight(
                                area_ratios,
                                min_area_ratio=float(min_area),
                                max_area_ratio=float(max_area),
                                ramp_width=float(ramp_width),
                                max_weight=float(max_weight),
                            )
                            probabilities = gate_weight * stack_probability + (1.0 - gate_weight) * full_probabilities
                            metrics = _best_threshold(
                                y_true,
                                probabilities,
                                min_sensitivity=min_sensitivity,
                            )
                            accepted = (
                                metrics["auc"] >= baseline["auc"] - max_auc_drop
                                and metrics["precision"] > baseline["precision"]
                                and metrics["f1_score"] > baseline["f1_score"]
                            )
                            results.append(
                                {
                                    "margins": [float(value) for value in margins],
                                    "margin_weights": [float(value) for value in margin_weights],
                                    "min_area_ratio": float(min_area),
                                    "max_area_ratio": float(max_area),
                                    "ramp_width": float(ramp_width),
                                    "max_weight": float(max_weight),
                                    "mean_gate_weight": float(np.mean(gate_weight)),
                                    "accepted_by_oof_protocol": bool(accepted),
                                    "metrics": metrics,
                                }
                            )
    return sorted(
        results,
        key=lambda row: (
            row["accepted_by_oof_protocol"],
            row["metrics"]["f1_score"],
            row["metrics"]["precision"],
            row["metrics"]["auc"],
            row["metrics"]["sensitivity"],
        ),
        reverse=True,
    )


def _apply_candidate(
    *,
    full_probabilities: np.ndarray,
    roi_probabilities_by_margin: dict[float, np.ndarray],
    area_ratios: np.ndarray,
    runtime_config: dict[str, Any],
    candidate: dict[str, Any],
) -> np.ndarray:
    """Apply candidate."""
    margins = tuple(float(value) for value in candidate["margins"])
    margin_weights = tuple(float(value) for value in candidate["margin_weights"])
    roi_probability = _weighted_average(roi_probabilities_by_margin, margins, margin_weights)
    stack_probability = _runtime_stack(full_probabilities, roi_probability, runtime_config)
    gate_weight = _soft_gate_weight(
        area_ratios,
        min_area_ratio=float(candidate["min_area_ratio"]),
        max_area_ratio=float(candidate["max_area_ratio"]),
        ramp_width=float(candidate["ramp_width"]),
        max_weight=float(candidate["max_weight"]),
    )
    return gate_weight * stack_probability + (1.0 - gate_weight) * full_probabilities


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
                _format_metric(metric, "recall"),
                _format_metric(metric, "specificity"),
                _format_metric(metric, "precision"),
                _format_metric(metric, "npv"),
                _format_metric(metric, "f1_score"),
                _format_metric(metric, "balanced_accuracy"),
                _format_metric(metric, "youden_j"),
                _format_metric(metric, "fpr"),
                _format_metric(metric, "fnr"),
                _confusion_text(metric),
            ]
        )
        + " |"
    )


def _u(text: str) -> str:
    """Return a short unique key for report labels."""
    return text.encode("utf-8").decode("unicode_escape")


def build_markdown(report: dict[str, Any]) -> list[str]:
    """Build the Markdown report body for this experiment."""
    selected = report["selected_candidate"]
    lines = [
        _u("# ROI \\u8f6f\\u95e8\\u63a7 + \\u591a\\u5c3a\\u5ea6\\u88c1\\u526a OOF \\u5b9e\\u9a8c"),
        "",
        _u("\\u65e5\\u671f\\uff1a2026-04-26"),
        "",
        _u("## \\u5b9e\\u9a8c\\u8fb9\\u754c"),
        "",
        _u("- \\u5019\\u9009\\u53c2\\u6570\\u53ea\\u4f7f\\u7528 BUSBRA OOF \\u9009\\u62e9\\u3002"),
        _u("- BUSI \\u4ec5\\u5728\\u5019\\u9009\\u56fa\\u5b9a\\u540e\\u505a\\u4e00\\u6b21\\u590d\\u6838\\uff0c\\u4e0d\\u53c2\\u4e0e\\u641c\\u7d22\\u3002"),
        _u("- \\u5f53\\u524d\\u4e3b\\u7ebf\\u662f ROI \\u9762\\u79ef\\u786c\\u95e8\\u63a7\\uff0c\\u672c\\u5b9e\\u9a8c\\u5c1d\\u8bd5\\u7528\\u8fde\\u7eed\\u6743\\u91cd\\u66ff\\u4ee3\\u786c\\u56de\\u9000\\u3002"),
        "",
        _u("## OOF \\u5165\\u9009\\u65b9\\u6848"),
        "",
        f"- margins: `{selected['margins']}`",
        f"- margin_weights: `{selected['margin_weights']}`",
        f"- min_area_ratio: `{selected['min_area_ratio']}`",
        f"- max_area_ratio: `{selected['max_area_ratio']}`",
        f"- ramp_width: `{selected['ramp_width']}`",
        f"- max_weight: `{selected['max_weight']}`",
        f"- mean_gate_weight: `{selected['mean_gate_weight']:.4f}`",
        f"- OOF threshold: `{selected['metrics']['threshold']:.3f}`",
        "",
        _u("## \\u5b8c\\u6574\\u6307\\u6807\\u5bf9\\u6bd4"),
        "",
        _u("| \\u65b9\\u6848 | \\u6837\\u672c\\u6570 | \\u9633\\u6027 | \\u9634\\u6027 | AUC | \\u9608\\u503c | Accuracy | Sensitivity | Recall | Specificity | Precision | NPV | F1-Score | Balanced Acc | Youden J | FPR | FNR | Confusion |"),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        _metrics_row("OOF current area gate", report["baseline_oof_metrics"]),
        _metrics_row("OOF soft gate multiscale", selected["metrics"]),
        _metrics_row("External current area gate", report["baseline_busi_metrics"]),
        _metrics_row("External soft gate multiscale", report["busi_metrics"]),
        "",
        _u("## OOF Top \\u5019\\u9009"),
        "",
        _u("| \\u6392\\u540d | margins | weights | min | max | ramp | max_weight | AUC | \\u9608\\u503c | Sensitivity | Specificity | Precision | F1 | FPR | FNR |"),
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(report["top_candidates"][:10], start=1):
        metric = row["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`{row['margins']}`",
                    f"`{row['margin_weights']}`",
                    f"{row['min_area_ratio']:.2f}",
                    f"{row['max_area_ratio']:.2f}",
                    f"{row['ramp_width']:.2f}",
                    f"{row['max_weight']:.2f}",
                    _format_metric(metric, "auc"),
                    f"{metric['threshold']:.3f}",
                    _format_metric(metric, "sensitivity"),
                    _format_metric(metric, "specificity"),
                    _format_metric(metric, "precision"),
                    _format_metric(metric, "f1_score"),
                    _format_metric(metric, "fpr"),
                    _format_metric(metric, "fnr"),
                ]
            )
            + " |"
        )
    lines.extend(["", _u("## \\u7ed3\\u8bba"), ""])
    if report["recommendation"] == "keep":
        lines.append(_u("- \\u56fa\\u5b9a\\u590d\\u6838\\u8d85\\u8fc7\\u5f53\\u524d\\u4e3b\\u7ebf\\uff0c\\u5efa\\u8bae\\u5408\\u5165 demo \\u4e3b\\u7ebf\\u3002"))
    else:
        lines.append(_u("- \\u56fa\\u5b9a\\u590d\\u6838\\u672a\\u8d85\\u8fc7\\u5f53\\u524d\\u4e3b\\u7ebf\\uff0c\\u5efa\\u8bae\\u653e\\u5f03\\u672c\\u8f6e\\u65b9\\u5411\\u3002"))
    return lines


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the experiment workflow and return the generated summary."""
    runtime_config = yaml.safe_load(Path(args.runtime_config).read_text(encoding="utf-8"))
    full_oof = _load_json(args.full_oof_cache)
    area_cache = _load_json(args.area_cache)
    _validate_order(full_oof["views"])
    y_true = np.asarray(
        [
            1 if str(row["pathology_label"]).lower() == "malignant" else 0
            for row in full_oof["views"]["eff_identity"]
        ],
        dtype=np.int32,
    )
    oof_area = np.asarray(area_cache["oof_lcc_area_ratio"], dtype=np.float64)
    full_oof_probability = _pair_blend(full_oof["views"])
    roi_oof_by_margin: dict[float, np.ndarray] = {}
    for margin in MARGIN_PRESETS:
        report = _load_or_generate_roi_oof_for_margin(args=args, margin_ratio=float(margin))
        roi_oof_by_margin[float(margin)] = _pair_blend(report["views"])

    stack_035 = _runtime_stack(full_oof_probability, roi_oof_by_margin[0.35], runtime_config)
    hard_gate_oof_probability = np.where(
        (oof_area < 0.08) | (oof_area > 0.75),
        full_oof_probability,
        stack_035,
    )
    baseline_oof_metrics = _metrics(y_true, hard_gate_oof_probability, threshold=0.51)
    baseline_oof_best = _best_threshold(
        y_true,
        hard_gate_oof_probability,
        min_sensitivity=float(args.min_sensitivity),
    )

    search_results = _search_candidates(
        y_true=y_true,
        full_probabilities=full_oof_probability,
        roi_probabilities_by_margin=roi_oof_by_margin,
        area_ratios=oof_area,
        runtime_config=runtime_config,
        baseline=baseline_oof_best,
        min_sensitivity=float(args.min_sensitivity),
        max_auc_drop=float(args.max_auc_drop),
    )
    selected = next((row for row in search_results if row["accepted_by_oof_protocol"]), search_results[0])

    busi_reference, busi_y_true, busi_full_views = _load_busi_full_views()
    busi_area = np.asarray(area_cache["busi_segmenter_lcc_area_ratio"], dtype=np.float64)
    if len(busi_area) != len(busi_y_true):
        raise ValueError("BUSI area ratio length mismatch.")
    busi_full_probability = _pair_blend(busi_full_views)
    busi_ref_masks: tuple[list[dict[str, str]], list[np.ndarray], list[np.ndarray], np.ndarray] | None = None
    busi_roi_by_margin: dict[float, np.ndarray] = {}
    for margin in selected["margins"]:
        margin_float = float(margin)
        path = _busi_roi_cache_path(margin_float)
        if not path.exists() and busi_ref_masks is None:
            busi_ref_masks = _load_or_build_busi_masks(args)
        if busi_ref_masks is not None:
            reference, images, masks, mask_y_true = busi_ref_masks
        else:
            reference, images, masks, mask_y_true = [], [], [], busi_y_true
        report = _load_or_generate_busi_roi_for_margin(
            args=args,
            margin_ratio=margin_float,
            reference=reference if reference else busi_reference,
            images=images,
            masks=masks,
            y_true=mask_y_true,
        )
        if [str(row["sample_id"]) for row in report["views"]["eff_identity"]] != [
            row["sample_id"] for row in busi_reference
        ]:
            raise ValueError(f"BUSI ROI cache order mismatch for margin {margin_float}.")
        busi_roi_by_margin[margin_float] = _pair_blend(report["views"])

    busi_probability = _apply_candidate(
        full_probabilities=busi_full_probability,
        roi_probabilities_by_margin=busi_roi_by_margin,
        area_ratios=busi_area,
        runtime_config=runtime_config,
        candidate=selected,
    )
    busi_metrics = _metrics(
        busi_y_true,
        busi_probability,
        threshold=float(selected["metrics"]["threshold"]),
    )
    baseline_busi_metrics = _normalise_metric(
        _load_json("artifacts/reports/busi_demo_roi_area_gate_mainline_eval.json")["metrics"]
    )
    recommendation = (
        "keep"
        if (
            busi_metrics["auc"] >= baseline_busi_metrics["auc"]
            and busi_metrics["precision"] > baseline_busi_metrics["precision"]
            and busi_metrics["f1_score"] > baseline_busi_metrics["f1_score"]
            and busi_metrics["sensitivity"] >= baseline_busi_metrics["sensitivity"] - 0.02
        )
        else "abandon"
    )
    report = {
        "method": "OOF-selected ROI soft gate plus multi-scale ROI crop",
        "data_boundary": {
            "selection": "BUSBRA OOF only",
            "external_review": "BUSI fixed review after candidate selection",
        },
        "margin_presets": [float(value) for value in MARGIN_PRESETS],
        "baseline_oof_metrics": baseline_oof_metrics,
        "baseline_oof_best_threshold_metrics": baseline_oof_best,
        "selected_candidate": selected,
        "top_candidates": search_results[:10],
        "baseline_busi_metrics": baseline_busi_metrics,
        "busi_metrics": busi_metrics,
        "recommendation": recommendation,
    }
    write_json_report(args.output, report)
    write_markdown_report(args.markdown, build_markdown(report))
    return report


def main() -> int:
    """Parse CLI arguments and run the script entry point."""
    args = build_parser().parse_args()
    report = run(args)
    print(
        {
            "recommendation": report["recommendation"],
            "selected": {
                key: report["selected_candidate"][key]
                for key in ("margins", "margin_weights", "min_area_ratio", "max_area_ratio", "ramp_width", "max_weight")
            },
            "oof_auc": report["selected_candidate"]["metrics"]["auc"],
            "oof_precision": report["selected_candidate"]["metrics"]["precision"],
            "oof_f1": report["selected_candidate"]["metrics"]["f1_score"],
            "busi_auc": report["busi_metrics"]["auc"],
            "busi_precision": report["busi_metrics"]["precision"],
            "busi_f1": report["busi_metrics"]["f1_score"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
