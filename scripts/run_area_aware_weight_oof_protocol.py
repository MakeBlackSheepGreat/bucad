"""Utility script for area aware weight oof protocol workflows."""

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

from scripts.run_roi_oof_experiment import (  # noqa: E402
    _busi_roi_images,
    _predict_view_on_images,
)
from src.utils.config import load_project_config  # noqa: E402
from src.utils.reporting import write_json_report, write_markdown_report  # noqa: E402
from src.utils.runtime import select_device  # noqa: E402


PAIR_VIEWS = ("eff_identity", "conv_crop_sweep")
DEFAULT_WEIGHTS = {"eff_identity": 0.427, "conv_crop_sweep": 0.573}
AREA_BINS = (
    {"name": "small", "min": 0.0, "max": 0.15},
    {"name": "medium", "min": 0.15, "max": 0.45},
    {"name": "large", "min": 0.45, "max": 0.75},
    {"name": "very_large", "min": 0.75, "max": 1.01},
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="Use BUSBRA OOF to study lesion-area-aware model fusion weights."
    )
    parser.add_argument("--runtime-config", default="configs/inference/demo.yml")
    parser.add_argument("--full-oof-cache", default="artifacts/reports/oof_two_model_predictions.json")
    parser.add_argument(
        "--roi-oof-cache",
        default="artifacts/reports/roi_oof_lcc_mask04_protocol_predictions.json",
    )
    parser.add_argument(
        "--area-cache",
        default="artifacts/reports/roi_precision_f1_area_cache.json",
    )
    parser.add_argument(
        "--busi-roi-cache",
        default="artifacts/reports/busi_roi_lcc_pair_view_predictions.json",
    )
    parser.add_argument("--segmenter-checkpoint", default="artifacts/checkpoints/segmenter_fold1.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--margin-ratio", type=float, default=0.35)
    parser.add_argument("--mask-threshold", type=float, default=0.4)
    parser.add_argument("--min-sensitivity", type=float, default=0.84)
    parser.add_argument("--max-auc-drop", type=float, default=0.002)
    parser.add_argument("--output", default="artifacts/reports/area_aware_weight_oof_protocol.json")
    parser.add_argument("--markdown", default="artifacts/reports/area_aware_weight_oof_protocol.md")
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
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    accuracy = float((tp + tn) / (tp + tn + fp + fn))
    f1_score = (
        float(2.0 * precision * sensitivity / (precision + sensitivity))
        if precision + sensitivity
        else 0.0
    )
    return {
        "auc": float(roc_auc_score(y_true, probabilities)),
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "precision": precision,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1_score": f1_score,
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
        ),
    )


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
        [row["malignant_probability"] for row in views[view_name]],
        dtype=np.float64,
    )


def _fixed_blend(views: dict[str, list[dict[str, Any]]]) -> np.ndarray:
    """Blend full and ROI probability vectors with fixed weights."""
    _validate_order(views)
    return (
        DEFAULT_WEIGHTS["eff_identity"] * _view_array(views, "eff_identity")
        + DEFAULT_WEIGHTS["conv_crop_sweep"] * _view_array(views, "conv_crop_sweep")
    )


def _runtime_stack(
    full_probabilities: np.ndarray,
    roi_probabilities: np.ndarray,
    *,
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


def _bin_indices(area_ratios: np.ndarray, bin_def: dict[str, float | str]) -> np.ndarray:
    """Return sample indices for one ROI area bin."""
    return (area_ratios >= float(bin_def["min"])) & (area_ratios < float(bin_def["max"]))


def _area_aware_roi_probability(
    *,
    eff_probabilities: np.ndarray,
    conv_probabilities: np.ndarray,
    area_ratios: np.ndarray,
    conv_weights: dict[str, float],
) -> np.ndarray:
    """Select ROI probabilities from area-aware candidate models."""
    output = np.zeros_like(eff_probabilities, dtype=np.float64)
    for bin_def in AREA_BINS:
        mask = _bin_indices(area_ratios, bin_def)
        weight = float(conv_weights[str(bin_def["name"])])
        output[mask] = (
            (1.0 - weight) * eff_probabilities[mask]
            + weight * conv_probabilities[mask]
        )
    return output


def _bin_model_metrics(
    *,
    y_true: np.ndarray,
    full_views: dict[str, list[dict[str, Any]]],
    roi_views: dict[str, list[dict[str, Any]]],
    area_ratios: np.ndarray,
) -> list[dict[str, Any]]:
    """Calculate metrics for each ROI area bin."""
    rows: list[dict[str, Any]] = []
    candidates = {
        "full_eff": _view_array(full_views, "eff_identity"),
        "full_conv": _view_array(full_views, "conv_crop_sweep"),
        "roi_eff": _view_array(roi_views, "eff_identity"),
        "roi_conv": _view_array(roi_views, "conv_crop_sweep"),
        "full_fixed": _fixed_blend(full_views),
        "roi_fixed": _fixed_blend(roi_views),
    }
    for bin_def in AREA_BINS:
        mask = _bin_indices(area_ratios, bin_def)
        bin_y = y_true[mask]
        row: dict[str, Any] = {
            "bin": bin_def["name"],
            "min_area": bin_def["min"],
            "max_area": bin_def["max"],
            "sample_count": int(mask.sum()),
            "positive_count": int(bin_y.sum()),
            "negative_count": int(mask.sum() - bin_y.sum()),
        }
        for name, probabilities in candidates.items():
            if len(np.unique(bin_y)) > 1:
                row[f"{name}_auc"] = float(roc_auc_score(bin_y, probabilities[mask]))
            else:
                row[f"{name}_auc"] = None
        rows.append(row)
    return rows


def _search_area_weights(
    *,
    y_true: np.ndarray,
    full_probabilities: np.ndarray,
    roi_views: dict[str, list[dict[str, Any]]],
    area_ratios: np.ndarray,
    runtime_config: dict[str, Any],
    baseline_metrics: dict[str, Any],
    min_sensitivity: float,
    max_auc_drop: float,
) -> list[dict[str, Any]]:
    """Search area weights."""
    eff_roi = _view_array(roi_views, "eff_identity")
    conv_roi = _view_array(roi_views, "conv_crop_sweep")
    grid = np.round(np.arange(0.0, 1.0001, 0.1), 2)
    results: list[dict[str, Any]] = []
    bin_names = [str(bin_def["name"]) for bin_def in AREA_BINS]
    for weights in itertools.product(grid, repeat=len(bin_names)):
        conv_weights = {name: float(weight) for name, weight in zip(bin_names, weights)}
        roi_probability = _area_aware_roi_probability(
            eff_probabilities=eff_roi,
            conv_probabilities=conv_roi,
            area_ratios=area_ratios,
            conv_weights=conv_weights,
        )
        stack_probability = _runtime_stack(
            full_probabilities,
            roi_probability,
            runtime_config=runtime_config,
        )
        metrics = _best_threshold(
            y_true,
            stack_probability,
            min_sensitivity=min_sensitivity,
        )
        accepted = (
            metrics["auc"] >= baseline_metrics["auc"] - max_auc_drop
            and metrics["precision"] > baseline_metrics["precision"]
            and metrics["f1_score"] > baseline_metrics["f1_score"]
        )
        results.append(
            {
                "conv_weights": conv_weights,
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
        ),
        reverse=True,
    )


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


def _load_or_generate_busi_roi_cache(args: argparse.Namespace) -> dict[str, Any]:
    """Load or generate busi roi cache."""
    path = Path(args.busi_roi_cache)
    if path.exists():
        return _load_json(path)
    _, paths = load_project_config("configs/classifier/convnext_tiny_timm_recipe.yml")
    reference, roi_images, y_true, roi_stats = _busi_roi_images(
        source="segmenter",
        segmenter_checkpoint=Path(args.segmenter_checkpoint),
        margin_ratio=float(args.margin_ratio),
        mask_threshold=float(args.mask_threshold),
    )
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
    area_cache = _load_json(args.area_cache)
    report = {
        "source": "BUSI segmenter ROI pair-view probabilities with LCC mask threshold 0.40",
        "sample_count": len(reference),
        "roi_stats": roi_stats,
        "roi_area_ratios": area_cache["busi_segmenter_lcc_area_ratio"],
        "views": views,
        "y_true": [int(value) for value in y_true],
    }
    write_json_report(path, report)
    return report


def _format_metric(metric: dict[str, Any], key: str) -> str:
    """Format one metric value for report tables."""
    return f"{float(metric[key]):.4f}"


def _confusion_text(metric: dict[str, Any]) -> str:
    """Format confusion-matrix counts for report tables."""
    c = metric["confusion"]
    return f"TN {c['tn']} / FP {c['fp']} / FN {c['fn']} / TP {c['tp']}"


def build_markdown(report: dict[str, Any]) -> list[str]:
    """Build the Markdown report body for this experiment."""
    selected = report["selected_candidate"]
    lines = [
        "# 病灶面积感知动态模型权重 OOF 实验",
        "",
        "日期：2026-04-26",
        "",
        "## 实验目标",
        "",
        "- 使用 BUSBRA OOF 预测分析病灶面积大小对 EfficientNetV2-S 和 ConvNeXt-Tiny 表现的影响。",
        "- 根据病灶面积分档动态调整 ROI 分支的模型融合权重。",
        "- 参数选择只使用 OOF；外部结果只做固定候选后的单次复核。",
        "",
        "## 面积分档模型表现",
        "",
        "| 面积分档 | 样本数 | 阳性 | 阴性 | full Eff AUC | full Conv AUC | ROI Eff AUC | ROI Conv AUC | full 固定 AUC | ROI 固定 AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["bin_model_metrics"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["bin"]),
                    str(row["sample_count"]),
                    str(row["positive_count"]),
                    str(row["negative_count"]),
                    "-" if row["full_eff_auc"] is None else f"{row['full_eff_auc']:.4f}",
                    "-" if row["full_conv_auc"] is None else f"{row['full_conv_auc']:.4f}",
                    "-" if row["roi_eff_auc"] is None else f"{row['roi_eff_auc']:.4f}",
                    "-" if row["roi_conv_auc"] is None else f"{row['roi_conv_auc']:.4f}",
                    "-" if row["full_fixed_auc"] is None else f"{row['full_fixed_auc']:.4f}",
                    "-" if row["roi_fixed_auc"] is None else f"{row['roi_fixed_auc']:.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## OOF 选择结果",
            "",
            "| 方案 | AUC | 阈值 | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for name, metric in (
        ("当前 OOF ROI stack", report["baseline_oof_metrics"]),
        ("面积动态权重 OOF 候选", selected["metrics"]),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _format_metric(metric, "auc"),
                    f"{metric['threshold']:.3f}",
                    _format_metric(metric, "sensitivity"),
                    _format_metric(metric, "specificity"),
                    _format_metric(metric, "accuracy"),
                    _format_metric(metric, "precision"),
                    _format_metric(metric, "f1_score"),
                    _confusion_text(metric),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "OOF 选中的 ConvNeXt ROI 权重：",
            "",
        ]
    )
    for bin_name, weight in selected["conv_weights"].items():
        lines.append(f"- `{bin_name}`：`{weight:.2f}`")
    lines.extend(
        [
            "",
            "## 固定候选复核",
            "",
            "| 方案 | AUC | 阈值 | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for name, metric in (
        ("当前主线", report["baseline_busi_metrics"]),
        ("面积动态权重复核", report["busi_metrics"]),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _format_metric(metric, "auc"),
                    f"{metric['threshold']:.3f}",
                    _format_metric(metric, "sensitivity"),
                    _format_metric(metric, "specificity"),
                    _format_metric(metric, "accuracy"),
                    _format_metric(metric, "precision"),
                    _format_metric(metric, "f1_score"),
                    _confusion_text(metric),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
        ]
    )
    if report["recommendation"] == "keep":
        lines.append("- 面积动态权重相对当前主线有综合提升，建议保留为候选。")
    else:
        lines.append("- 面积动态权重未超过当前主线，建议放弃该方向或仅保留为研究记录。")
    lines.append("- 如果要合并到 runtime，需要继续实现按 ROI 面积动态设置 `classifier_weight_overrides`。")
    return lines

def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the experiment workflow and return the generated summary."""
    runtime_config = yaml.safe_load(Path(args.runtime_config).read_text(encoding="utf-8"))
    full_oof = _load_json(args.full_oof_cache)
    roi_oof = _load_json(args.roi_oof_cache)
    area_cache = _load_json(args.area_cache)
    oof_area = np.asarray(area_cache["oof_lcc_area_ratio"], dtype=np.float64)
    _validate_order(full_oof["views"])
    _validate_order(roi_oof["views"])
    y_true = np.asarray(
        [
            1 if str(row["pathology_label"]).lower() == "malignant" else 0
            for row in full_oof["views"]["eff_identity"]
        ],
        dtype=np.int32,
    )
    full_fixed = _fixed_blend(full_oof["views"])
    roi_fixed = _fixed_blend(roi_oof["views"])
    baseline_oof_probabilities = _runtime_stack(
        full_fixed,
        roi_fixed,
        runtime_config=runtime_config,
    )
    baseline_oof_metrics = _metrics(y_true, baseline_oof_probabilities, threshold=0.55)
    bin_metrics = _bin_model_metrics(
        y_true=y_true,
        full_views=full_oof["views"],
        roi_views=roi_oof["views"],
        area_ratios=oof_area,
    )
    search_results = _search_area_weights(
        y_true=y_true,
        full_probabilities=full_fixed,
        roi_views=roi_oof["views"],
        area_ratios=oof_area,
        runtime_config=runtime_config,
        baseline_metrics=baseline_oof_metrics,
        min_sensitivity=float(args.min_sensitivity),
        max_auc_drop=float(args.max_auc_drop),
    )
    accepted = [row for row in search_results if row["accepted_by_oof_protocol"]]
    selected = accepted[0] if accepted else search_results[0]

    busi_reference, busi_y_true, busi_full_views = _load_busi_full_views()
    busi_roi = _load_or_generate_busi_roi_cache(args)
    busi_roi_views = busi_roi["views"]
    if [str(row["sample_id"]) for row in busi_roi_views["eff_identity"]] != [
        row["sample_id"] for row in busi_reference
    ]:
        raise ValueError("BUSI ROI cache order does not match full reports.")
    busi_area = np.asarray(busi_roi["roi_area_ratios"], dtype=np.float64)
    busi_full_fixed = _fixed_blend(busi_full_views)
    busi_roi_dynamic = _area_aware_roi_probability(
        eff_probabilities=_view_array(busi_roi_views, "eff_identity"),
        conv_probabilities=_view_array(busi_roi_views, "conv_crop_sweep"),
        area_ratios=busi_area,
        conv_weights=selected["conv_weights"],
    )
    busi_probabilities = _runtime_stack(
        busi_full_fixed,
        busi_roi_dynamic,
        runtime_config=runtime_config,
    )
    busi_metrics = _metrics(
        busi_y_true,
        busi_probabilities,
        threshold=float(selected["metrics"]["threshold"]),
    )
    baseline_busi_metrics = _load_json(
        "artifacts/reports/busi_demo_roi_oof_lcc_mask04_eval.json"
    )["metrics"]
    recommendation = (
        "keep"
        if (
            busi_metrics["auc"] >= baseline_busi_metrics["auc"]
            and busi_metrics["precision"] > baseline_busi_metrics["precision"]
            and busi_metrics["f1_score"] > baseline_busi_metrics["f1_score"]
        )
        else "abandon"
    )
    report = {
        "method": "area-aware ROI model fusion selected by BUSBRA OOF",
        "area_bins": AREA_BINS,
        "bin_model_metrics": bin_metrics,
        "baseline_oof_metrics": baseline_oof_metrics,
        "selected_candidate": selected,
        "top_candidates": search_results[:10],
        "baseline_busi_metrics": baseline_busi_metrics,
        "busi_metrics": busi_metrics,
        "recommendation": recommendation,
        "notes": [
            "OOF chooses area-bin-specific ConvNeXt weights for the ROI branch.",
            "BUSI is used only once after the OOF-selected candidate is fixed.",
        ],
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
            "selected_conv_weights": report["selected_candidate"]["conv_weights"],
            "oof_f1": report["selected_candidate"]["metrics"]["f1_score"],
            "busi_auc": report["busi_metrics"]["auc"],
            "busi_precision": report["busi_metrics"]["precision"],
            "busi_f1": report["busi_metrics"]["f1_score"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
