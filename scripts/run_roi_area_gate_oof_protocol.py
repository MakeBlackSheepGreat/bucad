"""Utility script for roi area gate oof protocol workflows."""

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
    _load_or_create_splits,
    _val_manifest_for_fold,
)
from scripts.run_roi_oof_experiment import _predict_view_on_images  # noqa: E402
from src.datasets.busbra import load_busbra_manifest  # noqa: E402
from src.preprocess.io import read_image, read_mask  # noqa: E402
from src.preprocess.roi import crop_to_mask_bbox, expand_bbox, mask_bbox  # noqa: E402
from src.utils.config import load_project_config  # noqa: E402
from src.utils.reporting import write_json_report, write_markdown_report  # noqa: E402
from src.utils.runtime import select_device  # noqa: E402


PAIR_VIEWS = ("eff_identity", "conv_crop_sweep")
PAIR_WEIGHTS = {"eff_identity": 0.427, "conv_crop_sweep": 0.573}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="Select ROI area gate parameters from BUSBRA OOF only."
    )
    parser.add_argument("--config", default="configs/classifier/convnext_tiny_timm_recipe.yml")
    parser.add_argument("--runtime-config", default="configs/inference/demo.yml")
    parser.add_argument("--full-oof-cache", default="artifacts/reports/oof_two_model_predictions.json")
    parser.add_argument(
        "--roi-oof-cache",
        default="artifacts/reports/roi_oof_lcc_mask04_protocol_predictions.json",
    )
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--margin-ratio", type=float, default=0.35)
    parser.add_argument("--mask-threshold", type=float, default=0.4)
    parser.add_argument("--min-sensitivity", type=float, default=0.84)
    parser.add_argument("--min-auc-drop", type=float, default=0.002)
    parser.add_argument("--reuse-roi-oof", action="store_true")
    parser.add_argument(
        "--candidate-config",
        default="configs/inference/demo_roi_area_gate_oof_protocol.yml",
    )
    parser.add_argument(
        "--output",
        default="artifacts/reports/roi_area_gate_oof_protocol.json",
    )
    parser.add_argument(
        "--markdown",
        default="artifacts/reports/roi_area_gate_oof_protocol.md",
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


def _combine_views(views: dict[str, list[dict[str, Any]]]) -> np.ndarray:
    """Combine views."""
    reference_ids: list[str] | None = None
    combined: np.ndarray | None = None
    total_weight = 0.0
    for view_name in PAIR_VIEWS:
        rows = views[view_name]
        current_ids = [str(row["sample_id"]) for row in rows]
        if reference_ids is None:
            reference_ids = current_ids
            combined = np.zeros(len(rows), dtype=np.float64)
        elif current_ids != reference_ids:
            raise ValueError(f"Sample order mismatch for {view_name}.")
        weight = float(PAIR_WEIGHTS[view_name])
        total_weight += weight
        combined += weight * np.asarray(
            [row["malignant_probability"] for row in rows],
            dtype=np.float64,
        )
    if combined is None or total_weight <= 0:
        raise ValueError("No probability views to combine.")
    return combined / total_weight


def _roi_area_ratio(mask: np.ndarray, *, threshold: float, margin_ratio: float) -> float:
    """Process ROI area ratio."""
    bbox = mask_bbox(
        mask,
        threshold=threshold,
        min_area_ratio=0.001,
        largest_component=True,
    )
    if bbox is None:
        return 1.0
    x1, y1, x2, y2 = expand_bbox(
        bbox,
        image_shape=mask.shape,
        margin_ratio=margin_ratio,
        square=True,
    )
    return float(max(1, (x2 - x1) * (y2 - y1)) / max(1, mask.shape[0] * mask.shape[1]))


def _roi_images_and_areas(
    manifest,
    *,
    margin_ratio: float,
    mask_threshold: float,
) -> tuple[list[np.ndarray], list[float]]:
    """Process ROI images and areas."""
    images: list[np.ndarray] = []
    area_ratios: list[float] = []
    for row in manifest.itertuples(index=False):
        image = read_image(row.image_path, grayscale=True)
        mask = read_mask(row.mask_path)
        area_ratios.append(
            _roi_area_ratio(
                mask,
                threshold=mask_threshold,
                margin_ratio=margin_ratio,
            )
        )
        images.append(
            crop_to_mask_bbox(
                image,
                mask,
                threshold=mask_threshold,
                margin_ratio=margin_ratio,
                largest_component=True,
            )
        )
    return images, area_ratios


def _generate_roi_oof_cache(args: argparse.Namespace) -> dict[str, Any]:
    """Generate roi oof cache."""
    config, paths = load_project_config(args.config)
    manifest = load_busbra_manifest(paths.busbra_root)
    seed = int(config.get("seed", 42))
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
        seed=seed,
    )
    resolved_device = select_device(str(args.device))
    views: dict[str, list[dict[str, Any]]] = {view_name: [] for view_name in PAIR_VIEWS}
    area_by_sample: dict[str, float] = {}
    for fold in range(1, int(args.fold_count) + 1):
        val_manifest = _val_manifest_for_fold(manifest, assignments, fold)
        roi_images, area_ratios = _roi_images_and_areas(
            val_manifest,
            margin_ratio=float(args.margin_ratio),
            mask_threshold=float(args.mask_threshold),
        )
        for row, area_ratio in zip(val_manifest.itertuples(index=False), area_ratios):
            area_by_sample[str(row.sample_id)] = float(area_ratio)
        for view_name in PAIR_VIEWS:
            probabilities = _predict_view_on_images(
                view_name,
                roi_images,
                fold=fold,
                project_root=paths.project_root,
                device=resolved_device,
                batch_size=int(args.batch_size),
            )
            for row, probability in zip(val_manifest.itertuples(index=False), probabilities):
                views[view_name].append(
                    {
                        "sample_id": row.sample_id,
                        "case_id": row.case_id,
                        "fold_id": int(fold),
                        "pathology_label": row.pathology_label,
                        "malignant_probability": float(probability),
                    }
                )
    for view_name in views:
        views[view_name] = sorted(views[view_name], key=lambda row: row["sample_id"])
    sample_ids = [str(row["sample_id"]) for row in views["eff_identity"]]
    return {
        "source": "BUSBRA OOF ROI predictions with mask_threshold=0.40 and largest_component=True",
        "fold_count": int(args.fold_count),
        "sample_count": len(sample_ids),
        "margin_ratio": float(args.margin_ratio),
        "mask_threshold": float(args.mask_threshold),
        "largest_component": True,
        "views": views,
        "roi_area_ratios": [float(area_by_sample[sample_id]) for sample_id in sample_ids],
    }


def _load_or_generate_roi_oof(args: argparse.Namespace) -> dict[str, Any]:
    """Load or generate roi oof."""
    path = Path(args.roi_oof_cache)
    if args.reuse_roi_oof and path.exists():
        return _load_json(path)
    report = _generate_roi_oof_cache(args)
    write_json_report(path, report)
    return report


def _apply_runtime_stacker(
    full_probabilities: np.ndarray,
    roi_probabilities: np.ndarray,
    runtime_config: dict[str, Any],
) -> np.ndarray:
    """Apply runtime stacker."""
    stacker = runtime_config["runtime"]["roi_enhancement"]["stacker"]
    matrix = np.vstack([_logit(full_probabilities), _logit(roi_probabilities)]).T
    scaled = (
        matrix - np.asarray(stacker["scaler_mean"], dtype=np.float64)
    ) / np.asarray(stacker["scaler_scale"], dtype=np.float64)
    return _sigmoid(
        scaled @ np.asarray(stacker["coef"], dtype=np.float64)
        + float(stacker["intercept"])
    )


def _scan_area_gates(
    *,
    y_true: np.ndarray,
    full_probabilities: np.ndarray,
    stack_probabilities: np.ndarray,
    area_ratios: np.ndarray,
    baseline: dict[str, Any],
    min_sensitivity: float,
    min_auc_drop: float,
) -> list[dict[str, Any]]:
    """Scan area gates."""
    results: list[dict[str, Any]] = []
    min_values = [0.0, 0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
    max_values = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.01]
    for min_area in min_values:
        for max_area in max_values:
            if min_area >= max_area:
                continue
            gated_probabilities = np.where(
                (area_ratios < min_area) | (area_ratios > max_area),
                full_probabilities,
                stack_probabilities,
            )
            metrics = _best_threshold(
                y_true,
                gated_probabilities,
                min_sensitivity=min_sensitivity,
            )
            accepted = (
                metrics["auc"] >= baseline["auc"] - min_auc_drop
                and metrics["precision"] > baseline["precision"]
                and metrics["f1_score"] > baseline["f1_score"]
            )
            results.append(
                {
                    "min_area_ratio": float(min_area),
                    "max_area_ratio": float(max_area),
                    "fallback_count": int(
                        np.sum((area_ratios < min_area) | (area_ratios > max_area))
                    ),
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


def _write_candidate_config(
    *,
    runtime_config_path: str | Path,
    destination: str | Path,
    candidate: dict[str, Any],
) -> None:
    """Write candidate config."""
    config = yaml.safe_load(Path(runtime_config_path).read_text(encoding="utf-8"))
    runtime = config["runtime"]
    runtime["ensemble_display_name"] = "ConvNeXt-Tiny + EfficientNetV2-S + ROI Area Gate OOF Protocol"
    runtime["default_threshold"] = float(candidate["metrics"]["threshold"])
    roi_config = runtime["roi_enhancement"]
    roi_config["quality_gate"] = {
        "enabled": True,
        "min_area_ratio": float(candidate["min_area_ratio"]),
        "max_area_ratio": float(candidate["max_area_ratio"]),
        "fallback_to_full": True,
    }
    Path(destination).write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _format_metric(metric: dict[str, Any], key: str) -> str:
    """Format one metric value for report tables."""
    return f"{float(metric[key]):.4f}"


def build_markdown(report: dict[str, Any]) -> list[str]:
    """Build the Markdown report body for this experiment."""
    baseline = report["baseline_oof_metrics"]
    selected = report["selected_candidate"]
    lines = [
        "# ROI 面积门控 OOF 固定协议",
        "",
        "日期：2026-04-26",
        "",
        "## 协议",
        "",
        "- 只使用 BUSBRA OOF 预测选择 ROI 面积门控参数和阈值。",
        "- 目标是在 Sensitivity 不低于设定下限的前提下，提高 Precision 和 F1-Score。",
        "- 选择过程不读取外部评估结果；外部结果只用于候选固定后的单次复核。",
        "",
        "## OOF 基线",
        "",
        "| AUC | 阈值 | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        "| "
        + " | ".join(
            [
                _format_metric(baseline, "auc"),
                f"{baseline['threshold']:.3f}",
                _format_metric(baseline, "sensitivity"),
                _format_metric(baseline, "specificity"),
                _format_metric(baseline, "accuracy"),
                _format_metric(baseline, "precision"),
                _format_metric(baseline, "f1_score"),
                (
                    f"TN {baseline['confusion']['tn']} / FP {baseline['confusion']['fp']} / "
                    f"FN {baseline['confusion']['fn']} / TP {baseline['confusion']['tp']}"
                ),
            ]
        )
        + " |",
        "",
        "## OOF 选中候选",
        "",
        "| min_area | max_area | fallback_count | AUC | 阈值 | Sensitivity | Specificity | Accuracy | Precision | F1-Score |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| "
        + " | ".join(
            [
                f"{selected['min_area_ratio']:.2f}",
                f"{selected['max_area_ratio']:.2f}",
                str(selected["fallback_count"]),
                _format_metric(selected["metrics"], "auc"),
                f"{selected['metrics']['threshold']:.3f}",
                _format_metric(selected["metrics"], "sensitivity"),
                _format_metric(selected["metrics"], "specificity"),
                _format_metric(selected["metrics"], "accuracy"),
                _format_metric(selected["metrics"], "precision"),
                _format_metric(selected["metrics"], "f1_score"),
            ]
        )
        + " |",
        "",
        "## Top OOF 候选",
        "",
        "| 排名 | min_area | max_area | accepted | AUC | 阈值 | Sensitivity | Precision | F1-Score |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, candidate in enumerate(report["top_candidates"], start=1):
        metrics = candidate["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"{candidate['min_area_ratio']:.2f}",
                    f"{candidate['max_area_ratio']:.2f}",
                    str(candidate["accepted_by_oof_protocol"]),
                    _format_metric(metrics, "auc"),
                    f"{metrics['threshold']:.3f}",
                    _format_metric(metrics, "sensitivity"),
                    _format_metric(metrics, "precision"),
                    _format_metric(metrics, "f1_score"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- OOF 协议选中的候选配置写入 `{report['candidate_config']}`。",
            "- 如果后续单次外部复核不优于当前主线，则该候选应保留为研究记录，不合并默认 demo。",
            "- 如果后续单次外部复核优于当前主线，也仍建议先由队内固定验证流程确认，再覆盖 `demo.yml`。",
        ]
    )
    return lines


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the experiment workflow and return the generated summary."""
    full_oof = _load_json(args.full_oof_cache)
    roi_oof = _load_or_generate_roi_oof(args)
    y_true = np.asarray(
        [
            1 if str(row["pathology_label"]).lower() == "malignant" else 0
            for row in full_oof["views"]["eff_identity"]
        ],
        dtype=np.int32,
    )
    full_probabilities = _combine_views(full_oof["views"])
    roi_probabilities = _combine_views(roi_oof["views"])
    runtime_config = yaml.safe_load(Path(args.runtime_config).read_text(encoding="utf-8"))
    stack_probabilities = _apply_runtime_stacker(
        full_probabilities,
        roi_probabilities,
        runtime_config,
    )
    area_ratios = np.asarray(roi_oof["roi_area_ratios"], dtype=np.float64)
    baseline_metrics = _metrics(y_true, stack_probabilities, threshold=0.55)
    scan_results = _scan_area_gates(
        y_true=y_true,
        full_probabilities=full_probabilities,
        stack_probabilities=stack_probabilities,
        area_ratios=area_ratios,
        baseline=baseline_metrics,
        min_sensitivity=float(args.min_sensitivity),
        min_auc_drop=float(args.min_auc_drop),
    )
    accepted = [row for row in scan_results if row["accepted_by_oof_protocol"]]
    selected = accepted[0] if accepted else scan_results[0]
    _write_candidate_config(
        runtime_config_path=args.runtime_config,
        destination=args.candidate_config,
        candidate=selected,
    )
    report = {
        "method": "BUSBRA OOF-only ROI area gate selection",
        "inputs": {
            "full_oof_cache": str(args.full_oof_cache),
            "roi_oof_cache": str(args.roi_oof_cache),
            "runtime_config": str(args.runtime_config),
        },
        "margin_ratio": float(args.margin_ratio),
        "mask_threshold": float(args.mask_threshold),
        "min_sensitivity": float(args.min_sensitivity),
        "min_auc_drop": float(args.min_auc_drop),
        "baseline_oof_metrics": baseline_metrics,
        "selected_candidate": selected,
        "accepted_candidate_count": len(accepted),
        "top_candidates": scan_results[:10],
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
            "selected_min_area": selected["min_area_ratio"],
            "selected_max_area": selected["max_area_ratio"],
            "selected_threshold": selected["metrics"]["threshold"],
            "selected_precision": selected["metrics"]["precision"],
            "selected_f1": selected["metrics"]["f1_score"],
            "candidate_config": report["candidate_config"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
