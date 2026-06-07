"""Utility script for three model roi oof experiment workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_oof_stacking import (  # noqa: E402
    MODEL_VIEWS,
    ModelView,
    _load_or_create_splits,
    _predict_view_for_manifest,
    _val_manifest_for_fold,
)
from scripts.run_roi_oof_experiment import (  # noqa: E402
    _busi_roi_images,
    _evaluate_stacker_on_busi,
    _fit_oof_stacker,
    _format_metric,
    _metrics,
    _predict_view_on_images,
    _roi_images_from_manifest,
    _slim_stacker,
)
from src.datasets.busbra import LABEL_TO_INDEX, load_busbra_manifest  # noqa: E402
from src.utils.config import load_project_config  # noqa: E402
from src.utils.metrics import classification_metrics  # noqa: E402
from src.utils.reporting import write_json_report, write_markdown_report  # noqa: E402
from src.utils.runtime import select_device  # noqa: E402


THREE_MODEL_VIEWS = ("eff_identity", "densenet_identity", "conv_crop_sweep")
THREE_MODEL_WEIGHTS = {
    "eff_identity": 0.358,
    "densenet_identity": 0.244,
    "conv_crop_sweep": 0.398,
}


MODEL_VIEWS["densenet_identity"] = ModelView(
    name="densenet_identity",
    model_name="densenet121",
    checkpoint_template="./artifacts/checkpoints/densenet121_fold{fold}.pt",
    image_size=224,
    apply_clahe=True,
    tta_variants=({"name": "identity"},),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="Run a three-model ROI + OOF side experiment without modifying demo.yml."
    )
    parser.add_argument(
        "--config",
        default="configs/classifier/convnext_tiny_timm_recipe.yml",
        help="Project config used for paths and BUSBRA split settings.",
    )
    parser.add_argument(
        "--source-full-oof-cache",
        default="artifacts/reports/oof_two_model_predictions.json",
        help="Existing full-image OOF cache to extend with DenseNet.",
    )
    parser.add_argument(
        "--source-roi-oof-cache",
        default="artifacts/reports/roi_oof_predictions.json",
        help="Existing ROI OOF cache to extend with DenseNet.",
    )
    parser.add_argument(
        "--full-oof-cache",
        default="artifacts/reports/oof_three_model_predictions.json",
        help="Output/reuse full-image three-model OOF cache.",
    )
    parser.add_argument(
        "--roi-oof-cache",
        default="artifacts/reports/roi_three_model_predictions.json",
        help="Output/reuse ROI three-model OOF cache.",
    )
    parser.add_argument(
        "--segmenter-checkpoint",
        default="artifacts/checkpoints/segmenter_fold1.pt",
        help="Segmenter checkpoint used for deployable ROI proxy.",
    )
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--margin-ratio", type=float, default=0.35)
    parser.add_argument("--mask-threshold", type=float, default=0.4)
    parser.add_argument(
        "--oof-mask-threshold",
        type=float,
        default=None,
        help="Mask threshold for BUSBRA ROI OOF generation. Defaults to --mask-threshold.",
    )
    parser.add_argument("--reuse-oof", action="store_true")
    parser.add_argument(
        "--output",
        default="artifacts/reports/three_model_roi_oof_experiment.json",
    )
    parser.add_argument(
        "--markdown",
        default="artifacts/reports/three_model_roi_oof_experiment.md",
    )
    return parser


def _load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON report and validate its top-level object."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _oof_mask_threshold(args: argparse.Namespace) -> float:
    """Return the OOF mask threshold for a named method."""
    return (
        float(args.oof_mask_threshold)
        if args.oof_mask_threshold is not None
        else float(args.mask_threshold)
    )


def _reference_ids(views: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Return reference sample identifiers from cached report rows."""
    return [str(row["sample_id"]) for row in views["eff_identity"]]


def _validate_view_order(views: dict[str, list[dict[str, Any]]], view_names: tuple[str, ...]) -> None:
    """Validate view order."""
    reference = _reference_ids(views)
    for view_name in view_names:
        current = [str(row["sample_id"]) for row in views[view_name]]
        if current != reference:
            raise ValueError(f"Sample order mismatch for {view_name}.")


def _combine_probabilities(
    views: dict[str, list[dict[str, Any]]],
    *,
    view_names: tuple[str, ...] = THREE_MODEL_VIEWS,
    weights: dict[str, float] = THREE_MODEL_WEIGHTS,
) -> np.ndarray:
    """Combine probabilities."""
    _validate_view_order(views, view_names)
    combined = np.zeros(len(views[view_names[0]]), dtype=np.float64)
    total_weight = 0.0
    for view_name in view_names:
        weight = float(weights[view_name])
        total_weight += weight
        combined += weight * np.asarray(
            [row["malignant_probability"] for row in views[view_name]],
            dtype=np.float64,
        )
    if total_weight <= 0:
        raise ValueError("Total ensemble weight must be positive.")
    return combined / total_weight


def _oof_targets_and_groups(report: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return OOF targets and group labels for stacking."""
    rows = report["views"]["eff_identity"]
    y_true = np.asarray(
        [LABEL_TO_INDEX[str(row["pathology_label"]).lower()] for row in rows],
        dtype=np.int32,
    )
    groups = np.asarray([str(row["case_id"]) for row in rows])
    return y_true, groups


def _generate_full_oof_view(
    *,
    view_name: str,
    config_path: str | Path,
    fold_count: int,
    device: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Generate full oof view."""
    config, paths = load_project_config(config_path)
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
        fold_count=fold_count,
        seed=seed,
    )
    resolved_device = select_device(device)
    rows: list[dict[str, Any]] = []
    for fold in range(1, fold_count + 1):
        val_manifest = _val_manifest_for_fold(manifest, assignments, fold)
        rows.extend(
            _predict_view_for_manifest(
                MODEL_VIEWS[view_name],
                val_manifest,
                fold=fold,
                project_root=paths.project_root,
                device=resolved_device,
                batch_size=batch_size,
            )
        )
    return sorted(rows, key=lambda row: row["sample_id"])


def _generate_roi_oof_view(
    *,
    view_name: str,
    config_path: str | Path,
    fold_count: int,
    device: str,
    batch_size: int,
    margin_ratio: float,
    mask_threshold: float,
) -> list[dict[str, Any]]:
    """Generate roi oof view."""
    config, paths = load_project_config(config_path)
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
        fold_count=fold_count,
        seed=seed,
    )
    resolved_device = select_device(device)
    rows: list[dict[str, Any]] = []
    for fold in range(1, fold_count + 1):
        val_manifest = _val_manifest_for_fold(manifest, assignments, fold)
        roi_images = _roi_images_from_manifest(
            val_manifest,
            margin_ratio=margin_ratio,
            mask_threshold=mask_threshold,
        )
        probabilities = _predict_view_on_images(
            view_name,
            roi_images,
            fold=fold,
            project_root=paths.project_root,
            device=resolved_device,
            batch_size=batch_size,
        )
        for row, probability in zip(val_manifest.itertuples(index=False), probabilities):
            rows.append(
                {
                    "sample_id": row.sample_id,
                    "case_id": row.case_id,
                    "fold_id": int(fold),
                    "pathology_label": row.pathology_label,
                    "malignant_probability": float(probability),
                }
            )
    return sorted(rows, key=lambda row: row["sample_id"])


def _ensure_full_oof_cache(args: argparse.Namespace) -> dict[str, Any]:
    """Ensure full oof cache."""
    output_path = Path(args.full_oof_cache)
    if args.reuse_oof and output_path.exists():
        report = _load_json(output_path)
    else:
        report = _load_json(args.source_full_oof_cache)
    report.setdefault("views", {})
    if "densenet_identity" not in report["views"]:
        report["views"]["densenet_identity"] = _generate_full_oof_view(
            view_name="densenet_identity",
            config_path=args.config,
            fold_count=int(args.fold_count),
            device=str(args.device),
            batch_size=int(args.batch_size),
        )
    _validate_view_order(report["views"], THREE_MODEL_VIEWS)
    report["source"] = "BUSBRA OOF validation predictions with DenseNet extension"
    report["fold_count"] = int(args.fold_count)
    report["sample_count"] = int(len(report["views"]["eff_identity"]))
    write_json_report(output_path, report)
    return report


def _ensure_roi_oof_cache(args: argparse.Namespace) -> dict[str, Any]:
    """Ensure roi oof cache."""
    output_path = Path(args.roi_oof_cache)
    target_mask_threshold = _oof_mask_threshold(args)
    if args.reuse_oof and output_path.exists():
        report = _load_json(output_path)
    else:
        report = _load_json(args.source_roi_oof_cache)
    existing_mask_threshold = report.get("mask_threshold")
    if (
        existing_mask_threshold is not None
        and abs(float(existing_mask_threshold) - target_mask_threshold) > 1e-9
    ):
        report = {
            "source": "BUSBRA GT-mask ROI OOF predictions regenerated for DenseNet extension",
            "views": {},
        }
    report.setdefault("views", {})
    for view_name in THREE_MODEL_VIEWS:
        if view_name not in report["views"]:
            report["views"][view_name] = _generate_roi_oof_view(
                view_name=view_name,
                config_path=args.config,
                fold_count=int(args.fold_count),
                device=str(args.device),
                batch_size=int(args.batch_size),
                margin_ratio=float(args.margin_ratio),
                mask_threshold=target_mask_threshold,
            )
    _validate_view_order(report["views"], THREE_MODEL_VIEWS)
    report["source"] = "BUSBRA GT-mask ROI OOF predictions with DenseNet extension"
    report["fold_count"] = int(args.fold_count)
    report["sample_count"] = int(len(report["views"]["eff_identity"]))
    report["margin_ratio"] = float(args.margin_ratio)
    report["mask_threshold"] = target_mask_threshold
    write_json_report(output_path, report)
    return report


def _load_busi_full_three() -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, dict[str, Any]]:
    """Load busi full three."""
    report = _load_json("artifacts/reports/busi_ensemble_effnet_densenet_convnext_optimized.json")
    rows = report["rows"]
    reference = [
        {
            "sample_id": str(row["sample_id"]),
            "pathology_label": str(row["pathology_label"]),
        }
        for row in rows
    ]
    y_true = np.asarray(
        [1 if row["pathology_label"] == "malignant" else 0 for row in rows],
        dtype=np.int32,
    )
    probabilities = np.asarray(
        [row["malignant_probability"] for row in rows],
        dtype=np.float64,
    )
    return reference, y_true, probabilities, report["metrics"]


def _predict_busi_roi_three(
    *,
    source: str,
    segmenter_checkpoint: Path,
    margin_ratio: float,
    mask_threshold: float,
    device: str,
    batch_size: int,
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, dict[str, Any]]:
    """Predict busi roi three."""
    _, paths = load_project_config("configs/classifier/convnext_tiny_timm_recipe.yml")
    reference, roi_images, y_true, roi_stats = _busi_roi_images(
        source=source,
        segmenter_checkpoint=segmenter_checkpoint,
        margin_ratio=margin_ratio,
        mask_threshold=mask_threshold,
    )
    resolved_device = select_device(device)
    views: dict[str, list[dict[str, Any]]] = {}
    for view_name in THREE_MODEL_VIEWS:
        fold_probabilities: list[np.ndarray] = []
        for fold in range(1, 6):
            fold_probabilities.append(
                _predict_view_on_images(
                    view_name,
                    roi_images,
                    fold=fold,
                    project_root=paths.project_root,
                    device=resolved_device,
                    batch_size=batch_size,
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
    return reference, y_true, _combine_probabilities(views), roi_stats


def _compact_metrics(metrics: dict[str, Any]) -> str:
    """Return compact metric fields used in reports."""
    return (
        f"AUC {metrics['auc']:.4f}, Sens {_format_metric(metrics, 'sensitivity')}, "
        f"Spec {_format_metric(metrics, 'specificity')}, Acc {_format_metric(metrics, 'accuracy')}, "
        f"Precision {_format_metric(metrics, 'precision')}, F1 {_format_metric(metrics, 'f1_score')}"
    )


def _comparison_metrics(path: str) -> dict[str, Any] | None:
    """Compare candidate probabilities against the baseline."""
    report_path = Path(path)
    if not report_path.exists():
        return None
    report = _load_json(report_path)
    return report.get("metrics")


def build_markdown(report: dict[str, Any]) -> list[str]:
    """Build the Markdown report body for this experiment."""
    stacker = report["stacker"]
    lines = [
        "# 三模型 ROI 与 OOF 融合旁路实验",
        "",
        "日期：2026-04-26",
        "",
        "## 实验目标",
        "",
        "- 在不修改主线 `demo.yml` 的前提下，检查三模型测试线是否能从 ROI 裁剪和 OOF 融合中受益。",
        "- 三模型使用 `EfficientNetV2-S + DenseNet121 + ConvNeXt-Tiny`，权重为 `0.358 / 0.244 / 0.398`。",
        f"- ROI OOF 使用 mask 阈值 `{report['oof_mask_threshold']:.2f}`；部署评估使用 mask 阈值 `{report['eval_mask_threshold']:.2f}` 和现有 `segmenter_fold1.pt`。",
        "",
        "## OOF 融合器",
        "",
        f"- 特征：三模型完整图概率 + 三模型 ROI 概率。",
        f"- 特征模式：`{stacker['feature_mode']}`。",
        f"- OOF CV AUC：`{stacker['cv_auc']:.4f}`。",
        f"- OOF 推荐阈值：`{stacker['threshold_from_oof']:.2f}`。",
        "",
        "## 结果对比",
        "",
        "| 方案 | AUC | 阈值 | Sensitivity | Specificity | Accuracy | Precision | F1-Score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def append_metric_row(name: str, metrics: dict[str, Any]) -> None:
        """Run append metric row."""
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{metrics['auc']:.4f}",
                    f"{float(metrics['threshold']):.3f}",
                    _format_metric(metrics, "sensitivity"),
                    _format_metric(metrics, "specificity"),
                    _format_metric(metrics, "accuracy"),
                    _format_metric(metrics, "precision"),
                    _format_metric(metrics, "f1_score"),
                ]
            )
            + " |"
        )

    if report.get("two_model_mainline_metrics"):
        append_metric_row("两模型 ROI OOF LCC 主线", report["two_model_mainline_metrics"])
    append_metric_row("三模型完整图基线", report["baseline"]["full_three_metrics"])
    append_metric_row(
        "三模型 ROI OOF Stacking@OOF阈值",
        report["busi_results"]["segmenter"]["stacking"]["at_oof_threshold"],
    )
    append_metric_row(
        "三模型 ROI OOF Stacking@Youden",
        report["busi_results"]["segmenter"]["stacking"]["best_by_youden"],
    )
    append_metric_row(
        "三模型 ROI-only@Youden",
        report["busi_results"]["segmenter"]["roi_only"]["best_by_youden"],
    )
    append_metric_row(
        "三模型 oracle ROI Stacking@Youden",
        report["busi_results"]["oracle"]["stacking"]["best_by_youden"],
    )

    lines.extend(
        [
            "",
            "## ROI 面积分布",
            "",
            "| ROI来源 | fallback数量 | 平均面积占比 | 中位面积占比 | P10 | P90 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in ("oracle", "segmenter"):
        stats = report["busi_results"][key]["roi_stats"]
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    str(stats["fallback_count"]),
                    f"{stats['mean_area_ratio']:.4f}",
                    f"{stats['median_area_ratio']:.4f}",
                    f"{stats['p10_area_ratio']:.4f}",
                    f"{stats['p90_area_ratio']:.4f}",
                ]
            )
            + " |"
        )

    segmenter_stack_auc = report["busi_results"]["segmenter"]["stacking"]["auc"]
    full_auc = report["baseline"]["full_three_auc"]
    two_model_auc = (
        report["two_model_mainline_metrics"]["auc"]
        if report.get("two_model_mainline_metrics")
        else None
    )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 三模型完整图基线：`{_compact_metrics(report['baseline']['full_three_metrics'])}`。",
            f"- 三模型 segmenter ROI + OOF Stacking AUC：`{segmenter_stack_auc:.4f}`，相对三模型完整图基线变化 `{segmenter_stack_auc - full_auc:+.4f}`。",
        ]
    )
    if two_model_auc is not None:
        lines.append(
            f"- 与当前两模型 ROI OOF LCC 主线 AUC `{two_model_auc:.4f}` 相比，三模型 ROI OOF 变化 `{segmenter_stack_auc - two_model_auc:+.4f}`。"
        )
    lines.extend(
        [
            "- 如果三模型 ROI OOF 没有同时提高 AUC、Sensitivity 和 F1，不建议合并到主线；保留为测试线更稳妥。",
            "- 如果后续继续做三模型方向，优先改进 DenseNet 分支的 ROI 表现或改为三模型多特征 stacking，而不是直接替换当前 demo。",
        ]
    )
    return lines


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the experiment workflow and return the generated summary."""
    full_oof = _ensure_full_oof_cache(args)
    roi_oof = _ensure_roi_oof_cache(args)
    y_true, groups = _oof_targets_and_groups(full_oof)
    full_oof_three = _combine_probabilities(full_oof["views"])
    roi_oof_three = _combine_probabilities(roi_oof["views"])
    stacker = _fit_oof_stacker(full_oof_three, roi_oof_three, y_true, groups)

    busi_reference, busi_y_true, busi_full_three, full_three_metrics = _load_busi_full_three()
    busi_results: dict[str, Any] = {}
    for source in ("oracle", "segmenter"):
        roi_reference, roi_y_true, busi_roi_three, roi_stats = _predict_busi_roi_three(
            source=source,
            segmenter_checkpoint=Path(args.segmenter_checkpoint),
            margin_ratio=float(args.margin_ratio),
            mask_threshold=float(args.mask_threshold),
            device=str(args.device),
            batch_size=int(args.batch_size),
        )
        if roi_reference != busi_reference or not np.array_equal(roi_y_true, busi_y_true):
            raise ValueError(f"BUSI ROI reference mismatch for {source}.")
        busi_results[source] = {
            "roi_stats": roi_stats,
            "roi_only": _metrics(busi_y_true, busi_roi_three),
            "stacking": _evaluate_stacker_on_busi(
                stacker,
                full_probabilities=busi_full_three,
                roi_probabilities=busi_roi_three,
                y_true=busi_y_true,
            ),
        }

    report = {
        "method": "three-model ROI OOF stacking side experiment",
        "view_names": list(THREE_MODEL_VIEWS),
        "view_weights": THREE_MODEL_WEIGHTS,
        "margin_ratio": float(args.margin_ratio),
        "oof_mask_threshold": _oof_mask_threshold(args),
        "eval_mask_threshold": float(args.mask_threshold),
        "oof_sample_count": int(len(y_true)),
        "baseline": {
            "full_three_auc": float(roc_auc_score(busi_y_true, busi_full_three)),
            "full_three_metrics": full_three_metrics,
            "oof_full_three": _metrics(y_true, full_oof_three),
            "oof_roi_three": _metrics(y_true, roi_oof_three),
        },
        "stacker": _slim_stacker(stacker),
        "busi_results": busi_results,
        "two_model_mainline_metrics": _comparison_metrics(
            "artifacts/reports/busi_demo_roi_oof_lcc_mask04_eval.json"
        ),
        "notes": [
            "This is a side experiment and does not modify configs/inference/demo.yml.",
            "OOF fitting uses BUSBRA validation predictions; deployable ROI uses the existing segmenter checkpoint.",
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
            "full_three_auc": report["baseline"]["full_three_auc"],
            "segmenter_stack_auc": report["busi_results"]["segmenter"]["stacking"]["auc"],
            "segmenter_stack_oof_threshold": report["busi_results"]["segmenter"]["stacking"][
                "at_oof_threshold"
            ]["threshold"],
            "segmenter_stack_oof_sensitivity": report["busi_results"]["segmenter"]["stacking"][
                "at_oof_threshold"
            ]["sensitivity"],
            "segmenter_stack_oof_specificity": report["busi_results"]["segmenter"]["stacking"][
                "at_oof_threshold"
            ]["specificity"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
