"""Utility script for convnext soup oof protocol workflows."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_oof_stacking import (  # noqa: E402
    ModelView,
    _load_or_create_splits,
    _predict_view_for_manifest,
    _val_manifest_for_fold,
)
from scripts.run_roi_area_gate_oof_protocol import _roi_images_and_areas  # noqa: E402
from scripts.run_seed_diversity_oof_protocol import (  # noqa: E402
    BASE_CONV_VIEW,
    CONV_WEIGHT,
    EFF_WEIGHT,
    _array_from_rows,
    _best_threshold,
    _candidate_probability,
    _confusion_text,
    _format_metric,
    _load_json,
    _load_labels,
    _metrics,
    _metrics_row,
    _predict_model_view_on_images,
    _runtime_stack,
    _validate_same_order,
    _with_sample_counts,
)
from src.datasets.busbra import load_busbra_manifest  # noqa: E402
from src.utils.config import load_project_config  # noqa: E402
from src.utils.reporting import write_json_report, write_markdown_report  # noqa: E402
from src.utils.runtime import select_device  # noqa: E402


SOUP_VIEW = ModelView(
    name="conv_soup42_123_a050_crop_sweep",
    model_name="convnext_tiny",
    checkpoint_template="./artifacts/checkpoints/convnext_tiny_timm_recipe_soup42_123_a050_fold{fold}.pt",
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
        description="Select a ConvNeXt-Tiny weight-soup candidate using BUSBRA OOF only."
    )
    parser.add_argument("--config", default="configs/classifier/convnext_tiny_timm_recipe.yml")
    parser.add_argument("--runtime-config", default="configs/inference/demo.yml")
    parser.add_argument("--full-oof-cache", default="artifacts/reports/oof_two_model_predictions.json")
    parser.add_argument(
        "--roi-oof-cache",
        default="artifacts/reports/roi_oof_lcc_mask04_protocol_predictions.json",
    )
    parser.add_argument(
        "--soup-full-cache",
        default="artifacts/reports/oof_convnext_tiny_soup42_123_a050_predictions.json",
    )
    parser.add_argument(
        "--soup-roi-cache",
        default="artifacts/reports/roi_oof_convnext_tiny_soup42_123_a050_lcc_mask04_predictions.json",
    )
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--margin-ratio", type=float, default=0.35)
    parser.add_argument("--mask-threshold", type=float, default=0.4)
    parser.add_argument("--min-sensitivity", type=float, default=0.84)
    parser.add_argument("--max-auc-drop", type=float, default=0.002)
    parser.add_argument("--output", default="artifacts/reports/convnext_soup_oof_protocol.json")
    parser.add_argument("--markdown", default="artifacts/reports/convnext_soup_oof_protocol.md")
    parser.add_argument(
        "--candidate-config",
        default="configs/inference/demo_convnext_soup_oof_candidate.yml",
    )
    return parser


def _load_or_generate_soup_full(args: argparse.Namespace) -> dict[str, Any]:
    """Load or generate soup full."""
    path = Path(args.soup_full_cache)
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
        rows.extend(
            _predict_view_for_manifest(
                SOUP_VIEW,
                _val_manifest_for_fold(manifest, assignments, fold),
                fold=fold,
                project_root=paths.project_root,
                device=resolved_device,
                batch_size=int(args.batch_size),
            )
        )
    report = {
        "source": "BUSBRA OOF full-image ConvNeXt-Tiny seed42/seed123 a=0.5 weight-soup predictions",
        "fold_count": int(args.fold_count),
        "sample_count": len(rows),
        "views": {SOUP_VIEW.name: sorted(rows, key=lambda row: str(row["sample_id"]))},
    }
    write_json_report(path, report)
    return report


def _load_or_generate_soup_roi(args: argparse.Namespace) -> dict[str, Any]:
    """Load or generate soup roi."""
    path = Path(args.soup_roi_cache)
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
            SOUP_VIEW,
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
        "source": "BUSBRA GT-mask ROI ConvNeXt-Tiny seed42/seed123 a=0.5 weight-soup OOF predictions",
        "fold_count": int(args.fold_count),
        "sample_count": len(rows),
        "margin_ratio": float(args.margin_ratio),
        "mask_threshold": float(args.mask_threshold),
        "largest_component": True,
        "views": {SOUP_VIEW.name: rows},
        "roi_area_ratios": [float(area_by_sample[str(row["sample_id"])]) for row in rows],
    }
    write_json_report(path, report)
    return report


def _soup_probability(
    *,
    eff_full: np.ndarray,
    soup_full: np.ndarray,
    eff_roi: np.ndarray,
    soup_roi: np.ndarray,
    area_ratios: np.ndarray,
    runtime_config: dict[str, Any],
    roi_stack_blend_weight: float,
) -> np.ndarray:
    """Blend seed-soup probabilities from full and ROI views."""
    full_probability = EFF_WEIGHT * eff_full + CONV_WEIGHT * soup_full
    roi_probability = EFF_WEIGHT * eff_roi + CONV_WEIGHT * soup_roi
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
    soup_full: np.ndarray,
    eff_roi: np.ndarray,
    soup_roi: np.ndarray,
    area_ratios: np.ndarray,
    runtime_config: dict[str, Any],
    baseline: dict[str, Any],
    min_sensitivity: float,
    max_auc_drop: float,
) -> list[dict[str, Any]]:
    """Scan candidates."""
    results: list[dict[str, Any]] = []
    for blend_weight in (0.75, 0.85, 0.95, 1.0):
        probabilities = _soup_probability(
            eff_full=eff_full,
            soup_full=soup_full,
            eff_roi=eff_roi,
            soup_roi=soup_roi,
            area_ratios=area_ratios,
            runtime_config=runtime_config,
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


def _fold_from_checkpoint(member: dict[str, Any], fallback: int) -> int:
    """Infer the fold number from a checkpoint filename."""
    match = re.search(r"fold(\d+)", str(member.get("checkpoint", "")))
    if match:
        return int(match.group(1))
    return fallback


def _write_candidate_config(
    *,
    runtime_config_path: str | Path,
    destination: str | Path,
    selected: dict[str, Any],
) -> None:
    """Write candidate config."""
    config = yaml.safe_load(Path(runtime_config_path).read_text(encoding="utf-8"))
    runtime = config["runtime"]
    members: list[dict[str, Any]] = []
    conv_index = 0
    for member in runtime["classifier_members"]:
        copied = dict(member)
        if copied.get("model") == "convnext_tiny":
            conv_index += 1
            fold = _fold_from_checkpoint(copied, conv_index)
            copied["checkpoint"] = (
                f"./artifacts/checkpoints/convnext_tiny_timm_recipe_soup42_123_a050_fold{fold}.pt"
            )
            copied["weight"] = float(CONV_WEIGHT)
        members.append(copied)
    runtime["classifier_members"] = members
    runtime["default_threshold"] = float(selected["metrics"]["threshold"])
    runtime["ensemble_display_name"] = "ConvNeXt-Tiny weight soup + EfficientNetV2-S + ROI Area Gate"
    runtime["primary_classifier_model"] = "ConvNeXt-Tiny Weight Soup"
    runtime["roi_enhancement"]["roi_stack_blend_weight"] = float(
        selected["roi_stack_blend_weight"]
    )
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def build_markdown(report: dict[str, Any]) -> list[str]:
    """Build the Markdown report body for this experiment."""
    selected = report["selected_candidate"]
    lines = [
        "# ConvNeXt-Tiny Weight Soup OOF Protocol",
        "",
        "## Boundary",
        "",
        "- Candidate selection used BUSBRA OOF data only.",
        "- BUSI is not read by this protocol.",
        "- The ConvNeXt branch is replaced by same-fold seed42/seed123 linear weight averaging at alpha 0.50.",
        "",
        "## Selected Candidate",
        "",
        f"- roi_stack_blend_weight: `{selected['roi_stack_blend_weight']:.2f}`",
        f"- threshold: `{selected['metrics']['threshold']:.3f}`",
        "",
        "## OOF Metrics",
        "",
        "| Scheme | Samples | Pos | Neg | AUC | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1 | Balanced Acc | Youden J | Confusion |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        _metrics_row("Current demo OOF", report["baseline_oof_metrics"]),
        _metrics_row("Current demo OOF best threshold", report["baseline_oof_best_threshold_metrics"]),
        _metrics_row("ConvNeXt soup selected OOF", selected["metrics"]),
        "",
        "## Top OOF Candidates",
        "",
        "| Rank | ROI blend | Accepted | AUC | Threshold | Sensitivity | Specificity | Precision | F1 |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(report["top_candidates"], start=1):
        metric = row["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
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
    soup_full = _load_or_generate_soup_full(args)
    soup_roi = _load_or_generate_soup_roi(args)

    eff_full_rows = full_oof["views"]["eff_identity"]
    conv42_full_rows = full_oof["views"][BASE_CONV_VIEW]
    soup_full_rows = soup_full["views"][SOUP_VIEW.name]
    eff_roi_rows = roi_oof["views"]["eff_identity"]
    conv42_roi_rows = roi_oof["views"][BASE_CONV_VIEW]
    soup_roi_rows = soup_roi["views"][SOUP_VIEW.name]
    _validate_same_order(
        eff_full_rows,
        conv42_full_rows,
        soup_full_rows,
        eff_roi_rows,
        conv42_roi_rows,
        soup_roi_rows,
    )
    y_true = _load_labels(eff_full_rows)
    area_ratios = np.asarray(roi_oof["roi_area_ratios"], dtype=np.float64)

    eff_full = _array_from_rows(eff_full_rows)
    conv42_full = _array_from_rows(conv42_full_rows)
    soup_full_array = _array_from_rows(soup_full_rows)
    eff_roi = _array_from_rows(eff_roi_rows)
    conv42_roi = _array_from_rows(conv42_roi_rows)
    soup_roi_array = _array_from_rows(soup_roi_rows)

    baseline_probability = _candidate_probability(
        eff_full=eff_full,
        conv42_full=conv42_full,
        conv123_full=soup_full_array,
        eff_roi=eff_roi,
        conv42_roi=conv42_roi,
        conv123_roi=soup_roi_array,
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
        soup_full=soup_full_array,
        eff_roi=eff_roi,
        soup_roi=soup_roi_array,
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
        "method": "ConvNeXt-Tiny seed42/seed123 weight soup selected on BUSBRA OOF only",
        "data_boundary": {
            "selection": "BUSBRA OOF only",
            "external_review": "Run BUSI separately after candidate config is frozen.",
        },
        "baseline_oof_metrics": _with_sample_counts(baseline_default),
        "baseline_oof_best_threshold_metrics": _with_sample_counts(baseline_best),
        "selected_candidate": selected,
        "top_candidates": candidates,
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
