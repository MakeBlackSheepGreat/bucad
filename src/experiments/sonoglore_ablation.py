"""SonoGloReNet ablation orchestration helpers."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.datasets.busi import load_busi_manifest
from src.engine.inference import BreastUltrasoundInferenceService
from src.engine.train_cls import run_classifier_training
from src.models.classifier import create_classifier
from src.utils.config import load_project_config, load_yaml, save_yaml
from src.utils.reporting import write_json_report, write_markdown_report
from src.utils.runtime import ensure_dir, optional_import, seed_everything


torch = optional_import("torch")


REFERENCE_ROWS = (
    {
        "candidate_id": "B0",
        "candidate_name": "convnext_tiny_timm_recipe_fold1",
        "role": "baseline_reference",
        "model_name": "convnext_tiny",
        "variant_name": "baseline_fold1",
        "train_report": "artifacts/reports/train_cls_convnext_tiny_timm_recipe_fold1.json",
        "busi_report": "artifacts/reports/busi_convnext_tiny_timm_recipe_fold1.json",
        "config_path": "configs/classifier/convnext_tiny_timm_recipe.yml",
        "inference_config": "configs/inference/convnext_tiny_timm_recipe_fold1.yml",
        "train_source": "reused",
        "busi_source": "reused",
        "eligible_for_promotion": False,
    },
    {
        "candidate_id": "R0",
        "candidate_name": "sonoglore_convnext_tiny_fold1",
        "role": "reference_sonoglore",
        "model_name": "sonoglore_convnext_tiny",
        "variant_name": "reference_fold1",
        "train_report": "artifacts/reports/train_cls_sonoglore_convnext_tiny_fold1.json",
        "busi_report": "artifacts/reports/busi_sonoglore_convnext_tiny_fold1.json",
        "config_path": "configs/classifier/sonoglore_convnext_tiny.yml",
        "inference_config": "configs/inference/sonoglore_convnext_tiny_fold1.yml",
        "train_source": "reused",
        "busi_source": "reused",
        "eligible_for_promotion": False,
    },
    {
        "candidate_id": "R1",
        "candidate_name": "sonoglore_convnext_tiny_relu_fold1",
        "role": "appendix_failed_relu",
        "model_name": "sonoglore_convnext_tiny",
        "variant_name": "relu_reference_fold1",
        "train_report": "artifacts/reports/train_cls_sonoglore_convnext_tiny_relu_fold1.json",
        "busi_report": "artifacts/reports/busi_sonoglore_convnext_tiny_relu_fold1.json",
        "config_path": "configs/classifier/sonoglore_convnext_tiny_relu.yml",
        "inference_config": "configs/inference/sonoglore_convnext_tiny_relu_fold1.yml",
        "train_source": "reused",
        "busi_source": "reused",
        "eligible_for_promotion": False,
    },
)

STRUCTURE_CANDIDATES = (
    {
        "candidate_id": "S1",
        "candidate_name": "ms_noattn",
        "stage": "structure",
        "model_overrides": {
            "active_stage_indices": [2, 3, 4],
            "use_stage4_attention": False,
        },
    },
    {
        "candidate_id": "S2",
        "candidate_name": "stage4_only",
        "stage": "structure",
        "model_overrides": {
            "active_stage_indices": [4],
            "use_stage4_attention": False,
        },
    },
    {
        "candidate_id": "S3",
        "candidate_name": "stage34_noattn",
        "stage": "structure",
        "model_overrides": {
            "active_stage_indices": [3, 4],
            "use_stage4_attention": False,
        },
    },
)

REGULARIZATION_CANDIDATES = (
    {
        "candidate_id": "T0",
        "candidate_name": "base",
        "stage": "regularization",
        "training_overrides": {
            "learning_rate": 3e-5,
            "weight_decay": 0.05,
            "scheduler": {
                "name": "warmup_cosine",
                "warmup_epochs": 3,
                "min_lr": 1e-6,
            },
        },
        "data_overrides": {
            "augmentation": {
                "horizontal_flip": True,
                "flip_probability": 0.5,
            }
        },
    },
    {
        "candidate_id": "T1",
        "candidate_name": "regularized",
        "stage": "regularization",
        "model_overrides": {
            "drop_path_rate": 0.10,
        },
        "training_overrides": {
            "learning_rate": 3e-5,
            "weight_decay": 0.08,
            "label_smoothing": 0.03,
            "scheduler": {
                "name": "warmup_cosine",
                "warmup_epochs": 4,
                "min_lr": 1e-6,
            },
        },
        "data_overrides": {
            "augmentation": {
                "horizontal_flip": True,
                "flip_probability": 0.5,
                "rotation_degrees": 6,
                "brightness": 0.04,
                "contrast": 0.08,
                "scale_min": 0.95,
                "scale_max": 1.05,
            }
        },
    },
    {
        "candidate_id": "T2",
        "candidate_name": "cutmix",
        "stage": "regularization",
        "training_overrides": {
            "learning_rate": 3e-5,
            "weight_decay": 0.05,
            "label_smoothing": 0.02,
            "cutmix_alpha": 0.6,
            "mix_probability": 0.5,
            "scheduler": {
                "name": "warmup_cosine",
                "warmup_epochs": 3,
                "min_lr": 1e-6,
            },
        },
    },
    {
        "candidate_id": "T3",
        "candidate_name": "regularized_cutmix",
        "stage": "regularization",
        "model_overrides": {
            "drop_path_rate": 0.10,
        },
        "training_overrides": {
            "learning_rate": 3e-5,
            "weight_decay": 0.08,
            "label_smoothing": 0.03,
            "cutmix_alpha": 0.6,
            "mix_probability": 0.5,
            "scheduler": {
                "name": "warmup_cosine",
                "warmup_epochs": 4,
                "min_lr": 1e-6,
            },
        },
        "data_overrides": {
            "augmentation": {
                "horizontal_flip": True,
                "flip_probability": 0.5,
                "rotation_degrees": 6,
                "brightness": 0.04,
                "contrast": 0.08,
                "scale_min": 0.95,
                "scale_max": 1.05,
            }
        },
    },
)


@dataclass(frozen=True)
class PathsBundle:
    """Resolved filesystem bundle for ablation artifacts."""

    project_root: Path
    tmp_root: Path
    generated_root: Path
    chinese_report: Path
    english_report: Path
    chinese_benchmark_report: Path
    english_benchmark_report: Path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the ablation workflow."""
    parser = argparse.ArgumentParser(description="Run the SonoGloReNet ablation matrix.")
    parser.add_argument(
        "--base-config",
        default="configs/classifier/sonoglore_convnext_tiny.yml",
        help="Base classifier config used to derive ablation variants.",
    )
    parser.add_argument(
        "--base-inference-config",
        default="configs/inference/sonoglore_convnext_tiny_fold1.yml",
        help="Base inference config used to derive ablation evaluation configs.",
    )
    parser.add_argument(
        "--baseline-inference-config",
        default="configs/inference/convnext_tiny_timm_recipe_fold1.yml",
        help="Fold1 baseline inference config reused for latency and comparison rows.",
    )
    parser.add_argument(
        "--baseline-5fold-report",
        default="artifacts/reports/busi_convnext_tiny_timm_recipe_5fold.json",
        help="Frozen 5-fold baseline report used in final benchmark comparisons.",
    )
    parser.add_argument("--fold", type=int, default=1, help="Classifier fold to run in stage 1.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed reused for generated configs and latency sampling.",
    )
    parser.add_argument(
        "--latency-sample-id",
        default="benign (1)",
        help="BUSI sample_id used for single-image latency comparison.",
    )
    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=2,
        help="Number of warm-up inference runs before timing.",
    )
    parser.add_argument(
        "--latency-runs",
        type=int,
        default=10,
        help="Number of timed inference runs for latency measurement.",
    )
    parser.add_argument(
        "--generated-prefix",
        default="sonoglore_ablation_matrix",
        help="Prefix used for generated temporary configs and summary files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs, constructors, and report generation without training.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full SonoGloReNet ablation workflow."""
    seed_everything(int(args.seed))
    base_config, project_paths = load_project_config(args.base_config)
    bundle = _paths_bundle(project_paths.project_root)
    ensure_dir(bundle.tmp_root)
    ensure_dir(bundle.generated_root)

    reference_rows = [_reference_row(project_paths.project_root, row) for row in REFERENCE_ROWS]
    r0_row = _row_by_candidate_id(reference_rows, "R0")
    b0_row = _row_by_candidate_id(reference_rows, "B0")
    if r0_row is None or b0_row is None:
        raise RuntimeError("Reference rows B0 and R0 must be available.")

    report: dict[str, Any] = {
        "method": "SonoGloReNet ablation matrix",
        "date": time.strftime("%Y-%m-%d"),
        "seed": int(args.seed),
        "fold": int(args.fold),
        "dry_run": bool(args.dry_run),
        "data_boundary": {
            "training": "BUSBRA only",
            "internal_validation": "BUSBRA fold validation only",
            "external_evaluation": "BUSI only",
            "busi_used_for_training": False,
        },
        "paths": {
            "base_config": str(Path(args.base_config)),
            "base_inference_config": str(Path(args.base_inference_config)),
            "baseline_inference_config": str(Path(args.baseline_inference_config)),
            "baseline_5fold_report": str(Path(args.baseline_5fold_report)),
        },
        "references": reference_rows,
    }

    structure_rows = _run_stage1_structure_matrix(
        args=args,
        base_config=base_config,
        project_root=project_paths.project_root,
        bundle=bundle,
    )
    report["stage1_structure_rows"] = structure_rows
    structure_ranking = rank_candidates(structure_rows)
    report["stage1_structure_ranking"] = structure_ranking
    structure_champion = structure_ranking[0] if structure_ranking else None
    report["stage1_structure_champion"] = structure_champion

    regularization_rows: list[dict[str, Any]] = []
    regularization_ranking: list[dict[str, Any]] = []
    regularization_champion: dict[str, Any] | None = None
    gate = {
        "balanced_accuracy_threshold": float(r0_row["busi_metrics"]["balanced_accuracy"]),
        "minimum_auc": float(r0_row["busi_metrics"]["auc"]) - 0.005,
        "passed": False,
    }

    if structure_champion is not None:
        regularization_rows = _run_stage2_regularization_matrix(
            args=args,
            base_config=base_config,
            structure_champion=structure_champion,
            project_root=project_paths.project_root,
            bundle=bundle,
        )
        regularization_ranking = rank_candidates(regularization_rows)
        regularization_champion = regularization_ranking[0] if regularization_ranking else None
        report["stage2_regularization_rows"] = regularization_rows
        report["stage2_regularization_ranking"] = regularization_ranking
        report["stage2_regularization_champion"] = regularization_champion
        if regularization_champion is not None:
            gate["candidate_id"] = regularization_champion["candidate_id"]
            gate["candidate_name"] = regularization_champion["candidate_name"]
            gate["balanced_accuracy"] = float(regularization_champion["busi_metrics"]["balanced_accuracy"])
            gate["auc"] = float(regularization_champion["busi_metrics"]["auc"])
            gate["passed"] = bool(
                gate["balanced_accuracy"] > gate["balanced_accuracy_threshold"]
                and gate["auc"] >= gate["minimum_auc"]
            )
    report["promotion_gate"] = gate

    final_benchmark: dict[str, Any] | None = None
    if gate["passed"] and regularization_champion is not None:
        final_benchmark = _run_stage3_final_benchmark(
            args=args,
            base_config=base_config,
            champion=regularization_champion,
            project_root=project_paths.project_root,
            bundle=bundle,
        )
    report["stage3_final_benchmark"] = final_benchmark

    report["candidate_error_summary"] = _candidate_error_summary(
        stage1_rows=structure_rows,
        stage2_rows=regularization_rows,
        baseline_row=b0_row,
    )

    final_candidate = regularization_champion or structure_champion
    if final_candidate is None:
        final_candidate = r0_row
    report["final_candidate"] = final_candidate

    latency = _latency_section(
        args=args,
        busi_root=project_paths.busi_root,
        project_root=project_paths.project_root,
        baseline_config=Path(args.baseline_inference_config),
        final_candidate=final_candidate,
    )
    report["latency"] = latency

    error_analysis = _error_analysis_section(
        final_candidate=final_candidate,
        baseline_row=b0_row,
        bundle=bundle,
        top_n=10,
        write_outputs=not bool(args.dry_run),
    )
    report["error_analysis"] = error_analysis

    summary_path = (
        bundle.tmp_root / f"{args.generated_prefix}_summary.json"
        if bool(args.dry_run)
        else bundle.generated_root / f"{args.generated_prefix}_summary.json"
    )
    report["summary_json"] = str(summary_path)

    fold1_report_paths = _report_output_paths(bundle, dry_run=bool(args.dry_run), benchmark=False)
    chinese_lines = build_chinese_report(report)
    english_lines = build_english_report(report)
    report["report_paths"] = {
        "chinese_fold1": str(write_markdown_report(fold1_report_paths["chinese"], chinese_lines)),
        "english_fold1": str(write_markdown_report(fold1_report_paths["english"], english_lines)),
    }

    if final_benchmark is not None:
        benchmark_report_paths = _report_output_paths(bundle, dry_run=bool(args.dry_run), benchmark=True)
        chinese_benchmark_lines = build_chinese_benchmark_report(report)
        english_benchmark_lines = build_english_benchmark_report(report)
        report["report_paths"]["chinese_benchmark"] = str(
            write_markdown_report(benchmark_report_paths["chinese"], chinese_benchmark_lines)
        )
        report["report_paths"]["english_benchmark"] = str(
            write_markdown_report(benchmark_report_paths["english"], english_benchmark_lines)
        )

    write_json_report(summary_path, _summary_payload(report))
    return report


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the dry-run validation path."""
    report = run(args)
    if report["stage1_structure_rows"]:
        for row in report["stage1_structure_rows"]:
            if row["train_source"] != "dry_run":
                raise AssertionError("Dry-run stage1 rows must be marked as dry_run.")
    return report


def rank_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank candidates by the fixed balanced-metrics-first rule."""
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["busi_metrics"]["balanced_accuracy"]),
            float(row["busi_metrics"]["auc"]),
            float(row["busi_metrics"]["f1_score"]),
            float(row["train_metrics"]["auc"]),
        ),
        reverse=True,
    )
    return [copy.deepcopy(row) for row in ranked]


def build_chinese_report(report: dict[str, Any]) -> list[str]:
    """Render the Chinese fold1 ablation report."""
    structure_table = _candidate_table_lines(report.get("stage1_structure_ranking", []), language="zh")
    regularization_table = _candidate_table_lines(report.get("stage2_regularization_ranking", []), language="zh")
    gate = report.get("promotion_gate", {})
    final_candidate = report.get("final_candidate") or {}
    error_analysis = report.get("error_analysis", {})
    candidate_error_summary = report.get("candidate_error_summary", [])
    latency = report.get("latency", {})
    lines = [
        "# SonoGloReNet 消融矩阵 fold1 筛选报告",
        "",
        "> 数据边界说明：BUSBRA 仅用于训练、内部验证和模型选择；BUSI 仅用于锁定外部评估。本报告复用现有 `B0 / R0 / R1` 参考行，不重复训练。",
        "",
        f"日期：{report.get('date', '-')}",
        "",
        "## 固定参考行",
        "",
        "| 编号 | 角色 | 内部验证 AUC | BUSI AUC | BUSI Balanced Accuracy | Sensitivity | Specificity | F1-Score |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("references", []):
        lines.append(_reference_markdown_row(row))
    lines.extend(
        [
            "",
            "## 第一阶段：结构消融排序",
            "",
            *structure_table,
            "",
            "## 第二阶段：正则矩阵排序",
            "",
            *regularization_table,
            "",
            "## 晋级门控",
            "",
            f"- 参考 `R0` 的 BUSI balanced accuracy：`{_fmt(gate.get('balanced_accuracy_threshold'))}`",
            f"- 参考 `R0` 的最低允许 AUC：`{_fmt(gate.get('minimum_auc'))}`",
            f"- 当前候选：`{gate.get('candidate_id', '-')}` / `{gate.get('candidate_name', '-')}`",
            f"- 当前候选 balanced accuracy：`{_fmt(gate.get('balanced_accuracy'))}`",
            f"- 当前候选 AUC：`{_fmt(gate.get('auc'))}`",
            f"- 门控结果：`{'晋级 5-fold' if gate.get('passed') else '终止在 fold1'}`",
            "",
            "## 最终候选概览",
            "",
            f"- 候选：`{final_candidate.get('candidate_id', '-')}` / `{final_candidate.get('candidate_name', '-')}`",
            f"- 配置：`{final_candidate.get('config_path', '-')}`",
            f"- 推理配置：`{final_candidate.get('inference_config', '-')}`",
            f"- 内部验证 AUC：`{_fmt((final_candidate.get('train_metrics') or {}).get('auc'))}`",
            f"- BUSI AUC：`{_fmt((final_candidate.get('busi_metrics') or {}).get('auc'))}`",
            f"- BUSI balanced accuracy：`{_fmt((final_candidate.get('busi_metrics') or {}).get('balanced_accuracy'))}`",
            "",
            "## 错判与概率分布",
            "",
            "| 候选 | newly_worse | newly_better | both_wrong |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in candidate_error_summary:
        lines.append(
            f"| {row['candidate_id']} / {row['candidate_name']} | {row['newly_worse_count']} | "
            f"{row['newly_better_count']} | {row['both_wrong_count']} |"
        )
    lines.extend(
        [
            "",
            f"- baseline 正确 -> 当前候选错误：`{error_analysis.get('newly_worse_count', '-')}`",
            f"- baseline 错误 -> 当前候选正确：`{error_analysis.get('newly_better_count', '-')}`",
            f"- 双方都错：`{error_analysis.get('both_wrong_count', '-')}`",
            f"- benign 恶性概率均值：baseline `{_fmt(error_analysis.get('baseline_benign_mean'))}`，当前 `{_fmt(error_analysis.get('candidate_benign_mean'))}`",
            f"- malignant 恶性概率均值：baseline `{_fmt(error_analysis.get('baseline_malignant_mean'))}`，当前 `{_fmt(error_analysis.get('candidate_malignant_mean'))}`",
            f"- 差异 CSV：`{error_analysis.get('diff_csv', '-')}`",
            f"- 新增误判 CSV：`{error_analysis.get('worsened_csv', '-')}`",
            "",
            "### 代表性新增误判 Top-N",
            "",
        ]
    )
    for item in error_analysis.get("top_newly_worse_examples", []):
        lines.append(
            f"- `{item['sample_id']}` / `{item['pathology_label']}`: "
            f"baseline `{_fmt(item['baseline_probability'])}` -> current `{_fmt(item['candidate_probability'])}` "
            f"(shift `{_fmt(item['probability_shift'])}`)"
        )
    lines.extend(
        [
            "",
            "## 单图推理耗时",
            "",
            f"- 测试样本：`{latency.get('sample_id', '-')}`",
            f"- 预热次数：`{latency.get('warmup_runs', '-')}`，统计次数：`{latency.get('measured_runs', '-')}`",
        ]
    )
    for entry in latency.get("rows", []):
        lines.append(
            f"- `{entry['label']}`: mean `{_fmt(entry['mean_seconds'], 5)}` s, "
            f"min `{_fmt(entry['min_seconds'], 5)}` s, max `{_fmt(entry['max_seconds'], 5)}` s"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 本轮排序固定按 BUSI balanced accuracy、BUSI AUC、BUSI F1、内部验证 AUC 依次判定。",
            f"- 当前最终候选门控结果：`{'已进入 5-fold' if gate.get('passed') else '未进入 5-fold'}`。",
            "- `R1` 仅保留在附录语境下作为失败参考，不参与晋级判断。",
        ]
    )
    return lines


def build_english_report(report: dict[str, Any]) -> list[str]:
    """Render the English fold1 ablation report."""
    structure_table = _candidate_table_lines(report.get("stage1_structure_ranking", []), language="en")
    regularization_table = _candidate_table_lines(report.get("stage2_regularization_ranking", []), language="en")
    gate = report.get("promotion_gate", {})
    final_candidate = report.get("final_candidate") or {}
    error_analysis = report.get("error_analysis", {})
    candidate_error_summary = report.get("candidate_error_summary", [])
    latency = report.get("latency", {})
    lines = [
        "# SonoGloReNet Ablation Matrix Fold1 Report",
        "",
        "> Data boundary: BUSBRA is used for training, internal validation, and model selection only. BUSI is used as the locked external evaluation set. Existing `B0 / R0 / R1` rows are reused without retraining.",
        "",
        f"Date: {report.get('date', '-')}",
        "",
        "## Fixed References",
        "",
        "| ID | Role | Internal AUC | BUSI AUC | BUSI Balanced Accuracy | Sensitivity | Specificity | F1-Score |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("references", []):
        lines.append(_reference_markdown_row(row))
    lines.extend(
        [
            "",
            "## Stage 1: Structure Ranking",
            "",
            *structure_table,
            "",
            "## Stage 2: Regularization Ranking",
            "",
            *regularization_table,
            "",
            "## Promotion Gate",
            "",
            f"- `R0` BUSI balanced accuracy: `{_fmt(gate.get('balanced_accuracy_threshold'))}`",
            f"- minimum allowed AUC from `R0`: `{_fmt(gate.get('minimum_auc'))}`",
            f"- candidate: `{gate.get('candidate_id', '-')}` / `{gate.get('candidate_name', '-')}`",
            f"- candidate balanced accuracy: `{_fmt(gate.get('balanced_accuracy'))}`",
            f"- candidate AUC: `{_fmt(gate.get('auc'))}`",
            f"- gate decision: `{'promote to 5-fold' if gate.get('passed') else 'stop at fold1'}`",
            "",
            "## Final Candidate Snapshot",
            "",
            f"- candidate: `{final_candidate.get('candidate_id', '-')}` / `{final_candidate.get('candidate_name', '-')}`",
            f"- config: `{final_candidate.get('config_path', '-')}`",
            f"- inference config: `{final_candidate.get('inference_config', '-')}`",
            f"- internal AUC: `{_fmt((final_candidate.get('train_metrics') or {}).get('auc'))}`",
            f"- BUSI AUC: `{_fmt((final_candidate.get('busi_metrics') or {}).get('auc'))}`",
            f"- BUSI balanced accuracy: `{_fmt((final_candidate.get('busi_metrics') or {}).get('balanced_accuracy'))}`",
            "",
            "## Error Analysis and Probability Shift",
            "",
            "| Candidate | newly_worse | newly_better | both_wrong |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in candidate_error_summary:
        lines.append(
            f"| {row['candidate_id']} / {row['candidate_name']} | {row['newly_worse_count']} | "
            f"{row['newly_better_count']} | {row['both_wrong_count']} |"
        )
    lines.extend(
        [
            "",
            f"- baseline correct -> candidate wrong: `{error_analysis.get('newly_worse_count', '-')}`",
            f"- baseline wrong -> candidate correct: `{error_analysis.get('newly_better_count', '-')}`",
            f"- both wrong: `{error_analysis.get('both_wrong_count', '-')}`",
            f"- benign malignant-prob mean: baseline `{_fmt(error_analysis.get('baseline_benign_mean'))}`, candidate `{_fmt(error_analysis.get('candidate_benign_mean'))}`",
            f"- malignant malignant-prob mean: baseline `{_fmt(error_analysis.get('baseline_malignant_mean'))}`, candidate `{_fmt(error_analysis.get('candidate_malignant_mean'))}`",
            f"- diff CSV: `{error_analysis.get('diff_csv', '-')}`",
            f"- worsened-case CSV: `{error_analysis.get('worsened_csv', '-')}`",
            "",
            "### Representative Newly Worse Cases Top-N",
            "",
        ]
    )
    for item in error_analysis.get("top_newly_worse_examples", []):
        lines.append(
            f"- `{item['sample_id']}` / `{item['pathology_label']}`: "
            f"baseline `{_fmt(item['baseline_probability'])}` -> current `{_fmt(item['candidate_probability'])}` "
            f"(shift `{_fmt(item['probability_shift'])}`)"
        )
    lines.extend(
        [
            "",
            "## Single-Image Latency",
            "",
            f"- sample: `{latency.get('sample_id', '-')}`",
            f"- warm-up runs: `{latency.get('warmup_runs', '-')}`, measured runs: `{latency.get('measured_runs', '-')}`",
        ]
    )
    for entry in latency.get("rows", []):
        lines.append(
            f"- `{entry['label']}`: mean `{_fmt(entry['mean_seconds'], 5)}` s, "
            f"min `{_fmt(entry['min_seconds'], 5)}` s, max `{_fmt(entry['max_seconds'], 5)}` s"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- Ranking is fixed by BUSI balanced accuracy, BUSI AUC, BUSI F1, then internal validation AUC.",
            f"- Final gate result in this round: `{'entered 5-fold' if gate.get('passed') else 'did not enter 5-fold'}`.",
            "- `R1` remains an appendix-only failed reference and is excluded from promotion decisions.",
        ]
    )
    return lines


def build_chinese_benchmark_report(report: dict[str, Any]) -> list[str]:
    """Render the Chinese final 5-fold benchmark report."""
    stage3 = report.get("stage3_final_benchmark") or {}
    baseline_5fold = stage3.get("baseline_5fold_metrics", {})
    champion = stage3.get("champion_5fold_metrics", {})
    lines = [
        "# SonoGloReNet 最终 5-fold Benchmark 报告",
        "",
        f"日期：{report.get('date', '-')}",
        "",
        "## 对照对象",
        "",
        f"- 冻结基线：`{stage3.get('baseline_5fold_report', '-')}`",
        f"- 当前冠军：`{stage3.get('champion_candidate_id', '-')}` / `{stage3.get('champion_candidate_name', '-')}`",
        f"- 5-fold inference config：`{stage3.get('inference_config_path', '-')}`",
        "",
        "## BUSI 5-fold 结果",
        "",
        "| 模型 | AUC | Balanced Accuracy | Sensitivity | Specificity | Accuracy | Precision | F1-Score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _metrics_markdown_row("ConvNeXt-Tiny 5-fold baseline", baseline_5fold),
        _metrics_markdown_row("SonoGloReNet champion 5-fold", champion),
        "",
        "## 结论",
        "",
        "- 本报告仅在冠军通过 fold1 门控后生成。",
        "- 所有 5-fold 指标均来自同一套 BUSI 外部评估口径。",
    ]
    return lines


def build_english_benchmark_report(report: dict[str, Any]) -> list[str]:
    """Render the English final 5-fold benchmark report."""
    stage3 = report.get("stage3_final_benchmark") or {}
    baseline_5fold = stage3.get("baseline_5fold_metrics", {})
    champion = stage3.get("champion_5fold_metrics", {})
    lines = [
        "# SonoGloReNet Final 5-Fold Benchmark Report",
        "",
        f"Date: {report.get('date', '-')}",
        "",
        "## Compared Systems",
        "",
        f"- frozen baseline: `{stage3.get('baseline_5fold_report', '-')}`",
        f"- current champion: `{stage3.get('champion_candidate_id', '-')}` / `{stage3.get('champion_candidate_name', '-')}`",
        f"- 5-fold inference config: `{stage3.get('inference_config_path', '-')}`",
        "",
        "## BUSI 5-Fold Results",
        "",
        "| Model | AUC | Balanced Accuracy | Sensitivity | Specificity | Accuracy | Precision | F1-Score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _metrics_markdown_row("ConvNeXt-Tiny 5-fold baseline", baseline_5fold),
        _metrics_markdown_row("SonoGloReNet champion 5-fold", champion),
        "",
        "## Conclusion",
        "",
        "- This report is generated only when the champion passes the fold1 promotion gate.",
        "- All 5-fold metrics use the same BUSI external evaluation protocol.",
    ]
    return lines


def _paths_bundle(project_root: Path) -> PathsBundle:
    """Build the path bundle used by the ablation workflow."""
    generated_root = project_root / "artifacts" / "reports" / "generated"
    temp_root = Path(tempfile.gettempdir()) / "sonoglore_ablation_matrix"
    return PathsBundle(
        project_root=project_root,
        tmp_root=temp_root,
        generated_root=generated_root,
        chinese_report=project_root
        / "artifacts"
        / "reports"
        / "Chinese reports"
        / "01_baseline_model_screening"
        / "sonoglore_ablation_matrix_fold1.md",
        english_report=project_root
        / "artifacts"
        / "reports"
        / "English reports"
        / "sonoglore_ablation_matrix_fold1.md",
        chinese_benchmark_report=project_root
        / "artifacts"
        / "reports"
        / "Chinese reports"
        / "01_baseline_model_screening"
        / "sonoglore_ablation_matrix_final_benchmark.md",
        english_benchmark_report=project_root
        / "artifacts"
        / "reports"
        / "English reports"
        / "sonoglore_ablation_matrix_final_benchmark.md",
    )


def _report_output_paths(
    bundle: PathsBundle,
    *,
    dry_run: bool,
    benchmark: bool,
) -> dict[str, Path]:
    """Return fold1 or benchmark report destinations, using temp paths for dry-run."""
    if benchmark:
        chinese_path = bundle.chinese_benchmark_report
        english_path = bundle.english_benchmark_report
        dry_run_dir_name = "benchmark_reports"
    else:
        chinese_path = bundle.chinese_report
        english_path = bundle.english_report
        dry_run_dir_name = "fold1_reports"
    if not dry_run:
        return {"chinese": chinese_path, "english": english_path}
    return {
        "chinese": bundle.tmp_root / dry_run_dir_name / "chinese" / chinese_path.name,
        "english": bundle.tmp_root / dry_run_dir_name / "english" / english_path.name,
    }


def _reference_row(project_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Load one frozen reference row from existing train/BUSI reports."""
    train_report = project_root / row["train_report"]
    busi_report = project_root / row["busi_report"]
    train_payload = json.loads(train_report.read_text(encoding="utf-8"))
    busi_payload = json.loads(busi_report.read_text(encoding="utf-8"))
    enriched = dict(row)
    enriched["train_report"] = str(train_report)
    enriched["busi_report"] = str(busi_report)
    enriched["checkpoint_path"] = str(train_payload.get("checkpoint_path", ""))
    enriched["train_metrics"] = _extract_metric_block(train_payload.get("metrics", {}))
    enriched["busi_metrics"] = _extract_metric_block(busi_payload.get("metrics", {}))
    enriched["train_report_payload"] = train_payload
    enriched["busi_report_payload"] = busi_payload
    return enriched


def _run_stage1_structure_matrix(
    *,
    args: argparse.Namespace,
    base_config: dict[str, Any],
    project_root: Path,
    bundle: PathsBundle,
) -> list[dict[str, Any]]:
    """Run or validate the structure-only candidate set."""
    rows: list[dict[str, Any]] = []
    for spec in STRUCTURE_CANDIDATES:
        rows.append(
            _run_fold1_candidate(
                args=args,
                base_config=base_config,
                candidate_group="structure",
                spec=spec,
                project_root=project_root,
                bundle=bundle,
            )
        )
    return rows


def _run_stage2_regularization_matrix(
    *,
    args: argparse.Namespace,
    base_config: dict[str, Any],
    structure_champion: dict[str, Any],
    project_root: Path,
    bundle: PathsBundle,
) -> list[dict[str, Any]]:
    """Run the regularization matrix on top of the stage1 champion structure."""
    rows: list[dict[str, Any]] = []
    for spec in REGULARIZATION_CANDIDATES:
        merged_spec = {
            "candidate_id": f"{structure_champion['candidate_id']}_{spec['candidate_id']}",
            "candidate_name": f"{structure_champion['candidate_name']}__{spec['candidate_name']}",
            "stage": "regularization",
            "model_overrides": {
                **structure_champion.get("model_overrides", {}),
                **spec.get("model_overrides", {}),
            },
            "training_overrides": spec.get("training_overrides", {}),
            "data_overrides": spec.get("data_overrides", {}),
            "parent_structure_candidate_id": structure_champion["candidate_id"],
            "parent_structure_candidate_name": structure_champion["candidate_name"],
        }
        rows.append(
            _run_fold1_candidate(
                args=args,
                base_config=base_config,
                candidate_group="regularization",
                spec=merged_spec,
                project_root=project_root,
                bundle=bundle,
            )
        )
    return rows


def _run_stage3_final_benchmark(
    *,
    args: argparse.Namespace,
    base_config: dict[str, Any],
    champion: dict[str, Any],
    project_root: Path,
    bundle: PathsBundle,
) -> dict[str, Any]:
    """Run the promoted champion through 5-fold training and BUSI evaluation."""
    fold_rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        fold_args = copy.deepcopy(args)
        fold_args.fold = fold
        fold_row = _run_fold_candidate(
            args=fold_args,
            base_config=base_config,
            candidate_group="final5fold",
            spec={
                "candidate_id": champion["candidate_id"],
                "candidate_name": champion["candidate_name"],
                "model_overrides": champion.get("model_overrides", {}),
                "training_overrides": champion.get("training_overrides", {}),
                "data_overrides": champion.get("data_overrides", {}),
            },
            project_root=project_root,
            bundle=bundle,
            fold=fold,
        )
        fold_rows.append(fold_row)

    inference_config_path = _write_5fold_inference_config(
        champion=champion,
        fold_rows=fold_rows,
        project_root=project_root,
        bundle=bundle,
    )
    busi_report_path = bundle.generated_root / f"{champion['candidate_name']}_5fold_busi.json"
    busi_report = _evaluate_busi_config(
        inference_config_path,
        output_path=busi_report_path,
        dry_run=bool(args.dry_run),
    )
    baseline_payload = json.loads((project_root / args.baseline_5fold_report).read_text(encoding="utf-8"))
    return {
        "champion_candidate_id": champion["candidate_id"],
        "champion_candidate_name": champion["candidate_name"],
        "fold_rows": fold_rows,
        "inference_config_path": str(inference_config_path),
        "busi_report_path": str(busi_report_path),
        "champion_5fold_metrics": _extract_metric_block(busi_report.get("metrics", {})),
        "baseline_5fold_report": str(project_root / args.baseline_5fold_report),
        "baseline_5fold_metrics": _extract_metric_block(baseline_payload.get("metrics", {})),
    }


def _run_fold1_candidate(
    *,
    args: argparse.Namespace,
    base_config: dict[str, Any],
    candidate_group: str,
    spec: dict[str, Any],
    project_root: Path,
    bundle: PathsBundle,
) -> dict[str, Any]:
    """Run one fold1 candidate and return its summary row."""
    return _run_fold_candidate(
        args=args,
        base_config=base_config,
        candidate_group=candidate_group,
        spec=spec,
        project_root=project_root,
        bundle=bundle,
        fold=int(args.fold),
    )


def _run_fold_candidate(
    *,
    args: argparse.Namespace,
    base_config: dict[str, Any],
    candidate_group: str,
    spec: dict[str, Any],
    project_root: Path,
    bundle: PathsBundle,
    fold: int,
) -> dict[str, Any]:
    """Run one fold for one candidate specification."""
    candidate_name = str(spec["candidate_name"])
    candidate_id = str(spec["candidate_id"])
    classifier_config = _build_candidate_training_config(
        base_config=base_config,
        spec=spec,
        candidate_group=candidate_group,
        fold=fold,
    )
    classifier_config_path = bundle.tmp_root / f"{candidate_name}_fold{fold}.yml"
    save_yaml(classifier_config_path, classifier_config)

    inference_config_path = bundle.tmp_root / f"{candidate_name}_fold{fold}_inference.yml"
    train_report_path = _expected_train_report_path(project_root, classifier_config, fold)
    checkpoint_path = _expected_checkpoint_path(project_root, classifier_config, fold)
    inference_config = _build_candidate_inference_config(
        project_root=project_root,
        base_inference_config=Path(args.base_inference_config),
        model_name=str(classifier_config["model"]["name"]),
        checkpoint_path=checkpoint_path,
        model_kwargs=_inference_model_kwargs(classifier_config["model"]),
        candidate_name=candidate_name,
    )
    save_yaml(inference_config_path, inference_config)
    busi_report_path = bundle.generated_root / f"busi_{candidate_name}_fold{fold}.json"

    if args.dry_run:
        _validate_candidate_constructor(classifier_config)
        train_report = _synthetic_dry_run_train_report(classifier_config, checkpoint_path)
        busi_report = _synthetic_dry_run_busi_report(candidate_name)
        train_source = "dry_run"
        busi_source = "dry_run"
    else:
        if _is_compatible_train_report(train_report_path, classifier_config, fold):
            train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
            train_source = "reused"
        else:
            train_report = run_classifier_training(classifier_config_path, fold=fold)
            train_source = "trained"
        if _is_compatible_busi_report(busi_report_path, candidate_name, classifier_config, fold):
            busi_report = json.loads(busi_report_path.read_text(encoding="utf-8"))
            busi_source = "reused"
        else:
            busi_report = _evaluate_busi_config(inference_config_path, output_path=busi_report_path, dry_run=False)
            busi_source = "evaluated"

    row = {
        "candidate_group": candidate_group,
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "fold": int(fold),
        "config_path": str(classifier_config_path),
        "inference_config": str(inference_config_path),
        "checkpoint_path": str(checkpoint_path),
        "train_report": str(train_report_path if train_report_path.exists() else train_report_path),
        "busi_report": str(busi_report_path),
        "train_source": train_source,
        "busi_source": busi_source,
        "eligible_for_promotion": True,
        "model_config": copy.deepcopy(classifier_config["model"]),
        "model_overrides": copy.deepcopy(spec.get("model_overrides", {})),
        "training_overrides": copy.deepcopy(spec.get("training_overrides", {})),
        "data_overrides": copy.deepcopy(spec.get("data_overrides", {})),
        "train_metrics": _extract_metric_block(train_report.get("metrics", {})),
        "busi_metrics": _extract_metric_block(busi_report.get("metrics", {})),
        "train_report_payload": train_report,
        "busi_report_payload": busi_report,
    }
    return row


def _build_candidate_training_config(
    *,
    base_config: dict[str, Any],
    spec: dict[str, Any],
    candidate_group: str,
    fold: int,
) -> dict[str, Any]:
    """Clone and override the base training config for one candidate."""
    config = copy.deepcopy(base_config)
    candidate_name = str(spec["candidate_name"])
    config["model"].update(copy.deepcopy(spec.get("model_overrides", {})))
    if spec.get("training_overrides"):
        _deep_update(config.setdefault("training", {}), copy.deepcopy(spec["training_overrides"]))
    if spec.get("data_overrides"):
        _deep_update(config.setdefault("data", {}), copy.deepcopy(spec["data_overrides"]))
    config.setdefault("output", {})
    config["output"]["checkpoint_name"] = f"{candidate_name}_fold{{fold}}.pt"
    config["output"]["report_name"] = f"train_cls_{candidate_name}_fold{{fold}}.json"
    config["experiment"] = {
        "candidate_group": candidate_group,
        "candidate_id": spec["candidate_id"],
        "candidate_name": candidate_name,
    }
    return config


def _build_candidate_inference_config(
    *,
    project_root: Path,
    base_inference_config: Path,
    model_name: str,
    checkpoint_path: Path,
    model_kwargs: dict[str, Any],
    candidate_name: str,
) -> dict[str, Any]:
    """Clone and override the base inference config for one candidate."""
    config = load_yaml(project_root / base_inference_config)
    runtime = config.setdefault("runtime", {})
    runtime["classifier_model"] = model_name
    runtime["classifier_checkpoint"] = _relative_to_project(project_root, checkpoint_path)
    runtime["classifier_checkpoints"] = []
    runtime["classifier_members"] = []
    runtime["classifier_model_kwargs"] = model_kwargs
    runtime["primary_classifier_model"] = candidate_name
    runtime["ensemble_display_name"] = f"SonoGloReNet ablation {candidate_name}"
    runtime["classifier_tta_horizontal_flip"] = True
    runtime["classifier_tta_variants"] = [
        {"name": "identity", "crop_pct": 0.95},
        {"name": "hflip", "crop_pct": 0.95},
    ]
    runtime["segmentation_enabled"] = False
    runtime["gradcam_enabled"] = False
    return config


def _evaluate_busi_config(
    config_path: Path,
    *,
    output_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Evaluate BUSI for one inference config, or synthesize a dry-run payload."""
    if dry_run:
        return _synthetic_dry_run_busi_report(config_path.stem)
    from src.engine.inference import evaluate_busi_dataset

    return evaluate_busi_dataset(config_path, output_path=output_path)


def _write_5fold_inference_config(
    *,
    champion: dict[str, Any],
    fold_rows: list[dict[str, Any]],
    project_root: Path,
    bundle: PathsBundle,
) -> Path:
    """Write the 5-fold inference config for the promoted champion."""
    base_config = load_yaml(project_root / "configs/inference/sonoglore_convnext_tiny_fold1.yml")
    runtime = base_config.setdefault("runtime", {})
    runtime["classifier_model"] = fold_rows[0]["model_config"].get("name", "sonoglore_convnext_tiny")
    runtime["classifier_pretrained"] = False
    runtime["classifier_checkpoint"] = None
    runtime["classifier_checkpoints"] = [
        _relative_to_project(project_root, Path(row["checkpoint_path"]))
        for row in fold_rows
    ]
    runtime["classifier_members"] = [
        {
            "model": runtime["classifier_model"],
            "checkpoint": _relative_to_project(project_root, Path(row["checkpoint_path"])),
            "weight": 1.0,
            "model_kwargs": _inference_model_kwargs(row["model_config"]),
            "tta_variants": [
                {"name": "identity", "crop_pct": 0.95},
                {"name": "hflip", "crop_pct": 0.95},
            ],
        }
        for row in fold_rows
    ]
    runtime["classifier_model_kwargs"] = _inference_model_kwargs(fold_rows[0]["model_config"])
    runtime["classifier_tta_horizontal_flip"] = False
    runtime["classifier_tta_variants"] = [
        {"name": "identity", "crop_pct": 0.95},
        {"name": "hflip", "crop_pct": 0.95},
    ]
    runtime["primary_classifier_model"] = champion["candidate_name"]
    runtime["ensemble_display_name"] = f"SonoGloReNet 5-fold {champion['candidate_name']}"
    runtime["segmentation_enabled"] = False
    runtime["gradcam_enabled"] = False
    path = bundle.tmp_root / f"{champion['candidate_name']}_5fold_inference.yml"
    save_yaml(path, base_config)
    return path


def _latency_section(
    *,
    args: argparse.Namespace,
    busi_root: Path,
    project_root: Path,
    baseline_config: Path,
    final_candidate: dict[str, Any],
) -> dict[str, Any]:
    """Measure or synthesize single-image latency comparisons."""
    if args.dry_run:
        return {
            "sample_id": str(args.latency_sample_id),
            "sample_path": None,
            "warmup_runs": int(args.latency_warmup),
            "measured_runs": int(args.latency_runs),
            "rows": [
                {"label": "B0 baseline", "mean_seconds": 0.0, "min_seconds": 0.0, "max_seconds": 0.0},
                {"label": "Final candidate", "mean_seconds": 0.0, "min_seconds": 0.0, "max_seconds": 0.0},
            ],
        }
    manifest = load_busi_manifest(busi_root, include_normal=False)
    if args.latency_sample_id:
        sample_row = manifest[manifest["sample_id"] == args.latency_sample_id]
        if sample_row.empty:
            sample_row = manifest.iloc[[0]]
    else:
        sample_row = manifest.iloc[[0]]
    sample = sample_row.iloc[0]
    sample_path = str(sample["image_path"])
    rows = []
    for label, config_path in (
        ("B0 baseline", project_root / baseline_config),
        ("Final candidate", Path(final_candidate["inference_config"])),
    ):
        service = BreastUltrasoundInferenceService.from_config(config_path)
        for _ in range(int(args.latency_warmup)):
            service.diagnose(sample_path, need_segmentation=False, need_explanation=False)
        measurements = []
        for _ in range(int(args.latency_runs)):
            start = time.perf_counter()
            service.diagnose(sample_path, need_segmentation=False, need_explanation=False)
            measurements.append(time.perf_counter() - start)
        rows.append(
            {
                "label": label,
                "mean_seconds": float(np.mean(measurements)),
                "min_seconds": float(np.min(measurements)),
                "max_seconds": float(np.max(measurements)),
            }
        )
    return {
        "sample_id": str(sample["sample_id"]),
        "sample_path": sample_path,
        "warmup_runs": int(args.latency_warmup),
        "measured_runs": int(args.latency_runs),
        "rows": rows,
    }


def _error_analysis_section(
    *,
    final_candidate: dict[str, Any],
    baseline_row: dict[str, Any],
    bundle: PathsBundle,
    top_n: int,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Generate candidate-vs-baseline error analysis outputs."""
    baseline_rows = baseline_row["busi_report_payload"]["rows"]
    candidate_rows = final_candidate["busi_report_payload"]["rows"]
    baseline_df = pd.DataFrame(baseline_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    worsened_csv = bundle.generated_root / f"{final_candidate['candidate_name']}_worsened_cases.csv"
    diff_csv = bundle.generated_root / f"{final_candidate['candidate_name']}_vs_baseline_case_diff.csv"
    required_columns = {"sample_id", "pathology_label", "malignant_probability", "final_label"}
    if (
        baseline_df.empty
        or candidate_df.empty
        or not required_columns.issubset(set(baseline_df.columns))
        or not required_columns.issubset(set(candidate_df.columns))
    ):
        return {
            "newly_worse_count": 0,
            "newly_better_count": 0,
            "both_wrong_count": 0,
            "baseline_benign_mean": None,
            "candidate_benign_mean": None,
            "baseline_malignant_mean": None,
            "candidate_malignant_mean": None,
            "diff_csv": str(diff_csv),
            "worsened_csv": str(worsened_csv),
            "top_newly_worse_examples": [],
        }
    merged = baseline_df.merge(
        candidate_df,
        on=["sample_id", "pathology_label"],
        suffixes=("_baseline", "_candidate"),
        how="inner",
    )
    if merged.empty:
        if write_outputs:
            newly_worse = pd.DataFrame(columns=list(baseline_df.columns) + list(candidate_df.columns))
            newly_worse.to_csv(worsened_csv, index=False, encoding="utf-8")
            merged.to_csv(diff_csv, index=False, encoding="utf-8")
        return {
            "newly_worse_count": 0,
            "newly_better_count": 0,
            "both_wrong_count": 0,
            "baseline_benign_mean": None,
            "candidate_benign_mean": None,
            "baseline_malignant_mean": None,
            "candidate_malignant_mean": None,
            "diff_csv": str(diff_csv),
            "worsened_csv": str(worsened_csv),
            "top_newly_worse_examples": [],
        }
    truth = (merged["pathology_label"].astype(str).str.lower() == "malignant").astype(int)
    baseline_wrong = (merged["final_label_baseline"].astype(str).str.lower() != merged["pathology_label"].astype(str).str.lower())
    candidate_wrong = (merged["final_label_candidate"].astype(str).str.lower() != merged["pathology_label"].astype(str).str.lower())
    merged["probability_shift"] = merged["malignant_probability_candidate"] - merged["malignant_probability_baseline"]
    newly_worse = merged[(~baseline_wrong) & candidate_wrong].copy()
    newly_better = merged[baseline_wrong & (~candidate_wrong)].copy()
    both_wrong = merged[baseline_wrong & candidate_wrong].copy()
    if write_outputs:
        newly_worse.sort_values("probability_shift", ascending=False).to_csv(
            worsened_csv,
            index=False,
            encoding="utf-8",
        )
        merged.sort_values("probability_shift", ascending=False).to_csv(
            diff_csv,
            index=False,
            encoding="utf-8",
        )
    benign_mask = truth == 0
    malignant_mask = truth == 1
    top_records = newly_worse.assign(
        abs_shift=lambda frame: frame["probability_shift"].abs()
    ).sort_values("abs_shift", ascending=False).head(top_n)
    top_examples = [
        {
            "sample_id": str(row.sample_id),
            "pathology_label": str(row.pathology_label),
            "baseline_probability": float(row.malignant_probability_baseline),
            "candidate_probability": float(row.malignant_probability_candidate),
            "probability_shift": float(row.probability_shift),
        }
        for row in top_records.itertuples(index=False)
    ]
    return {
        "newly_worse_count": int(len(newly_worse)),
        "newly_better_count": int(len(newly_better)),
        "both_wrong_count": int(len(both_wrong)),
        "baseline_benign_mean": _safe_series_mean(merged.loc[benign_mask, "malignant_probability_baseline"]),
        "candidate_benign_mean": _safe_series_mean(merged.loc[benign_mask, "malignant_probability_candidate"]),
        "baseline_malignant_mean": _safe_series_mean(merged.loc[malignant_mask, "malignant_probability_baseline"]),
        "candidate_malignant_mean": _safe_series_mean(merged.loc[malignant_mask, "malignant_probability_candidate"]),
        "diff_csv": str(diff_csv),
        "worsened_csv": str(worsened_csv),
        "top_newly_worse_examples": top_examples,
    }


def _validate_candidate_constructor(config: dict[str, Any]) -> None:
    """Construct the configured model once to validate dry-run candidate assembly."""
    if torch is None:
        return
    model_cfg = config["model"]
    model = create_classifier(
        model_name=str(model_cfg["name"]),
        pretrained=False,
        in_chans=int(model_cfg.get("in_chans", 3)),
        num_classes=int(model_cfg.get("num_classes", 2)),
        **_inference_model_kwargs(model_cfg),
    )
    batch = torch.zeros((2, 3, int(config["data"].get("image_size", 224)), int(config["data"].get("image_size", 224))))
    logits = model(batch)
    if tuple(logits.shape) != (2, 2):
        raise AssertionError("Dry-run validation expected logits shape (2, 2).")


def _synthetic_dry_run_train_report(config: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    """Create a small synthetic train report for dry-run validation."""
    return {
        "checkpoint_path": str(checkpoint_path),
        "metrics": {
            "auc": 0.0,
            "threshold": 0.5,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "f1_score": 0.0,
            "confusion": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
        },
        "model_config": copy.deepcopy(config["model"]),
    }


def _synthetic_dry_run_busi_report(candidate_name: str) -> dict[str, Any]:
    """Create a small synthetic BUSI report for dry-run validation."""
    return {
        "model_identifier": candidate_name,
        "metrics": {
            "auc": 0.0,
            "threshold": 0.5,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "f1_score": 0.0,
            "confusion": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
        },
        "rows": [],
    }


def _extract_metric_block(metrics: dict[str, Any]) -> dict[str, Any]:
    """Normalize one metric block and add balanced accuracy."""
    normalized = dict(metrics or {})
    sensitivity = float(normalized.get("sensitivity", 0.0) or 0.0)
    specificity = float(normalized.get("specificity", 0.0) or 0.0)
    normalized["balanced_accuracy"] = (sensitivity + specificity) / 2.0
    normalized["f1_score"] = float(normalized.get("f1_score", 0.0) or 0.0)
    normalized["precision"] = float(normalized.get("precision", 0.0) or 0.0)
    normalized["accuracy"] = float(normalized.get("accuracy", 0.0) or 0.0)
    normalized["auc"] = float(normalized.get("auc", 0.0) or 0.0)
    normalized["sensitivity"] = sensitivity
    normalized["specificity"] = specificity
    return normalized


def _deep_update(target: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a nested mapping in place."""
    for key, value in override.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _is_compatible_train_report(report_path: Path, classifier_config: dict[str, Any], fold: int) -> bool:
    """Return True only when an existing train report matches the current candidate config."""
    if not report_path.exists():
        return False
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected_training = dict(classifier_config.get("training", {}))
    expected_output = dict(classifier_config.get("output", {}))
    expected_data = dict(classifier_config.get("data", {}))
    expected_checkpoint_name = str(expected_output.get("checkpoint_name", "")).format(fold=fold)
    return (
        int(payload.get("fold", -1)) == int(fold)
        and Path(str(payload.get("checkpoint_path", ""))).name == expected_checkpoint_name
        and payload.get("checkpoint_strategy") == expected_training.get("checkpoint_strategy", payload.get("checkpoint_strategy"))
        and payload.get("loss", expected_training.get("loss", "cross_entropy")) == expected_training.get("loss", "cross_entropy")
        and payload.get("min_specificity", 0.0) == float(expected_training.get("min_specificity", 0.0) or 0.0)
        and payload.get("label_smoothing", 0.0) == float(expected_training.get("label_smoothing", 0.0) or 0.0)
        and payload.get("mixup_alpha", 0.0) == float(expected_training.get("mixup_alpha", 0.0) or 0.0)
        and payload.get("cutmix_alpha", 0.0) == float(expected_training.get("cutmix_alpha", 0.0) or 0.0)
        and payload.get("mix_probability", 0.0) == float(expected_training.get("mix_probability", 0.0) or 0.0)
        and payload.get("scheduler", {}) == (expected_training.get("scheduler", {}) or {})
        and int((payload.get("preprocess_settings") or {}).get("image_size", 0) or 0)
        == int(expected_data.get("image_size", 224))
    )


def _is_compatible_busi_report(
    report_path: Path,
    candidate_name: str,
    classifier_config: dict[str, Any],
    fold: int,
) -> bool:
    """Return True only when an existing BUSI report matches the current candidate config."""
    if not report_path.exists():
        return False
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    runtime_summary = payload.get("runtime_summary", {})
    expected_checkpoint_name = str(classifier_config.get("output", {}).get("checkpoint_name", "")).format(fold=fold)
    return (
        Path(str(payload.get("model_identifier", ""))).name == expected_checkpoint_name
        and int(runtime_summary.get("classifier_member_count", 0)) >= 0
        and candidate_name in str(payload.get("config_path", ""))
        and f"fold{fold}" in str(report_path.name)
    )


def _expected_checkpoint_path(project_root: Path, config: dict[str, Any], fold: int) -> Path:
    """Resolve the checkpoint path that train_cls will produce for one fold."""
    checkpoint_name = str(config["output"]["checkpoint_name"]).format(fold=fold)
    return project_root / "artifacts" / "checkpoints" / checkpoint_name


def _expected_train_report_path(project_root: Path, config: dict[str, Any], fold: int) -> Path:
    """Resolve the train JSON report path that train_cls will produce for one fold."""
    report_name = str(config["output"]["report_name"]).format(fold=fold)
    return project_root / "artifacts" / "reports" / report_name


def _inference_model_kwargs(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract model kwargs that should pass into load_classifier during inference."""
    return {
        key: copy.deepcopy(value)
        for key, value in model_cfg.items()
        if key not in {"name", "pretrained", "in_chans", "num_classes"}
    }


def _relative_to_project(project_root: Path, path: Path) -> str:
    """Return a stable project-relative path string for YAML configs."""
    return "./" + str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")


def _fmt(value: Any, digits: int = 4) -> str:
    """Format a scalar for Markdown tables."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _candidate_table_lines(rows: list[dict[str, Any]], *, language: str) -> list[str]:
    """Build one Markdown table for ranked candidates."""
    if not rows:
        return ["- no candidate rows"]
    lines = [
        "| Rank | Candidate | Internal AUC | BUSI AUC | BUSI Balanced Accuracy | Sensitivity | Specificity | F1-Score |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['candidate_id']} / {row['candidate_name']} | "
            f"{_fmt(row['train_metrics']['auc'])} | {_fmt(row['busi_metrics']['auc'])} | "
            f"{_fmt(row['busi_metrics']['balanced_accuracy'])} | {_fmt(row['busi_metrics']['sensitivity'])} | "
            f"{_fmt(row['busi_metrics']['specificity'])} | {_fmt(row['busi_metrics']['f1_score'])} |"
        )
    return lines


def _reference_markdown_row(row: dict[str, Any]) -> str:
    """Format one fixed reference row for Markdown output."""
    role = row.get("role", "-")
    return (
        f"| {row['candidate_id']} | {role} | {_fmt(row['train_metrics']['auc'])} | "
        f"{_fmt(row['busi_metrics']['auc'])} | {_fmt(row['busi_metrics']['balanced_accuracy'])} | "
        f"{_fmt(row['busi_metrics']['sensitivity'])} | {_fmt(row['busi_metrics']['specificity'])} | "
        f"{_fmt(row['busi_metrics']['f1_score'])} |"
    )


def _metrics_markdown_row(label: str, metrics: dict[str, Any]) -> str:
    """Format one metric row for final 5-fold reports."""
    return (
        f"| {label} | {_fmt(metrics.get('auc'))} | {_fmt(metrics.get('balanced_accuracy'))} | "
        f"{_fmt(metrics.get('sensitivity'))} | {_fmt(metrics.get('specificity'))} | "
        f"{_fmt(metrics.get('accuracy'))} | {_fmt(metrics.get('precision'))} | {_fmt(metrics.get('f1_score'))} |"
    )


def _row_by_candidate_id(rows: list[dict[str, Any]], candidate_id: str) -> dict[str, Any] | None:
    """Return the first row matching the requested candidate id."""
    for row in rows:
        if row.get("candidate_id") == candidate_id:
            return row
    return None


def _candidate_error_summary(
    *,
    stage1_rows: list[dict[str, Any]],
    stage2_rows: list[dict[str, Any]],
    baseline_row: dict[str, Any],
) -> list[dict[str, Any]]:
    """Summarize candidate-vs-baseline error counts across all evaluated candidates."""
    summary_rows: list[dict[str, Any]] = []
    for row in [*stage1_rows, *stage2_rows]:
        summary_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "candidate_name": row["candidate_name"],
                **_pairwise_error_counts(
                    baseline_payload=baseline_row.get("busi_report_payload", {}),
                    candidate_payload=row.get("busi_report_payload", {}),
                ),
            }
        )
    return summary_rows


def _pairwise_error_counts(
    *,
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
) -> dict[str, int]:
    """Return newly worse, newly better, and both-wrong counts between two BUSI payloads."""
    baseline_rows = pd.DataFrame(baseline_payload.get("rows", []))
    candidate_rows = pd.DataFrame(candidate_payload.get("rows", []))
    if baseline_rows.empty or candidate_rows.empty:
        return {
            "newly_worse_count": 0,
            "newly_better_count": 0,
            "both_wrong_count": 0,
        }
    merged = baseline_rows.merge(
        candidate_rows,
        on=["sample_id", "pathology_label"],
        suffixes=("_baseline", "_candidate"),
        how="inner",
    )
    if merged.empty:
        return {
            "newly_worse_count": 0,
            "newly_better_count": 0,
            "both_wrong_count": 0,
        }
    baseline_wrong = (
        merged["final_label_baseline"].astype(str).str.lower()
        != merged["pathology_label"].astype(str).str.lower()
    )
    candidate_wrong = (
        merged["final_label_candidate"].astype(str).str.lower()
        != merged["pathology_label"].astype(str).str.lower()
    )
    return {
        "newly_worse_count": int(((~baseline_wrong) & candidate_wrong).sum()),
        "newly_better_count": int((baseline_wrong & (~candidate_wrong)).sum()),
        "both_wrong_count": int((baseline_wrong & candidate_wrong).sum()),
    }


def _safe_series_mean(series: pd.Series) -> float | None:
    """Return one finite series mean, or None for empty/NaN-only inputs."""
    if series.empty:
        return None
    value = float(series.mean())
    if np.isnan(value):
        return None
    return value


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Return a report copy without embedded per-sample payloads."""
    payload = copy.deepcopy(report)

    def strip_rows(items: list[dict[str, Any]]) -> None:
        for item in items:
            item.pop("train_report_payload", None)
            item.pop("busi_report_payload", None)

    strip_rows(payload.get("references", []))
    strip_rows(payload.get("stage1_structure_rows", []))
    strip_rows(payload.get("stage1_structure_ranking", []))
    strip_rows(payload.get("stage2_regularization_rows", []))
    strip_rows(payload.get("stage2_regularization_ranking", []))
    if isinstance(payload.get("stage1_structure_champion"), dict):
        payload["stage1_structure_champion"].pop("train_report_payload", None)
        payload["stage1_structure_champion"].pop("busi_report_payload", None)
    if isinstance(payload.get("stage2_regularization_champion"), dict):
        payload["stage2_regularization_champion"].pop("train_report_payload", None)
        payload["stage2_regularization_champion"].pop("busi_report_payload", None)
    if isinstance(payload.get("final_candidate"), dict):
        payload["final_candidate"].pop("train_report_payload", None)
        payload["final_candidate"].pop("busi_report_payload", None)
    stage3 = payload.get("stage3_final_benchmark")
    if isinstance(stage3, dict):
        strip_rows(stage3.get("fold_rows", []))
    return payload
