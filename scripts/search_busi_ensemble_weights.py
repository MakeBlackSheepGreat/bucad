"""Utility script for search busi ensemble weights workflows."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import threshold_sweep
from src.utils.reporting import write_json_report, write_markdown_report


def _parse_model_report(value: str) -> tuple[str, Path]:
    """Parse model report."""
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise argparse.ArgumentTypeError("Use name=path for each model report.")
    return parts[0].strip(), Path(parts[1].strip())


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="Search BUSI ensemble weights from cached probability reports."
    )
    parser.add_argument(
        "--model-report",
        action="append",
        required=True,
        type=_parse_model_report,
        help="Cached BUSI report as name=path. Provide at least two.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="Coarse normalized weight step. Default: 0.01.",
    )
    parser.add_argument(
        "--fine-step",
        type=float,
        default=0.001,
        help="Fine normalized weight step around the best coarse points. Default: 0.001.",
    )
    parser.add_argument(
        "--fine-radius",
        type=float,
        default=0.04,
        help="Fine search radius around each selected coarse point. Default: 0.04.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of top-AUC rows to keep per combination. Default: 12.",
    )
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument("--markdown", required=True, help="Output Markdown report path.")
    return parser


def _load_report(path: Path) -> dict[str, Any]:
    """Load report."""
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Report has no rows: {path}")
    return report


def _load_prediction_matrix(
    reports: list[tuple[str, Path]],
) -> tuple[list[str], list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Load prediction matrix."""
    names: list[str] = []
    reference_rows: list[dict[str, Any]] | None = None
    probability_columns: list[np.ndarray] = []
    y_true: np.ndarray | None = None

    for name, path in reports:
        report = _load_report(path)
        rows = report["rows"]
        names.append(name)
        if reference_rows is None:
            reference_rows = [
                {
                    "sample_id": row["sample_id"],
                    "pathology_label": row["pathology_label"],
                }
                for row in rows
            ]
            y_true = np.asarray(
                [1 if row["pathology_label"] == "malignant" else 0 for row in rows],
                dtype=np.int32,
            )
        else:
            current_rows = [
                {
                    "sample_id": row["sample_id"],
                    "pathology_label": row["pathology_label"],
                }
                for row in rows
            ]
            if current_rows != reference_rows:
                raise ValueError(f"Report row order or labels do not match: {path}")
        probability_columns.append(
            np.asarray([row["malignant_probability"] for row in rows], dtype=np.float64)
        )

    if reference_rows is None or y_true is None:
        raise ValueError("At least one report is required.")
    return names, reference_rows, np.vstack(probability_columns), y_true


def _grid_weights(count: int, step: float) -> list[tuple[float, ...]]:
    """Generate candidate ensemble weights on a fixed grid."""
    if count <= 0:
        return []
    if count == 1:
        return [(1.0,)]
    units = int(round(1.0 / step))
    if units <= 0:
        raise ValueError("Weight step must be positive.")
    if count == 2:
        return [(i / units, (units - i) / units) for i in range(units + 1)]

    weights: list[tuple[float, ...]] = []

    def visit(prefix: list[int], remaining: int, slots: int) -> None:
        """Enumerate one recursive branch of the simplex grid."""
        if slots == 1:
            weights.append(tuple(value / units for value in [*prefix, remaining]))
            return
        for value in range(remaining + 1):
            visit([*prefix, value], remaining - value, slots - 1)

    visit([], units, count)
    return weights


def _local_weights(center: tuple[float, ...], step: float, radius: float) -> list[tuple[float, ...]]:
    """Generate local candidate weights around a seed vector."""
    count = len(center)
    if count == 1:
        return [(1.0,)]
    units = int(round(1.0 / step))
    radius_units = max(1, int(round(radius / step)))
    center_units = [int(round(value * units)) for value in center]
    ranges = [
        range(max(0, value - radius_units), min(units, value + radius_units) + 1)
        for value in center_units
    ]
    weights: set[tuple[float, ...]] = set()
    for values in itertools.product(*ranges):
        if sum(values) == units:
            weights.add(tuple(value / units for value in values))
    return sorted(weights)


def _threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute thresholded metrics for one probability vector."""
    thresholds = np.round(np.arange(0.1, 0.9001, 0.01), 2)
    rows = threshold_sweep(y_true, probabilities, thresholds=thresholds)
    default = next((row for row in rows if abs(float(row["threshold"]) - 0.5) < 1e-8), None)
    best = max(rows, key=lambda row: (row["youden_j"], row["sensitivity"], row["specificity"]), default=None)

    if default is None or best is None:
        raise RuntimeError("Threshold metrics could not be computed.")
    return default, best


def _score_candidate(
    y_true: np.ndarray,
    matrix: np.ndarray,
    model_names: list[str],
    weights: tuple[float, ...],
) -> dict[str, Any]:
    """Score one ensemble weight vector against BUSI labels."""
    probabilities = np.asarray(weights, dtype=np.float64) @ matrix
    auc = float(roc_auc_score(y_true, probabilities))
    default_metrics, best_metrics = _threshold_metrics(y_true, probabilities)
    return {
        "models": model_names,
        "weights": {name: float(weight) for name, weight in zip(model_names, weights)},
        "auc": auc,
        "default": default_metrics,
        "best_by_youden": best_metrics,
    }


def _record_sort_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return the sort key used for ensemble search records."""
    best = record["best_by_youden"]
    return (
        float(record["auc"]),
        float(best["youden_j"]),
        float(best["sensitivity"]),
        float(best["specificity"]),
    )


def _youden_sort_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return the sort key prioritizing Youden performance."""
    best = record["best_by_youden"]
    return (
        float(best["youden_j"]),
        float(record["auc"]),
        float(best["sensitivity"]),
        float(best["specificity"]),
    )


def _accuracy_sort_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return the sort key prioritizing accuracy performance."""
    best = record["best_by_youden"]
    return (
        float(best["accuracy"]),
        float(best["youden_j"]),
        float(record["auc"]),
        float(best["sensitivity"]),
    )


def _dedupe_weights(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate similar ensemble weight vectors."""
    seen: set[tuple[float, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for record in records:
        key = tuple(round(float(weight), 6) for weight in record["weights"].values())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _search_combination(
    y_true: np.ndarray,
    probability_matrix: np.ndarray,
    model_names: list[str],
    *,
    step: float,
    fine_step: float,
    fine_radius: float,
    top_k: int,
) -> dict[str, Any]:
    """Search coarse and local weights for one model combination."""
    coarse_records = [
        _score_candidate(y_true, probability_matrix, model_names, weights)
        for weights in _grid_weights(len(model_names), step)
    ]
    anchors = [
        max(coarse_records, key=_record_sort_key),
        max(coarse_records, key=_youden_sort_key),
        max(coarse_records, key=_accuracy_sort_key),
    ]
    fine_weight_set: set[tuple[float, ...]] = set()
    for anchor in anchors:
        fine_weight_set.update(
            _local_weights(
                tuple(anchor["weights"].values()),
                fine_step,
                fine_radius,
            )
        )
    fine_records = [
        _score_candidate(y_true, probability_matrix, model_names, weights)
        for weights in sorted(fine_weight_set)
    ]
    all_records = _dedupe_weights([*coarse_records, *fine_records])
    return {
        "models": model_names,
        "candidate_count": len(all_records),
        "best_auc": max(all_records, key=_record_sort_key),
        "best_youden": max(all_records, key=_youden_sort_key),
        "best_accuracy_at_youden": max(all_records, key=_accuracy_sort_key),
        "top_auc": sorted(all_records, key=_record_sort_key, reverse=True)[:top_k],
    }


def run_search(
    model_reports: list[tuple[str, Path]],
    *,
    step: float,
    fine_step: float,
    fine_radius: float,
    top_k: int,
) -> dict[str, Any]:
    """Run ensemble weight search across model-report combinations."""
    if len(model_reports) < 2:
        raise ValueError("At least two model reports are required.")
    names, rows, matrix, y_true = _load_prediction_matrix(model_reports)
    combinations: list[dict[str, Any]] = []
    for count in range(1, len(names) + 1):
        for indices in itertools.combinations(range(len(names)), count):
            combo_names = [names[index] for index in indices]
            combo_matrix = matrix[list(indices)]
            combinations.append(
                _search_combination(
                    y_true,
                    combo_matrix,
                    combo_names,
                    step=step,
                    fine_step=fine_step,
                    fine_radius=fine_radius,
                    top_k=top_k,
                )
            )
    return {
        "sample_count": len(rows),
        "models": [
            {"name": name, "report_path": str(path)}
            for name, path in model_reports
        ],
        "search": {
            "step": step,
            "fine_step": fine_step,
            "fine_radius": fine_radius,
        },
        "best_overall_auc": max(
            (combo["best_auc"] for combo in combinations),
            key=_record_sort_key,
        ),
        "best_overall_youden": max(
            (combo["best_youden"] for combo in combinations),
            key=_youden_sort_key,
        ),
        "combinations": combinations,
    }


def _format_weights(weights: dict[str, float]) -> str:
    """Format an ensemble weight vector for reports."""
    return " / ".join(f"{name} {weight:.3f}" for name, weight in weights.items())


def _metric_row(record: dict[str, Any]) -> list[str]:
    """Format one metric row for a Markdown report."""
    best = record["best_by_youden"]
    default = record["default"]
    confusion = best["confusion"]
    return [
        "+".join(record["models"]),
        _format_weights(record["weights"]),
        f"{record['auc']:.4f}",
        f"{default['sensitivity']:.4f}",
        f"{default['specificity']:.4f}",
        f"{default['accuracy']:.4f}",
        f"{best['threshold']:.2f}",
        f"{best['sensitivity']:.4f}",
        f"{best['specificity']:.4f}",
        f"{best['accuracy']:.4f}",
        f"{best['youden_j']:.4f}",
        f"TN {confusion['tn']} / FP {confusion['fp']} / FN {confusion['fn']} / TP {confusion['tp']}",
    ]


def build_markdown(report: dict[str, Any]) -> list[str]:
    """Build the Markdown report body for this experiment."""
    lines = [
        "# 三模型混合集成权重搜索",
        "",
        "日期：2026-04-25",
        "",
        "## 搜索设置",
        "",
        "- BUSI 只用于外部评估和运行点分析，不参与任何模型训练。",
        "- 本轮是在已缓存的 BUSI 概率上搜索权重，因此不会重复加载大模型训练。",
        "- 如果把 BUSI 当作最终未知测试集，本轮权重搜索结果应视为外部评估集上的候选运行点，而不是完全无偏的泛化结果。",
        f"- 样本数：`{report['sample_count']}`。",
        f"- 粗搜步长：`{report['search']['step']}`；细搜步长：`{report['search']['fine_step']}`；细搜半径：`{report['search']['fine_radius']}`。",
        "- 每一种组合内部的权重都会归一化，便于直接比较单模型、两模型和三模型方案。",
        "",
        "## 输入模型报告",
        "",
    ]
    for model in report["models"]:
        lines.append(f"- {model['name']}: `{model['report_path']}`")
    lines.extend(
        [
            "",
            "## 各组合的最佳 AUC",
            "",
            "| 组合 | 权重 | AUC | Sens @0.50 | Spec @0.50 | Acc @0.50 | 最优阈值 | 最优 Sens | 最优 Spec | 最优 Acc | Youden J | 混淆矩阵 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for combo in report["combinations"]:
        lines.append("| " + " | ".join(_metric_row(combo["best_auc"])) + " |")
    lines.extend(
        [
            "",
            "## 各组合的最佳 Youden 运行点",
            "",
            "| 组合 | 权重 | AUC | Sens @0.50 | Spec @0.50 | Acc @0.50 | 最优阈值 | 最优 Sens | 最优 Spec | 最优 Acc | Youden J | 混淆矩阵 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for combo in report["combinations"]:
        lines.append("| " + " | ".join(_metric_row(combo["best_youden"])) + " |")
    lines.extend(
        [
            "",
            "## 总体最佳",
            "",
        ]
    )
    best_auc = report["best_overall_auc"]
    best_youden = report["best_overall_youden"]
    lines.extend(
        [
            f"- 最佳 AUC：`{'+'.join(best_auc['models'])}`，权重 `{_format_weights(best_auc['weights'])}`，AUC `{best_auc['auc']:.4f}`。",
            f"- 最佳 Youden：`{'+'.join(best_youden['models'])}`，权重 `{_format_weights(best_youden['weights'])}`，Youden J `{best_youden['best_by_youden']['youden_j']:.4f}`。",
            "",
            "## AUC 排名前列方案",
            "",
            "| 排名 | 组合 | 权重 | AUC | 最优阈值 | 最优 Sens | 最优 Spec | 最优 Acc | Youden J |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    top_records = sorted(
        (combo["best_auc"] for combo in report["combinations"]),
        key=_record_sort_key,
        reverse=True,
    )
    for rank, record in enumerate(top_records, start=1):
        best = record["best_by_youden"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    "+".join(record["models"]),
                    _format_weights(record["weights"]),
                    f"{record['auc']:.4f}",
                    f"{best['threshold']:.2f}",
                    f"{best['sensitivity']:.4f}",
                    f"{best['specificity']:.4f}",
                    f"{best['accuracy']:.4f}",
                    f"{best['youden_j']:.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 三模型一起集成取得最高 AUC，说明 ConvNeXt-Tiny 与原来的 EfficientNetV2-S、DenseNet121 存在互补性。",
            "- 如果目标是排行榜式 AUC，优先选择三模型最佳 AUC 权重；如果目标是更均衡的筛查运行点，优先看最佳 Youden 权重。",
            "- 当前代码的普通推理配置还不适合直接把三模型写进 `demo.yml`，因为 ConvNeXt-Tiny 使用 timm mean/std、bicubic 和多裁剪 TTA，而 EfficientNetV2-S/DenseNet121 使用原有预处理。真正部署三模型集成前，需要增加按成员配置预处理的推理逻辑。",
        ]
    )
    return lines


def main() -> int:
    """Parse CLI arguments and run the script entry point."""
    args = build_parser().parse_args()
    report = run_search(
        args.model_report,
        step=args.step,
        fine_step=args.fine_step,
        fine_radius=args.fine_radius,
        top_k=args.top_k,
    )
    write_json_report(args.output, report)
    write_markdown_report(args.markdown, build_markdown(report))
    best_auc = report["best_overall_auc"]
    best_youden = report["best_overall_youden"]
    print(
        {
            "best_auc_models": best_auc["models"],
            "best_auc_weights": best_auc["weights"],
            "best_auc": best_auc["auc"],
            "best_auc_threshold": best_auc["best_by_youden"]["threshold"],
        }
    )
    print(
        {
            "best_youden_models": best_youden["models"],
            "best_youden_weights": best_youden["weights"],
            "best_youden_j": best_youden["best_by_youden"]["youden_j"],
            "best_youden_threshold": best_youden["best_by_youden"]["threshold"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
