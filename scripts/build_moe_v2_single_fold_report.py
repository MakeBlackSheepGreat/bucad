"""Build Chinese/English single-fold benchmark reports for MoE v2."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = ("auc", "sensitivity", "specificity", "f1_score", "accuracy", "precision")

MODEL_ROWS = (
    {
        "name": "ConvNeXt-Tiny",
        "train_report": "artifacts/reports/train_cls_convnext_tiny_timm_recipe_fold1.json",
        "busi_report": "artifacts/reports/busi_convnext_tiny_timm_recipe_fold1.json",
    },
    {
        "name": "Base MoE v1",
        "train_report": "artifacts/reports/train_cls_sonoglore_lesion_moe_convnext_tiny_fold1.json",
        "busi_report": "artifacts/reports/busi_sonoglore_lesion_moe_convnext_tiny_fold1_tta_crop_sweep.json",
    },
    {
        "name": "Optimized MoE v2",
        "train_report": "artifacts/reports/train_cls_sonoglore_lesion_moe_v2_convnext_tiny_fold1.json",
        "busi_report": "artifacts/reports/busi_sonoglore_lesion_moe_v2_convnext_tiny_fold1_tta_crop_sweep.json",
    },
)


def _load_json(path: str | Path) -> dict[str, Any] | None:
    """Load a JSON object when it exists."""
    json_path = Path(path)
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metrics(payload: dict[str, Any] | None) -> dict[str, float | None]:
    """Return normalized metric fields from a report payload."""
    values = (payload or {}).get("metrics", {})
    return {key: values.get(key) for key in METRIC_KEYS}


def _fmt(value: Any) -> str:
    """Format a metric value for markdown tables."""
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _rows() -> list[dict[str, Any]]:
    """Collect benchmark rows from existing report artifacts."""
    rows = []
    for item in MODEL_ROWS:
        train_report = _load_json(item["train_report"])
        busi_report = _load_json(item["busi_report"])
        internal = _metrics(train_report)
        external = _metrics(busi_report)
        row = {
            "model": item["name"],
            "train_report": item["train_report"],
            "busi_report": item["busi_report"],
            "internal_sample_count": (train_report or {}).get("val_size"),
            "external_sample_count": (busi_report or {}).get("sample_count"),
        }
        for key, value in internal.items():
            row[f"internal_{key}"] = value
        for key, value in external.items():
            row[f"external_{key}"] = value
        rows.append(row)
    return rows


def _markdown_table(rows: list[dict[str, Any]], *, prefix: str) -> list[str]:
    """Build one markdown metrics table."""
    columns = [f"{prefix}_{key}" for key in METRIC_KEYS]
    lines = [
        "| Model | AUC | Sensitivity | Specificity | F1 | Accuracy | Precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = " | ".join(_fmt(row.get(column)) for column in columns)
        lines.append(f"| {row['model']} | {values} |")
    return lines


def _build_zh(rows: list[dict[str, Any]]) -> list[str]:
    """Render the Chinese benchmark report."""
    lines = [
        "# MoE v2 单折基准性能对比",
        "",
        "- 训练/内部验证：BUSBRA fold1 validation。",
        "- 外部测试：BUSI benign/malignant，未用于训练或调参。",
        "- 对照范围：单模型 ConvNeXt-Tiny、基础 MoE v1、优化 MoE v2；不包含主线 ensemble、OOF stacking、area gate 或混合权重。",
        "",
        "## 内部验证指标",
        "",
        *_markdown_table(rows, prefix="internal"),
        "",
        "## BUSI 外部测试指标",
        "",
        *_markdown_table(rows, prefix="external"),
        "",
        "## 结构差异",
        "",
        "- MoE v2 在基础 MoE v1 上增加 area-prior router、load-balance 正则、small-expert 空间分支和 baseline residual path。",
        "- MoE v2 训练 ROI 使用 BUSBRA fold 外 segmenter 预测 mask 缓存，使训练 ROI 更接近推理期分布。",
    ]
    return lines


def _build_en(rows: list[dict[str, Any]]) -> list[str]:
    """Render the English benchmark report."""
    lines = [
        "# MoE v2 Single-Fold Benchmark",
        "",
        "- Internal validation: BUSBRA fold1 validation.",
        "- External test: BUSI benign/malignant only, not used for training or tuning.",
        "- Scope: single-model ConvNeXt-Tiny, base MoE v1, and optimized MoE v2; no mainline ensemble, OOF stacking, area gate, or mixed weights.",
        "",
        "## Internal Validation",
        "",
        *_markdown_table(rows, prefix="internal"),
        "",
        "## BUSI External Test",
        "",
        *_markdown_table(rows, prefix="external"),
        "",
        "## Architecture Delta",
        "",
        "- MoE v2 adds an area-prior router, load-balance regularization, a spatial small-expert branch, and a baseline residual path on top of MoE v1.",
        "- MoE v2 trains with BUSBRA out-of-fold segmenter-predicted ROI mask caches to better match deployment-time ROI distribution.",
    ]
    return lines


def main() -> int:
    """Write CSV plus bilingual markdown reports."""
    rows = _rows()
    output_dir = Path("artifacts/reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "moe_v2_single_fold_benchmark.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "moe_v2_single_fold_benchmark_zh.md").write_text(
        "\n".join(_build_zh(rows)) + "\n",
        encoding="utf-8",
    )
    (output_dir / "moe_v2_single_fold_benchmark_en.md").write_text(
        "\n".join(_build_en(rows)) + "\n",
        encoding="utf-8",
    )
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
