"""Build a bilingual all-dataset metric comparison including LesioNeXt-LENS v1a."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "artifacts" / "reports" / "fixed_classification_benchmark"
INDEX_PATH = REPORT_ROOT / "benchmark_index.json"

MODEL_NAMES = {
    "resnet18": "ResNet18",
    "densenet121": "DenseNet121",
    "efficientnetv2_s": "EfficientNetV2-S",
    "convnext_tiny": "ConvNeXt-Tiny",
    "convnext_small": "ConvNeXt-Small",
    "swin_tiny": "Swin-Tiny",
    "lesionext_moe_v3": "LesioNeXt-MoE V3",
    "lesionext_lens_v1a": "LesioNeXt-LENS v1a",
}
MODEL_ORDER = list(MODEL_NAMES)
DATASET_NAMES = {
    "busbra": "BUSBRA pooled OOF",
    "busi": "BUSI",
    "bus_uclm": "BUS-UCLM",
    "busi_whu": "BUSI-WHU",
    "tcia_breast_us": "TCIA BrEaST",
}
METRICS = ("auc", "accuracy", "sensitivity", "specificity", "precision", "f1_score")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _v1a_path(dataset_id: str) -> Path:
    if dataset_id == "busbra":
        return REPORT_ROOT.parent / "lesionext_lens_v1a_evidence_only_5fold_oof.json"
    names = {
        "busi": "busi_lesionext_lens_v1a_5fold_identity_external.json",
        "bus_uclm": "external_bus_uclm_lesionext_lens_v1a_5fold_identity.json",
        "busi_whu": "external_busi_whu_lesionext_lens_v1a_5fold_identity.json",
        "tcia_breast_us": "external_tcia_breast_us_lesionext_lens_v1a_5fold_identity.json",
    }
    return REPORT_ROOT.parent / names[dataset_id]


def _collect() -> list[dict]:
    index = _load(INDEX_PATH)
    rows: list[dict] = []
    for model_id in MODEL_ORDER:
        for dataset_id in DATASET_NAMES:
            if model_id == "lesionext_lens_v1a":
                path = _v1a_path(dataset_id)
            elif dataset_id == "busbra":
                path = REPORT_ROOT / "oof" / f"{model_id}.json"
            else:
                match = next(
                    row
                    for row in index
                    if row.get("model_id") == model_id and row.get("dataset_id") == dataset_id
                )
                path = Path(match["output"])
            report = _load(path)
            metrics = report.get("metrics", {})
            ci = report.get("auc_bootstrap_ci") or {}
            row = {
                "model_id": model_id,
                "model": MODEL_NAMES[model_id],
                "dataset_id": dataset_id,
                "dataset": DATASET_NAMES[dataset_id],
                "sample_count": report.get("sample_count", "-"),
                "source": str(path),
                "auc_ci": (
                    f"[{float(ci['lower']):.4f}, {float(ci['upper']):.4f}]"
                    if ci.get("lower") is not None and ci.get("upper") is not None
                    else "-"
                ),
            }
            row.update({metric: metrics.get(metric) for metric in METRICS})
            rows.append(row)
    return rows


def _fmt(value) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _write_csv(rows: list[dict]) -> Path:
    path = REPORT_ROOT / "all_model_comparison_v1a.csv"
    fields = ["model_id", "model", "dataset_id", "dataset", "sample_count", *METRICS, "auc_ci", "source"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_markdown(rows: list[dict], *, english: bool) -> Path:
    if english:
        path = REPORT_ROOT / "all_model_comparison_v1a_en.md"
        lines = [
            "# All-Model, All-Dataset Classification Comparison",
            "",
            "The table combines the fixed seven-model benchmark and the newly evaluated LesioNeXt-LENS v1a. BUSBRA uses pooled five-fold OOF; the four independent cohorts use frozen identity-only inference and fixed threshold 0.50. External AUC intervals are bootstrap 95% CIs.",
            "",
        ]
        header = "| Model | n | AUC | AUC 95% CI | Accuracy | Sensitivity | Specificity | Precision | F1 |"
    else:
        path = REPORT_ROOT / "all_model_comparison_v1a_zh.md"
        lines = [
            "# 全模型全数据集分类性能对比",
            "",
            "本表合并固定七模型基准与新增 LesioNeXt-LENS v1a。BUSBRA 使用 pooled 五折 OOF；四个独立队列使用冻结的 identity-only 推理和固定阈值 0.50。外部 AUC 区间为 bootstrap 95% CI。",
            "",
        ]
        header = "| 模型 | n | AUC | AUC 95% CI | Accuracy | Sensitivity | Specificity | Precision | F1 |"
    for dataset_id in DATASET_NAMES:
        dataset = DATASET_NAMES[dataset_id]
        lines.extend([f"## {dataset}", "", header, "|---|---:|---:|---|---:|---:|---:|---:|---:|"])
        for row in rows:
            if row["dataset_id"] != dataset_id:
                continue
            lines.append(
                f"| {row['model']} | {row['sample_count']} | {_fmt(row['auc'])} | {row['auc_ci']} | "
                f"{_fmt(row['accuracy'])} | {_fmt(row['sensitivity'])} | {_fmt(row['specificity'])} | "
                f"{_fmt(row['precision'])} | {_fmt(row['f1_score'])} |"
            )
        lines.append("")
    if english:
        lines.extend([
            "## Reading Notes",
            "",
            "- `LesioNeXt-LENS v1a` is the five-fold evidence-only candidate. Its internal pooled OOF Sensitivity is 0.7694, slightly below the ConvNeXt-Tiny control at 0.7743; this caveat remains part of the comparison.",
            "- Existing baseline rows come from the fixed benchmark artifacts and follow the same external dataset manifests.",
            "- Precision and F1 are threshold-dependent at the frozen threshold; AUC is the ranking metric.",
        ])
    else:
        lines.extend([
            "## 阅读说明",
            "",
            "- `LesioNeXt-LENS v1a` 为五折 evidence-only 候选，内部 pooled OOF Sensitivity 为 0.7694，略低于 ConvNeXt-Tiny 的 0.7743；该限制保留在比较中。",
            "- 其他基线来自固定模型基准报告，使用同一批外部数据清单。",
            "- Precision 与 F1 依赖固定阈值；AUC 用于排序性能比较。",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    rows = _collect()
    csv_path = _write_csv(rows)
    zh_path = _write_markdown(rows, english=False)
    en_path = _write_markdown(rows, english=True)
    archive_zh = ROOT / "artifacts" / "reports" / "Chinese reports" / "07_fixed_classification_benchmark" / "all_model_comparison_v1a.md"
    archive_en = ROOT / "artifacts" / "reports" / "English reports" / "07_fixed_classification_benchmark" / "all_model_comparison_v1a.md"
    archive_zh.parent.mkdir(parents=True, exist_ok=True)
    archive_en.parent.mkdir(parents=True, exist_ok=True)
    archive_zh.write_text(zh_path.read_text(encoding="utf-8"), encoding="utf-8")
    archive_en.write_text(en_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "zh": str(zh_path), "en": str(en_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
