"""Run the fixed six-model classification benchmark on all locked datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.inference import evaluate_busi_dataset, evaluate_external_bus_dataset


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "artifacts" / "reports" / "fixed_classification_benchmark"

FIXED_MODELS: dict[str, dict[str, str]] = {
    "resnet18": {
        "display_name": "ResNet18",
        "config": "configs/inference/resnet18_5fold_identity.yml",
    },
    "densenet121": {
        "display_name": "DenseNet121",
        "config": "configs/inference/densenet121_5fold_tta_identity.yml",
    },
    "efficientnetv2_s": {
        "display_name": "EfficientNetV2-S",
        "config": "configs/inference/efficientnetv2_s_5fold_identity.yml",
    },
    "convnext_tiny": {
        "display_name": "ConvNeXt-Tiny",
        "config": "configs/inference/convnext_tiny_timm_recipe_5fold_tta_identity.yml",
    },
    "swin_tiny": {
        "display_name": "Swin-Tiny",
        "config": "configs/inference/swin_tiny_timm_recipe_5fold_tta_identity.yml",
    },
    "lesionext_lens_v1a": {
        "display_name": "LesioNeXt-LENS v1a",
        "config": "configs/inference/lesionext_lens_v1a_5fold_identity.yml",
    },
}

FIXED_DATASETS: dict[str, dict[str, str]] = {
    "busi": {"display_name": "BUSI", "kind": "busi"},
    "bus_uclm": {
        "display_name": "BUS-UCLM",
        "kind": "external",
        "root": "data/external/bus_uclm",
        "manifest": "data/external/bus_uclm/manifest.csv",
    },
    "busi_whu": {
        "display_name": "BUSI-WHU",
        "kind": "external",
        "root": "data/external/busi_whu",
        "manifest": "data/external/busi_whu/manifests/manifest_external_788.csv",
    },
    "tcia_breast_us": {
        "display_name": "TCIA BrEaST",
        "kind": "external",
        "root": "data/external/tcia_breast_us",
        "manifest": "data/external/tcia_breast_us/manifests/manifest.csv",
    },
}


def _write_prediction_csv(report: dict[str, Any], output_path: Path) -> Path:
    """Write per-sample predictions beside a benchmark JSON report."""
    rows = report.get("rows", [])
    destination = output_path.with_name(f"{output_path.stem}_predictions.csv")
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of prediction rows in {output_path}")

    fieldnames: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Expected a mapping prediction row in {output_path}")
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _bootstrap_auc(report: dict[str, Any], replicates: int) -> dict[str, Any] | None:
    rows = report.get("rows", [])
    if not rows or replicates <= 0:
        return None
    from sklearn.metrics import roc_auc_score

    labels = np.asarray([int(row["pathology_label"] == "malignant") for row in rows], dtype=np.int32)
    scores = np.asarray([float(row["malignant_probability"]) for row in rows], dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return None
    rng = np.random.default_rng(20260806)
    values: list[float] = []
    for _ in range(replicates):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled = labels[indices]
        if len(np.unique(sampled)) == 2:
            values.append(float(roc_auc_score(sampled, scores[indices])))
    if not values:
        return None
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {"confidence": 0.95, "lower": float(lower), "upper": float(upper), "replicates": len(values)}


def _run_one(model_id: str, dataset_id: str, bootstrap: int, rerun: bool) -> dict[str, Any]:
    model = FIXED_MODELS[model_id]
    dataset = FIXED_DATASETS[dataset_id]
    output_path = REPORT_ROOT / "raw" / f"{model_id}__{dataset_id}.json"
    if output_path.exists() and not rerun:
        report = json.loads(output_path.read_text(encoding="utf-8"))
        _write_prediction_csv(report, output_path)
        return {
            "model_id": model_id,
            "model_name": model["display_name"],
            "dataset_id": dataset_id,
            "dataset_name": dataset["display_name"],
            "status": "completed",
            "output": str(output_path),
            "sample_count": report.get("sample_count"),
            "metrics": report.get("metrics", {}),
            "auc_bootstrap_ci": report.get("auc_bootstrap_ci"),
        }
    config_path = ROOT / model["config"]
    if dataset["kind"] == "busi":
        report = evaluate_busi_dataset(config_path, output_path=output_path)
        report["auc_bootstrap_ci"] = _bootstrap_auc(report, bootstrap)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_prediction_csv(report, output_path)
    else:
        report = evaluate_external_bus_dataset(
            dataset_id,
            ROOT / dataset["root"],
            config_path=config_path,
            manifest_path=ROOT / dataset["manifest"],
            output_path=output_path,
            bootstrap_replicates=bootstrap,
        )
    return {
        "model_id": model_id,
        "model_name": model["display_name"],
        "dataset_id": dataset_id,
        "dataset_name": dataset["display_name"],
        "status": "completed",
        "output": str(output_path),
        "sample_count": report.get("sample_count"),
        "metrics": report.get("metrics", {}),
        "auc_bootstrap_ci": report.get("auc_bootstrap_ci"),
    }


def _load_completed(index_path: Path) -> list[dict[str, Any]]:
    if not index_path.exists():
        return []
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in benchmark index: {index_path}")
    completed: list[dict[str, Any]] = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("model_id") not in FIXED_MODELS
            or row.get("dataset_id") not in FIXED_DATASETS
        ):
            continue
        normalized = dict(row)
        local_output = REPORT_ROOT / "raw" / f"{normalized['model_id']}__{normalized['dataset_id']}.json"
        if local_output.exists():
            normalized["output"] = str(local_output)
        completed.append(normalized)
    return completed


def _write_index(rows: list[dict[str, Any]]) -> Path:
    path = REPORT_ROOT / "benchmark_index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _metric_value(row: dict[str, Any], key: str) -> str:
    value = row.get("metrics", {}).get(key)
    return "-" if value is None else f"{float(value):.4f}"


def _ci_value(row: dict[str, Any]) -> str:
    ci = row.get("auc_bootstrap_ci")
    if not ci:
        return "-"
    return f"[{float(ci['lower']):.4f}, {float(ci['upper']):.4f}]"


def _write_markdown(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {dataset_id: [] for dataset_id in FIXED_DATASETS}
    for row in rows:
        if row.get("model_id") not in FIXED_MODELS or row.get("dataset_id") not in FIXED_DATASETS:
            continue
        if row.get("status") in {"completed", "reused"}:
            source = row
            output = Path(row["output"])
            if output.exists():
                source = {**row, **json.loads(output.read_text(encoding="utf-8"))}
            rows_by_dataset.setdefault(row["dataset_id"], []).append(source)

    zh: list[str] = [
        "# 固定分类模型全数据集外部推理报告",
        "",
        "本报告严格遵循 AGENTS.md 中固定的六模型、固定数据划分和 identity-only TTA 协议。",
        "主任务为 benign/malignant 二分类；外部集未参与模型、阈值、TTA或checkpoint选择。",
        "",
        "## 统一协议",
        "",
        "- BUSBRA：训练与内部五折 OOF。",
        "- BUSI、BUS-UCLM、BUSI-WHU、TCIA BrEaST：锁定外部测试。",
        "- 默认阈值：0.50；AUC置信区间：bootstrap 95%。",
        "- 分割结果不进入分类主表；本报告只呈现分类指标。",
        "",
    ]
    en: list[str] = [
        "# Fixed Classification Benchmark: All-Dataset Inference",
        "",
        "This report follows the fixed six-model, fixed split, and identity-only TTA protocol in AGENTS.md.",
        "The primary task is benign/malignant classification; external datasets were not used for model, threshold, TTA, or checkpoint selection.",
        "",
        "## Unified Protocol",
        "",
        "- BUSBRA: training and internal five-fold OOF.",
        "- BUSI, BUS-UCLM, BUSI-WHU, and TCIA BrEaST: locked external evaluation.",
        "- Default threshold: 0.50; AUC confidence interval: bootstrap 95%.",
        "- Segmentation outputs are excluded from the classification main table.",
        "",
    ]
    for dataset_id, dataset in FIXED_DATASETS.items():
        dataset_rows = sorted(rows_by_dataset.get(dataset_id, []), key=lambda row: list(FIXED_MODELS).index(row["model_id"]))
        zh.extend([f"## {dataset['display_name']}", "", "| 模型 | 样本数 | AUC | AUC 95% CI | Accuracy | Sensitivity | Specificity | F1 |", "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |"])
        en.extend([f"## {dataset['display_name']}", "", "| Model | Samples | AUC | AUC 95% CI | Accuracy | Sensitivity | Specificity | F1 |", "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |"])
        for row in dataset_rows:
            name = row.get("model_name", FIXED_MODELS[row["model_id"]]["display_name"])
            sample_count = row.get("sample_count", "-")
            zh.append(f"| {name} | {sample_count} | {_metric_value(row, 'auc')} | {_ci_value(row)} | {_metric_value(row, 'accuracy')} | {_metric_value(row, 'sensitivity')} | {_metric_value(row, 'specificity')} | {_metric_value(row, 'f1_score')} |")
            en.append(f"| {name} | {sample_count} | {_metric_value(row, 'auc')} | {_ci_value(row)} | {_metric_value(row, 'accuracy')} | {_metric_value(row, 'sensitivity')} | {_metric_value(row, 'specificity')} | {_metric_value(row, 'f1_score')} |")
        zh.append("")
        en.append("")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    zh_path = REPORT_ROOT / "fixed_classification_benchmark_zh.md"
    en_path = REPORT_ROOT / "fixed_classification_benchmark_en.md"
    zh_archive = ROOT / "artifacts" / "reports" / "Chinese reports" / "07_fixed_classification_benchmark" / "fixed_classification_benchmark.md"
    en_archive = ROOT / "artifacts" / "reports" / "English reports" / "07_fixed_classification_benchmark" / "fixed_classification_benchmark.md"
    zh_path.write_text("\n".join(zh) + "\n", encoding="utf-8")
    en_path.write_text("\n".join(en) + "\n", encoding="utf-8")
    zh_archive.parent.mkdir(parents=True, exist_ok=True)
    en_archive.parent.mkdir(parents=True, exist_ok=True)
    zh_archive.write_text("\n".join(zh) + "\n", encoding="utf-8")
    en_archive.write_text("\n".join(en) + "\n", encoding="utf-8")
    return zh_path, en_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=sorted(FIXED_MODELS), default=sorted(FIXED_MODELS))
    parser.add_argument("--datasets", nargs="+", choices=sorted(FIXED_DATASETS), default=sorted(FIXED_DATASETS))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    rows = _load_completed(REPORT_ROOT / "benchmark_index.json")
    for model_id in args.models:
        for dataset_id in args.datasets:
            rows = [row for row in rows if not (row.get("model_id") == model_id and row.get("dataset_id") == dataset_id)]
            try:
                print(f"=== {FIXED_MODELS[model_id]['display_name']} / {FIXED_DATASETS[dataset_id]['display_name']} ===", flush=True)
                result = _run_one(model_id, dataset_id, args.bootstrap, args.rerun)
            except Exception as exc:
                result = {"model_id": model_id, "dataset_id": dataset_id, "status": "error", "error": repr(exc)}
                print(result, flush=True)
            rows.append(result)
            _write_index(rows)
    zh_path, en_path = _write_markdown(rows)
    print(json.dumps({"index": str(REPORT_ROOT / 'benchmark_index.json'), "zh": str(zh_path), "en": str(en_path)}, ensure_ascii=False))
    return 0 if not any(row.get("status") == "error" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
