"""Utility script for analyze busbra oof demo errors workflows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.busi import load_busi_manifest
from src.datasets.busbra import load_busbra_manifest
from src.preprocess.io import cv2, read_image, read_mask, save_image
from src.preprocess.roi import mask_bbox
from src.utils.config import load_project_config
from src.utils.metrics import classification_metrics
from src.utils.reporting import write_json_report, write_markdown_report


PAIR_WEIGHTS = {"eff_identity": 0.427, "conv_crop_sweep": 0.573}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze current demo mainline errors on BUSBRA OOF/internal validation data, "
            "then summarize locked BUSI external validation."
        )
    )
    parser.add_argument("--config", default="configs/inference/demo.yml")
    parser.add_argument(
        "--full-oof-cache",
        default="artifacts/reports/oof_two_model_predictions.json",
    )
    parser.add_argument(
        "--roi-oof-cache",
        default="artifacts/reports/roi_oof_lcc_mask04_protocol_predictions.json",
    )
    parser.add_argument(
        "--busi-report",
        default="artifacts/reports/busi_demo_roi_area_gate_current_recheck_after_seg5fold.json",
    )
    parser.add_argument(
        "--csv",
        default="artifacts/reports/busbra_oof_demo_error_analysis.csv",
    )
    parser.add_argument(
        "--json",
        default="artifacts/reports/busbra_oof_demo_error_analysis.json",
    )
    parser.add_argument(
        "--markdown",
        default="artifacts/reports/busbra_oof_demo_error_analysis.md",
    )
    parser.add_argument(
        "--case-dir",
        default="artifacts/reports/error_cases_busbra_oof",
    )
    parser.add_argument("--top-n", type=int, default=12)
    return parser


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _rows_for_view(report: dict[str, Any], view_name: str) -> list[dict[str, Any]]:
    views = report.get("views")
    if not isinstance(views, dict) or view_name not in views:
        raise ValueError(f"Report does not contain view {view_name!r}.")
    rows = views[view_name]
    if not isinstance(rows, list):
        raise ValueError(f"View {view_name!r} is not a list.")
    return rows


def _row_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["sample_id"]) for row in rows]


def _probabilities(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["malignant_probability"]) for row in rows], dtype=np.float64)


def _weighted_pair(eff: np.ndarray, conv: np.ndarray) -> np.ndarray:
    total = PAIR_WEIGHTS["eff_identity"] + PAIR_WEIGHTS["conv_crop_sweep"]
    return (
        PAIR_WEIGHTS["eff_identity"] * eff
        + PAIR_WEIGHTS["conv_crop_sweep"] * conv
    ) / total


def _stacker_probability(
    full_probability: float,
    roi_probability: float,
    stacker: dict[str, Any],
) -> float:
    values = np.asarray([full_probability, roi_probability], dtype=np.float64)
    if str(stacker.get("feature_mode", "probability")) == "logit":
        values = np.clip(values, 1e-6, 1.0 - 1e-6)
        values = np.log(values / (1.0 - values))
    mean = np.asarray(stacker.get("scaler_mean", [0.0, 0.0]), dtype=np.float64)
    scale = np.asarray(stacker.get("scaler_scale", [1.0, 1.0]), dtype=np.float64)
    coef = np.asarray(stacker.get("coef", [1.0, 0.0]), dtype=np.float64)
    scaled = (values - mean) / np.maximum(scale, 1e-6)
    logit = float(np.dot(coef, scaled) + float(stacker.get("intercept", 0.0)))
    return float(1.0 / (1.0 + math.exp(-logit)))


def _prediction(probability: float, threshold: float) -> int:
    return int(float(probability) >= float(threshold))


def _wrong_flag(probability: float, y_true: int, threshold: float) -> bool:
    return _prediction(probability, threshold) != int(y_true)


def _gate_reason(area_ratio: float, min_area: float, max_area: float) -> str:
    if area_ratio < min_area:
        return "too_small"
    if area_ratio > max_area:
        return "too_large"
    return "none"


def _area_bin(area_ratio: float, min_area: float, max_area: float) -> str:
    if area_ratio < min_area:
        return "small_or_empty"
    if area_ratio < 0.20:
        return "small"
    if area_ratio < 0.50:
        return "medium"
    if area_ratio <= max_area:
        return "large"
    return "very_large"


def _confidence_band(distance: float, borderline_margin: float) -> str:
    if distance <= 0.03:
        return "near_threshold"
    if distance <= borderline_margin:
        return "borderline"
    return "confident"


def _mechanism(
    *,
    y_true: int,
    final_probability: float,
    full_probability: float,
    ungated_probability: float,
    gate_fallback: bool,
    threshold: float,
) -> str:
    final_pred = _prediction(final_probability, threshold)
    full_pred = _prediction(full_probability, threshold)
    ungated_pred = _prediction(ungated_probability, threshold)
    if y_true == 0 and final_pred == 1:
        if gate_fallback:
            return "full_image_error_area_gate_fallback"
        if full_pred == 0 and ungated_pred == 1:
            return "roi_stacker_pushed_to_malignant"
        if full_pred == 1:
            return "full_image_already_malignant"
        return "other_false_positive"
    if y_true == 1 and final_pred == 0:
        if gate_fallback:
            return "full_image_error_area_gate_fallback"
        if full_pred == 1 and ungated_pred == 0:
            return "roi_stacker_pulled_to_benign"
        if full_pred == 0:
            return "full_image_already_benign"
        return "other_false_negative"
    return "correct"


def _image_stats(image_path: str) -> tuple[float, float]:
    image = read_image(image_path, grayscale=True).astype(np.float32) / 255.0
    return float(image.mean()), float(image.std())


def _mask_stats(mask_path: str | None) -> tuple[float, float]:
    mask = read_mask(mask_path) if mask_path else None
    if mask is None or mask.size == 0:
        return 0.0, 0.0
    binary = np.asarray(mask) > 0
    area_ratio = float(binary.mean())
    bbox = mask_bbox(binary.astype(np.uint8), threshold=0.5, min_area_ratio=0.0)
    if bbox is None:
        return area_ratio, 0.0
    x1, y1, x2, y2 = bbox
    bbox_ratio = float(max(0, x2 - x1) * max(0, y2 - y1)) / float(mask.shape[0] * mask.shape[1])
    return area_ratio, bbox_ratio


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    width = max(1, int(round(image.shape[1] * (height / image.shape[0]))))
    if cv2 is None:
        from PIL import Image

        return np.asarray(Image.fromarray(image).resize((width, height)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _overlay_mask(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    rgb = np.repeat(image[..., None], 3, axis=2) if image.ndim == 2 else image.copy()
    if mask is None:
        return rgb
    if mask.shape[:2] != rgb.shape[:2]:
        if cv2 is None:
            return rgb
        mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    binary = mask > 0
    overlay = rgb.copy()
    overlay[binary] = (0.55 * overlay[binary] + 0.45 * np.array([255, 32, 32])).astype(np.uint8)
    if cv2 is not None:
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
    return overlay


def _draw_text_block(width: int, lines: list[str]) -> np.ndarray:
    height = 24 + 22 * len(lines)
    block = np.full((height, width, 3), 245, dtype=np.uint8)
    if cv2 is None:
        from PIL import Image, ImageDraw

        canvas = Image.fromarray(block)
        draw = ImageDraw.Draw(canvas)
        y = 10
        for line in lines:
            draw.text((10, y), line[:130], fill=(20, 20, 20))
            y += 18
        return np.asarray(canvas)
    y = 26
    for line in lines:
        cv2.putText(
            block,
            line[:130],
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return block


def _export_case_image(row: dict[str, Any], output_path: Path) -> None:
    image = read_image(str(row["image_path"]), grayscale=True)
    mask = read_mask(str(row["mask_path"])) if row.get("mask_path") else None
    original = np.repeat(image[..., None], 3, axis=2) if image.ndim == 2 else image
    overlay = _overlay_mask(image, mask)
    height = min(420, max(240, original.shape[0]))
    original = _resize_to_height(original, height)
    overlay = _resize_to_height(overlay, height)
    panel = np.concatenate([original, overlay], axis=1)
    text = _draw_text_block(
        panel.shape[1],
        [
            f"{row['sample_id']} fold={row['fold_id']} {row['error_type']} p={row['final_probability']:.3f} full={row['full_pair_probability']:.3f} roi={row['roi_pair_probability']:.3f}",
            f"area={row['roi_area_ratio']:.3f} gt={row['gt_mask_area_ratio']:.3f} birads={row['birads']} side={row['view_side']} | {row['mechanism']}",
        ],
    )
    save_image(output_path, np.concatenate([text, panel], axis=0))


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "case_id",
        "fold_id",
        "pathology_label",
        "y_true",
        "final_prediction",
        "error_type",
        "final_probability",
        "threshold",
        "distance_to_threshold",
        "confidence_band",
        "mechanism",
        "full_pair_probability",
        "roi_pair_probability",
        "ungated_roi_stacker_probability",
        "roi_delta_vs_full",
        "final_delta_vs_full",
        "full_eff_probability",
        "full_conv_probability",
        "roi_eff_probability",
        "roi_conv_probability",
        "full_eff_wrong",
        "full_conv_wrong",
        "roi_eff_wrong",
        "roi_conv_wrong",
        "full_pair_wrong",
        "roi_pair_wrong",
        "gate_fallback",
        "gate_fallback_reason",
        "roi_area_ratio",
        "roi_area_bin",
        "gt_mask_area_ratio",
        "gt_bbox_area_ratio",
        "image_mean",
        "image_std",
        "view_side",
        "birads",
        "image_path",
        "mask_path",
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_case_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "group",
        "sample_id",
        "case_id",
        "fold_id",
        "error_type",
        "final_probability",
        "mechanism",
        "case_image",
        "source_image",
        "source_mask",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _counter_table(counter: Counter[str], total: int) -> list[str]:
    lines = ["| 类别 | 数量 | 占比 |", "| --- | ---: | ---: |"]
    for key, count in counter.most_common():
        ratio = count / total if total else 0.0
        lines.append(f"| {key} | {count} | {ratio:.1%} |")
    return lines


def _branch_wrong_rates(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| 错误类型 | full_eff | full_conv | roi_eff | roi_conv | full_pair | roi_pair | gate_fallback |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for error_type in ("FP", "FN"):
        current = [row for row in rows if row["error_type"] == error_type]
        total = max(1, len(current))
        lines.append(
            "| "
            + " | ".join(
                [
                    error_type,
                    f"{sum(row['full_eff_wrong'] for row in current) / total:.1%}",
                    f"{sum(row['full_conv_wrong'] for row in current) / total:.1%}",
                    f"{sum(row['roi_eff_wrong'] for row in current) / total:.1%}",
                    f"{sum(row['roi_conv_wrong'] for row in current) / total:.1%}",
                    f"{sum(row['full_pair_wrong'] for row in current) / total:.1%}",
                    f"{sum(row['roi_pair_wrong'] for row in current) / total:.1%}",
                    f"{sum(row['gate_fallback'] for row in current) / total:.1%}",
                ]
            )
            + " |"
        )
    return lines


def _group_stats(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| 错误类型 | final均值 | full均值 | ROI均值 | ROI面积均值 | GT mask面积均值 | 图像均值 | 图像std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for error_type in ("FP", "FN"):
        current = [row for row in rows if row["error_type"] == error_type]
        if not current:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    error_type,
                    f"{float(np.mean([row['final_probability'] for row in current])):.4f}",
                    f"{float(np.mean([row['full_pair_probability'] for row in current])):.4f}",
                    f"{float(np.mean([row['roi_pair_probability'] for row in current])):.4f}",
                    f"{float(np.mean([row['roi_area_ratio'] for row in current])):.4f}",
                    f"{float(np.mean([row['gt_mask_area_ratio'] for row in current])):.4f}",
                    f"{float(np.mean([row['image_mean'] for row in current])):.4f}",
                    f"{float(np.mean([row['image_std'] for row in current])):.4f}",
                ]
            )
            + " |"
        )
    return lines


def _top_case_table(rows: list[dict[str, Any]], error_type: str) -> list[str]:
    current = [row for row in rows if row["error_type"] == error_type][:10]
    lines = [
        "| sample_id | fold | p(final) | full | ROI | ROI面积 | BIRADS | 机制 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in current:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["sample_id"]),
                    str(row["fold_id"]),
                    f"{row['final_probability']:.3f}",
                    f"{row['full_pair_probability']:.3f}",
                    f"{row['roi_pair_probability']:.3f}",
                    f"{row['roi_area_ratio']:.3f}",
                    str(row["birads"]),
                    str(row["mechanism"]),
                ]
            )
            + " |"
        )
    return lines


def _distribution_table(rows: list[dict[str, Any]], field: str) -> list[str]:
    lines = [f"| {field} | FP | FN |", "| --- | ---: | ---: |"]
    values = sorted({str(row.get(field, "")) for row in rows})
    for value in values:
        fp_count = sum(row["error_type"] == "FP" and str(row.get(field, "")) == value for row in rows)
        fn_count = sum(row["error_type"] == "FN" and str(row.get(field, "")) == value for row in rows)
        lines.append(f"| {value} | {fp_count} | {fn_count} |")
    return lines


def build_markdown(report: dict[str, Any], error_rows: list[dict[str, Any]]) -> list[str]:
    metrics = report["internal_oof_metrics"]
    confusion = metrics["confusion"]
    busi_metrics = report["busi_external_metrics"]
    busi_confusion = busi_metrics["confusion"]
    total_errors = len(error_rows)
    fp_rows = [row for row in error_rows if row["error_type"] == "FP"]
    fn_rows = [row for row in error_rows if row["error_type"] == "FN"]
    mechanism_counter = Counter(row["mechanism"] for row in error_rows)
    confidence_counter = Counter(row["confidence_band"] for row in error_rows)
    area_counter = Counter(row["roi_area_bin"] for row in error_rows)
    fp_mechanism = Counter(row["mechanism"] for row in fp_rows)
    fn_mechanism = Counter(row["mechanism"] for row in fn_rows)
    lines = [
        "# BUSBRA OOF Demo 主线错误样本分析",
        "",
        "## 数据边界",
        "",
        "- 内部分析使用 BUSBRA OOF 预测缓存：每个样本由对应折外的分类器预测，适合作为训练集内部测试/误差分析口径。",
        "- ROI OOF 缓存使用 BUSBRA 真值 mask 生成 ROI，再跑同一组分类器；因此它分析的是当前 ROI 融合/面积门逻辑，不等同于分割器预测 mask 的端到端内部成绩。",
        "- BUSI 只在本报告末尾作为锁定外部验证结果列出，不参与阈值、gate、模型或样本选择。",
        "",
        "## 内部 OOF 当前 demo 运行点",
        "",
        f"- 配置：`{report['config_path']}`",
        f"- 阈值：`{metrics['threshold']:.2f}`",
        f"- AUC：`{metrics['auc']:.4f}`，Sensitivity：`{metrics['sensitivity']:.4f}`，Specificity：`{metrics['specificity']:.4f}`，Precision：`{metrics['precision']:.4f}`，F1：`{metrics['f1_score']:.4f}`",
        f"- 混淆矩阵：TN `{confusion['tn']}`，FP `{confusion['fp']}`，FN `{confusion['fn']}`，TP `{confusion['tp']}`",
        f"- 错误样本：`{total_errors}` 个，其中 FP `{len(fp_rows)}` 个，FN `{len(fn_rows)}` 个。",
        "",
        "## 错误机制",
        "",
    ]
    lines.extend(_counter_table(mechanism_counter, total_errors))
    lines.extend(["", "### FP 机制", ""])
    lines.extend(_counter_table(fp_mechanism, len(fp_rows)))
    lines.extend(["", "### FN 机制", ""])
    lines.extend(_counter_table(fn_mechanism, len(fn_rows)))
    lines.extend(["", "## 置信度与面积", "", "### 置信度分布", ""])
    lines.extend(_counter_table(confidence_counter, total_errors))
    lines.extend(["", "### ROI 面积分布", ""])
    lines.extend(_counter_table(area_counter, total_errors))
    lines.extend(["", "### 分组均值", ""])
    lines.extend(_group_stats(error_rows))
    lines.extend(["", "## 分支错因定位", ""])
    lines.extend(_branch_wrong_rates(error_rows))
    lines.extend(["", "## 元数据分布", "", "### BIRADS", ""])
    lines.extend(_distribution_table(error_rows, "birads"))
    lines.extend(["", "### view_side", ""])
    lines.extend(_distribution_table(error_rows, "view_side"))
    lines.extend(["", "## 代表性内部错误样本", "", "### 最高置信 FP", ""])
    lines.extend(_top_case_table(fp_rows, "FP"))
    lines.extend(["", "### 最高置信 FN", ""])
    lines.extend(_top_case_table(fn_rows, "FN"))
    lines.extend(
        [
            "",
            "## BUSI 锁定外部验证",
            "",
            f"- BUSI 配置：`{report['busi_config_path']}`",
            f"- AUC：`{busi_metrics['auc']:.4f}`，Sensitivity：`{busi_metrics['sensitivity']:.4f}`，Specificity：`{busi_metrics['specificity']:.4f}`，Precision：`{busi_metrics['precision']:.4f}`，F1：`{busi_metrics['f1_score']:.4f}`",
            f"- 混淆矩阵：TN `{busi_confusion['tn']}`，FP `{busi_confusion['fp']}`，FN `{busi_confusion['fn']}`，TP `{busi_confusion['tp']}`",
            "- 该外部验证只用于确认锁定配置的泛化表现，不用于本轮错误样本选择或参数调整。",
            "",
            "## 结论",
            "",
            f"- 内部 OOF 下，FP 中 `{fp_mechanism.get('roi_stacker_pushed_to_malignant', 0)}` 个是 ROI/stacker 把良性推过阈值，`{fp_mechanism.get('full_image_error_area_gate_fallback', 0)}` 个是面积门回退后完整图误报。",
            f"- 内部 OOF 下，FN 中 `{fn_mechanism.get('full_image_already_benign', 0)}` 个在完整图分支已经偏良性，优先说明分类表征/训练样本覆盖问题，而不是单纯 ROI 后处理问题。",
            "- 后续优化应只基于 BUSBRA OOF 的这些内部错误模式做 hard-case mining、增强和融合器重训；完成后再对 BUSI 做一次锁定外部复核。",
        ]
    )
    return lines


def _busi_metrics_from_report(path: str | Path) -> tuple[str, dict[str, Any]]:
    report = _load_json(path)
    return str(report.get("config_path", path)), dict(report["metrics"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, paths = load_project_config(args.config)
    runtime = dict(config.get("runtime", {}))
    roi_config = dict(runtime.get("roi_enhancement", {}))
    stacker = dict(roi_config.get("stacker", {}))
    quality_gate = dict(roi_config.get("quality_gate", {}))
    threshold = float(runtime.get("default_threshold", 0.5))
    borderline_margin = float(runtime.get("borderline_margin", 0.08))
    min_area = float(quality_gate.get("min_area_ratio", 0.08))
    max_area = float(quality_gate.get("max_area_ratio", 0.75))

    full_oof = _load_json(args.full_oof_cache)
    roi_oof = _load_json(args.roi_oof_cache)
    full_eff_rows = _rows_for_view(full_oof, "eff_identity")
    full_conv_rows = _rows_for_view(full_oof, "conv_crop_sweep")
    roi_eff_rows = _rows_for_view(roi_oof, "eff_identity")
    roi_conv_rows = _rows_for_view(roi_oof, "conv_crop_sweep")
    ids = _row_ids(full_eff_rows)
    for name, rows in (
        ("full_conv", full_conv_rows),
        ("roi_eff", roi_eff_rows),
        ("roi_conv", roi_conv_rows),
    ):
        if _row_ids(rows) != ids:
            raise ValueError(f"Sample order mismatch for {name}.")

    y_true = np.asarray(
        [1 if str(row["pathology_label"]).lower() == "malignant" else 0 for row in full_eff_rows],
        dtype=np.int32,
    )
    full_eff = _probabilities(full_eff_rows)
    full_conv = _probabilities(full_conv_rows)
    roi_eff = _probabilities(roi_eff_rows)
    roi_conv = _probabilities(roi_conv_rows)
    full_pair = _weighted_pair(full_eff, full_conv)
    roi_pair = _weighted_pair(roi_eff, roi_conv)
    area = np.asarray(roi_oof["roi_area_ratios"], dtype=np.float64)
    if len(area) != len(ids):
        raise ValueError("ROI area ratio length does not match OOF rows.")

    ungated = np.asarray(
        [
            _stacker_probability(float(full_prob), float(roi_prob), stacker)
            for full_prob, roi_prob in zip(full_pair, roi_pair)
        ],
        dtype=np.float64,
    )
    gate_reasons = [_gate_reason(float(value), min_area, max_area) for value in area]
    final = np.where(
        np.asarray([reason != "none" for reason in gate_reasons], dtype=bool),
        full_pair,
        ungated,
    )
    metrics = classification_metrics(y_true, final, threshold=threshold)

    manifest = load_busbra_manifest(paths.busbra_root)
    manifest_by_id = {str(row.sample_id): row for row in manifest.itertuples(index=False)}
    all_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(ids):
        source_row = full_eff_rows[index]
        manifest_row = manifest_by_id.get(sample_id)
        if manifest_row is None:
            raise ValueError(f"BUSBRA manifest missing sample {sample_id}.")
        y_value = int(y_true[index])
        final_prediction = _prediction(float(final[index]), threshold)
        error_type = "FP" if y_value == 0 and final_prediction == 1 else "FN" if y_value == 1 and final_prediction == 0 else "correct"
        image_mean, image_std = _image_stats(str(manifest_row.image_path))
        gt_area, gt_bbox_area = _mask_stats(str(manifest_row.mask_path) if manifest_row.mask_path else None)
        gate_reason = gate_reasons[index]
        distance = abs(float(final[index]) - threshold)
        record = {
            "sample_id": sample_id,
            "case_id": str(source_row["case_id"]),
            "fold_id": int(source_row["fold_id"]),
            "pathology_label": str(source_row["pathology_label"]),
            "y_true": y_value,
            "final_prediction": final_prediction,
            "error_type": error_type,
            "final_probability": float(final[index]),
            "threshold": threshold,
            "distance_to_threshold": float(distance),
            "confidence_band": _confidence_band(distance, borderline_margin),
            "mechanism": _mechanism(
                y_true=y_value,
                final_probability=float(final[index]),
                full_probability=float(full_pair[index]),
                ungated_probability=float(ungated[index]),
                gate_fallback=gate_reason != "none",
                threshold=threshold,
            ),
            "full_pair_probability": float(full_pair[index]),
            "roi_pair_probability": float(roi_pair[index]),
            "ungated_roi_stacker_probability": float(ungated[index]),
            "roi_delta_vs_full": float(roi_pair[index] - full_pair[index]),
            "final_delta_vs_full": float(final[index] - full_pair[index]),
            "full_eff_probability": float(full_eff[index]),
            "full_conv_probability": float(full_conv[index]),
            "roi_eff_probability": float(roi_eff[index]),
            "roi_conv_probability": float(roi_conv[index]),
            "full_eff_wrong": _wrong_flag(float(full_eff[index]), y_value, threshold),
            "full_conv_wrong": _wrong_flag(float(full_conv[index]), y_value, threshold),
            "roi_eff_wrong": _wrong_flag(float(roi_eff[index]), y_value, threshold),
            "roi_conv_wrong": _wrong_flag(float(roi_conv[index]), y_value, threshold),
            "full_pair_wrong": _wrong_flag(float(full_pair[index]), y_value, threshold),
            "roi_pair_wrong": _wrong_flag(float(roi_pair[index]), y_value, threshold),
            "gate_fallback": gate_reason != "none",
            "gate_fallback_reason": gate_reason,
            "roi_area_ratio": float(area[index]),
            "roi_area_bin": _area_bin(float(area[index]), min_area, max_area),
            "gt_mask_area_ratio": gt_area,
            "gt_bbox_area_ratio": gt_bbox_area,
            "image_mean": image_mean,
            "image_std": image_std,
            "view_side": str(getattr(manifest_row, "view_side", "")),
            "birads": str(getattr(manifest_row, "birads", "")),
            "image_path": str(manifest_row.image_path),
            "mask_path": str(manifest_row.mask_path) if manifest_row.mask_path else "",
        }
        all_rows.append(record)
        if error_type != "correct":
            error_rows.append(record)

    fp_rows = sorted(
        [row for row in error_rows if row["error_type"] == "FP"],
        key=lambda row: float(row["final_probability"]) - threshold,
        reverse=True,
    )
    fn_rows = sorted(
        [row for row in error_rows if row["error_type"] == "FN"],
        key=lambda row: threshold - float(row["final_probability"]),
        reverse=True,
    )
    ordered_errors = fp_rows + fn_rows
    _write_csv(args.csv, ordered_errors)

    case_root = Path(args.case_dir)
    case_manifest: list[dict[str, Any]] = []
    for group_name, rows in (("fp_top", fp_rows), ("fn_top", fn_rows)):
        output_dir = case_root / group_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for rank, row in enumerate(rows[: max(0, int(args.top_n))], start=1):
            output_path = output_dir / f"{rank:02d}_{_safe_name(str(row['sample_id']))}.png"
            _export_case_image(row, output_path)
            case_manifest.append(
                {
                    "rank": rank,
                    "group": group_name,
                    "sample_id": row["sample_id"],
                    "case_id": row["case_id"],
                    "fold_id": row["fold_id"],
                    "error_type": row["error_type"],
                    "final_probability": row["final_probability"],
                    "mechanism": row["mechanism"],
                    "case_image": str(output_path),
                    "source_image": row["image_path"],
                    "source_mask": row["mask_path"],
                }
            )
    _write_case_manifest(case_root / "manifest.csv", case_manifest)

    busi_config_path, busi_metrics = _busi_metrics_from_report(args.busi_report)
    summary = {
        "method": "Current demo mainline on BUSBRA OOF/internal validation",
        "config_path": str(args.config),
        "full_oof_cache": str(args.full_oof_cache),
        "roi_oof_cache": str(args.roi_oof_cache),
        "roi_cache_note": "BUSBRA ROI OOF uses ground-truth masks for ROI generation.",
        "sample_count": int(len(ids)),
        "class_counts": {
            "benign": int(np.sum(y_true == 0)),
            "malignant": int(np.sum(y_true == 1)),
        },
        "internal_oof_metrics": metrics,
        "error_count": int(len(ordered_errors)),
        "fp_count": int(len(fp_rows)),
        "fn_count": int(len(fn_rows)),
        "mechanism_counts": dict(Counter(row["mechanism"] for row in ordered_errors)),
        "confidence_counts": dict(Counter(row["confidence_band"] for row in ordered_errors)),
        "roi_area_bin_counts": dict(Counter(row["roi_area_bin"] for row in ordered_errors)),
        "busi_report": str(args.busi_report),
        "busi_config_path": busi_config_path,
        "busi_external_metrics": busi_metrics,
        "outputs": {
            "csv": str(Path(args.csv)),
            "markdown": str(Path(args.markdown)),
            "case_dir": str(case_root),
        },
    }
    write_json_report(args.json, summary)
    write_markdown_report(args.markdown, build_markdown(summary, ordered_errors))
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "csv": summary["outputs"]["csv"],
                "markdown": summary["outputs"]["markdown"],
                "case_dir": summary["outputs"]["case_dir"],
                "internal_oof_metrics": summary["internal_oof_metrics"],
                "error_count": summary["error_count"],
                "fp_count": summary["fp_count"],
                "fn_count": summary["fn_count"],
                "busi_external_metrics": summary["busi_external_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
