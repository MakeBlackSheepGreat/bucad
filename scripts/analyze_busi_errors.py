"""Utility script for analyze busi errors workflows."""

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
from src.preprocess.io import cv2, read_image, read_mask, save_image
from src.preprocess.roi import mask_bbox
from src.utils.config import load_project_config
from src.utils.reporting import write_markdown_report


PAIR_WEIGHTS = {"eff_identity": 0.427, "conv_crop_sweep": 0.573}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze BUSI wrong samples for the current ROI area-gated demo."
    )
    parser.add_argument("--config", default="configs/inference/demo.yml")
    parser.add_argument(
        "--main-report",
        default="artifacts/reports/busi_demo_roi_area_gate_current_recheck_after_seg5fold.json",
    )
    parser.add_argument(
        "--full-eff-report",
        default="artifacts/reports/busi_efficientnetv2_s_5fold_identity.json",
    )
    parser.add_argument(
        "--full-conv-report",
        default="artifacts/reports/busi_convnext_tiny_tta_crop_sweep.json",
    )
    parser.add_argument(
        "--roi-report",
        default="artifacts/reports/busi_roi_lcc_mask04_margin035_predictions.json",
    )
    parser.add_argument(
        "--area-cache",
        default="artifacts/reports/roi_precision_f1_area_cache.json",
    )
    parser.add_argument(
        "--csv",
        default="artifacts/reports/busi_error_analysis.csv",
    )
    parser.add_argument(
        "--markdown",
        default="artifacts/reports/busi_error_analysis.md",
    )
    parser.add_argument(
        "--case-dir",
        default="artifacts/reports/error_cases",
    )
    parser.add_argument("--top-n", type=int, default=12)
    return parser


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _report_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Report does not contain a rows list.")
    return rows


def _probabilities(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["malignant_probability"]) for row in rows], dtype=np.float64)


def _row_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["sample_id"]) for row in rows]


def _view_probabilities(
    roi_report: dict[str, Any],
    view_name: str,
    expected_ids: list[str],
) -> np.ndarray:
    views = roi_report.get("views")
    if not isinstance(views, dict) or view_name not in views:
        raise ValueError(f"ROI report does not contain view {view_name!r}.")
    rows = views[view_name]
    current_ids = _row_ids(rows)
    if current_ids != expected_ids:
        raise ValueError(f"Sample order mismatch for ROI view {view_name}.")
    return _probabilities(rows)


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
    feature_mode = str(stacker.get("feature_mode", "probability"))
    if feature_mode == "logit":
        values = np.clip(values, 1e-6, 1.0 - 1e-6)
        values = np.log(values / (1.0 - values))
    mean = np.asarray(stacker.get("scaler_mean", [0.0, 0.0]), dtype=np.float64)
    scale = np.asarray(stacker.get("scaler_scale", [1.0, 1.0]), dtype=np.float64)
    coef = np.asarray(stacker.get("coef", [1.0, 0.0]), dtype=np.float64)
    scaled = (values - mean) / np.maximum(scale, 1e-6)
    logit = float(np.dot(coef, scaled) + float(stacker.get("intercept", 0.0)))
    return float(1.0 / (1.0 + math.exp(-logit)))


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


def _prediction(probability: float, threshold: float) -> int:
    return int(probability >= threshold)


def _wrong_flag(probability: float, y_true: int, threshold: float) -> bool:
    return _prediction(probability, threshold) != int(y_true)


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


def _image_stats(image_path: str) -> tuple[float, float]:
    image = read_image(image_path, grayscale=True).astype(np.float32) / 255.0
    return float(image.mean()), float(image.std())


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
    else:
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
            f"{row['sample_id']} | {row['error_type']} | p={row['final_probability']:.3f} | full={row['full_pair_probability']:.3f} | roi={row['roi_pair_probability']:.3f}",
            f"area={row['roi_area_ratio']:.3f} {row['roi_area_bin']} | gate={row['gate_fallback_reason']} | {row['mechanism']}",
        ],
    )
    save_image(output_path, np.concatenate([text, panel], axis=0))


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
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
        "image_mean",
        "image_std",
        "gt_mask_area_ratio",
        "gt_bbox_area_ratio",
        "image_path",
        "mask_path",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _format_float(value: Any, digits: int = 4) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _counter_table(counter: Counter[str], total: int) -> list[str]:
    lines = ["| 类别 | 数量 | 占比 |", "| --- | ---: | ---: |"]
    for key, count in counter.most_common():
        ratio = count / total if total else 0.0
        lines.append(f"| {key} | {count} | {ratio:.1%} |")
    return lines


def _branch_wrong_rates(error_rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| 错误类型 | full_eff | full_conv | roi_eff | roi_conv | full_pair | roi_pair | gate_fallback |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for error_type in ("FP", "FN"):
        rows = [row for row in error_rows if row["error_type"] == error_type]
        total = max(1, len(rows))
        lines.append(
            "| "
            + " | ".join(
                [
                    error_type,
                    f"{sum(row['full_eff_wrong'] for row in rows) / total:.1%}",
                    f"{sum(row['full_conv_wrong'] for row in rows) / total:.1%}",
                    f"{sum(row['roi_eff_wrong'] for row in rows) / total:.1%}",
                    f"{sum(row['roi_conv_wrong'] for row in rows) / total:.1%}",
                    f"{sum(row['full_pair_wrong'] for row in rows) / total:.1%}",
                    f"{sum(row['roi_pair_wrong'] for row in rows) / total:.1%}",
                    f"{sum(row['gate_fallback'] for row in rows) / total:.1%}",
                ]
            )
            + " |"
        )
    return lines


def _top_case_table(rows: list[dict[str, Any]], error_type: str) -> list[str]:
    selected = [row for row in rows if row["error_type"] == error_type][:10]
    lines = [
        "| sample_id | p(final) | full | ROI | 面积 | 机制 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in selected:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["sample_id"]),
                    f"{row['final_probability']:.3f}",
                    f"{row['full_pair_probability']:.3f}",
                    f"{row['roi_pair_probability']:.3f}",
                    f"{row['roi_area_ratio']:.3f}",
                    str(row["mechanism"]),
                ]
            )
            + " |"
        )
    return lines


def _group_stats(rows: list[dict[str, Any]]) -> list[str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["error_type"]].append(row)
    lines = [
        "| 错误类型 | final均值 | full均值 | ROI均值 | ROI面积均值 | GT mask面积均值 | 图像均值 | 图像std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("FP", "FN"):
        current = groups.get(key, [])
        if not current:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    _format_float(float(np.mean([row["final_probability"] for row in current]))),
                    _format_float(float(np.mean([row["full_pair_probability"] for row in current]))),
                    _format_float(float(np.mean([row["roi_pair_probability"] for row in current]))),
                    _format_float(float(np.mean([row["roi_area_ratio"] for row in current]))),
                    _format_float(float(np.mean([row["gt_mask_area_ratio"] for row in current]))),
                    _format_float(float(np.mean([row["image_mean"] for row in current]))),
                    _format_float(float(np.mean([row["image_std"] for row in current]))),
                ]
            )
            + " |"
        )
    return lines


def build_markdown(report: dict[str, Any], error_rows: list[dict[str, Any]]) -> list[str]:
    metrics = report["metrics"]
    confusion = metrics["confusion"]
    threshold = float(metrics["threshold"])
    total_errors = len(error_rows)
    fp_rows = [row for row in error_rows if row["error_type"] == "FP"]
    fn_rows = [row for row in error_rows if row["error_type"] == "FN"]
    mechanism_counter = Counter(row["mechanism"] for row in error_rows)
    fp_mechanism = Counter(row["mechanism"] for row in fp_rows)
    fn_mechanism = Counter(row["mechanism"] for row in fn_rows)
    confidence_counter = Counter(row["confidence_band"] for row in error_rows)
    area_counter = Counter(row["roi_area_bin"] for row in error_rows)
    lines = [
        "# BUSI Demo 错误样本分析",
        "",
        "## 数据边界",
        "",
        "- BUSI 是锁定外部测试/评估集，本报告只能用于事后错误归因和答辩说明。",
        "- 禁止用本报告中的 BUSI 样本、错误类型、阈值距离或 ROI 面积分布来调阈值、调 ROI gate、选择分割器、选择模型、训练 hard negative 或决定合入。",
        "- 任何性能优化都必须回到 BUSBRA 的训练/验证/OOF 流程内完成；BUSI 只在配置锁定后做外部复核。",
        "",
        "## 当前运行点",
        "",
        f"- 配置：`{report['config_path']}`",
        f"- 阈值：`{threshold:.2f}`",
        f"- AUC：`{metrics['auc']:.4f}`，Sensitivity：`{metrics['sensitivity']:.4f}`，Specificity：`{metrics['specificity']:.4f}`，Precision：`{metrics['precision']:.4f}`，F1：`{metrics['f1_score']:.4f}`",
        f"- 混淆矩阵：TN `{confusion['tn']}`，FP `{confusion['fp']}`，FN `{confusion['fn']}`，TP `{confusion['tp']}`",
        f"- 错误样本：`{total_errors}` 个，其中 FP `{len(fp_rows)}` 个，FN `{len(fn_rows)}` 个。",
        f"- 主线概率重建最大误差：`{report['reconstruction']['max_abs_diff']:.6f}`；该误差只来自缓存/浮点差异。",
        "",
        "## 主要错误机制",
        "",
    ]
    lines.extend(_counter_table(mechanism_counter, total_errors))
    lines.extend(["", "### FP 机制", ""])
    lines.extend(_counter_table(fp_mechanism, len(fp_rows)))
    lines.extend(["", "### FN 机制", ""])
    lines.extend(_counter_table(fn_mechanism, len(fn_rows)))
    lines.extend(
        [
            "",
            "## 置信度与 ROI 面积",
            "",
            "### 置信度分布",
            "",
        ]
    )
    lines.extend(_counter_table(confidence_counter, total_errors))
    lines.extend(["", "### ROI 面积分布", ""])
    lines.extend(_counter_table(area_counter, total_errors))
    lines.extend(["", "### 分组均值", ""])
    lines.extend(_group_stats(error_rows))
    lines.extend(["", "## 分支错因定位", ""])
    lines.extend(_branch_wrong_rates(error_rows))
    lines.extend(
        [
            "",
            "## 代表性错误样本",
            "",
            "### 最高置信 FP",
            "",
        ]
    )
    lines.extend(_top_case_table(fp_rows, "FP"))
    lines.extend(["", "### 最高置信 FN", ""])
    lines.extend(_top_case_table(fn_rows, "FN"))
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- FN 不是简单阈值问题：`{sum(row['confidence_band'] == 'confident' for row in fn_rows)}/{len(fn_rows)}` 个 FN 离阈值超过 0.08，且 full/ROI 分支大多已经把恶性压低。",
            f"- FP 主要分两类：ROI/stacker 把良性推到恶性的 `{fp_mechanism.get('roi_stacker_pushed_to_malignant', 0)}` 个，以及 ROI 面积门回退后完整图已经误报的 `{fp_mechanism.get('full_image_error_area_gate_fallback', 0)}` 个。",
            "- 5 折分割器 Dice 更高但没有提升 demo 指标，原因更像是 ROI 面积分布和融合器校准不匹配；诊断分支不是单纯吃分割 Dice。",
            "- 下一步如果要优化，只能在 BUSBRA OOF 内做恶性 FN 风格的增强、良性 hard negative、ROI 面积门和融合器重选；BUSI 错误样本不得回灌训练或参与配置选择。",
        ]
    )
    return lines


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

    main_report = _load_json(args.main_report)
    full_eff_report = _load_json(args.full_eff_report)
    full_conv_report = _load_json(args.full_conv_report)
    roi_report = _load_json(args.roi_report)
    area_cache = _load_json(args.area_cache)

    main_rows = _report_rows(main_report)
    ids = _row_ids(main_rows)
    for name, rows in (
        ("full_eff", _report_rows(full_eff_report)),
        ("full_conv", _report_rows(full_conv_report)),
    ):
        if _row_ids(rows) != ids:
            raise ValueError(f"Sample order mismatch for {name}.")

    full_eff = _probabilities(_report_rows(full_eff_report))
    full_conv = _probabilities(_report_rows(full_conv_report))
    roi_eff = _view_probabilities(roi_report, "eff_identity", ids)
    roi_conv = _view_probabilities(roi_report, "conv_crop_sweep", ids)
    final = _probabilities(main_rows)
    full_pair = _weighted_pair(full_eff, full_conv)
    roi_pair = _weighted_pair(roi_eff, roi_conv)
    area = np.asarray(area_cache["busi_segmenter_lcc_area_ratio"], dtype=np.float64)
    if len(area) != len(ids):
        raise ValueError("Area cache length does not match BUSI report rows.")

    ungated = np.asarray(
        [
            _stacker_probability(float(full_prob), float(roi_prob), stacker)
            for full_prob, roi_prob in zip(full_pair, roi_pair)
        ],
        dtype=np.float64,
    )
    gate_reasons = [_gate_reason(float(value), min_area, max_area) for value in area]
    reconstructed = np.where(
        np.asarray([reason != "none" for reason in gate_reasons], dtype=bool),
        full_pair,
        ungated,
    )

    manifest = load_busi_manifest(paths.busi_root, include_normal=False)
    manifest_by_id = {str(row.sample_id): row for row in manifest.itertuples(index=False)}
    all_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for index, row in enumerate(main_rows):
        sample_id = str(row["sample_id"])
        label = str(row["pathology_label"])
        y_true = 1 if label == "malignant" else 0
        final_prediction = _prediction(float(final[index]), threshold)
        is_error = final_prediction != y_true
        manifest_row = manifest_by_id.get(sample_id)
        if manifest_row is None:
            raise ValueError(f"BUSI manifest missing sample {sample_id}.")
        image_mean, image_std = _image_stats(str(manifest_row.image_path))
        gt_area, gt_bbox_area = _mask_stats(str(manifest_row.mask_path) if manifest_row.mask_path else None)
        distance = abs(float(final[index]) - threshold)
        gate_reason = gate_reasons[index]
        record = {
            "sample_id": sample_id,
            "pathology_label": label,
            "y_true": y_true,
            "final_prediction": final_prediction,
            "error_type": "FP" if y_true == 0 and final_prediction == 1 else "FN" if y_true == 1 and final_prediction == 0 else "correct",
            "final_probability": float(final[index]),
            "threshold": threshold,
            "distance_to_threshold": float(distance),
            "confidence_band": _confidence_band(distance, borderline_margin),
            "mechanism": _mechanism(
                y_true=y_true,
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
            "full_eff_wrong": _wrong_flag(float(full_eff[index]), y_true, threshold),
            "full_conv_wrong": _wrong_flag(float(full_conv[index]), y_true, threshold),
            "roi_eff_wrong": _wrong_flag(float(roi_eff[index]), y_true, threshold),
            "roi_conv_wrong": _wrong_flag(float(roi_conv[index]), y_true, threshold),
            "full_pair_wrong": _wrong_flag(float(full_pair[index]), y_true, threshold),
            "roi_pair_wrong": _wrong_flag(float(roi_pair[index]), y_true, threshold),
            "gate_fallback": gate_reason != "none",
            "gate_fallback_reason": gate_reason,
            "roi_area_ratio": float(area[index]),
            "roi_area_bin": _area_bin(float(area[index]), min_area, max_area),
            "image_mean": image_mean,
            "image_std": image_std,
            "gt_mask_area_ratio": gt_area,
            "gt_bbox_area_ratio": gt_bbox_area,
            "image_path": str(manifest_row.image_path),
            "mask_path": str(manifest_row.mask_path) if manifest_row.mask_path else "",
        }
        all_rows.append(record)
        if is_error:
            error_rows.append(record)

    error_rows.sort(
        key=lambda item: (
            item["error_type"],
            -abs(float(item["final_probability"]) - threshold),
        )
    )
    fp_rows = sorted(
        [row for row in error_rows if row["error_type"] == "FP"],
        key=lambda item: float(item["final_probability"]) - threshold,
        reverse=True,
    )
    fn_rows = sorted(
        [row for row in error_rows if row["error_type"] == "FN"],
        key=lambda item: threshold - float(item["final_probability"]),
        reverse=True,
    )
    ordered_errors = fp_rows + fn_rows

    _write_csv(args.csv, ordered_errors)
    main_report["reconstruction"] = {
        "max_abs_diff": float(np.max(np.abs(reconstructed - final))),
        "mean_abs_diff": float(np.mean(np.abs(reconstructed - final))),
        "fallback_count": int(sum(reason != "none" for reason in gate_reasons)),
    }
    write_markdown_report(args.markdown, build_markdown(main_report, ordered_errors))

    case_root = Path(args.case_dir)
    case_rows: list[dict[str, Any]] = []
    for group_name, rows in (("fp_top", fp_rows), ("fn_top", fn_rows)):
        output_dir = case_root / group_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for rank, row in enumerate(rows[: max(0, int(args.top_n))], start=1):
            output_path = output_dir / f"{rank:02d}_{_safe_name(str(row['sample_id']))}.png"
            _export_case_image(row, output_path)
            case_rows.append(
                {
                    "rank": rank,
                    "group": group_name,
                    "sample_id": row["sample_id"],
                    "error_type": row["error_type"],
                    "final_probability": row["final_probability"],
                    "mechanism": row["mechanism"],
                    "case_image": str(output_path),
                    "source_image": row["image_path"],
                    "source_mask": row["mask_path"],
                }
            )
    _write_case_manifest(case_root / "manifest.csv", case_rows)

    return {
        "csv": str(Path(args.csv)),
        "markdown": str(Path(args.markdown)),
        "case_dir": str(case_root),
        "error_count": len(ordered_errors),
        "fp_count": len(fp_rows),
        "fn_count": len(fn_rows),
        "reconstruction_max_abs_diff": main_report["reconstruction"]["max_abs_diff"],
    }


def _write_case_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "group",
        "sample_id",
        "error_type",
        "final_probability",
        "mechanism",
        "case_image",
        "source_image",
        "source_mask",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
