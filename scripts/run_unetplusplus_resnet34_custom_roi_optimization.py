"""Utility script for unetplusplus resnet34 custom roi optimization workflows."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.busbra import LABEL_TO_INDEX  # noqa: E402
from src.utils.config import load_project_config  # noqa: E402
from src.utils.metrics import best_threshold_by_youden, classification_metrics  # noqa: E402
from src.utils.reporting import write_json_report, write_markdown_report  # noqa: E402
from src.utils.runtime import timestamp_now  # noqa: E402


REPORT_DIR = Path("artifacts/reports/Chinese reports/09_unetplusplus_resnet34_custom_roi")
PAIR_VIEWS = ("eff_identity", "conv_crop_sweep")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search deployable BUSBRA OOF calibration settings for the "
            "U-Net++ ResNet34 segmenter mainline. BUSI is not loaded here."
        )
    )
    parser.add_argument("--config", default="configs/inference/demo.yml")
    parser.add_argument("--full-oof-cache", default="artifacts/reports/oof_two_model_predictions.json")
    parser.add_argument(
        "--roi-oof-cache",
        default=(
            "artifacts/reports/Chinese reports/08_segmenter_recalibrated_roi/"
            "oof_predictions/unetplusplus_resnet34_bce_dice_roi_oof.json"
        ),
    )
    parser.add_argument("--output", default=str(REPORT_DIR / "unetplusplus_resnet34_custom_roi_optimization.json"))
    parser.add_argument("--markdown", default=str(REPORT_DIR / "unetplusplus_resnet34_custom_roi_optimization.md"))
    parser.add_argument(
        "--frozen-config",
        default=str(REPORT_DIR / "frozen_configs/demo_unetplusplus_resnet34_custom_roi.yml"),
    )
    parser.add_argument("--top-stackers", type=int, default=5)
    return parser


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object: {path}")
    return data


def _write_yaml(path: str | Path, data: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    return target


def _rows(report: dict[str, Any], view: str) -> list[dict[str, Any]]:
    views = report.get("views")
    if not isinstance(views, dict) or view not in views:
        raise ValueError(f"Missing view {view!r}.")
    rows = views[view]
    if not isinstance(rows, list):
        raise ValueError(f"View {view!r} is not a list.")
    return rows


def _validate_order(*groups: list[dict[str, Any]]) -> list[str]:
    if not groups:
        raise ValueError("No OOF rows provided.")
    reference = [str(row["sample_id"]) for row in groups[0]]
    for rows in groups[1:]:
        current = [str(row["sample_id"]) for row in rows]
        if current != reference:
            raise ValueError("OOF sample order mismatch.")
    return reference


def _probabilities(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["malignant_probability"]) for row in rows], dtype=np.float64)


def _weighted_pair(eff: np.ndarray, conv: np.ndarray, conv_weight: float) -> np.ndarray:
    weight = float(np.clip(conv_weight, 0.0, 1.0))
    return (1.0 - weight) * eff + weight * conv


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _transform_pair(full_probabilities: np.ndarray, roi_probabilities: np.ndarray, mode: str) -> np.ndarray:
    matrix = np.vstack([full_probabilities, roi_probabilities]).T
    if mode == "logit":
        return _logit(matrix)
    if mode == "probability":
        return matrix
    raise ValueError(f"Unsupported feature mode: {mode}")


def _metrics_at_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    best = best_threshold_by_youden(
        y_true,
        probabilities,
        thresholds=np.round(np.arange(0.1, 0.9001, 0.01), 2),
    )
    metrics = classification_metrics(y_true, probabilities, threshold=float(best["threshold"]))
    metrics["auc"] = float(roc_auc_score(y_true, probabilities))
    metrics["youden_j"] = float(metrics["sensitivity"] + metrics["specificity"] - 1.0)
    return metrics


def _runtime_stack(
    full_probabilities: np.ndarray,
    roi_probabilities: np.ndarray,
    stacker: dict[str, Any],
) -> np.ndarray:
    features = _transform_pair(
        full_probabilities,
        roi_probabilities,
        str(stacker.get("feature_mode", "probability")),
    )
    mean = np.asarray(stacker.get("scaler_mean", [0.0, 0.0]), dtype=np.float64)
    scale = np.asarray(stacker.get("scaler_scale", [1.0, 1.0]), dtype=np.float64)
    coef = np.asarray(stacker.get("coef", [1.0, 0.0]), dtype=np.float64)
    return _sigmoid(((features - mean) / np.maximum(scale, 1e-6)) @ coef + float(stacker.get("intercept", 0.0)))


def _current_oof_probabilities(
    *,
    full_eff: np.ndarray,
    full_conv: np.ndarray,
    roi_eff: np.ndarray,
    roi_conv: np.ndarray,
    area_ratios: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    runtime = dict(config.get("runtime", {}))
    roi_config = dict(runtime.get("roi_enhancement", {}))
    stacker = dict(roi_config.get("stacker", {}))
    full_pair = _weighted_pair(full_eff, full_conv, _full_conv_weight_from_config(runtime))
    overrides = dict(roi_config.get("classifier_weight_overrides", {}))
    roi_conv_weight = _roi_conv_weight_from_overrides(overrides, default=_full_conv_weight_from_config(runtime))
    roi_pair = _weighted_pair(roi_eff, roi_conv, roi_conv_weight)
    stacked = _runtime_stack(full_pair, roi_pair, stacker)
    blend_weight = float(roi_config.get("roi_stack_blend_weight", 1.0))
    if blend_weight < 1.0:
        stacked = float(np.clip(blend_weight, 0.0, 1.0)) * stacked + (1.0 - float(np.clip(blend_weight, 0.0, 1.0))) * full_pair
    gate = dict(roi_config.get("quality_gate", {}))
    min_area = float(gate.get("min_area_ratio", 0.0))
    max_area = float(gate.get("max_area_ratio", 1.01))
    fallback = (area_ratios < min_area) | (area_ratios > max_area)
    return np.where(fallback, full_pair, stacked)


def _full_conv_weight_from_config(runtime: dict[str, Any]) -> float:
    conv_weight = 0.0
    eff_weight = 0.0
    for member in runtime.get("classifier_members", []):
        if not isinstance(member, dict):
            continue
        weight = float(member.get("weight", 1.0))
        model = str(member.get("model", ""))
        if model == "convnext_tiny":
            conv_weight += weight
        elif model == "tf_efficientnetv2_s":
            eff_weight += weight
    total = conv_weight + eff_weight
    if total <= 0.0:
        return 0.573
    return float(conv_weight / total)


def _roi_conv_weight_from_overrides(overrides: dict[str, Any], *, default: float) -> float:
    conv = overrides.get("convnext_tiny")
    eff = overrides.get("tf_efficientnetv2_s")
    if conv is None or eff is None:
        return float(default)
    total = float(conv) + float(eff)
    if total <= 0.0:
        return float(default)
    return float(float(conv) / total)


def _fit_stacker_candidates(
    *,
    full_probabilities: np.ndarray,
    roi_probabilities: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    full_conv_weight: float,
    roi_conv_weight: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=5)
    for feature_mode in ("probability", "logit"):
        features = _transform_pair(full_probabilities, roi_probabilities, feature_mode)
        for class_weight in (None, "balanced"):
            for c_value in (1.0, 3.0, 10.0):
                cv_probabilities = np.zeros(len(y_true), dtype=np.float64)
                fold_aucs: list[float] = []
                for train_index, val_index in splitter.split(features, y_true, groups):
                    model = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "classifier",
                                SGDClassifier(
                                    loss="log_loss",
                                    alpha=1.0 / (float(c_value) * 10000.0),
                                    class_weight=class_weight,
                                    max_iter=1000,
                                    random_state=42,
                                    tol=1e-4,
                                ),
                            ),
                        ]
                    )
                    model.fit(features[train_index], y_true[train_index])
                    probabilities = model.predict_proba(features[val_index])[:, 1]
                    cv_probabilities[val_index] = probabilities
                    if len(np.unique(y_true[val_index])) > 1:
                        fold_aucs.append(float(roc_auc_score(y_true[val_index], probabilities)))
                candidates.append(
                    {
                        "full_conv_weight": float(full_conv_weight),
                        "roi_conv_weight": float(roi_conv_weight),
                        "feature_mode": feature_mode,
                        "C": float(c_value),
                        "class_weight": class_weight,
                        "cv_auc": float(roc_auc_score(y_true, cv_probabilities)),
                        "fold_auc_mean": float(np.mean(fold_aucs)),
                        "fold_auc_std": float(np.std(fold_aucs)),
                        "cv_probabilities": cv_probabilities,
                    }
                )
    return candidates


def _fit_final_stacker(
    *,
    full_probabilities: np.ndarray,
    roi_probabilities: np.ndarray,
    y_true: np.ndarray,
    selected: dict[str, Any],
) -> Pipeline:
    features = _transform_pair(full_probabilities, roi_probabilities, str(selected["feature_mode"]))
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    alpha=1.0 / (float(selected["C"]) * 10000.0),
                    class_weight=selected["class_weight"],
                    max_iter=1000,
                    random_state=42,
                    tol=1e-4,
                ),
            ),
        ]
    )
    model.fit(features, y_true)
    return model


def _slim_stacker(selected: dict[str, Any], model: Pipeline, y_true: np.ndarray) -> dict[str, Any]:
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    best = best_threshold_by_youden(
        y_true,
        selected["cv_probabilities"],
        thresholds=np.round(np.arange(0.1, 0.9001, 0.01), 2),
    )
    return {
        "feature_mode": str(selected["feature_mode"]),
        "C": float(selected["C"]),
        "class_weight": selected["class_weight"],
        "cv_auc": float(selected.get("cv_auc", selected["stacker_cv_auc"])),
        "fold_auc_mean": float(selected["fold_auc_mean"]),
        "fold_auc_std": float(selected["fold_auc_std"]),
        "threshold_from_oof": float(best["threshold"]),
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) for value in scaler.scale_],
        "coef": [float(value) for value in classifier.coef_[0]],
        "intercept": float(classifier.intercept_[0]),
    }


def _scan_final_candidates(
    *,
    y_true: np.ndarray,
    area_ratios: np.ndarray,
    full_probabilities_by_key: dict[float, np.ndarray],
    stacker_candidates: list[dict[str, Any]],
    top_stackers: int,
) -> list[dict[str, Any]]:
    blend_values = [0.75, 0.85, 0.95, 1.0]
    min_values = [0.0, 0.05, 0.08, 0.1, 0.15, 0.2]
    max_values = [0.65, 0.75, 0.85, 0.95, 1.01]
    selected_stackers = sorted(
        stacker_candidates,
        key=lambda row: (row["cv_auc"], row["fold_auc_mean"], -row["fold_auc_std"]),
        reverse=True,
    )[: int(top_stackers)]
    final_rows: list[dict[str, Any]] = []
    for stacker in selected_stackers:
        full_probabilities = full_probabilities_by_key[float(stacker["full_conv_weight"])]
        stack_probabilities = np.asarray(stacker["cv_probabilities"], dtype=np.float64)
        for blend_weight in blend_values:
            blended = float(blend_weight) * stack_probabilities + (1.0 - float(blend_weight)) * full_probabilities
            for min_area in min_values:
                for max_area in max_values:
                    if min_area >= max_area:
                        continue
                    fallback = (area_ratios < min_area) | (area_ratios > max_area)
                    probabilities = np.where(fallback, full_probabilities, blended)
                    metrics = _metrics_at_best_threshold(y_true, probabilities)
                    row = {
                        "full_conv_weight": float(stacker["full_conv_weight"]),
                        "roi_conv_weight": float(stacker["roi_conv_weight"]),
                        "feature_mode": stacker["feature_mode"],
                        "C": float(stacker["C"]),
                        "class_weight": stacker["class_weight"],
                        "stacker_cv_auc": float(stacker["cv_auc"]),
                        "fold_auc_mean": float(stacker["fold_auc_mean"]),
                        "fold_auc_std": float(stacker["fold_auc_std"]),
                        "roi_stack_blend_weight": float(blend_weight),
                        "min_area_ratio": float(min_area),
                        "max_area_ratio": float(max_area),
                        "fallback_count": int(fallback.sum()),
                        "metrics": metrics,
                        "cv_probabilities": stack_probabilities,
                    }
                    final_rows.append(row)
    return sorted(
        final_rows,
        key=lambda row: (
            row["metrics"]["auc"],
            row["metrics"]["f1_score"],
            row["metrics"]["sensitivity"],
            row["metrics"]["specificity"],
            -row["fallback_count"],
        ),
        reverse=True,
    )


def _json_slim_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key != "cv_probabilities"
    }


def _freeze_config(
    *,
    base_config_path: str | Path,
    output_path: str | Path,
    selected: dict[str, Any],
    stacker: dict[str, Any],
) -> Path:
    config = _load_yaml(base_config_path)
    frozen = copy.deepcopy(config)
    runtime = frozen.setdefault("runtime", {})
    for member in runtime.get("classifier_members", []):
        if not isinstance(member, dict):
            continue
        model_name = str(member.get("model", ""))
        if model_name == "convnext_tiny":
            member["weight"] = float(selected["full_conv_weight"])
        elif model_name == "tf_efficientnetv2_s":
            member["weight"] = float(1.0 - float(selected["full_conv_weight"]))
    runtime["default_threshold"] = float(selected["metrics"]["threshold"])
    runtime["ensemble_display_name"] = (
        "ConvNeXt-Tiny + EfficientNetV2-S + ROI Area Gate + "
        "U-Net++ ResNet34 segmenter + BUSBRA custom calibration"
    )
    roi_config = runtime.setdefault("roi_enhancement", {})
    roi_config["enabled"] = True
    roi_config["stacker"] = stacker
    roi_config["roi_stack_blend_weight"] = float(selected["roi_stack_blend_weight"])
    roi_config["classifier_weight_overrides"] = {
        "convnext_tiny": float(selected["roi_conv_weight"]),
        "tf_efficientnetv2_s": float(1.0 - float(selected["roi_conv_weight"])),
    }
    roi_config["quality_gate"] = {
        "enabled": True,
        "min_area_ratio": float(selected["min_area_ratio"]),
        "max_area_ratio": float(selected["max_area_ratio"]),
        "fallback_to_full": True,
    }
    frozen["unetplusplus_resnet34_custom_roi_optimization"] = {
        "method_id": "unetplusplus_resnet34_custom_roi",
        "selected_at": timestamp_now(),
        "data_boundary": "All calibration choices selected from BUSBRA OOF only; BUSI is frozen external review only.",
        "full_conv_weight": float(selected["full_conv_weight"]),
        "roi_conv_weight": float(selected["roi_conv_weight"]),
        "oof_auc": float(selected["metrics"]["auc"]),
        "oof_threshold": float(selected["metrics"]["threshold"]),
    }
    return _write_yaml(output_path, frozen)


def _build_markdown(report: dict[str, Any]) -> list[str]:
    selected = report["selected_candidate"]
    baseline = report["baseline_current_unetplusplus_oof_metrics"]
    lines = [
        "# U-Net++ ResNet34 主线定制 ROI 标定实验",
        "",
        "## 数据边界",
        "",
        "- 本实验只读取 BUSBRA OOF 分类概率、U-Net++ ResNet34 预测 mask 的 ROI OOF 概率与 ROI 面积缓存。",
        "- BUSI 不参与本脚本中的权重、stacker、area gate、blend 或阈值选择。",
        "- 输出的 frozen config 是冻结候选，后续只能做一次外部复核，不能根据外验结果反向调参。",
        "",
        "## 当前 U-Net++ 配置 OOF",
        "",
        (
            f"- AUC `{baseline['auc']:.6f}`，threshold `{baseline['threshold']:.2f}`，"
            f"Sensitivity `{baseline['sensitivity']:.4f}`，Specificity `{baseline['specificity']:.4f}`，"
            f"F1 `{baseline['f1_score']:.4f}`。"
        ),
        "",
        "## 入选定制配置",
        "",
        (
            f"- AUC `{selected['metrics']['auc']:.6f}`，threshold `{selected['metrics']['threshold']:.2f}`，"
            f"Sensitivity `{selected['metrics']['sensitivity']:.4f}`，"
            f"Specificity `{selected['metrics']['specificity']:.4f}`，"
            f"F1 `{selected['metrics']['f1_score']:.4f}`。"
        ),
        f"- Full-image ConvNeXt 权重：`{selected['full_conv_weight']:.3f}`；ROI ConvNeXt 权重：`{selected['roi_conv_weight']:.3f}`。",
        (
            f"- Stacker：`{selected['feature_mode']}` 特征，`C={selected['C']}`，"
            f"`class_weight={selected['class_weight']}`，CV AUC `{selected['stacker_cv_auc']:.6f}`。"
        ),
        (
            f"- ROI blend：`{selected['roi_stack_blend_weight']:.2f}`；"
            f"area gate：`{selected['min_area_ratio']:.2f}-{selected['max_area_ratio']:.2f}`，"
            f"fallback `{selected['fallback_count']}` 个样本。"
        ),
        f"- 冻结配置：`{report['frozen_config']}`。",
        "",
        "## Top 10 候选",
        "",
        "| Rank | OOF AUC | Threshold | Sens | Spec | F1 | Full Conv | ROI Conv | Blend | Gate | Stacker |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for rank, row in enumerate(report["top_candidates"][:10], start=1):
        metrics = row["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    f"{metrics['auc']:.6f}",
                    f"{metrics['threshold']:.2f}",
                    f"{metrics['sensitivity']:.4f}",
                    f"{metrics['specificity']:.4f}",
                    f"{metrics['f1_score']:.4f}",
                    f"{row['full_conv_weight']:.3f}",
                    f"{row['roi_conv_weight']:.3f}",
                    f"{row['roi_stack_blend_weight']:.2f}",
                    f"{row['min_area_ratio']:.2f}-{row['max_area_ratio']:.2f}",
                    f"{row['feature_mode']} C={row['C']} {row['class_weight']}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 内部 OOF AUC 增量：`{report['delta_auc_vs_current_oof']:+.6f}`。",
            f"- 建议：`{report['recommendation']}`。",
        ]
    )
    return lines


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, paths = load_project_config(args.config)
    output_root = paths.project_root / REPORT_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    full_oof = _load_json(args.full_oof_cache)
    roi_oof = _load_json(args.roi_oof_cache)
    full_eff_rows = _rows(full_oof, "eff_identity")
    full_conv_rows = _rows(full_oof, "conv_crop_sweep")
    roi_eff_rows = _rows(roi_oof, "eff_identity")
    roi_conv_rows = _rows(roi_oof, "conv_crop_sweep")
    sample_ids = _validate_order(full_eff_rows, full_conv_rows, roi_eff_rows, roi_conv_rows)

    y_true = np.asarray(
        [LABEL_TO_INDEX[str(row["pathology_label"]).lower()] for row in full_eff_rows],
        dtype=np.int32,
    )
    groups = np.asarray([str(row["case_id"]) for row in full_eff_rows])
    full_eff = _probabilities(full_eff_rows)
    full_conv = _probabilities(full_conv_rows)
    roi_eff = _probabilities(roi_eff_rows)
    roi_conv = _probabilities(roi_conv_rows)
    area_ratios = np.asarray(roi_oof["roi_area_ratios"], dtype=np.float64)
    if len(area_ratios) != len(sample_ids):
        raise ValueError("ROI area ratio length mismatch.")

    current_probabilities = _current_oof_probabilities(
        full_eff=full_eff,
        full_conv=full_conv,
        roi_eff=roi_eff,
        roi_conv=roi_conv,
        area_ratios=area_ratios,
        config=config,
    )
    current_metrics = _metrics_at_best_threshold(y_true, current_probabilities)

    base_full_weight = _full_conv_weight_from_config(dict(config.get("runtime", {})))
    weight_grid = sorted({round(base_full_weight, 3), 0.573, 0.6})
    roi_weight_grid = sorted({round(base_full_weight, 3), 0.573, 0.6, 0.625})
    full_probabilities_by_key = {
        float(weight): _weighted_pair(full_eff, full_conv, float(weight))
        for weight in weight_grid
    }
    roi_probabilities_by_key = {
        float(weight): _weighted_pair(roi_eff, roi_conv, float(weight))
        for weight in roi_weight_grid
    }

    stacker_candidates: list[dict[str, Any]] = []
    total_pair_count = len(full_probabilities_by_key) * len(roi_probabilities_by_key)
    pair_index = 0
    for full_weight, full_probabilities in full_probabilities_by_key.items():
        for roi_weight, roi_probabilities in roi_probabilities_by_key.items():
            pair_index += 1
            print(
                f"stacker search {pair_index}/{total_pair_count}: "
                f"full_conv={full_weight:.3f} roi_conv={roi_weight:.3f}",
                flush=True,
            )
            stacker_candidates.extend(
                _fit_stacker_candidates(
                    full_probabilities=full_probabilities,
                    roi_probabilities=roi_probabilities,
                    y_true=y_true,
                    groups=groups,
                    full_conv_weight=full_weight,
                    roi_conv_weight=roi_weight,
                )
            )

    final_candidates = _scan_final_candidates(
        y_true=y_true,
        area_ratios=area_ratios,
        full_probabilities_by_key=full_probabilities_by_key,
        stacker_candidates=stacker_candidates,
        top_stackers=int(args.top_stackers),
    )
    selected = final_candidates[0]
    selected_full = full_probabilities_by_key[float(selected["full_conv_weight"])]
    selected_roi = roi_probabilities_by_key[float(selected["roi_conv_weight"])]
    final_model = _fit_final_stacker(
        full_probabilities=selected_full,
        roi_probabilities=selected_roi,
        y_true=y_true,
        selected=selected,
    )
    stacker = _slim_stacker(selected, final_model, y_true)
    frozen_config = _freeze_config(
        base_config_path=args.config,
        output_path=args.frozen_config,
        selected=selected,
        stacker=stacker,
    )

    selected_slim = _json_slim_candidate(selected)
    delta_auc = float(selected_slim["metrics"]["auc"]) - float(current_metrics["auc"])
    report = {
        "generated_at": timestamp_now(),
        "method": "U-Net++ ResNet34 custom ROI calibration from BUSBRA OOF",
        "data_boundary": "BUSBRA OOF only for selection; BUSI is not loaded by this script.",
        "config": str(args.config),
        "full_oof_cache": str(args.full_oof_cache),
        "roi_oof_cache": str(args.roi_oof_cache),
        "sample_count": int(len(sample_ids)),
        "searched": {
            "full_conv_weight_grid": list(full_probabilities_by_key.keys()),
            "roi_conv_weight_grid": list(roi_probabilities_by_key.keys()),
            "stacker_candidate_count": int(len(stacker_candidates)),
            "final_candidate_count": int(len(final_candidates)),
            "top_stackers_scanned": int(args.top_stackers),
        },
        "baseline_current_unetplusplus_oof_metrics": current_metrics,
        "selected_candidate": selected_slim,
        "top_candidates": [_json_slim_candidate(row) for row in final_candidates[:20]],
        "stacker": stacker,
        "frozen_config": str(frozen_config),
        "delta_auc_vs_current_oof": delta_auc,
        "recommendation": (
            "frozen_external_review_candidate"
            if delta_auc > 0.0
            else "report_only_no_internal_oof_gain"
        ),
        "notes": [
            "The selected candidate is deployable with the existing inference stack.",
            "Classifier member weights affect the full-image path; classifier_weight_overrides affect only the ROI path.",
            "The frozen config should receive at most one BUSI review before any merge decision.",
        ],
    }
    write_json_report(args.output, report)
    write_markdown_report(args.markdown, _build_markdown(report))
    print(
        {
            "selected_auc": selected_slim["metrics"]["auc"],
            "current_auc": current_metrics["auc"],
            "delta_auc": delta_auc,
            "frozen_config": str(frozen_config),
            "recommendation": report["recommendation"],
        }
    )
    return report


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
