"""Utility script for segmenter recalibrated roi protocol workflows."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_oof_stacking import (  # noqa: E402
    _load_or_create_splits,
    _resolve_project_path,
    _transform_features,
    _val_manifest_for_fold,
)
from scripts.run_roi_oof_experiment import (  # noqa: E402
    PAIR_VIEWS,
    _combine_pair_probabilities,
    _load_full_oof_pair,
    _predict_view_on_images,
)
from scripts.run_segmenter_architecture_replacement_protocol import (  # noqa: E402
    MAINLINE_METRICS,
    METHODS,
    SegmenterMethod,
)
from src.datasets.busbra import load_busbra_manifest  # noqa: E402
from src.engine.train_seg import run_segmentation_training  # noqa: E402
from src.models.segmenter import load_segmenter  # noqa: E402
from src.preprocess.io import read_image  # noqa: E402
from src.preprocess.roi import crop_to_mask_bbox, expand_bbox, mask_bbox  # noqa: E402
from src.preprocess.transforms import prepare_classifier_input  # noqa: E402
from src.utils.config import load_project_config  # noqa: E402
from src.utils.metrics import best_threshold_by_youden, classification_metrics  # noqa: E402
from src.utils.reporting import write_json_report, write_markdown_report  # noqa: E402
from src.utils.runtime import optional_import, require_dependency, select_device, timestamp_now  # noqa: E402


torch = optional_import("torch")

REPORT_DIR = Path("artifacts/reports/Chinese reports/08_segmenter_recalibrated_roi")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a full ROI recalibration protocol for non-SAM segmenters: train 5-fold "
            "segmenters, generate BUSBRA predicted-mask ROI OOF, fit stacker/gate on BUSBRA, "
            "freeze config, then run one BUSI external review."
        )
    )
    parser.add_argument("--base-segmenter-config", default="configs/segmenter/unet_5fold.yml")
    parser.add_argument("--runtime-config", default="configs/inference/demo.yml")
    parser.add_argument("--classifier-config", default="configs/classifier/convnext_tiny_timm_recipe.yml")
    parser.add_argument("--full-oof-cache", default="artifacts/reports/oof_two_model_predictions.json")
    parser.add_argument("--methods", default="all", help="Comma-separated method ids, or all.")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--segmenter-image-size", type=int, default=256)
    parser.add_argument("--margin-ratio", type=float, default=0.35)
    parser.add_argument("--mask-threshold", type=float, default=0.4)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-oof", action="store_true")
    parser.add_argument("--force-busi", action="store_true")
    parser.add_argument("--no-busi", action="store_true")
    parser.add_argument("--output", default=str(REPORT_DIR / "segmenter_recalibrated_roi_protocol.json"))
    parser.add_argument("--markdown", default=str(REPORT_DIR / "segmenter_recalibrated_roi_protocol.md"))
    return parser


def _selected_methods(value: str) -> list[SegmenterMethod]:
    method_by_id = {method.method_id: method for method in METHODS}
    if value == "all":
        return list(METHODS)
    selected = {item.strip() for item in value.split(",") if item.strip()}
    missing = sorted(selected - set(method_by_id))
    if missing:
        raise ValueError(f"Unknown method ids: {missing}")
    return [method_by_id[method_id] for method_id in selected]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def _write_yaml(path: str | Path, data: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    return target


def _method_config(
    *,
    base_config: dict[str, Any],
    method: SegmenterMethod,
    output_root: Path,
    fold_count: int,
    device: str | None,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["model"] = copy.deepcopy(method.model)
    config["loss"] = copy.deepcopy(method.loss)
    if device:
        config["device"] = device
    config.setdefault("training", {})
    config["training"]["fold_count"] = int(fold_count)
    config.setdefault("output", {})
    config["output"]["checkpoint_name"] = f"segmenter_recal_{method.method_id}_fold{{fold}}.pt"
    config["output"]["report_name"] = f"segmenter_recal_{method.method_id}_train_fold{{fold}}.json"
    config["segmenter_recalibrated_roi"] = {
        "method_id": method.method_id,
        "category": method.category,
        "rationale": method.rationale,
        "generated_at": timestamp_now(),
        "data_boundary": "BUSBRA for training and recalibration; BUSI frozen external review only.",
    }
    config["report_dir"] = str(output_root)
    return config


def _checkpoint_path(paths, config: dict[str, Any], fold: int) -> Path:
    checkpoint_name = config["output"]["checkpoint_name"].format(fold=fold)
    return paths.checkpoints_root / checkpoint_name


def _train_report_path(paths, config: dict[str, Any], fold: int) -> Path:
    report_name = config["output"]["report_name"].format(fold=fold)
    return paths.reports_root / report_name


def _load_cached_train_report(paths, config: dict[str, Any], fold: int) -> dict[str, Any] | None:
    report_path = _train_report_path(paths, config, fold)
    checkpoint_path = _checkpoint_path(paths, config, fold)
    if not report_path.exists() or not checkpoint_path.exists():
        return None
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    report["checkpoint_path"] = str(checkpoint_path)
    report["cached"] = True
    return report


def _train_missing_folds(
    *,
    config_path: Path,
    config: dict[str, Any],
    paths,
    fold_count: int,
    epochs: int,
    force_train: bool,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for fold in range(1, fold_count + 1):
        cached = None if force_train else _load_cached_train_report(paths, config, fold)
        if cached is not None:
            reports.append(cached)
            continue
        reports.append(run_segmentation_training(config_path, fold=fold, epochs_override=epochs))
    return reports


def _predict_segmenter_mask(
    segmenter,
    image: np.ndarray,
    *,
    image_size: int,
    device: str,
) -> np.ndarray:
    require_dependency("torch", torch)
    input_tensor = prepare_classifier_input(image, image_size)
    with torch.no_grad():
        logits = segmenter(input_tensor.unsqueeze(0).to(device=device, dtype=torch.float32))
    return torch.sigmoid(logits)[0, 0].cpu().numpy()


def _roi_area_ratio(
    mask: np.ndarray | None,
    *,
    threshold: float,
    margin_ratio: float,
    largest_component: bool = True,
) -> tuple[float, bool]:
    if mask is None:
        return 1.0, True
    bbox = mask_bbox(
        mask,
        threshold=threshold,
        min_area_ratio=0.001,
        largest_component=largest_component,
    )
    if bbox is None:
        return 1.0, True
    x1, y1, x2, y2 = expand_bbox(
        bbox,
        image_shape=mask.shape,
        margin_ratio=margin_ratio,
        square=True,
    )
    area = max(1, (x2 - x1) * (y2 - y1))
    total = max(1, int(mask.shape[0]) * int(mask.shape[1]))
    return float(area / total), False


def _load_segmenter_for_fold(checkpoint_path: Path, *, device: str):
    model = load_segmenter(
        {
            "architecture": "unet",
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "in_channels": 3,
            "classes": 1,
        },
        checkpoint_path=checkpoint_path,
        map_location="cpu",
    )
    model.to(device)
    model.eval()
    return model


def _generate_roi_oof(
    *,
    classifier_config: str | Path,
    checkpoint_paths: list[Path],
    fold_count: int,
    device: str,
    batch_size: int,
    margin_ratio: float,
    mask_threshold: float,
    segmenter_image_size: int,
) -> dict[str, Any]:
    config, paths = load_project_config(classifier_config)
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
        fold_count=fold_count,
        seed=int(config.get("seed", 42)),
    )
    resolved_device = select_device(device)
    views: dict[str, list[dict[str, Any]]] = {view_name: [] for view_name in PAIR_VIEWS}
    area_by_sample: dict[str, float] = {}
    fallback_by_sample: dict[str, bool] = {}
    for fold in range(1, fold_count + 1):
        checkpoint_path = checkpoint_paths[fold - 1]
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing fold {fold} segmenter checkpoint: {checkpoint_path}")
        segmenter = _load_segmenter_for_fold(checkpoint_path, device=resolved_device)
        val_manifest = _val_manifest_for_fold(manifest, assignments, fold)
        roi_images: list[np.ndarray] = []
        for row in val_manifest.itertuples(index=False):
            image = read_image(row.image_path, grayscale=True)
            mask = _predict_segmenter_mask(
                segmenter,
                image,
                image_size=segmenter_image_size,
                device=resolved_device,
            )
            area_ratio, used_fallback = _roi_area_ratio(
                mask,
                threshold=mask_threshold,
                margin_ratio=margin_ratio,
                largest_component=True,
            )
            area_by_sample[str(row.sample_id)] = area_ratio
            fallback_by_sample[str(row.sample_id)] = used_fallback
            roi_images.append(
                crop_to_mask_bbox(
                    image,
                    mask,
                    threshold=mask_threshold,
                    margin_ratio=margin_ratio,
                    largest_component=True,
                )
            )
        for view_name in PAIR_VIEWS:
            probabilities = _predict_view_on_images(
                view_name,
                roi_images,
                fold=fold,
                project_root=paths.project_root,
                device=resolved_device,
                batch_size=batch_size,
            )
            for row, probability in zip(val_manifest.itertuples(index=False), probabilities):
                views[view_name].append(
                    {
                        "sample_id": row.sample_id,
                        "case_id": row.case_id,
                        "fold_id": int(fold),
                        "pathology_label": row.pathology_label,
                        "malignant_probability": float(probability),
                    }
                )
        del segmenter
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    for view_name in views:
        views[view_name] = sorted(views[view_name], key=lambda row: str(row["sample_id"]))
    sample_ids = [str(row["sample_id"]) for row in views[PAIR_VIEWS[0]]]
    return {
        "source": "BUSBRA predicted-mask ROI OOF predictions",
        "fold_count": int(fold_count),
        "sample_count": len(sample_ids),
        "margin_ratio": float(margin_ratio),
        "mask_threshold": float(mask_threshold),
        "largest_component": True,
        "segmenter_checkpoints": [str(path) for path in checkpoint_paths],
        "views": views,
        "roi_area_ratios": [float(area_by_sample[sample_id]) for sample_id in sample_ids],
        "roi_fallback_flags": [bool(fallback_by_sample[sample_id]) for sample_id in sample_ids],
    }


def _load_or_generate_roi_oof(
    *,
    cache_path: Path,
    force_oof: bool,
    classifier_config: str | Path,
    checkpoint_paths: list[Path],
    fold_count: int,
    device: str,
    batch_size: int,
    margin_ratio: float,
    mask_threshold: float,
    segmenter_image_size: int,
) -> dict[str, Any]:
    if cache_path.exists() and not force_oof:
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    report = _generate_roi_oof(
        classifier_config=classifier_config,
        checkpoint_paths=checkpoint_paths,
        fold_count=fold_count,
        device=device,
        batch_size=batch_size,
        margin_ratio=margin_ratio,
        mask_threshold=mask_threshold,
        segmenter_image_size=segmenter_image_size,
    )
    write_json_report(cache_path, report)
    return report


def _fit_oof_stacker_with_cv(
    full_probabilities: np.ndarray,
    roi_probabilities: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    matrix = np.vstack([full_probabilities, roi_probabilities]).T
    candidates: list[dict[str, Any]] = []
    for feature_mode in ("probability", "logit"):
        features = _transform_features(matrix, feature_mode)
        for class_weight in (None, "balanced"):
            for c_value in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
                cv_probabilities = np.zeros(len(y_true), dtype=np.float64)
                fold_aucs: list[float] = []
                splitter = GroupKFold(n_splits=5)
                for train_index, val_index in splitter.split(features, y_true, groups):
                    model = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "classifier",
                                LogisticRegression(
                                    C=float(c_value),
                                    class_weight=class_weight,
                                    max_iter=2000,
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
                        "feature_mode": feature_mode,
                        "C": float(c_value),
                        "class_weight": class_weight,
                        "cv_auc": float(roc_auc_score(y_true, cv_probabilities)),
                        "fold_auc_mean": float(np.mean(fold_aucs)),
                        "fold_auc_std": float(np.std(fold_aucs)),
                        "cv_probabilities": cv_probabilities,
                    }
                )
    selected = max(
        candidates,
        key=lambda item: (item["cv_auc"], item["fold_auc_mean"], -item["fold_auc_std"]),
    )
    final_features = _transform_features(matrix, str(selected["feature_mode"]))
    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(selected["C"]),
                    class_weight=selected["class_weight"],
                    max_iter=2000,
                ),
            ),
        ]
    )
    final_model.fit(final_features, y_true)
    selected["model"] = final_model
    selected["threshold_from_oof"] = float(
        best_threshold_by_youden(
            y_true,
            selected["cv_probabilities"],
            thresholds=np.round(np.arange(0.1, 0.9001, 0.01), 2),
        )["threshold"]
    )
    return selected


def _metrics_at_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    best = best_threshold_by_youden(
        y_true,
        probabilities,
        thresholds=np.round(np.arange(0.1, 0.9001, 0.01), 2),
    )
    metrics = classification_metrics(y_true, probabilities, threshold=float(best["threshold"]))
    metrics["auc"] = float(roc_auc_score(y_true, probabilities))
    return metrics


def _scan_area_gates(
    *,
    y_true: np.ndarray,
    full_probabilities: np.ndarray,
    stack_probabilities: np.ndarray,
    area_ratios: np.ndarray,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    min_values = [0.0, 0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
    max_values = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.01]
    for min_area in min_values:
        for max_area in max_values:
            if min_area >= max_area:
                continue
            fallback = (area_ratios < min_area) | (area_ratios > max_area)
            probabilities = np.where(fallback, full_probabilities, stack_probabilities)
            metrics = _metrics_at_best_threshold(y_true, probabilities)
            candidates.append(
                {
                    "min_area_ratio": float(min_area),
                    "max_area_ratio": float(max_area),
                    "fallback_count": int(fallback.sum()),
                    "metrics": metrics,
                }
            )
    return sorted(
        candidates,
        key=lambda row: (
            row["metrics"]["auc"],
            row["metrics"]["f1_score"],
            row["metrics"]["sensitivity"],
            row["metrics"]["specificity"],
        ),
        reverse=True,
    )


def _slim_stacker(stacker: dict[str, Any]) -> dict[str, Any]:
    model = stacker["model"]
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    return {
        "feature_mode": stacker["feature_mode"],
        "C": float(stacker["C"]),
        "class_weight": stacker["class_weight"],
        "cv_auc": float(stacker["cv_auc"]),
        "fold_auc_mean": float(stacker["fold_auc_mean"]),
        "fold_auc_std": float(stacker["fold_auc_std"]),
        "threshold_from_oof": float(stacker["threshold_from_oof"]),
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) for value in scaler.scale_],
        "coef": [float(value) for value in classifier.coef_[0]],
        "intercept": float(classifier.intercept_[0]),
    }


def _freeze_runtime_config(
    *,
    runtime_config_path: str | Path,
    method: SegmenterMethod,
    checkpoint_paths: list[Path],
    stacker: dict[str, Any],
    gate: dict[str, Any],
    margin_ratio: float,
    mask_threshold: float,
    output_root: Path,
) -> Path:
    runtime = _load_yaml(runtime_config_path)
    runtime.setdefault("runtime", {})
    runtime["runtime"]["segmenter_checkpoint"] = None
    runtime["runtime"]["segmenter_checkpoints"] = [str(path) for path in checkpoint_paths]
    runtime["runtime"]["ensemble_display_name"] = (
        f"{runtime['runtime'].get('ensemble_display_name', 'BUCAD')} + recalibrated {method.method_id}"
    )
    runtime["runtime"]["default_threshold"] = float(gate["metrics"]["threshold"])
    roi_config = runtime["runtime"].setdefault("roi_enhancement", {})
    roi_config["enabled"] = True
    roi_config["margin_ratio"] = float(margin_ratio)
    roi_config["mask_threshold"] = float(mask_threshold)
    roi_config["largest_component"] = True
    roi_config["fallback_to_full"] = True
    roi_config["stacker"] = _slim_stacker(stacker)
    roi_config["quality_gate"] = {
        "enabled": True,
        "min_area_ratio": float(gate["min_area_ratio"]),
        "max_area_ratio": float(gate["max_area_ratio"]),
        "fallback_to_full": True,
    }
    roi_config["roi_stack_blend_weight"] = 1.0
    runtime["segmenter_recalibrated_roi"] = {
        "method_id": method.method_id,
        "category": method.category,
        "frozen_at": timestamp_now(),
        "data_boundary": "Stacker, threshold, and area gate selected from BUSBRA OOF only.",
    }
    return _write_yaml(output_root / "frozen_configs" / f"demo_{method.method_id}_recalibrated.yml", runtime)


def _run_busi_review(config_path: Path, output_root: Path, method_id: str, *, force: bool) -> dict[str, Any]:
    output_path = output_root / "external_reviews" / f"busi_{method_id}_recalibrated_frozen_review.json"
    if output_path.exists() and not force:
        with output_path.open("r", encoding="utf-8") as handle:
            return {"returncode": 0, "cached": True, "output_path": str(output_path), "report": json.load(handle)}
    command = [
        sys.executable,
        "scripts/eval_busi.py",
        "--config",
        str(config_path),
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    result: dict[str, Any] = {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "output_path": str(output_path),
    }
    if completed.returncode == 0 and output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            result["report"] = json.load(handle)
    return result


def _external_metrics(item: dict[str, Any]) -> dict[str, Any]:
    review = item.get("external_review")
    if not isinstance(review, dict):
        return {}
    report = review.get("report")
    if not isinstance(report, dict):
        return {}
    metrics = report.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _candidate_decision(item: dict[str, Any]) -> str:
    metrics = _external_metrics(item)
    auc = metrics.get("auc")
    if not isinstance(auc, (int, float)):
        return "report_only"
    if (
        float(auc) > MAINLINE_METRICS["auc"]
        and float(metrics.get("sensitivity", 0.0)) >= MAINLINE_METRICS["sensitivity"] - 0.02
        and float(metrics.get("specificity", 0.0)) >= MAINLINE_METRICS["specificity"] - 0.02
        and float(metrics.get("f1_score", 0.0)) >= MAINLINE_METRICS["f1_score"] - 0.02
    ):
        return "candidate_hold_for_user"
    return "report_only"


def _mean_internal_metrics(fold_reports: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted(
        {
            key
            for report in fold_reports
            for key, value in report.get("metrics", {}).items()
            if isinstance(value, (int, float)) and np.isfinite(float(value))
        }
    )
    return {
        key: float(np.mean([float(report["metrics"][key]) for report in fold_reports if key in report["metrics"]]))
        for key in keys
    }


def _markdown(report: dict[str, Any]) -> list[str]:
    lines = [
        "# 分割器替换后的 ROI OOF 重新标定实验",
        "",
        "## 协议边界",
        "",
        "- BUSBRA 用于分割器训练、预测 mask OOF、ROI stacker 拟合、area gate 扫描和阈值选择。",
        "- BUSI 只在每个配置冻结后做一次外部验证，不根据 BUSI 回调参数。",
        "- 每个方法使用 fold-specific segmenter 生成 BUSBRA OOF mask；外部部署配置使用 5 个 segmenter checkpoint 的 mask ensemble。",
        "- 主线对照为当前 `configs/inference/demo.yml`：BUSI AUC `0.9256`，阈值 `0.51`。",
        "",
        "## 总表",
        "",
        "| 方法 | 状态 | Val Dice | OOF ROI AUC | OOF Stack AUC | Gate | OOF Gated AUC | BUSI AUC | Sens | Spec | F1 | 决策 |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["results"]:
        internal = item.get("internal_mean_metrics", {})
        oof = item.get("oof", {})
        gate = item.get("selected_gate", {})
        gate_metrics = gate.get("metrics", {}) if isinstance(gate, dict) else {}
        metrics = _external_metrics(item)
        lines.append(
            "| "
            + " | ".join(
                [
                    item["method_id"],
                    item["status"],
                    f"{float(internal['dice']):.4f}" if "dice" in internal else "-",
                    f"{float(oof['roi_pair']['auc']):.4f}" if "roi_pair" in oof else "-",
                    f"{float(oof['stacker_cv_auc']):.4f}" if "stacker_cv_auc" in oof else "-",
                    (
                        f"{gate.get('min_area_ratio', 0):.2f}-{gate.get('max_area_ratio', 0):.2f}"
                        if gate
                        else "-"
                    ),
                    f"{float(gate_metrics['auc']):.4f}" if "auc" in gate_metrics else "-",
                    f"{float(metrics['auc']):.4f}" if "auc" in metrics else "-",
                    f"{float(metrics['sensitivity']):.4f}" if "sensitivity" in metrics else "-",
                    f"{float(metrics['specificity']):.4f}" if "specificity" in metrics else "-",
                    f"{float(metrics['f1_score']):.4f}" if "f1_score" in metrics else "-",
                    item.get("decision", "report_only"),
                ]
            )
            + " |"
        )
    best = report.get("best_by_busi_auc")
    lines.extend(["", "## 结论", ""])
    if isinstance(best, dict):
        delta = float(best["busi_auc"]) - MAINLINE_METRICS["auc"]
        lines.append(
            f"- 本轮重新标定后最佳方法为 `{best['method_id']}`，BUSI AUC `{best['busi_auc']:.4f}`，相对主线 `{delta:+.4f}`。"
        )
    lines.extend(
        [
            "- 只有 `candidate_hold_for_user` 才表示可供人工决定是否合入；`report_only` 只保留实验记录。",
            "- 本脚本不修改 `configs/inference/demo.yml`，所有冻结候选配置均保存在本目录 `frozen_configs/` 下。",
        ]
    )
    return lines


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_config = _load_yaml(args.base_segmenter_config)
    _, paths = load_project_config(args.base_segmenter_config)
    output_root = (paths.project_root / REPORT_DIR).resolve()
    generated_config_root = output_root / "generated_configs"
    oof_root = output_root / "oof_predictions"
    output_root.mkdir(parents=True, exist_ok=True)
    generated_config_root.mkdir(parents=True, exist_ok=True)
    oof_root.mkdir(parents=True, exist_ok=True)

    selected_methods = _selected_methods(str(args.methods))
    sample_ids, y_true, groups, full_oof_pair = _load_full_oof_pair(args.full_oof_cache)
    baseline_full = _metrics_at_best_threshold(y_true, full_oof_pair)
    results: list[dict[str, Any]] = []

    for method in selected_methods:
        item: dict[str, Any] = {
            "method_id": method.method_id,
            "category": method.category,
            "status": "pending",
            "decision": "report_only",
        }
        try:
            config = _method_config(
                base_config=base_config,
                method=method,
                output_root=output_root,
                fold_count=int(args.fold_count),
                device=args.device,
            )
            config_path = _write_yaml(generated_config_root / f"{method.method_id}.yml", config)
            item["config_path"] = str(config_path)
            fold_reports = _train_missing_folds(
                config_path=config_path,
                config=config,
                paths=paths,
                fold_count=int(args.fold_count),
                epochs=int(args.epochs),
                force_train=bool(args.force_train),
            )
            checkpoint_paths = [Path(report["checkpoint_path"]) for report in sorted(fold_reports, key=lambda row: row["fold"])]
            item["fold_reports"] = fold_reports
            item["internal_mean_metrics"] = _mean_internal_metrics(fold_reports)
            roi_oof_path = oof_root / f"{method.method_id}_roi_oof.json"
            roi_oof = _load_or_generate_roi_oof(
                cache_path=roi_oof_path,
                force_oof=bool(args.force_oof),
                classifier_config=args.classifier_config,
                checkpoint_paths=checkpoint_paths,
                fold_count=int(args.fold_count),
                device=str(args.device),
                batch_size=int(args.batch_size),
                margin_ratio=float(args.margin_ratio),
                mask_threshold=float(args.mask_threshold),
                segmenter_image_size=int(args.segmenter_image_size),
            )
            roi_oof_pair = _combine_pair_probabilities(roi_oof["views"])
            roi_sample_ids = [str(row["sample_id"]) for row in roi_oof["views"][PAIR_VIEWS[0]]]
            if roi_sample_ids != [str(sample_id) for sample_id in sample_ids]:
                raise ValueError(f"OOF sample order mismatch for {method.method_id}.")
            area_ratios = np.asarray(roi_oof["roi_area_ratios"], dtype=np.float64)
            stacker = _fit_oof_stacker_with_cv(full_oof_pair, roi_oof_pair, y_true, groups)
            gate_candidates = _scan_area_gates(
                y_true=y_true,
                full_probabilities=full_oof_pair,
                stack_probabilities=stacker["cv_probabilities"],
                area_ratios=area_ratios,
            )
            selected_gate = gate_candidates[0]
            frozen_config = _freeze_runtime_config(
                runtime_config_path=args.runtime_config,
                method=method,
                checkpoint_paths=checkpoint_paths,
                stacker=stacker,
                gate=selected_gate,
                margin_ratio=float(args.margin_ratio),
                mask_threshold=float(args.mask_threshold),
                output_root=output_root,
            )
            item["oof_roi_cache"] = str(roi_oof_path)
            item["frozen_config"] = str(frozen_config)
            item["oof"] = {
                "full_pair": baseline_full,
                "roi_pair": _metrics_at_best_threshold(y_true, roi_oof_pair),
                "stacker_cv_auc": float(stacker["cv_auc"]),
                "stacker_threshold": float(stacker["threshold_from_oof"]),
                "area_ratio_mean": float(area_ratios.mean()),
                "area_ratio_median": float(np.median(area_ratios)),
            }
            item["stacker"] = _slim_stacker(stacker)
            item["selected_gate"] = selected_gate
            item["top_gates"] = gate_candidates[:10]
            item["status"] = "trained_recalibrated"
            if not args.no_busi:
                item["external_review"] = _run_busi_review(
                    frozen_config,
                    output_root,
                    method.method_id,
                    force=bool(args.force_busi),
                )
                item["decision"] = _candidate_decision(item)
        except Exception as exc:
            item["status"] = "failed"
            item["error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)
        write_json_report(args.output, _build_report(args, results))
        write_markdown_report(args.markdown, _markdown(_build_report(args, results)))

    report = _build_report(args, results)
    write_json_report(args.output, report)
    write_markdown_report(args.markdown, _markdown(report))
    print(
        {
            "methods": len(results),
            "completed": sum(1 for item in results if item["status"] == "trained_recalibrated"),
            "failed": sum(1 for item in results if item["status"] == "failed"),
            "best_by_busi_auc": report.get("best_by_busi_auc"),
            "markdown": args.markdown,
        }
    )
    return report


def _build_report(args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    best = None
    for item in results:
        metrics = _external_metrics(item)
        auc = metrics.get("auc")
        if isinstance(auc, (int, float)) and (best is None or float(auc) > float(best["busi_auc"])):
            best = {"method_id": item["method_id"], "busi_auc": float(auc)}
    return {
        "generated_at": timestamp_now(),
        "method": "non-SAM segmenter replacement with BUSBRA OOF ROI recalibration",
        "data_boundary": "BUSBRA trains segmenters and selects stacker/gate/threshold; BUSI is frozen external review only.",
        "runtime_config": str(args.runtime_config),
        "full_oof_cache": str(args.full_oof_cache),
        "fold_count": int(args.fold_count),
        "epochs": int(args.epochs),
        "margin_ratio": float(args.margin_ratio),
        "mask_threshold": float(args.mask_threshold),
        "mainline_metrics": MAINLINE_METRICS,
        "results": results,
        "best_by_busi_auc": best,
    }


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
