"""Utility script for oof stacking workflows."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.busbra import (
    LABEL_TO_INDEX,
    generate_busbra_split_assignments,
    load_busbra_manifest,
)
from src.models.classifier import classifier_probabilities, load_classifier
from src.preprocess.io import read_image
from src.preprocess.transforms import prepare_classifier_input
from src.utils.config import load_project_config
from src.utils.metrics import best_threshold_by_youden, classification_metrics
from src.utils.reporting import write_json_report, write_markdown_report
from src.utils.runtime import optional_import, require_dependency, select_device


torch = optional_import("torch")


@dataclass(frozen=True)
class ModelView:
    """Represent ModelView for this module."""
    name: str
    model_name: str
    checkpoint_template: str
    image_size: int = 224
    apply_clahe: bool = True
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    interpolation: str = "area"
    crop_pct: float = 1.0
    tta_variants: tuple[dict[str, Any], ...] = ({"name": "identity"},)


MODEL_VIEWS: dict[str, ModelView] = {
    "eff_identity": ModelView(
        name="eff_identity",
        model_name="tf_efficientnetv2_s",
        checkpoint_template="./artifacts/checkpoints/efficientnetv2_s_fold{fold}.pt",
        tta_variants=({"name": "identity"},),
    ),
    "conv_hflip": ModelView(
        name="conv_hflip",
        model_name="convnext_tiny",
        checkpoint_template="./artifacts/checkpoints/convnext_tiny_timm_recipe_fold{fold}.pt",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation="bicubic",
        crop_pct=0.95,
        tta_variants=(
            {"name": "identity", "crop_pct": 0.95},
            {"name": "hflip", "crop_pct": 0.95},
        ),
    ),
    "conv_crop_sweep": ModelView(
        name="conv_crop_sweep",
        model_name="convnext_tiny",
        checkpoint_template="./artifacts/checkpoints/convnext_tiny_timm_recipe_fold{fold}.pt",
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
    ),
}


FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "current_crop_sweep": ("eff_identity", "conv_crop_sweep"),
    "auc_hflip": ("eff_identity", "conv_hflip"),
    "multi_view": ("eff_identity", "conv_crop_sweep", "conv_hflip"),
}


BUSI_REPORTS: dict[str, str] = {
    "eff_identity": "artifacts/reports/busi_efficientnetv2_s_5fold_identity.json",
    "conv_crop_sweep": "artifacts/reports/busi_convnext_tiny_tta_crop_sweep.json",
    "conv_hflip": "artifacts/reports/busi_convnext_tiny_tta_hflip.json",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="Generate BUSBRA OOF probabilities and evaluate lightweight stackers."
    )
    parser.add_argument(
        "--config",
        default="configs/classifier/convnext_tiny_timm_recipe.yml",
        help="Project config used only for paths and split location.",
    )
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        default="artifacts/reports/oof_two_model_stacking.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--markdown",
        default="artifacts/reports/oof_two_model_stacking.md",
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--oof-cache",
        default="artifacts/reports/oof_two_model_predictions.json",
        help="OOF probability cache path.",
    )
    parser.add_argument(
        "--reuse-oof",
        action="store_true",
        help="Reuse cached OOF predictions when available.",
    )
    return parser


def _resolve_project_path(project_root: Path, value: str | Path) -> Path:
    """Resolve project path."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _load_or_create_splits(
    manifest: pd.DataFrame,
    split_path: Path,
    *,
    fold_count: int,
    seed: int,
) -> pd.DataFrame:
    """Load or create splits."""
    if split_path.exists():
        return pd.read_csv(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    assignments = generate_busbra_split_assignments(
        manifest,
        n_splits=fold_count,
        seed=seed,
    )
    assignments.to_csv(split_path, index=False)
    return assignments


def _val_manifest_for_fold(
    manifest: pd.DataFrame,
    assignments: pd.DataFrame,
    fold: int,
) -> pd.DataFrame:
    """Return validation manifest rows for one fold."""
    fold_assignments = assignments[assignments["fold_id"] == fold]
    val_ids = set(fold_assignments.loc[fold_assignments["stage"] == "val", "sample_id"])
    return manifest[manifest["sample_id"].isin(val_ids)].reset_index(drop=True)


def _apply_tta(image: np.ndarray, variant: dict[str, Any]) -> np.ndarray:
    """Apply a configured test-time augmentation variant to an image."""
    name = str(variant.get("name", "identity")).lower()
    if name in {"identity", "none", "original"}:
        return image
    if name in {"hflip", "horizontal_flip"}:
        return np.fliplr(image).copy()
    return image


def _prepare_view_tensor(image: np.ndarray, view: ModelView, variant: dict[str, Any]):
    """Prepare view tensor."""
    return prepare_classifier_input(
        _apply_tta(image, variant),
        view.image_size,
        apply_clahe_enabled=view.apply_clahe,
        mean=view.mean,
        std=view.std,
        interpolation=view.interpolation,
        crop_pct=float(variant.get("crop_pct", view.crop_pct)),
    )


def _load_view_model(view: ModelView, checkpoint_path: Path):
    """Load view model."""
    return load_classifier(
        {
            "name": view.model_name,
            "pretrained": False,
            "in_chans": 3,
            "num_classes": 2,
        },
        checkpoint_path=checkpoint_path,
        map_location="cpu",
    )


def _predict_view_for_manifest(
    view: ModelView,
    manifest: pd.DataFrame,
    *,
    fold: int,
    project_root: Path,
    device: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Predict view for manifest."""
    require_dependency("torch", torch)
    checkpoint_path = _resolve_project_path(
        project_root,
        view.checkpoint_template.format(fold=fold),
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {view.name} fold {fold}: {checkpoint_path}")
    model = _load_view_model(view, checkpoint_path)
    model.eval()

    rows: list[dict[str, Any]] = []
    for start in range(0, len(manifest), batch_size):
        subset = manifest.iloc[start:start + batch_size]
        tensors = []
        for item in subset.itertuples(index=False):
            image = read_image(item.image_path, grayscale=True)
            for variant in view.tta_variants:
                tensors.append(_prepare_view_tensor(image, view, variant))
        batch = torch.stack(tensors)
        probabilities = classifier_probabilities(model, batch, device=device)
        probabilities = probabilities.reshape(len(subset), len(view.tta_variants), 2).mean(axis=1)
        for item, probability in zip(subset.itertuples(index=False), probabilities):
            rows.append(
                {
                    "sample_id": item.sample_id,
                    "case_id": item.case_id,
                    "fold_id": int(fold),
                    "pathology_label": item.pathology_label,
                    "malignant_probability": float(probability[1]),
                }
            )
    return rows


def _generate_oof_predictions(
    *,
    config_path: str | Path,
    fold_count: int,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Generate oof predictions."""
    config, paths = load_project_config(config_path)
    seed = int(config.get("seed", 42))
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
        seed=seed,
    )
    resolved_device = select_device(device)

    predictions: dict[str, list[dict[str, Any]]] = {}
    for view in MODEL_VIEWS.values():
        view_rows: list[dict[str, Any]] = []
        for fold in range(1, fold_count + 1):
            val_manifest = _val_manifest_for_fold(manifest, assignments, fold)
            if val_manifest.empty:
                raise ValueError(f"Fold {fold} has no validation samples.")
            view_rows.extend(
                _predict_view_for_manifest(
                    view,
                    val_manifest,
                    fold=fold,
                    project_root=paths.project_root,
                    device=resolved_device,
                    batch_size=batch_size,
                )
            )
        predictions[view.name] = sorted(view_rows, key=lambda row: row["sample_id"])

    return {
        "source": "BUSBRA OOF validation predictions",
        "fold_count": fold_count,
        "sample_count": int(len(predictions["eff_identity"])),
        "views": predictions,
    }


def _load_busi_view_probabilities() -> tuple[list[dict[str, str]], dict[str, np.ndarray], np.ndarray]:
    """Load busi view probabilities."""
    reference_rows: list[dict[str, str]] | None = None
    y_true: np.ndarray | None = None
    probabilities: dict[str, np.ndarray] = {}
    for view_name, report_path in BUSI_REPORTS.items():
        with Path(report_path).open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        rows = report["rows"]
        current_reference = [
            {
                "sample_id": str(row["sample_id"]),
                "pathology_label": str(row["pathology_label"]),
            }
            for row in rows
        ]
        if reference_rows is None:
            reference_rows = current_reference
            y_true = np.asarray(
                [1 if row["pathology_label"] == "malignant" else 0 for row in rows],
                dtype=np.int32,
            )
        elif current_reference != reference_rows:
            raise ValueError(f"BUSI report row order mismatch: {report_path}")
        probabilities[view_name] = np.asarray(
            [row["malignant_probability"] for row in rows],
            dtype=np.float64,
        )
    if reference_rows is None or y_true is None:
        raise ValueError("No BUSI reports loaded.")
    return reference_rows, probabilities, y_true


def _rows_to_matrix(
    oof_report: dict[str, Any],
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Convert report rows into a model-feature matrix."""
    reference = oof_report["views"][feature_names[0]]
    sample_ids = [row["sample_id"] for row in reference]
    y_true = np.asarray(
        [LABEL_TO_INDEX[str(row["pathology_label"]).lower()] for row in reference],
        dtype=np.int32,
    )
    groups = np.asarray([str(row["case_id"]) for row in reference])
    columns = []
    for feature_name in feature_names:
        rows = oof_report["views"][feature_name]
        if [row["sample_id"] for row in rows] != sample_ids:
            raise ValueError(f"OOF row order mismatch for {feature_name}")
        columns.append(np.asarray([row["malignant_probability"] for row in rows], dtype=np.float64))
    return np.vstack(columns).T, y_true, groups, sample_ids


def _transform_features(matrix: np.ndarray, mode: str) -> np.ndarray:
    """Transform stacked probability features for the meta learner."""
    if mode == "probability":
        return matrix
    if mode == "logit":
        clipped = np.clip(matrix, 1e-6, 1.0 - 1e-6)
        return np.log(clipped / (1.0 - clipped))
    raise ValueError(f"Unsupported feature mode: {mode}")


def _make_pipeline(C: float, class_weight: str | None) -> Pipeline:
    """Create pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(C),
                    class_weight=class_weight,
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _cv_score_stacker(
    features: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    *,
    C: float,
    class_weight: str | None,
) -> dict[str, Any]:
    """Cross-validate a stacker and return its score summary."""
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Need at least two groups for OOF stacker CV.")
    splitter = GroupKFold(n_splits=n_splits)
    probabilities = np.zeros(len(y_true), dtype=np.float64)
    fold_aucs: list[float] = []
    for train_index, val_index in splitter.split(features, y_true, groups):
        model = _make_pipeline(C, class_weight)
        model.fit(features[train_index], y_true[train_index])
        fold_probabilities = model.predict_proba(features[val_index])[:, 1]
        probabilities[val_index] = fold_probabilities
        if len(np.unique(y_true[val_index])) > 1:
            fold_aucs.append(float(roc_auc_score(y_true[val_index], fold_probabilities)))
    return {
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "cv_probabilities": probabilities,
    }


def _select_stacker(
    matrix: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    """Select stacker."""
    candidates: list[dict[str, Any]] = []
    for feature_mode in ("probability", "logit"):
        transformed = _transform_features(matrix, feature_mode)
        for class_weight in (None, "balanced"):
            for C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
                cv_result = _cv_score_stacker(
                    transformed,
                    y_true,
                    groups,
                    C=C,
                    class_weight=class_weight,
                )
                candidates.append(
                    {
                        "feature_mode": feature_mode,
                        "class_weight": class_weight,
                        "C": float(C),
                        "mean_auc": cv_result["mean_auc"],
                        "std_auc": cv_result["std_auc"],
                        "cv_probabilities": cv_result["cv_probabilities"],
                    }
                )
    return max(candidates, key=lambda row: (row["mean_auc"], -row["std_auc"]))


def _metrics_with_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Calculate metrics with best threshold."""
    default = classification_metrics(y_true, probabilities, threshold=0.5)
    best = best_threshold_by_youden(
        y_true,
        probabilities,
        thresholds=np.round(np.arange(0.1, 0.9001, 0.01), 2),
    )
    return {
        "default": default,
        "best_by_youden": best,
    }


def _evaluate_feature_set(
    name: str,
    feature_names: tuple[str, ...],
    oof_report: dict[str, Any],
    busi_probabilities: dict[str, np.ndarray],
    busi_y_true: np.ndarray,
) -> dict[str, Any]:
    """Evaluate feature set."""
    matrix, y_true, groups, sample_ids = _rows_to_matrix(oof_report, feature_names)
    selected = _select_stacker(matrix, y_true, groups)
    feature_mode = str(selected["feature_mode"])
    features = _transform_features(matrix, feature_mode)
    model = _make_pipeline(float(selected["C"]), selected["class_weight"])
    model.fit(features, y_true)
    oof_probabilities = model.predict_proba(features)[:, 1]
    cv_probabilities = selected["cv_probabilities"]
    threshold_from_oof = float(
        best_threshold_by_youden(
            y_true,
            cv_probabilities,
            thresholds=np.round(np.arange(0.1, 0.9001, 0.01), 2),
        )["threshold"]
    )

    busi_matrix = np.vstack([busi_probabilities[feature_name] for feature_name in feature_names]).T
    busi_features = _transform_features(busi_matrix, feature_mode)
    busi_stack_probabilities = model.predict_proba(busi_features)[:, 1]

    logistic = model.named_steps["classifier"]
    scaler = model.named_steps["scaler"]
    return {
        "name": name,
        "features": list(feature_names),
        "selected_stacker": {
            "feature_mode": feature_mode,
            "C": float(selected["C"]),
            "class_weight": selected["class_weight"],
            "cv_mean_auc": float(selected["mean_auc"]),
            "cv_std_auc": float(selected["std_auc"]),
            "threshold_from_oof_cv": threshold_from_oof,
            "coef": [float(value) for value in logistic.coef_[0]],
            "intercept": float(logistic.intercept_[0]),
            "scaler_mean": [float(value) for value in scaler.mean_],
            "scaler_scale": [float(value) for value in scaler.scale_],
        },
        "oof_fit_metrics": _metrics_with_best_threshold(y_true, oof_probabilities),
        "oof_cv_metrics": _metrics_with_best_threshold(y_true, cv_probabilities),
        "busi_metrics_at_0_5": classification_metrics(busi_y_true, busi_stack_probabilities, threshold=0.5),
        "busi_metrics_at_oof_threshold": classification_metrics(
            busi_y_true,
            busi_stack_probabilities,
            threshold=threshold_from_oof,
        ),
        "busi_best_by_youden": best_threshold_by_youden(
            busi_y_true,
            busi_stack_probabilities,
            thresholds=np.round(np.arange(0.1, 0.9001, 0.01), 2),
        ),
        "busi_auc": float(roc_auc_score(busi_y_true, busi_stack_probabilities)),
        "oof_sample_ids": sample_ids,
        "busi_probabilities": [float(value) for value in busi_stack_probabilities],
    }


def _strip_large_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Remove bulky arrays from a report payload before writing."""
    slim = dict(report)
    slim["feature_set_results"] = []
    for result in report["feature_set_results"]:
        copied = dict(result)
        copied.pop("oof_sample_ids", None)
        copied.pop("busi_probabilities", None)
        slim["feature_set_results"].append(copied)
    return slim


def _format_metric(metric: dict[str, Any], key: str) -> str:
    """Format one metric value for report tables."""
    value = metric.get(key)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _confusion_text(metric: dict[str, Any]) -> str:
    """Format confusion-matrix counts for report tables."""
    confusion = metric.get("confusion", {})
    return (
        f"TN {confusion.get('tn')} / FP {confusion.get('fp')} / "
        f"FN {confusion.get('fn')} / TP {confusion.get('tp')}"
    )


def build_markdown(report: dict[str, Any]) -> list[str]:
    """Build the Markdown report body for this experiment."""
    lines = [
        "# 两模型 OOF Stacking 实验报告",
        "",
        "日期：2026-04-25",
        "",
        "## 实验边界",
        "",
        "- OOF 融合器只使用 BUSBRA 五折验证样本的 out-of-fold 概率训练。",
        "- BUSI 只用于外部评估，不参与融合器训练、交叉验证或阈值选择。",
        "- 本报告是旁路优化实验，不修改 `configs/inference/demo.yml`。",
        "",
        "## 输入视图",
        "",
        "- `eff_identity`：EfficientNetV2-S 五折，identity TTA。",
        "- `conv_crop_sweep`：ConvNeXt-Tiny 五折，crop-sweep TTA，对齐当前 Demo 主线。",
        "- `conv_hflip`：ConvNeXt-Tiny 五折，hflip TTA，对齐上一轮 AUC 候选。",
        "",
        "## OOF 融合结果",
        "",
        "| 方案 | 特征 | CV AUC | OOF阈值 | BUSI AUC | BUSI@OOF阈值 Sens | BUSI@OOF阈值 Spec | BUSI@OOF阈值 Acc | BUSI最佳Youden阈值 | BUSI最佳Youden Sens | BUSI最佳Youden Spec |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report["feature_set_results"]:
        stacker = result["selected_stacker"]
        busi_oof_threshold = result["busi_metrics_at_oof_threshold"]
        busi_best = result["busi_best_by_youden"]
        lines.append(
            "| "
            + " | ".join(
                [
                    result["name"],
                    " + ".join(result["features"]),
                    f"{stacker['cv_mean_auc']:.4f}",
                    f"{stacker['threshold_from_oof_cv']:.2f}",
                    f"{result['busi_auc']:.4f}",
                    _format_metric(busi_oof_threshold, "sensitivity"),
                    _format_metric(busi_oof_threshold, "specificity"),
                    _format_metric(busi_oof_threshold, "accuracy"),
                    f"{busi_best['threshold']:.2f}",
                    _format_metric(busi_best, "sensitivity"),
                    _format_metric(busi_best, "specificity"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 外部混淆矩阵",
            "",
            "| 方案 | BUSI@OOF阈值 | BUSI最佳Youden点 |",
            "| --- | --- | --- |",
        ]
    )
    for result in report["feature_set_results"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    result["name"],
                    _confusion_text(result["busi_metrics_at_oof_threshold"]),
                    _confusion_text(result["busi_best_by_youden"]),
                ]
            )
            + " |"
        )

    best = report["best_by_busi_auc"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 本轮 BUSI AUC 最高的是 `{best['name']}`，AUC `{best['busi_auc']:.4f}`。",
            "- OOF Stacking 的训练流程比直接在 BUSI 上搜权重更规范，但是否替换主线仍要看 BUSI 外部指标是否真正提升。",
            "- 如果 OOF 阈值迁移到 BUSI 后敏感性下降明显，应保留当前 Demo 主线，把该结果作为实验记录。",
        ]
    )
    return lines


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the experiment workflow and return the generated summary."""
    oof_cache_path = Path(args.oof_cache)
    if args.reuse_oof and oof_cache_path.exists():
        with oof_cache_path.open("r", encoding="utf-8") as handle:
            oof_report = json.load(handle)
    else:
        oof_report = _generate_oof_predictions(
            config_path=args.config,
            fold_count=int(args.fold_count),
            device=str(args.device),
            batch_size=int(args.batch_size),
        )
        write_json_report(oof_cache_path, oof_report)

    busi_rows, busi_probabilities, busi_y_true = _load_busi_view_probabilities()
    feature_set_results = [
        _evaluate_feature_set(name, features, oof_report, busi_probabilities, busi_y_true)
        for name, features in FEATURE_SETS.items()
    ]
    best_by_busi_auc = max(
        feature_set_results,
        key=lambda result: (
            float(result["busi_auc"]),
            float(result["busi_best_by_youden"]["youden_j"]),
        ),
    )
    report = {
        "method": "OOF logistic stacking",
        "oof_cache": str(oof_cache_path),
        "busi_sample_count": len(busi_rows),
        "feature_sets": {name: list(features) for name, features in FEATURE_SETS.items()},
        "feature_set_results": feature_set_results,
        "best_by_busi_auc": {
            "name": best_by_busi_auc["name"],
            "features": best_by_busi_auc["features"],
            "busi_auc": best_by_busi_auc["busi_auc"],
            "busi_metrics_at_oof_threshold": best_by_busi_auc["busi_metrics_at_oof_threshold"],
            "busi_best_by_youden": best_by_busi_auc["busi_best_by_youden"],
            "selected_stacker": best_by_busi_auc["selected_stacker"],
        },
    }
    write_json_report(args.output, _strip_large_payload(report))
    write_markdown_report(args.markdown, build_markdown(report))
    return report


def main() -> int:
    """Parse CLI arguments and run the script entry point."""
    args = build_parser().parse_args()
    report = run(args)
    best = report["best_by_busi_auc"]
    print(
        {
            "best_feature_set": best["name"],
            "busi_auc": best["busi_auc"],
            "oof_threshold": best["busi_metrics_at_oof_threshold"]["threshold"],
            "busi_sensitivity_at_oof_threshold": best["busi_metrics_at_oof_threshold"]["sensitivity"],
            "busi_specificity_at_oof_threshold": best["busi_metrics_at_oof_threshold"]["specificity"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
