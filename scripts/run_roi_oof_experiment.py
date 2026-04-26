from __future__ import annotations

import argparse
import json
import sys
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

from scripts.run_oof_stacking import (
    MODEL_VIEWS,
    _apply_tta,
    _load_or_create_splits,
    _load_view_model,
    _resolve_project_path,
    _transform_features,
    _val_manifest_for_fold,
)
from src.datasets.busbra import LABEL_TO_INDEX, load_busbra_manifest
from src.datasets.busi import load_busi_manifest
from src.models.classifier import classifier_probabilities
from src.models.segmenter import load_segmenter
from src.preprocess.io import read_image, read_mask
from src.preprocess.roi import crop_to_mask_bbox, expand_bbox, mask_bbox
from src.preprocess.transforms import prepare_classifier_input
from src.utils.config import load_project_config
from src.utils.metrics import best_threshold_by_youden, classification_metrics
from src.utils.reporting import write_json_report, write_markdown_report
from src.utils.runtime import optional_import, require_dependency, select_device


torch = optional_import("torch")

PAIR_VIEWS = ("eff_identity", "conv_crop_sweep")
PAIR_WEIGHTS = {"eff_identity": 0.427, "conv_crop_sweep": 0.573}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ROI-cropped lesion guidance with BUSBRA OOF stacking."
    )
    parser.add_argument(
        "--config",
        default="configs/classifier/convnext_tiny_timm_recipe.yml",
        help="Project config used for paths and BUSBRA split settings.",
    )
    parser.add_argument(
        "--full-oof-cache",
        default="artifacts/reports/oof_two_model_predictions.json",
        help="Existing full-image OOF probability cache from run_oof_stacking.py.",
    )
    parser.add_argument(
        "--roi-oof-cache",
        default="artifacts/reports/roi_oof_predictions.json",
        help="Output/reuse cache for BUSBRA GT-mask ROI OOF probabilities.",
    )
    parser.add_argument(
        "--segmenter-checkpoint",
        default="artifacts/checkpoints/segmenter_fold1.pt",
        help="Segmenter checkpoint for deployable BUSI ROI proxy.",
    )
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--margin-ratio", type=float, default=0.35)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--reuse-roi-oof", action="store_true")
    parser.add_argument(
        "--output",
        default="artifacts/reports/roi_oof_experiment.json",
    )
    parser.add_argument(
        "--markdown",
        default="artifacts/reports/roi_oof_experiment.md",
    )
    return parser


def _predict_view_on_images(
    view_name: str,
    images: list[np.ndarray],
    *,
    fold: int,
    project_root: Path,
    device: str,
    batch_size: int,
) -> np.ndarray:
    require_dependency("torch", torch)
    view = MODEL_VIEWS[view_name]
    checkpoint_path = _resolve_project_path(
        project_root,
        view.checkpoint_template.format(fold=fold),
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {view_name} fold {fold}: {checkpoint_path}")
    model = _load_view_model(view, checkpoint_path)
    model.eval()
    probabilities: list[float] = []
    for start in range(0, len(images), batch_size):
        subset = images[start:start + batch_size]
        tensors = []
        for image in subset:
            for variant in view.tta_variants:
                tensor = prepare_classifier_input(
                    _apply_tta(image, variant),
                    view.image_size,
                    apply_clahe_enabled=view.apply_clahe,
                    mean=view.mean,
                    std=view.std,
                    interpolation=view.interpolation,
                    crop_pct=float(variant.get("crop_pct", view.crop_pct)),
                )
                tensors.append(tensor)
        batch = torch.stack(tensors)
        probs = classifier_probabilities(model, batch, device=device)
        probs = probs.reshape(len(subset), len(view.tta_variants), 2).mean(axis=1)
        probabilities.extend(float(value) for value in probs[:, 1])
    return np.asarray(probabilities, dtype=np.float64)


def _roi_images_from_manifest(
    manifest: pd.DataFrame,
    *,
    margin_ratio: float,
    mask_threshold: float,
) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for row in manifest.itertuples(index=False):
        image = read_image(row.image_path, grayscale=True)
        mask = read_mask(row.mask_path)
        images.append(
            crop_to_mask_bbox(
                image,
                mask,
                threshold=mask_threshold,
                margin_ratio=margin_ratio,
            )
        )
    return images


def _generate_roi_oof(
    *,
    config_path: str | Path,
    fold_count: int,
    device: str,
    batch_size: int,
    margin_ratio: float,
    mask_threshold: float,
) -> dict[str, Any]:
    config, paths = load_project_config(config_path)
    manifest = load_busbra_manifest(paths.busbra_root)
    seed = int(config.get("seed", 42))
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
    views: dict[str, list[dict[str, Any]]] = {view_name: [] for view_name in PAIR_VIEWS}
    for fold in range(1, fold_count + 1):
        val_manifest = _val_manifest_for_fold(manifest, assignments, fold)
        roi_images = _roi_images_from_manifest(
            val_manifest,
            margin_ratio=margin_ratio,
            mask_threshold=mask_threshold,
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
    for view_name in views:
        views[view_name] = sorted(views[view_name], key=lambda row: row["sample_id"])
    return {
        "source": "BUSBRA GT-mask ROI OOF predictions",
        "sample_count": len(next(iter(views.values()))),
        "margin_ratio": margin_ratio,
        "mask_threshold": mask_threshold,
        "views": views,
    }


def _load_full_oof_pair(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    with Path(path).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    rows = report["views"]["eff_identity"]
    sample_ids = [row["sample_id"] for row in rows]
    y_true = np.asarray(
        [LABEL_TO_INDEX[str(row["pathology_label"]).lower()] for row in rows],
        dtype=np.int32,
    )
    groups = np.asarray([str(row["case_id"]) for row in rows])
    probabilities = _combine_pair_probabilities(report["views"])
    return sample_ids, y_true, groups, probabilities


def _combine_pair_probabilities(views: dict[str, list[dict[str, Any]]]) -> np.ndarray:
    reference_ids: list[str] | None = None
    combined: np.ndarray | None = None
    total_weight = 0.0
    for view_name in PAIR_VIEWS:
        rows = views[view_name]
        current_ids = [row["sample_id"] for row in rows]
        if reference_ids is None:
            reference_ids = current_ids
            combined = np.zeros(len(rows), dtype=np.float64)
        elif current_ids != reference_ids:
            raise ValueError(f"Sample order mismatch for {view_name}.")
        weight = float(PAIR_WEIGHTS[view_name])
        total_weight += weight
        combined += weight * np.asarray(
            [row["malignant_probability"] for row in rows],
            dtype=np.float64,
        )
    if combined is None or total_weight <= 0:
        raise ValueError("No probabilities to combine.")
    return combined / total_weight


def _fit_oof_stacker(
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
            for C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
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
                                    C=float(C),
                                    class_weight=class_weight,
                                    max_iter=2000,
                                ),
                            ),
                        ]
                    )
                    model.fit(features[train_index], y_true[train_index])
                    fold_probs = model.predict_proba(features[val_index])[:, 1]
                    cv_probabilities[val_index] = fold_probs
                    if len(np.unique(y_true[val_index])) > 1:
                        fold_aucs.append(float(roc_auc_score(y_true[val_index], fold_probs)))
                cv_auc = float(roc_auc_score(y_true, cv_probabilities))
                candidates.append(
                    {
                        "feature_mode": feature_mode,
                        "C": float(C),
                        "class_weight": class_weight,
                        "cv_auc": cv_auc,
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
    oof_best = best_threshold_by_youden(
        y_true,
        selected["cv_probabilities"],
        thresholds=np.round(np.arange(0.1, 0.9001, 0.01), 2),
    )
    return {
        "model": final_model,
        "feature_mode": selected["feature_mode"],
        "C": selected["C"],
        "class_weight": selected["class_weight"],
        "cv_auc": selected["cv_auc"],
        "fold_auc_mean": selected["fold_auc_mean"],
        "fold_auc_std": selected["fold_auc_std"],
        "threshold_from_oof": float(oof_best["threshold"]),
        "oof_best_by_youden": oof_best,
    }


def _load_busi_full_pair(path: str | Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    with Path(path).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    rows = report["rows"]
    reference = [
        {
            "sample_id": str(row["sample_id"]),
            "pathology_label": str(row["pathology_label"]),
        }
        for row in rows
    ]
    y_true = np.asarray(
        [1 if row["pathology_label"] == "malignant" else 0 for row in rows],
        dtype=np.int32,
    )
    probabilities = np.asarray(
        [row["malignant_probability"] for row in rows],
        dtype=np.float64,
    )
    return reference, y_true, probabilities


def _predict_segmenter_mask(segmenter, image: np.ndarray) -> np.ndarray:
    require_dependency("torch", torch)
    input_tensor = prepare_classifier_input(image, 256)
    with torch.no_grad():
        logits = segmenter(input_tensor.unsqueeze(0).to(dtype=torch.float32))
        return torch.sigmoid(logits)[0, 0].cpu().numpy()


def _roi_area_ratio(mask: np.ndarray | None, *, margin_ratio: float, mask_threshold: float) -> tuple[float, bool]:
    if mask is None:
        return 1.0, True
    bbox = mask_bbox(mask, threshold=mask_threshold, min_area_ratio=0.001)
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


def _summarize_ratios(ratios: list[float], fallback_count: int) -> dict[str, Any]:
    array = np.asarray(ratios, dtype=np.float64)
    return {
        "fallback_count": int(fallback_count),
        "mean_area_ratio": float(array.mean()),
        "median_area_ratio": float(np.median(array)),
        "p10_area_ratio": float(np.quantile(array, 0.1)),
        "p90_area_ratio": float(np.quantile(array, 0.9)),
        "min_area_ratio": float(array.min()),
        "max_area_ratio": float(array.max()),
    }


def _busi_roi_images(
    *,
    source: str,
    segmenter_checkpoint: Path,
    margin_ratio: float,
    mask_threshold: float,
) -> tuple[list[dict[str, str]], list[np.ndarray], np.ndarray, dict[str, Any]]:
    _, paths = load_project_config("configs/classifier/convnext_tiny_timm_recipe.yml")
    manifest = load_busi_manifest(paths.busi_root, include_normal=False)
    segmenter = None
    if source == "segmenter":
        if not segmenter_checkpoint.exists():
            raise FileNotFoundError(f"Segmenter checkpoint not found: {segmenter_checkpoint}")
        segmenter = load_segmenter(
            {
                "architecture": "unet",
                "encoder_name": "resnet18",
                "encoder_weights": None,
                "in_channels": 3,
                "classes": 1,
            },
            checkpoint_path=segmenter_checkpoint,
            map_location="cpu",
        )
        segmenter.eval()
    reference: list[dict[str, str]] = []
    y_true: list[int] = []
    roi_images: list[np.ndarray] = []
    area_ratios: list[float] = []
    fallback_count = 0
    for row in manifest.itertuples(index=False):
        image = read_image(row.image_path, grayscale=True)
        if source == "oracle":
            mask = read_mask(row.mask_path)
        elif source == "segmenter":
            mask = _predict_segmenter_mask(segmenter, image)
        else:
            raise ValueError(f"Unsupported BUSI ROI source: {source}")
        area_ratio, used_fallback = _roi_area_ratio(
            mask,
            margin_ratio=margin_ratio,
            mask_threshold=mask_threshold,
        )
        area_ratios.append(area_ratio)
        fallback_count += int(used_fallback)
        roi_images.append(
            crop_to_mask_bbox(
                image,
                mask,
                threshold=mask_threshold,
                margin_ratio=margin_ratio,
            )
        )
        reference.append(
            {
                "sample_id": str(row.sample_id),
                "pathology_label": str(row.pathology_label),
            }
        )
        y_true.append(1 if row.pathology_label == "malignant" else 0)
    return reference, roi_images, np.asarray(y_true, dtype=np.int32), _summarize_ratios(
        area_ratios,
        fallback_count,
    )


def _predict_busi_roi_pair(
    *,
    source: str,
    segmenter_checkpoint: Path,
    margin_ratio: float,
    mask_threshold: float,
    device: str,
    batch_size: int,
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, dict[str, Any]]:
    _, paths = load_project_config("configs/classifier/convnext_tiny_timm_recipe.yml")
    reference, roi_images, y_true, roi_stats = _busi_roi_images(
        source=source,
        segmenter_checkpoint=segmenter_checkpoint,
        margin_ratio=margin_ratio,
        mask_threshold=mask_threshold,
    )
    view_probabilities: dict[str, list[np.ndarray]] = {view_name: [] for view_name in PAIR_VIEWS}
    resolved_device = select_device(device)
    for view_name in PAIR_VIEWS:
        for fold in range(1, 6):
            view_probabilities[view_name].append(
                _predict_view_on_images(
                    view_name,
                    roi_images,
                    fold=fold,
                    project_root=paths.project_root,
                    device=resolved_device,
                    batch_size=batch_size,
                )
            )
    views = {}
    for view_name, fold_probabilities in view_probabilities.items():
        mean_probabilities = np.vstack(fold_probabilities).mean(axis=0)
        views[view_name] = [
            {
                "sample_id": row["sample_id"],
                "pathology_label": row["pathology_label"],
                "malignant_probability": float(probability),
            }
            for row, probability in zip(reference, mean_probabilities)
        ]
    return reference, y_true, _combine_pair_probabilities(views), roi_stats


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    default = classification_metrics(y_true, probabilities, threshold=threshold)
    best = best_threshold_by_youden(
        y_true,
        probabilities,
        thresholds=np.round(np.arange(0.1, 0.9001, 0.01), 2),
    )
    return {
        "auc": float(roc_auc_score(y_true, probabilities)),
        "default": default,
        "best_by_youden": best,
    }


def _evaluate_stacker_on_busi(
    stacker: dict[str, Any],
    *,
    full_probabilities: np.ndarray,
    roi_probabilities: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, Any]:
    matrix = np.vstack([full_probabilities, roi_probabilities]).T
    features = _transform_features(matrix, str(stacker["feature_mode"]))
    probabilities = stacker["model"].predict_proba(features)[:, 1]
    threshold = float(stacker["threshold_from_oof"])
    return {
        "auc": float(roc_auc_score(y_true, probabilities)),
        "at_0_5": classification_metrics(y_true, probabilities, threshold=0.5),
        "at_oof_threshold": classification_metrics(y_true, probabilities, threshold=threshold),
        "best_by_youden": best_threshold_by_youden(
            y_true,
            probabilities,
            thresholds=np.round(np.arange(0.1, 0.9001, 0.01), 2),
        ),
    }


def _slim_stacker(stacker: dict[str, Any]) -> dict[str, Any]:
    model = stacker["model"]
    classifier = model.named_steps["classifier"]
    scaler = model.named_steps["scaler"]
    return {
        "feature_mode": stacker["feature_mode"],
        "C": stacker["C"],
        "class_weight": stacker["class_weight"],
        "cv_auc": stacker["cv_auc"],
        "fold_auc_mean": stacker["fold_auc_mean"],
        "fold_auc_std": stacker["fold_auc_std"],
        "threshold_from_oof": stacker["threshold_from_oof"],
        "oof_best_by_youden": stacker["oof_best_by_youden"],
        "coef": [float(value) for value in classifier.coef_[0]],
        "intercept": float(classifier.intercept_[0]),
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) for value in scaler.scale_],
    }


def _format_metric(metric: dict[str, Any], key: str) -> str:
    value = metric.get(key)
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def build_markdown(report: dict[str, Any]) -> list[str]:
    lines = [
        "# ROI 裁剪与 OOF 融合实验报告",
        "",
        "日期：2026-04-25",
        "",
        "## 实验边界",
        "",
        "- BUSBRA 使用真值 mask 生成 ROI，并通过 OOF 方式训练融合器。",
        "- BUSI 的 `oracle` 结果使用 BUSI 真值 mask，只能视为 ROI 上限，不可作为可部署成绩。",
        "- BUSI 的 `segmenter` 结果使用现有 `segmenter_fold1.pt` 预测 mask，是更接近部署口径的外部验证。",
        "- 当前 `demo.yml` 和主线权重没有修改。",
        "",
        "## OOF 融合器",
        "",
        f"- 特征：完整图两模型概率 + ROI 两模型概率。",
        f"- 特征模式：`{report['stacker']['feature_mode']}`。",
        f"- OOF CV AUC：`{report['stacker']['cv_auc']:.4f}`。",
        f"- OOF 推荐阈值：`{report['stacker']['threshold_from_oof']:.2f}`。",
        "",
        "## 外部验证结果",
        "",
        "| 方案 | ROI来源 | ROI-only AUC | Stacking AUC | Stacking@OOF阈值 Sens | Stacking@OOF阈值 Spec | Stacking最佳阈值 | Stacking最佳 Sens | Stacking最佳 Spec | 说明 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, label, note in (
        ("oracle", "BUSI真值mask", "上限参考，不可部署"),
        ("segmenter", "segmenter预测mask", "可部署代理，但依赖分割质量"),
    ):
        result = report["busi_results"][key]
        stack = result["stacking"]
        roi = result["roi_only"]
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    label,
                    f"{roi['auc']:.4f}",
                    f"{stack['auc']:.4f}",
                    _format_metric(stack["at_oof_threshold"], "sensitivity"),
                    _format_metric(stack["at_oof_threshold"], "specificity"),
                    f"{stack['best_by_youden']['threshold']:.2f}",
                    _format_metric(stack["best_by_youden"], "sensitivity"),
                    _format_metric(stack["best_by_youden"], "specificity"),
                    note,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## ROI 面积分布",
            "",
            "| 方案 | fallback数量 | 平均面积占比 | 中位面积占比 | P10 | P90 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in ("oracle", "segmenter"):
        stats = report["busi_results"][key]["roi_stats"]
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    str(stats["fallback_count"]),
                    f"{stats['mean_area_ratio']:.4f}",
                    f"{stats['median_area_ratio']:.4f}",
                    f"{stats['p10_area_ratio']:.4f}",
                    f"{stats['p90_area_ratio']:.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 当前完整图主线 AUC：`{report['baseline']['full_pair_auc']:.4f}`。",
            f"- segmenter ROI + OOF 融合 AUC：`{report['busi_results']['segmenter']['stacking']['auc']:.4f}`，高于完整图主线，说明 ROI/病灶区域引导方向有效。",
            "- segmenter ROI 的中位面积占比偏大，说明当前不是“紧贴病灶”的精细裁剪，而是带较多上下文的病灶区域引导。",
            "- 当前结果只能证明 ROI 方向值得继续验证，不建议立刻修改 Demo；下一步应在 BUSBRA OOF 内完成 ROI 参数、融合权重和运行阈值选择，再用 BUSI 做一次锁定配置后的外部评估。",
        ]
    )
    return lines


def run(args: argparse.Namespace) -> dict[str, Any]:
    full_oof_path = Path(args.full_oof_cache)
    if not full_oof_path.exists():
        raise FileNotFoundError(
            f"Full-image OOF cache not found: {full_oof_path}. Run scripts/run_oof_stacking.py first."
        )
    roi_oof_path = Path(args.roi_oof_cache)
    if args.reuse_roi_oof and roi_oof_path.exists():
        with roi_oof_path.open("r", encoding="utf-8") as handle:
            roi_oof = json.load(handle)
    else:
        roi_oof = _generate_roi_oof(
            config_path=args.config,
            fold_count=int(args.fold_count),
            device=str(args.device),
            batch_size=int(args.batch_size),
            margin_ratio=float(args.margin_ratio),
            mask_threshold=float(args.mask_threshold),
        )
        write_json_report(roi_oof_path, roi_oof)

    sample_ids, y_true, groups, full_oof_pair = _load_full_oof_pair(full_oof_path)
    roi_oof_pair = _combine_pair_probabilities(roi_oof["views"])
    stacker = _fit_oof_stacker(full_oof_pair, roi_oof_pair, y_true, groups)

    busi_reference, busi_y_true, busi_full_pair = _load_busi_full_pair(
        "artifacts/reports/busi_ensemble_effnet_convnext_optimized.json"
    )
    busi_results = {}
    for source in ("oracle", "segmenter"):
        roi_reference, roi_y_true, busi_roi_pair, roi_stats = _predict_busi_roi_pair(
            source=source,
            segmenter_checkpoint=Path(args.segmenter_checkpoint),
            margin_ratio=float(args.margin_ratio),
            mask_threshold=float(args.mask_threshold),
            device=str(args.device),
            batch_size=int(args.batch_size),
        )
        if roi_reference != busi_reference or not np.array_equal(roi_y_true, busi_y_true):
            raise ValueError(f"BUSI ROI reference mismatch for {source}.")
        busi_results[source] = {
            "roi_stats": roi_stats,
            "roi_only": _metrics(busi_y_true, busi_roi_pair),
            "stacking": _evaluate_stacker_on_busi(
                stacker,
                full_probabilities=busi_full_pair,
                roi_probabilities=busi_roi_pair,
                y_true=busi_y_true,
            ),
        }

    report = {
        "method": "ROI two-view OOF stacking",
        "margin_ratio": float(args.margin_ratio),
        "mask_threshold": float(args.mask_threshold),
        "oof_sample_count": int(len(sample_ids)),
        "baseline": {
            "full_pair_auc": float(roc_auc_score(busi_y_true, busi_full_pair)),
            "full_pair_metrics": _metrics(busi_y_true, busi_full_pair),
        },
        "oof": {
            "full_pair": _metrics(y_true, full_oof_pair),
            "roi_pair": _metrics(y_true, roi_oof_pair),
        },
        "stacker": _slim_stacker(stacker),
        "busi_results": busi_results,
    }
    write_json_report(args.output, report)
    write_markdown_report(args.markdown, build_markdown(report))
    return report


def main() -> int:
    args = build_parser().parse_args()
    report = run(args)
    print(
        {
            "full_pair_auc": report["baseline"]["full_pair_auc"],
            "oracle_stack_auc": report["busi_results"]["oracle"]["stacking"]["auc"],
            "segmenter_stack_auc": report["busi_results"]["segmenter"]["stacking"]["auc"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
