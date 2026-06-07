"""Utility script for descriptor router oof protocol workflows."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.busbra import load_busbra_manifest  # noqa: E402
from src.engine.descriptors import (  # noqa: E402
    build_router_feature_map,
    extract_roi_descriptors,
    router_feature_vector,
)
from src.models.segmenter import load_segmenter  # noqa: E402
from src.preprocess.io import read_image, read_mask  # noqa: E402
from src.preprocess.transforms import prepare_classifier_input  # noqa: E402
from src.utils.config import load_project_config  # noqa: E402
from src.utils.metrics import classification_metrics, threshold_sweep  # noqa: E402
from src.utils.reporting import write_json_report, write_markdown_report  # noqa: E402
from src.utils.runtime import optional_import, select_device  # noqa: E402


PAIR_WEIGHTS = {"eff_identity": 0.427, "conv_crop_sweep": 0.573}
PAIR_VIEWS = ("eff_identity", "conv_crop_sweep")
torch = optional_import("torch")

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "descriptor_minimal": (
        "stacked_logit",
        "full_logit",
        "roi_logit",
        "roi_area_ratio",
        "mask_area_ratio",
        "mask_compactness",
        "image_std",
        "image_sharpness",
        "abs_full_roi_delta",
    ),
    "descriptor_boundary_quality": (
        "stacked_logit",
        "full_logit",
        "roi_logit",
        "full_roi_delta",
        "abs_full_roi_delta",
        "full_uncertainty",
        "roi_uncertainty",
        "roi_area_ratio",
        "mask_area_ratio",
        "lesion_bbox_area_ratio",
        "lesion_aspect_ratio",
        "mask_extent",
        "mask_compactness",
        "boundary_complexity",
        "component_count",
        "edge_contrast",
        "image_std",
        "image_sharpness",
    ),
    "descriptor_full": (
        "full_probability",
        "roi_probability",
        "stacked_probability",
        "full_logit",
        "roi_logit",
        "stacked_logit",
        "full_roi_delta",
        "abs_full_roi_delta",
        "full_uncertainty",
        "roi_uncertainty",
        "stacked_uncertainty",
        "roi_valid",
        "roi_area_ratio",
        "mask_area_ratio",
        "lesion_bbox_area_ratio",
        "lesion_aspect_ratio",
        "mask_extent",
        "mask_compactness",
        "boundary_complexity",
        "component_count",
        "mask_mean_probability",
        "edge_contrast",
        "image_mean",
        "image_std",
        "image_sharpness",
    ),
}


@dataclass(frozen=True)
class CandidateSpec:
    """Represent CandidateSpec for this module."""
    name: str
    feature_set: str
    c_value: float
    class_weight: str | None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit a deployable descriptor-guided ROI router using BUSBRA OOF caches only. "
            "The external BUSI set is intentionally not loaded here."
        )
    )
    parser.add_argument("--config", default="configs/inference/demo.yml")
    parser.add_argument("--full-oof-cache", default="artifacts/reports/oof_two_model_predictions.json")
    parser.add_argument(
        "--roi-oof-cache",
        default="artifacts/reports/roi_oof_lcc_mask04_protocol_predictions.json",
    )
    parser.add_argument(
        "--descriptor-cache",
        default="artifacts/reports/descriptor_router_busbra_segmenter_oof_descriptors.json",
    )
    parser.add_argument(
        "--descriptor-mask-source",
        choices=("segmenter_oof", "segmenter_fold1", "gt_mask"),
        default="segmenter_oof",
        help=(
            "Mask source for descriptor fitting. segmenter_oof is the default because it "
            "matches deployment better and avoids using ground-truth shape descriptors."
        ),
    )
    parser.add_argument(
        "--segmenter-checkpoint",
        default="artifacts/checkpoints/segmenter_fold1.pt",
    )
    parser.add_argument(
        "--segmenter-checkpoint-pattern",
        default="artifacts/checkpoints/segmenter_5fold_fold{fold}.pt",
    )
    parser.add_argument("--segmenter-image-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="artifacts/reports/descriptor_router_oof_protocol.json")
    parser.add_argument(
        "--markdown",
        default="artifacts/reports/Chinese reports/03_ensemble_oof_stacking/descriptor_router_oof_protocol.md",
    )
    parser.add_argument("--experimental-config", default="configs/inference/demo_descriptor_router.yml")
    parser.add_argument("--min-auc-gain", type=float, default=0.005)
    parser.add_argument("--min-f1-gain", type=float, default=0.005)
    parser.add_argument("--max-metric-drop", type=float, default=0.01)
    return parser


def _load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON report and validate its top-level object."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _rows_for_view(report: dict[str, Any], view_name: str) -> list[dict[str, Any]]:
    """Return rows for a named prediction view after validation."""
    views = report.get("views")
    if not isinstance(views, dict) or view_name not in views:
        raise ValueError(f"Missing view {view_name!r}.")
    rows = views[view_name]
    if not isinstance(rows, list):
        raise ValueError(f"View {view_name!r} is not a list.")
    return rows


def _validate_order(*row_groups: list[dict[str, Any]]) -> list[str]:
    """Validate order."""
    if not row_groups:
        raise ValueError("No row groups provided.")
    reference = [str(row["sample_id"]) for row in row_groups[0]]
    for rows in row_groups[1:]:
        current = [str(row["sample_id"]) for row in rows]
        if current != reference:
            raise ValueError("OOF row order mismatch.")
    return reference


def _probabilities(rows: list[dict[str, Any]]) -> np.ndarray:
    """Return malignant probabilities as a NumPy vector."""
    return np.asarray([float(row["malignant_probability"]) for row in rows], dtype=np.float64)


def _weighted_pair(eff: np.ndarray, conv: np.ndarray) -> np.ndarray:
    """Blend paired model probability vectors with fixed weights."""
    total = PAIR_WEIGHTS["eff_identity"] + PAIR_WEIGHTS["conv_crop_sweep"]
    return (PAIR_WEIGHTS["eff_identity"] * eff + PAIR_WEIGHTS["conv_crop_sweep"] * conv) / total


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities with the sigmoid transform."""
    return 1.0 / (1.0 + np.exp(-values))


def _logit(probabilities: np.ndarray) -> np.ndarray:
    """Convert probabilities to clipped logits."""
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _runtime_stack(
    full_probabilities: np.ndarray,
    roi_probabilities: np.ndarray,
    *,
    stacker: dict[str, Any],
) -> np.ndarray:
    """Apply runtime stacker settings to paired probabilities."""
    values = np.vstack([full_probabilities, roi_probabilities]).T
    if str(stacker.get("feature_mode", "probability")) == "logit":
        values = _logit(values)
    mean = np.asarray(stacker.get("scaler_mean", [0.0, 0.0]), dtype=np.float64)
    scale = np.asarray(stacker.get("scaler_scale", [1.0, 1.0]), dtype=np.float64)
    coef = np.asarray(stacker.get("coef", [1.0, 0.0]), dtype=np.float64)
    scaled = (values - mean) / np.maximum(scale, 1e-6)
    return _sigmoid(scaled @ coef + float(stacker.get("intercept", 0.0)))


def _best_threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Calculate metrics at the best Youden threshold."""
    rows = threshold_sweep(y_true, probabilities, thresholds=np.round(np.arange(0.1, 0.9001, 0.01), 2))
    if not rows:
        raise ValueError("No threshold rows generated.")
    return max(
        rows,
        key=lambda row: (
            row["sensitivity"] + row["specificity"] - 1.0,
            row["f1_score"],
            row["specificity"],
        ),
    )


def _metrics_at_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Calculate metrics at the best Youden threshold."""
    best = _best_threshold_metrics(y_true, probabilities)
    metrics = classification_metrics(y_true, probabilities, threshold=float(best["threshold"]))
    metrics["youden_j"] = float(metrics["sensitivity"] + metrics["specificity"] - 1.0)
    return metrics


def _gate_reasons(area_ratios: np.ndarray, min_area: float, max_area: float) -> np.ndarray:
    """Return ROI area gate reasons for each sample."""
    return np.asarray(
        [
            "too_small" if value < min_area else "too_large" if value > max_area else "none"
            for value in area_ratios
        ],
        dtype=object,
    )


def _descriptor_rows(
    *,
    sample_ids: list[str],
    fold_ids: np.ndarray,
    manifest_by_id: dict[str, Any],
    roi_config: dict[str, Any],
    area_ratios: np.ndarray,
    cache_path: str | Path,
    mask_source: str,
    segmenter_checkpoint: str | Path,
    segmenter_checkpoint_pattern: str,
    segmenter_image_size: int,
    device: str,
    project_root: Path,
) -> list[dict[str, float]]:
    """Format descriptor rows."""
    path = Path(cache_path)
    resolved_segmenter_checkpoint = _resolve_project_path(segmenter_checkpoint, project_root)
    resolved_pattern = str(segmenter_checkpoint_pattern)
    if path.exists():
        cached = _load_json(path)
        rows = cached.get("rows")
        cached_ids = cached.get("sample_ids")
        if (
            isinstance(rows, list)
            and cached_ids == sample_ids
            and cached.get("mask_source") == mask_source
            and cached.get("segmenter_checkpoint") == str(resolved_segmenter_checkpoint)
            and cached.get("segmenter_checkpoint_pattern") == resolved_pattern
            and cached.get("segmenter_image_size") == int(segmenter_image_size)
        ):
            return [{str(key): float(value) for key, value in row.items()} for row in rows]

    resolved_device = select_device(str(device))
    segmenters: dict[int, Any] = {}
    single_segmenter = None
    if mask_source == "segmenter_fold1":
        single_segmenter = _load_segmenter_model(
            resolved_segmenter_checkpoint,
            device=resolved_device,
        )
    elif mask_source == "segmenter_oof":
        for fold_id in sorted(int(value) for value in np.unique(fold_ids)):
            checkpoint = _resolve_project_path(
                segmenter_checkpoint_pattern.format(fold=fold_id),
                project_root,
            )
            segmenters[fold_id] = _load_segmenter_model(checkpoint, device=resolved_device)

    rows: list[dict[str, float]] = []
    for index, sample_id in enumerate(sample_ids):
        row = manifest_by_id.get(sample_id)
        if row is None:
            raise ValueError(f"BUSBRA manifest missing sample {sample_id}.")
        image = read_image(row.image_path, grayscale=True)
        if mask_source == "gt_mask":
            mask = read_mask(row.mask_path) if row.mask_path else None
        elif mask_source == "segmenter_fold1":
            mask = _predict_segmenter_mask(
                single_segmenter,
                image,
                image_size=int(segmenter_image_size),
                device=resolved_device,
            )
        elif mask_source == "segmenter_oof":
            mask = _predict_segmenter_mask(
                segmenters[int(fold_ids[index])],
                image,
                image_size=int(segmenter_image_size),
                device=resolved_device,
            )
        else:
            raise ValueError(f"Unsupported descriptor mask source: {mask_source}")
        descriptors = extract_roi_descriptors(
            image,
            mask,
            threshold=float(roi_config.get("mask_threshold", 0.5)),
            margin_ratio=float(roi_config.get("margin_ratio", 0.35)),
            min_area_ratio=float(roi_config.get("min_mask_area_ratio", 0.001)),
            largest_component=bool(roi_config.get("largest_component", False)),
        )
        descriptors["roi_oof_cache_area_ratio"] = float(area_ratios[index])
        rows.append(descriptors)

    write_json_report(
        path,
        {
            "source": "BUSBRA descriptor cache",
            "mask_source": mask_source,
            "segmenter_checkpoint": str(resolved_segmenter_checkpoint),
            "segmenter_checkpoint_pattern": resolved_pattern,
            "segmenter_image_size": int(segmenter_image_size),
            "sample_ids": sample_ids,
            "fold_ids": [int(value) for value in fold_ids],
            "rows": rows,
        },
    )
    return rows


def _resolve_project_path(path: str | Path, project_root: Path) -> Path:
    """Resolve project path."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (project_root / resolved).resolve()
    return resolved


def _load_segmenter_model(checkpoint_path: str | Path, *, device: str):
    """Load segmenter model."""
    if torch is None:
        raise RuntimeError("Torch is required for segmenter descriptor masks.")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Segmenter checkpoint not found: {checkpoint}")
    model = load_segmenter(
        {
            "architecture": "unet",
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "in_channels": 3,
            "classes": 1,
        },
        checkpoint_path=checkpoint,
        map_location="cpu",
    )
    model.eval()
    model.to(device)
    return model


def _predict_segmenter_mask(
    model,
    image: np.ndarray,
    *,
    image_size: int,
    device: str,
) -> np.ndarray:
    """Predict segmenter mask."""
    if torch is None:
        raise RuntimeError("Torch is required for segmenter descriptor masks.")
    input_tensor = prepare_classifier_input(image, int(image_size))
    with torch.no_grad():
        logits = model(input_tensor.unsqueeze(0).to(device=device, dtype=torch.float32))
        return torch.sigmoid(logits)[0, 0].cpu().numpy()


def _feature_matrix(
    *,
    full_probabilities: np.ndarray,
    roi_probabilities: np.ndarray,
    stacked_probabilities: np.ndarray,
    descriptors: list[dict[str, float]],
    feature_names: tuple[str, ...],
) -> np.ndarray:
    """Build a model feature matrix for candidate scoring."""
    vectors = []
    for full_probability, roi_probability, stacked_probability, descriptor in zip(
        full_probabilities,
        roi_probabilities,
        stacked_probabilities,
        descriptors,
    ):
        feature_map = build_router_feature_map(
            full_probability=float(full_probability),
            roi_probability=float(roi_probability),
            stacked_probability=float(stacked_probability),
            descriptors=descriptor,
        )
        vectors.append(router_feature_vector(feature_map, feature_names))
    return np.vstack(vectors).astype(np.float64)


def _make_model(spec: CandidateSpec) -> Pipeline:
    """Create model."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(spec.c_value),
                    class_weight=spec.class_weight,
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def _candidate_specs() -> list[CandidateSpec]:
    """Generate candidate model or stacker specifications."""
    specs: list[CandidateSpec] = []
    for feature_set in FEATURE_SETS:
        for c_value in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
            for class_weight in (None, "balanced"):
                specs.append(
                    CandidateSpec(
                        name=f"logistic_{feature_set}_C{c_value:g}_{class_weight or 'none'}",
                        feature_set=feature_set,
                        c_value=float(c_value),
                        class_weight=class_weight,
                    )
                )
    return specs


def _nested_router_probabilities(
    *,
    spec: CandidateSpec,
    matrix: np.ndarray,
    y_true: np.ndarray,
    fold_ids: np.ndarray,
    valid_mask: np.ndarray,
    fallback_probabilities: np.ndarray,
) -> np.ndarray:
    """Generate nested router probabilities for a candidate."""
    probabilities = fallback_probabilities.astype(np.float64).copy()
    for fold_id in sorted(np.unique(fold_ids)):
        val_mask = (fold_ids == fold_id) & valid_mask
        train_mask = (fold_ids != fold_id) & valid_mask
        if not np.any(val_mask):
            continue
        if len(np.unique(y_true[train_mask])) < 2:
            probabilities[val_mask] = fallback_probabilities[val_mask]
            continue
        model = _make_model(spec)
        model.fit(matrix[train_mask], y_true[train_mask])
        probabilities[val_mask] = model.predict_proba(matrix[val_mask])[:, 1]
    return probabilities


def _fit_final_router(
    *,
    spec: CandidateSpec,
    matrix: np.ndarray,
    y_true: np.ndarray,
    valid_mask: np.ndarray,
) -> Pipeline:
    """Fit final router."""
    model = _make_model(spec)
    model.fit(matrix[valid_mask], y_true[valid_mask])
    return model


def _router_config_from_model(
    *,
    model: Pipeline,
    spec: CandidateSpec,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    """Serialize a trained router into runtime config."""
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    return {
        "enabled": True,
        "source": "BUSBRA_OOF_descriptor_router",
        "candidate_name": spec.name,
        "feature_set": spec.feature_set,
        "feature_names": list(feature_names),
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) for value in scaler.scale_],
        "coef": [float(value) for value in classifier.coef_[0]],
        "intercept": float(classifier.intercept_[0]),
        "blend_weight": 1.0,
        "training_boundary": "BUSBRA OOF only; BUSI is not used by this script.",
    }


def _write_experimental_config(
    *,
    base_config_path: str | Path,
    output_path: str | Path,
    router_config: dict[str, Any],
    threshold: float,
    enabled: bool,
) -> None:
    """Write experimental config."""
    with Path(base_config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = dict(config.get("runtime", {}))
    roi_config = dict(runtime.get("roi_enhancement", {}))
    roi_config["descriptor_router"] = {**router_config, "enabled": bool(enabled)}
    runtime["roi_enhancement"] = roi_config
    if enabled:
        runtime["default_threshold"] = float(threshold)
    runtime["ensemble_display_name"] = (
        "ConvNeXt-Tiny + EfficientNetV2-S + ROI Descriptor Router"
        if enabled
        else "ConvNeXt-Tiny + EfficientNetV2-S + ROI Area Gate"
    )
    config["runtime"] = runtime
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def _candidate_row(
    spec: CandidateSpec,
    metrics: dict[str, Any],
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    """Build one candidate summary row for reporting."""
    return {
        "name": spec.name,
        "feature_set": spec.feature_set,
        "feature_names": list(feature_names),
        "params": {"C": spec.c_value, "class_weight": spec.class_weight},
        "metrics": metrics,
    }


def _format_metric(row: dict[str, Any], key: str) -> str:
    """Format one metric value for report tables."""
    return f"{float(row[key]):.4f}"


def _confusion_text(row: dict[str, Any]) -> str:
    """Format confusion-matrix counts for report tables."""
    c = row["confusion"]
    return f"TN {c['tn']} / FP {c['fp']} / FN {c['fn']} / TP {c['tp']}"


def _metrics_table_row(name: str, metrics: dict[str, Any]) -> str:
    """Format one metric row for a Markdown report."""
    return (
        "| "
        + " | ".join(
            [
                name,
                _format_metric(metrics, "auc"),
                f"{metrics['threshold']:.3f}",
                _format_metric(metrics, "accuracy"),
                _format_metric(metrics, "sensitivity"),
                _format_metric(metrics, "specificity"),
                _format_metric(metrics, "precision"),
                _format_metric(metrics, "f1_score"),
                _format_metric(metrics, "youden_j"),
                _confusion_text(metrics),
            ]
        )
        + " |"
    )


def _build_markdown(report: dict[str, Any]) -> list[str]:
    """Build the Markdown report body for this experiment."""
    selected = report["selected_candidate"]
    lines = [
        "# Descriptor-guided ROI Router OOF 实验",
        "",
        "## 实验边界",
        "",
        "- 本脚本只读取 BUSBRA OOF 预测缓存、BUSBRA 图像，以及由 `descriptor_mask_source` 指定的内部 mask 口径，不读取 BUSI。",
        "- Router 使用原始 fold_id 做嵌套 OOF：每个样本的 router 概率来自未见过该 fold 的轻量逻辑回归模型。",
        "- 生成的 `demo_descriptor_router.yml` 是新分支实验配置，不等同于主线 `demo.yml`。",
        "",
        "## 入选 Router",
        "",
        f"- 候选：`{selected['name']}`",
        f"- 特征集：`{selected['feature_set']}`",
        f"- 参数：`{json.dumps(selected['params'], ensure_ascii=False)}`",
        f"- 是否满足内部合入门槛：`{selected['accepted_by_oof_protocol']}`",
        f"- 描述符 mask 来源：`{report['descriptor_mask_source']}`",
        f"- 实验配置：`{report['experimental_config']}`",
        f"- 实验配置中 router 启用状态：`{report['router_config']['enabled']}`",
        f"- 内部门槛：AUC 至少提升 `{report['acceptance_criteria']['min_auc_gain']:.4f}`，F1 至少提升 `{report['acceptance_criteria']['min_f1_gain']:.4f}`，Sensitivity/Specificity 最大下降 `{report['acceptance_criteria']['max_metric_drop']:.4f}`",
        "",
        "使用特征：",
        "",
    ]
    for feature in selected["feature_names"]:
        lines.append(f"- `{feature}`")
    lines.extend(
        [
            "",
            "## OOF 指标对比",
            "",
            "| 方案 | AUC | 阈值 | Accuracy | Sensitivity | Specificity | Precision | F1-Score | Youden J | Confusion |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            _metrics_table_row("当前 ROI stacker（demo 阈值）", report["baseline_metrics_at_demo_threshold"]),
            _metrics_table_row("当前 ROI stacker（OOF Youden 阈值）", report["baseline_metrics_at_best_threshold"]),
            _metrics_table_row("Descriptor router（嵌套 OOF）", selected["metrics"]),
            "",
            "## Top 候选",
            "",
            "| 排名 | 候选 | 特征集 | AUC | 阈值 | Sensitivity | Specificity | Precision | F1-Score | Confusion |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for rank, row in enumerate(report["top_candidates"][:10], start=1):
        metrics = row["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    f"`{row['name']}`",
                    f"`{row['feature_set']}`",
                    _format_metric(metrics, "auc"),
                    f"{metrics['threshold']:.3f}",
                    _format_metric(metrics, "sensitivity"),
                    _format_metric(metrics, "specificity"),
                    _format_metric(metrics, "precision"),
                    _format_metric(metrics, "f1_score"),
                    _confusion_text(metrics),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 建议：`{report['recommendation']}`",
            "- 若内部 OOF 未明显超过当前 ROI stacker，实验配置会写入 `descriptor_router.enabled: false`，避免误用。",
            "- 若后续要继续推进，下一步应在固定 router 配置后只做一次 BUSI 外部复核，不能用 BUSI 继续反向调参。",
        ]
    )
    return lines


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the experiment workflow and return the generated summary."""
    config, paths = load_project_config(args.config)
    runtime = dict(config.get("runtime", {}))
    roi_config = dict(runtime.get("roi_enhancement", {}))
    stacker = dict(roi_config.get("stacker", {}))
    gate = dict(roi_config.get("quality_gate", {}))
    min_area = float(gate.get("min_area_ratio", 0.08))
    max_area = float(gate.get("max_area_ratio", 0.75))
    demo_threshold = float(runtime.get("default_threshold", 0.5))

    full_oof = _load_json(args.full_oof_cache)
    roi_oof = _load_json(args.roi_oof_cache)
    full_eff_rows = _rows_for_view(full_oof, "eff_identity")
    full_conv_rows = _rows_for_view(full_oof, "conv_crop_sweep")
    roi_eff_rows = _rows_for_view(roi_oof, "eff_identity")
    roi_conv_rows = _rows_for_view(roi_oof, "conv_crop_sweep")
    sample_ids = _validate_order(full_eff_rows, full_conv_rows, roi_eff_rows, roi_conv_rows)

    y_true = np.asarray(
        [1 if str(row["pathology_label"]).lower() == "malignant" else 0 for row in full_eff_rows],
        dtype=np.int32,
    )
    fold_ids = np.asarray([int(row["fold_id"]) for row in full_eff_rows], dtype=np.int32)
    full_pair = _weighted_pair(_probabilities(full_eff_rows), _probabilities(full_conv_rows))
    roi_pair = _weighted_pair(_probabilities(roi_eff_rows), _probabilities(roi_conv_rows))
    stacked = _runtime_stack(full_pair, roi_pair, stacker=stacker)
    area_ratios = np.asarray(roi_oof["roi_area_ratios"], dtype=np.float64)
    if len(area_ratios) != len(sample_ids):
        raise ValueError("ROI area ratio length mismatch.")

    manifest = load_busbra_manifest(paths.busbra_root)
    manifest_by_id = {str(row.sample_id): row for row in manifest.itertuples(index=False)}
    descriptors = _descriptor_rows(
        sample_ids=sample_ids,
        fold_ids=fold_ids,
        manifest_by_id=manifest_by_id,
        roi_config=roi_config,
        area_ratios=area_ratios,
        cache_path=args.descriptor_cache,
        mask_source=str(args.descriptor_mask_source),
        segmenter_checkpoint=args.segmenter_checkpoint,
        segmenter_checkpoint_pattern=str(args.segmenter_checkpoint_pattern),
        segmenter_image_size=int(args.segmenter_image_size),
        device=str(args.device),
        project_root=paths.project_root,
    )
    descriptor_area_ratios = np.asarray(
        [float(row["roi_area_ratio"]) for row in descriptors],
        dtype=np.float64,
    )
    gate_reasons = _gate_reasons(descriptor_area_ratios, min_area, max_area)
    valid_mask = gate_reasons == "none"
    baseline_final = np.where(valid_mask, stacked, full_pair)
    baseline_demo_metrics = classification_metrics(y_true, baseline_final, threshold=demo_threshold)
    baseline_demo_metrics["youden_j"] = float(
        baseline_demo_metrics["sensitivity"] + baseline_demo_metrics["specificity"] - 1.0
    )
    baseline_best_metrics = _metrics_at_best_threshold(y_true, baseline_final)

    candidate_rows: list[dict[str, Any]] = []
    candidate_probabilities: dict[str, np.ndarray] = {}
    for spec in _candidate_specs():
        feature_names = FEATURE_SETS[spec.feature_set]
        matrix = _feature_matrix(
            full_probabilities=full_pair,
            roi_probabilities=roi_pair,
            stacked_probabilities=stacked,
            descriptors=descriptors,
            feature_names=feature_names,
        )
        probabilities = _nested_router_probabilities(
            spec=spec,
            matrix=matrix,
            y_true=y_true,
            fold_ids=fold_ids,
            valid_mask=valid_mask,
            fallback_probabilities=baseline_final,
        )
        metrics = _metrics_at_best_threshold(y_true, probabilities)
        candidate_rows.append(_candidate_row(spec, metrics, feature_names))
        candidate_probabilities[spec.name] = probabilities

    candidate_rows.sort(
        key=lambda row: (
            row["metrics"]["auc"],
            row["metrics"]["youden_j"],
            row["metrics"]["f1_score"],
            row["metrics"]["specificity"],
        ),
        reverse=True,
    )
    selected_row = dict(candidate_rows[0])
    selected_spec = next(spec for spec in _candidate_specs() if spec.name == selected_row["name"])
    selected_feature_names = FEATURE_SETS[selected_spec.feature_set]
    selected_matrix = _feature_matrix(
        full_probabilities=full_pair,
        roi_probabilities=roi_pair,
        stacked_probabilities=stacked,
        descriptors=descriptors,
        feature_names=selected_feature_names,
    )
    final_model = _fit_final_router(
        spec=selected_spec,
        matrix=selected_matrix,
        y_true=y_true,
        valid_mask=valid_mask,
    )
    router_config = _router_config_from_model(
        model=final_model,
        spec=selected_spec,
        feature_names=selected_feature_names,
    )
    selected_metrics = selected_row["metrics"]
    accepted = (
        float(selected_metrics["auc"]) >= float(baseline_best_metrics["auc"]) + float(args.min_auc_gain)
        and float(selected_metrics["f1_score"])
        >= float(baseline_best_metrics["f1_score"]) + float(args.min_f1_gain)
        and float(selected_metrics["sensitivity"])
        >= float(baseline_best_metrics["sensitivity"]) - float(args.max_metric_drop)
        and float(selected_metrics["specificity"])
        >= float(baseline_best_metrics["specificity"]) - float(args.max_metric_drop)
    )
    selected_row["accepted_by_oof_protocol"] = bool(accepted)
    experimental_router_config = {**router_config, "enabled": bool(accepted)}

    _write_experimental_config(
        base_config_path=args.config,
        output_path=args.experimental_config,
        router_config=experimental_router_config,
        threshold=float(selected_metrics["threshold"]),
        enabled=bool(accepted),
    )

    report = {
        "method": "descriptor-guided ROI router fitted on BUSBRA OOF only",
        "data_boundary": {
            "training_and_selection": "BUSBRA OOF predictions plus BUSBRA training images/masks",
            "external_busi_used": False,
        },
        "config_path": str(args.config),
        "full_oof_cache": str(args.full_oof_cache),
        "roi_oof_cache": str(args.roi_oof_cache),
        "descriptor_cache": str(args.descriptor_cache),
        "descriptor_mask_source": str(args.descriptor_mask_source),
        "experimental_config": str(args.experimental_config),
        "sample_count": int(len(sample_ids)),
        "valid_router_sample_count": int(np.sum(valid_mask)),
        "fallback_sample_count": int(len(valid_mask) - np.sum(valid_mask)),
        "descriptor_area_summary": {
            "mean": float(np.mean(descriptor_area_ratios)),
            "median": float(np.median(descriptor_area_ratios)),
            "p10": float(np.quantile(descriptor_area_ratios, 0.1)),
            "p90": float(np.quantile(descriptor_area_ratios, 0.9)),
        },
        "feature_sets": {key: list(value) for key, value in FEATURE_SETS.items()},
        "acceptance_criteria": {
            "min_auc_gain": float(args.min_auc_gain),
            "min_f1_gain": float(args.min_f1_gain),
            "max_metric_drop": float(args.max_metric_drop),
        },
        "baseline_metrics_at_demo_threshold": baseline_demo_metrics,
        "baseline_metrics_at_best_threshold": baseline_best_metrics,
        "selected_candidate": selected_row,
        "top_candidates": candidate_rows[:10],
        "router_config": experimental_router_config,
        "recommendation": "candidate_for_frozen_external_review" if accepted else "do_not_merge_without_more_internal_evidence",
        "notes": [
            "The router is only fitted and nested-evaluated on internal BUSBRA OOF data.",
            "The descriptor mask source is recorded explicitly to avoid mixing ground-truth mask descriptors with predicted-mask deployment.",
            "The generated config is experimental and does not modify configs/inference/demo.yml.",
            "If the OOF acceptance gate fails, descriptor_router.enabled is written as false in the experimental config.",
        ],
    }
    write_json_report(args.output, report)
    write_markdown_report(args.markdown, _build_markdown(report))
    return report


def main() -> None:
    """Parse CLI arguments and run the script entry point."""
    report = run(build_parser().parse_args())
    selected = report["selected_candidate"]
    metrics = selected["metrics"]
    print(
        f"selected={selected['name']} auc={metrics['auc']:.6f} "
        f"threshold={metrics['threshold']:.3f} recommendation={report['recommendation']}"
    )


if __name__ == "__main__":
    main()
