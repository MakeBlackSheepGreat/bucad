"""Utility script for model zoo oof protocol workflows."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_oof_stacking import (
    ModelView,
    _load_or_create_splits,
    _predict_view_for_manifest,
    _val_manifest_for_fold,
)
from src.datasets.busbra import LABEL_TO_INDEX, load_busbra_manifest
from src.utils.config import load_project_config
from src.utils.metrics import classification_metrics, threshold_sweep
from src.utils.reporting import write_json_report, write_markdown_report
from src.utils.runtime import select_device


FULL_VIEW_NAMES = (
    "eff_identity",
    "conv_crop_sweep",
    "densenet_identity",
    "convsmall_crop_sweep",
    "swin_crop_sweep",
)
ROI_VIEW_NAMES = ("eff_identity", "conv_crop_sweep")
ROI_PAIR_WEIGHTS = {"eff_identity": 0.427, "conv_crop_sweep": 0.573}


EXTRA_MODEL_VIEWS: dict[str, ModelView] = {
    "convsmall_crop_sweep": ModelView(
        name="convsmall_crop_sweep",
        model_name="convnext_small",
        checkpoint_template="./artifacts/checkpoints/convnext_small_timm_recipe_fold{fold}.pt",
        image_size=224,
        apply_clahe=True,
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
    "swin_crop_sweep": ModelView(
        name="swin_crop_sweep",
        model_name="swin_tiny_patch4_window7_224",
        checkpoint_template="./artifacts/checkpoints/swin_tiny_patch4_window7_224_timm_recipe_fold{fold}.pt",
        image_size=224,
        apply_clahe=True,
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


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    weights: tuple[float, ...]
    min_area_ratio: float
    max_area_ratio: float
    c_value: float
    class_weight: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a model-zoo ROI stack candidate using BUSBRA OOF only."
    )
    parser.add_argument("--config", default="configs/classifier/convnext_tiny_timm_recipe.yml")
    parser.add_argument("--runtime-config", default="configs/inference/demo.yml")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--base-full-oof-cache",
        default="artifacts/reports/oof_three_model_predictions.json",
    )
    parser.add_argument(
        "--roi-oof-cache",
        default="artifacts/reports/roi_oof_lcc_mask04_protocol_predictions.json",
    )
    parser.add_argument(
        "--model-zoo-oof-cache",
        default="artifacts/reports/oof_model_zoo_predictions.json",
    )
    parser.add_argument("--random-candidates", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument(
        "--output",
        default="artifacts/reports/model_zoo_oof_protocol.json",
    )
    parser.add_argument(
        "--markdown",
        default="artifacts/reports/model_zoo_oof_protocol.md",
    )
    parser.add_argument(
        "--candidate-config",
        default="configs/inference/demo_model_zoo_oof_candidate.yml",
    )
    return parser


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return data


def _sample_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["sample_id"]) for row in rows]


def _view_array(report: dict[str, Any], view_name: str, sample_ids: list[str]) -> np.ndarray:
    rows = report["views"][view_name]
    if _sample_ids(rows) != sample_ids:
        raise ValueError(f"Sample order mismatch for view {view_name}.")
    return np.asarray([float(row["malignant_probability"]) for row in rows], dtype=np.float64)


def _generate_extra_oof_view(
    *,
    view: ModelView,
    config_path: str | Path,
    fold_count: int,
    device: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    config, paths = load_project_config(config_path)
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
    rows: list[dict[str, Any]] = []
    for fold in range(1, fold_count + 1):
        val_manifest = _val_manifest_for_fold(manifest, assignments, fold)
        rows.extend(
            _predict_view_for_manifest(
                view,
                val_manifest,
                fold=fold,
                project_root=paths.project_root,
                device=resolved_device,
                batch_size=batch_size,
            )
        )
    return sorted(rows, key=lambda row: str(row["sample_id"]))


def _load_or_generate_model_zoo_oof(args: argparse.Namespace) -> dict[str, Any]:
    destination = Path(args.model_zoo_oof_cache)
    if destination.exists():
        report = _load_json(destination)
    else:
        report = _load_json(args.base_full_oof_cache)
    report.setdefault("views", {})
    for view_name, view in EXTRA_MODEL_VIEWS.items():
        if view_name in report["views"]:
            continue
        report["views"][view_name] = _generate_extra_oof_view(
            view=view,
            config_path=args.config,
            fold_count=int(args.fold_count),
            device=str(args.device),
            batch_size=int(args.batch_size),
        )
        report["sample_count"] = len(report["views"][view_name])
        write_json_report(destination, report)
    report["source"] = "BUSBRA OOF model-zoo validation predictions"
    report["fold_count"] = int(args.fold_count)
    write_json_report(destination, report)
    return report


def _label_arrays(report: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    reference = report["views"][FULL_VIEW_NAMES[0]]
    sample_ids = _sample_ids(reference)
    y_true = np.asarray(
        [LABEL_TO_INDEX[str(row["pathology_label"]).lower()] for row in reference],
        dtype=np.int32,
    )
    fold_ids = np.asarray([int(row["fold_id"]) for row in reference], dtype=np.int32)
    return sample_ids, y_true, fold_ids


def _weighted_probability(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 0.0, None)
    if float(weights.sum()) <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return np.average(matrix, axis=1, weights=weights)


def _roi_pair_probability(roi_report: dict[str, Any], sample_ids: list[str]) -> np.ndarray:
    eff = _view_array(roi_report, "eff_identity", sample_ids)
    conv = _view_array(roi_report, "conv_crop_sweep", sample_ids)
    total = ROI_PAIR_WEIGHTS["eff_identity"] + ROI_PAIR_WEIGHTS["conv_crop_sweep"]
    return (
        ROI_PAIR_WEIGHTS["eff_identity"] * eff
        + ROI_PAIR_WEIGHTS["conv_crop_sweep"] * conv
    ) / total


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _feature_matrix(full_probability: np.ndarray, roi_probability: np.ndarray) -> np.ndarray:
    return np.vstack([_logit(full_probability), _logit(roi_probability)]).T


def _fit_stacker(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    c_value: float,
    class_weight: str | None,
) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    class_weight=class_weight,
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def _threshold_for_constraints(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    min_sensitivity: float,
) -> dict[str, Any]:
    rows = threshold_sweep(y_true, probabilities)
    feasible = [row for row in rows if row["sensitivity"] >= min_sensitivity] or rows
    return max(
        feasible,
        key=lambda row: (
            row["f1_score"],
            row["precision"],
            row["specificity"],
            row["sensitivity"],
        ),
    )


def _candidate_nested_predictions(
    spec: CandidateSpec,
    *,
    full_matrix: np.ndarray,
    roi_probability: np.ndarray,
    area_ratios: np.ndarray,
    y_true: np.ndarray,
    fold_ids: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(spec.weights, dtype=np.float64)
    full_probability = _weighted_probability(full_matrix, weights)
    fallback = (area_ratios < spec.min_area_ratio) | (area_ratios > spec.max_area_ratio)
    x = _feature_matrix(full_probability, roi_probability)
    predictions = np.zeros_like(full_probability)
    for fold in sorted(np.unique(fold_ids)):
        val_mask = fold_ids == fold
        train_mask = ~val_mask
        train_for_stacker = train_mask & ~fallback
        if int(train_for_stacker.sum()) < 20 or len(np.unique(y_true[train_for_stacker])) < 2:
            train_for_stacker = train_mask
        model = _fit_stacker(
            x[train_for_stacker],
            y_true[train_for_stacker],
            c_value=spec.c_value,
            class_weight=spec.class_weight,
        )
        fold_probabilities = model.predict_proba(x[val_mask])[:, 1]
        predictions[val_mask] = np.where(
            fallback[val_mask],
            full_probability[val_mask],
            fold_probabilities,
        )
    return predictions


def _normalise_weights(values: tuple[float, ...] | list[float] | np.ndarray) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    array = np.clip(array, 0.0, None)
    if float(array.sum()) <= 0:
        array = np.ones_like(array)
    array = array / float(array.sum())
    return tuple(float(value) for value in array)


def _candidate_specs(random_count: int, seed: int) -> list[CandidateSpec]:
    manual_weights = [
        (0.427, 0.573, 0.0, 0.0, 0.0),
        (0.358, 0.398, 0.244, 0.0, 0.0),
        (0.25, 0.35, 0.0, 0.25, 0.15),
        (0.20, 0.35, 0.0, 0.30, 0.15),
        (0.20, 0.30, 0.10, 0.25, 0.15),
        (0.15, 0.35, 0.0, 0.35, 0.15),
        (0.15, 0.30, 0.0, 0.25, 0.30),
        (0.10, 0.45, 0.0, 0.30, 0.15),
        (0.0, 0.50, 0.0, 0.30, 0.20),
        (0.0, 0.40, 0.0, 0.40, 0.20),
        (0.0, 0.30, 0.0, 0.40, 0.30),
    ]
    rng = np.random.default_rng(seed)
    for alpha in (
        np.array([1.5, 2.5, 0.7, 1.4, 1.0]),
        np.array([0.8, 2.0, 0.4, 2.0, 1.0]),
        np.ones(5),
    ):
        for _ in range(max(0, random_count // 3)):
            manual_weights.append(tuple(float(value) for value in rng.dirichlet(alpha)))
    gates = [
        (0.00, 1.01),
        (0.05, 1.01),
        (0.08, 1.01),
        (0.10, 1.01),
        (0.08, 0.90),
        (0.08, 0.75),
        (0.12, 0.85),
    ]
    c_values = (0.03, 0.1, 0.3, 1.0)
    class_weights: tuple[str | None, ...] = (None, "balanced")
    specs: list[CandidateSpec] = []
    seen: set[tuple[Any, ...]] = set()
    for weights in manual_weights:
        norm = _normalise_weights(weights)
        rounded = tuple(round(value, 4) for value in norm)
        for min_area, max_area in gates:
            for c_value in c_values:
                for class_weight in class_weights:
                    key = (rounded, min_area, max_area, c_value, class_weight)
                    if key in seen:
                        continue
                    seen.add(key)
                    name = (
                        "w_"
                        + "_".join(f"{value:.2f}" for value in norm)
                        + f"_gate{min_area:.2f}-{max_area:.2f}_C{c_value:g}_{class_weight or 'none'}"
                    )
                    specs.append(
                        CandidateSpec(
                            name=name,
                            weights=norm,
                            min_area_ratio=float(min_area),
                            max_area_ratio=float(max_area),
                            c_value=float(c_value),
                            class_weight=class_weight,
                        )
                    )
    return specs


def _evaluate_candidate(
    spec: CandidateSpec,
    *,
    full_matrix: np.ndarray,
    roi_probability: np.ndarray,
    area_ratios: np.ndarray,
    y_true: np.ndarray,
    fold_ids: np.ndarray,
    min_sensitivity: float,
) -> dict[str, Any]:
    predictions = _candidate_nested_predictions(
        spec,
        full_matrix=full_matrix,
        roi_probability=roi_probability,
        area_ratios=area_ratios,
        y_true=y_true,
        fold_ids=fold_ids,
    )
    threshold_metrics = _threshold_for_constraints(
        y_true,
        predictions,
        min_sensitivity=min_sensitivity,
    )
    return {
        "name": spec.name,
        "weights": {
            view_name: float(weight)
            for view_name, weight in zip(FULL_VIEW_NAMES, spec.weights)
        },
        "min_area_ratio": float(spec.min_area_ratio),
        "max_area_ratio": float(spec.max_area_ratio),
        "c_value": float(spec.c_value),
        "class_weight": spec.class_weight,
        "fallback_count": int(
            np.sum((area_ratios < spec.min_area_ratio) | (area_ratios > spec.max_area_ratio))
        ),
        "metrics": {
            **threshold_metrics,
            "auc": float(roc_auc_score(y_true, predictions)),
        },
    }


def _fit_final_model(
    selected: dict[str, Any],
    *,
    full_matrix: np.ndarray,
    roi_probability: np.ndarray,
    area_ratios: np.ndarray,
    y_true: np.ndarray,
) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    weights = np.asarray([selected["weights"][name] for name in FULL_VIEW_NAMES], dtype=np.float64)
    full_probability = _weighted_probability(full_matrix, weights)
    fallback = (
        (area_ratios < float(selected["min_area_ratio"]))
        | (area_ratios > float(selected["max_area_ratio"]))
    )
    train_mask = ~fallback
    if int(train_mask.sum()) < 20 or len(np.unique(y_true[train_mask])) < 2:
        train_mask = np.ones_like(fallback, dtype=bool)
    x = _feature_matrix(full_probability, roi_probability)
    model = _fit_stacker(
        x[train_mask],
        y_true[train_mask],
        c_value=float(selected["c_value"]),
        class_weight=selected["class_weight"],
    )
    stack_prob = model.predict_proba(x)[:, 1]
    final_prob = np.where(fallback, full_probability, stack_prob)
    return model, full_probability, final_prob


def _slim_stacker(model: Pipeline) -> dict[str, Any]:
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    return {
        "feature_mode": "logit",
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) for value in scaler.scale_],
        "coef": [float(value) for value in classifier.coef_[0]],
        "intercept": float(classifier.intercept_[0]),
    }


def _member_specs_for_config(weights: dict[str, float]) -> list[dict[str, Any]]:
    crop_sweep_variants = [
        {"name": "identity", "crop_pct": 0.90},
        {"name": "hflip", "crop_pct": 0.90},
        {"name": "identity", "crop_pct": 0.95},
        {"name": "hflip", "crop_pct": 0.95},
        {"name": "identity", "crop_pct": 1.00},
        {"name": "hflip", "crop_pct": 1.00},
    ]
    groups = {
        "eff_identity": {
            "model": "tf_efficientnetv2_s",
            "checkpoint": "./artifacts/checkpoints/efficientnetv2_s_fold{fold}.pt",
            "weight": weights["eff_identity"],
            "image_size": 224,
            "apply_clahe": True,
            "mean": None,
            "std": None,
            "interpolation": "area",
            "crop_pct": 1.0,
            "tta_variants": [{"name": "identity"}],
        },
        "conv_crop_sweep": {
            "model": "convnext_tiny",
            "checkpoint": "./artifacts/checkpoints/convnext_tiny_timm_recipe_fold{fold}.pt",
            "weight": weights["conv_crop_sweep"],
            "image_size": 224,
            "apply_clahe": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "bicubic",
            "crop_pct": 0.95,
            "tta_variants": crop_sweep_variants,
        },
        "densenet_identity": {
            "model": "densenet121",
            "checkpoint": "./artifacts/checkpoints/densenet121_fold{fold}.pt",
            "weight": weights["densenet_identity"],
            "image_size": 224,
            "apply_clahe": True,
            "mean": None,
            "std": None,
            "interpolation": "area",
            "crop_pct": 1.0,
            "tta_variants": [{"name": "identity"}],
        },
        "convsmall_crop_sweep": {
            "model": "convnext_small",
            "checkpoint": "./artifacts/checkpoints/convnext_small_timm_recipe_fold{fold}.pt",
            "weight": weights["convsmall_crop_sweep"],
            "image_size": 224,
            "apply_clahe": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "bicubic",
            "crop_pct": 0.95,
            "tta_variants": crop_sweep_variants,
        },
        "swin_crop_sweep": {
            "model": "swin_tiny_patch4_window7_224",
            "checkpoint": "./artifacts/checkpoints/swin_tiny_patch4_window7_224_timm_recipe_fold{fold}.pt",
            "weight": weights["swin_crop_sweep"],
            "image_size": 224,
            "apply_clahe": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "bicubic",
            "crop_pct": 0.95,
            "tta_variants": crop_sweep_variants,
        },
    }
    members: list[dict[str, Any]] = []
    for name in FULL_VIEW_NAMES:
        spec = groups[name]
        include = float(spec["weight"]) > 1e-6 or name in {"eff_identity", "conv_crop_sweep"}
        if not include:
            continue
        for fold in range(1, 6):
            member = dict(spec)
            member["checkpoint"] = str(member["checkpoint"]).format(fold=fold)
            members.append(member)
    return members


def _write_candidate_config(
    *,
    runtime_config_path: str | Path,
    destination: str | Path,
    selected: dict[str, Any],
    stacker: dict[str, Any],
) -> None:
    config = yaml.safe_load(Path(runtime_config_path).read_text(encoding="utf-8"))
    runtime = config["runtime"]
    runtime["ensemble_display_name"] = "Model-Zoo OOF Candidate + ROI Area Gate"
    runtime["primary_classifier_model"] = "Model-Zoo OOF Candidate"
    runtime["default_threshold"] = float(selected["metrics"]["threshold"])
    runtime["classifier_members"] = _member_specs_for_config(selected["weights"])
    runtime["classifier_checkpoints"] = []
    runtime["classifier_checkpoint"] = None
    roi_config = runtime["roi_enhancement"]
    roi_config["stacker"] = stacker
    roi_config["quality_gate"] = {
        "enabled": True,
        "min_area_ratio": float(selected["min_area_ratio"]),
        "max_area_ratio": float(selected["max_area_ratio"]),
        "fallback_to_full": True,
    }
    roi_config["classifier_weight_overrides"] = {
        "tf_efficientnetv2_s": ROI_PAIR_WEIGHTS["eff_identity"],
        "convnext_tiny": ROI_PAIR_WEIGHTS["conv_crop_sweep"],
        "densenet121": 0.0,
        "convnext_small": 0.0,
        "swin_tiny_patch4_window7_224": 0.0,
    }
    Path(destination).write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _format_metric(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    return f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value)


def build_markdown(report: dict[str, Any]) -> list[str]:
    baseline = report["baseline_current_demo_oof_metrics"]
    selected = report["selected_candidate"]
    final = report["selected_refit_oof_metrics"]
    lines = [
        "# Model-Zoo OOF Candidate Protocol",
        "",
        "## Data Boundary",
        "",
        "- Selection uses BUSBRA OOF predictions only.",
        "- BUSI is not read by this script and must only be used after this candidate is frozen.",
        "- ROI branch keeps the deployable EfficientNetV2-S + ConvNeXt-Tiny ROI pair; new model-zoo members affect the full-image branch.",
        "",
        "## Views",
        "",
        "| View | Role |",
        "| --- | --- |",
    ]
    for view_name in FULL_VIEW_NAMES:
        lines.append(f"| {view_name} | full-image model-zoo candidate |")
    lines.extend(
        [
            "",
            "## OOF Metrics",
            "",
            "| Scheme | AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1 | Confusion |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for label, metrics in (
        ("current demo OOF", baseline),
        ("selected nested OOF", selected["metrics"]),
        ("selected refit OOF", final),
    ):
        confusion = metrics["confusion"]
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _format_metric(metrics, "auc"),
                    f"{float(metrics['threshold']):.3f}",
                    _format_metric(metrics, "sensitivity"),
                    _format_metric(metrics, "specificity"),
                    _format_metric(metrics, "accuracy"),
                    _format_metric(metrics, "precision"),
                    _format_metric(metrics, "f1_score"),
                    f"TN {confusion['tn']} / FP {confusion['fp']} / FN {confusion['fn']} / TP {confusion['tp']}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Selected Candidate",
            "",
            f"- Name: `{selected['name']}`",
            f"- Candidate config: `{report['candidate_config']}`",
            f"- Area gate: min `{selected['min_area_ratio']:.2f}`, max `{selected['max_area_ratio']:.2f}`, fallback `{selected['fallback_count']}` OOF samples.",
            f"- Logistic C: `{selected['c_value']}`, class_weight: `{selected['class_weight']}`",
            "",
            "| Full-image view | Weight |",
            "| --- | ---: |",
        ]
    )
    for view_name, weight in selected["weights"].items():
        lines.append(f"| {view_name} | {float(weight):.4f} |")
    lines.extend(
        [
            "",
            "## Top Candidates",
            "",
            "| Rank | AUC | Threshold | Sens | Spec | Precision | F1 | Gate | Weights |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for index, row in enumerate(report["top_candidates"][:10], start=1):
        metrics = row["metrics"]
        weights = ", ".join(f"{key}={value:.2f}" for key, value in row["weights"].items())
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    _format_metric(metrics, "auc"),
                    f"{float(metrics['threshold']):.3f}",
                    _format_metric(metrics, "sensitivity"),
                    _format_metric(metrics, "specificity"),
                    _format_metric(metrics, "precision"),
                    _format_metric(metrics, "f1_score"),
                    f"{row['min_area_ratio']:.2f}-{row['max_area_ratio']:.2f}",
                    weights,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- If the frozen BUSI external check improves AUC without materially hurting sensitivity, precision, specificity, or F1, this candidate can be considered for demo merge.",
            "- If BUSI does not improve, keep this as an internal experiment and do not tune the candidate using BUSI feedback.",
        ]
    )
    return lines


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_zoo_oof = _load_or_generate_model_zoo_oof(args)
    roi_oof = _load_json(args.roi_oof_cache)
    sample_ids, y_true, fold_ids = _label_arrays(model_zoo_oof)
    full_matrix = np.vstack(
        [_view_array(model_zoo_oof, view_name, sample_ids) for view_name in FULL_VIEW_NAMES]
    ).T
    roi_probability = _roi_pair_probability(roi_oof, sample_ids)
    area_ratios = np.asarray(roi_oof["roi_area_ratios"], dtype=np.float64)
    if len(area_ratios) != len(sample_ids):
        raise ValueError("ROI area ratio length does not match OOF sample count.")

    runtime_config = yaml.safe_load(Path(args.runtime_config).read_text(encoding="utf-8"))
    runtime = runtime_config["runtime"]
    current_weights = np.asarray([0.427, 0.573, 0.0, 0.0, 0.0], dtype=np.float64)
    current_full = _weighted_probability(full_matrix, current_weights)
    current_stacker = runtime["roi_enhancement"]["stacker"]
    current_x = _feature_matrix(current_full, roi_probability)
    current_scaled = (
        current_x - np.asarray(current_stacker["scaler_mean"], dtype=np.float64)
    ) / np.asarray(current_stacker["scaler_scale"], dtype=np.float64)
    current_logit = (
        current_scaled @ np.asarray(current_stacker["coef"], dtype=np.float64)
        + float(current_stacker["intercept"])
    )
    current_stack = 1.0 / (1.0 + np.exp(-current_logit))
    current_gate = runtime["roi_enhancement"]["quality_gate"]
    current_fallback = (
        (area_ratios < float(current_gate["min_area_ratio"]))
        | (area_ratios > float(current_gate["max_area_ratio"]))
    )
    current_probability = np.where(current_fallback, current_full, current_stack)
    baseline_metrics = classification_metrics(
        y_true,
        current_probability,
        threshold=float(runtime.get("default_threshold", 0.5)),
    )

    min_sensitivity = max(0.84, float(baseline_metrics["sensitivity"]) - 0.01)
    candidates = _candidate_specs(int(args.random_candidates), int(args.seed))
    evaluated: list[dict[str, Any]] = []
    for index, spec in enumerate(candidates, start=1):
        result = _evaluate_candidate(
            spec,
            full_matrix=full_matrix,
            roi_probability=roi_probability,
            area_ratios=area_ratios,
            y_true=y_true,
            fold_ids=fold_ids,
            min_sensitivity=min_sensitivity,
        )
        metrics = result["metrics"]
        result["accepted_by_oof_protocol"] = bool(
            metrics["sensitivity"] >= min_sensitivity
            and metrics["specificity"] >= float(baseline_metrics["specificity"]) - 0.03
            and metrics["f1_score"] >= float(baseline_metrics["f1_score"]) - 0.015
        )
        evaluated.append(result)
        if index % 500 == 0:
            print(f"evaluated {index}/{len(candidates)} candidates")
    evaluated.sort(
        key=lambda row: (
            row["accepted_by_oof_protocol"],
            row["metrics"]["auc"],
            row["metrics"]["f1_score"],
            row["metrics"]["precision"],
            row["metrics"]["sensitivity"],
        ),
        reverse=True,
    )
    selected = evaluated[0]
    model, _, refit_probability = _fit_final_model(
        selected,
        full_matrix=full_matrix,
        roi_probability=roi_probability,
        area_ratios=area_ratios,
        y_true=y_true,
    )
    refit_metrics = classification_metrics(
        y_true,
        refit_probability,
        threshold=float(selected["metrics"]["threshold"]),
    )
    refit_metrics["auc"] = float(roc_auc_score(y_true, refit_probability))
    stacker = _slim_stacker(model)
    _write_candidate_config(
        runtime_config_path=args.runtime_config,
        destination=args.candidate_config,
        selected=selected,
        stacker=stacker,
    )
    report = {
        "method": "model-zoo full branch plus ROI pair stacker selected on BUSBRA OOF only",
        "data_boundary": {
            "selection": "BUSBRA OOF only",
            "external_review": "BUSI only after candidate freeze",
        },
        "full_view_names": list(FULL_VIEW_NAMES),
        "roi_view_names": list(ROI_VIEW_NAMES),
        "sample_count": int(len(y_true)),
        "positive_count": int(np.sum(y_true == 1)),
        "negative_count": int(np.sum(y_true == 0)),
        "baseline_current_demo_oof_metrics": baseline_metrics,
        "selected_candidate": selected,
        "selected_refit_oof_metrics": refit_metrics,
        "selected_final_stacker": stacker,
        "top_candidates": evaluated[:20],
        "candidate_config": str(args.candidate_config),
        "model_zoo_oof_cache": str(args.model_zoo_oof_cache),
        "roi_oof_cache": str(args.roi_oof_cache),
    }
    write_json_report(args.output, report)
    write_markdown_report(args.markdown, build_markdown(report))
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    selected = report["selected_candidate"]
    print(
        json.dumps(
            {
                "candidate_config": report["candidate_config"],
                "baseline_oof_auc": report["baseline_current_demo_oof_metrics"]["auc"],
                "selected_nested_oof_auc": selected["metrics"]["auc"],
                "selected_refit_oof_auc": report["selected_refit_oof_metrics"]["auc"],
                "selected_threshold": selected["metrics"]["threshold"],
                "selected_weights": selected["weights"],
                "accepted": selected["accepted_by_oof_protocol"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
