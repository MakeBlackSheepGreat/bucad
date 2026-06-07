"""Utility script for oof meta learner protocol workflows."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.reporting import write_json_report, write_markdown_report  # noqa: E402


PAIR_VIEWS = ("eff_identity", "conv_crop_sweep")
PAIR_WEIGHTS = {"eff_identity": 0.427, "conv_crop_sweep": 0.573}
AREA_BINS = (
    ("small", 0.0, 0.15),
    ("medium", 0.15, 0.45),
    ("large", 0.45, 0.75),
    ("very_large", 0.75, 1.01),
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    feature_set: str
    model_family: str
    build_model: Callable[[], Any]
    params: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an OOF-only meta-learner for ROI-aware fusion."
    )
    parser.add_argument("--runtime-config", default="configs/inference/demo.yml")
    parser.add_argument("--full-oof-cache", default="artifacts/reports/oof_two_model_predictions.json")
    parser.add_argument(
        "--roi-oof-cache",
        default="artifacts/reports/roi_oof_lcc_mask04_protocol_predictions.json",
    )
    parser.add_argument(
        "--area-cache",
        default="artifacts/reports/roi_precision_f1_area_cache.json",
    )
    parser.add_argument(
        "--busi-roi-cache",
        default="artifacts/reports/busi_roi_lcc_pair_view_predictions.json",
    )
    parser.add_argument(
        "--baseline-busi-report",
        default="artifacts/reports/busi_demo_roi_oof_lcc_mask04_eval.json",
    )
    parser.add_argument("--min-sensitivity", type=float, default=0.84)
    parser.add_argument("--max-auc-drop", type=float, default=0.002)
    parser.add_argument("--output", default="artifacts/reports/oof_meta_learner_protocol.json")
    parser.add_argument("--markdown", default="artifacts/reports/oof_meta_learner_protocol.md")
    return parser


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= float(threshold)).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else 0.0
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    npv = float(tn / (tn + fn)) if tn + fn else 0.0
    accuracy = float((tp + tn) / (tp + tn + fp + fn)) if tp + tn + fp + fn else 0.0
    f1_score = (
        float(2.0 * precision * sensitivity / (precision + sensitivity))
        if precision + sensitivity
        else 0.0
    )
    fpr = float(fp / (fp + tn)) if fp + tn else 0.0
    fnr = float(fn / (fn + tp)) if fn + tp else 0.0
    balanced_accuracy = float((sensitivity + specificity) / 2.0)
    return {
        "auc": float(roc_auc_score(y_true, probabilities)),
        "threshold": float(threshold),
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1_score": f1_score,
        "balanced_accuracy": balanced_accuracy,
        "youden_j": float(sensitivity + specificity - 1.0),
        "fpr": fpr,
        "fnr": fnr,
        "sample_count": int(len(y_true)),
        "positive_count": int(np.asarray(y_true).sum()),
        "negative_count": int(len(y_true) - np.asarray(y_true).sum()),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _best_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    min_sensitivity: float,
) -> dict[str, Any]:
    rows = [
        _metrics(y_true, probabilities, float(threshold))
        for threshold in np.round(np.arange(0.1, 0.9001, 0.01), 2)
    ]
    feasible = [row for row in rows if row["sensitivity"] >= min_sensitivity] or rows
    return max(
        feasible,
        key=lambda row: (
            row["f1_score"],
            row["precision"],
            row["specificity"],
            row["auc"],
            row["sensitivity"],
        ),
    )


def _validate_order(views: dict[str, list[dict[str, Any]]], *, view_names: tuple[str, ...]) -> list[str]:
    sample_ids = [str(row["sample_id"]) for row in views[view_names[0]]]
    for view_name in view_names[1:]:
        current = [str(row["sample_id"]) for row in views[view_name]]
        if current != sample_ids:
            raise ValueError(f"Sample order mismatch for {view_name}.")
    return sample_ids


def _view_array(views: dict[str, list[dict[str, Any]]], view_name: str) -> np.ndarray:
    return np.asarray(
        [float(row["malignant_probability"]) for row in views[view_name]],
        dtype=np.float64,
    )


def _pair_blend(views: dict[str, list[dict[str, Any]]]) -> np.ndarray:
    _validate_order(views, view_names=PAIR_VIEWS)
    return (
        PAIR_WEIGHTS["eff_identity"] * _view_array(views, "eff_identity")
        + PAIR_WEIGHTS["conv_crop_sweep"] * _view_array(views, "conv_crop_sweep")
    )


def _runtime_stack(
    full_probabilities: np.ndarray,
    roi_probabilities: np.ndarray,
    *,
    runtime_config: dict[str, Any],
) -> np.ndarray:
    stacker = runtime_config["runtime"]["roi_enhancement"]["stacker"]
    matrix = np.vstack([_logit(full_probabilities), _logit(roi_probabilities)]).T
    scaled = (
        matrix - np.asarray(stacker["scaler_mean"], dtype=np.float64)
    ) / np.asarray(stacker["scaler_scale"], dtype=np.float64)
    return _sigmoid(
        scaled @ np.asarray(stacker["coef"], dtype=np.float64)
        + float(stacker["intercept"])
    )


def _bin_one_hot(area_ratios: np.ndarray) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name, min_area, max_area in AREA_BINS:
        output[f"area_bin_{name}"] = (
            (area_ratios >= min_area) & (area_ratios < max_area)
        ).astype(np.float64)
    return output


def _base_feature_map(
    *,
    full_views: dict[str, list[dict[str, Any]]],
    roi_views: dict[str, list[dict[str, Any]]],
    area_ratios: np.ndarray,
    runtime_config: dict[str, Any],
) -> dict[str, np.ndarray]:
    full_eff = _view_array(full_views, "eff_identity")
    full_conv = _view_array(full_views, "conv_crop_sweep")
    roi_eff = _view_array(roi_views, "eff_identity")
    roi_conv = _view_array(roi_views, "conv_crop_sweep")
    full_blend = _pair_blend(full_views)
    roi_blend = _pair_blend(roi_views)
    runtime_stack = _runtime_stack(
        full_blend,
        roi_blend,
        runtime_config=runtime_config,
    )
    clipped_area = np.clip(area_ratios, 1e-6, 1.0)
    features: dict[str, np.ndarray] = {
        "full_eff": full_eff,
        "full_conv": full_conv,
        "roi_eff": roi_eff,
        "roi_conv": roi_conv,
        "full_blend": full_blend,
        "roi_blend": roi_blend,
        "runtime_stack": runtime_stack,
        "area_ratio": clipped_area,
        "log_area": np.log(clipped_area),
        "sqrt_area": np.sqrt(clipped_area),
        "full_roi_delta": roi_blend - full_blend,
        "abs_full_roi_delta": np.abs(roi_blend - full_blend),
        "full_model_delta": full_conv - full_eff,
        "roi_model_delta": roi_conv - roi_eff,
        "abs_full_model_delta": np.abs(full_conv - full_eff),
        "abs_roi_model_delta": np.abs(roi_conv - roi_eff),
        "full_x_area": full_blend * clipped_area,
        "roi_x_area": roi_blend * clipped_area,
        "runtime_x_area": runtime_stack * clipped_area,
        "roi_delta_x_area": (roi_blend - full_blend) * clipped_area,
        "logit_full_blend": _logit(full_blend),
        "logit_roi_blend": _logit(roi_blend),
        "logit_runtime_stack": _logit(runtime_stack),
        "logit_full_eff": _logit(full_eff),
        "logit_full_conv": _logit(full_conv),
        "logit_roi_eff": _logit(roi_eff),
        "logit_roi_conv": _logit(roi_conv),
    }
    features.update(_bin_one_hot(clipped_area))
    return features


FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "minimal": (
        "runtime_stack",
        "full_blend",
        "roi_blend",
        "area_ratio",
        "abs_full_roi_delta",
    ),
    "probability_rich": (
        "runtime_stack",
        "full_eff",
        "full_conv",
        "roi_eff",
        "roi_conv",
        "full_blend",
        "roi_blend",
        "area_ratio",
        "log_area",
        "sqrt_area",
        "full_roi_delta",
        "abs_full_roi_delta",
        "full_model_delta",
        "roi_model_delta",
        "abs_full_model_delta",
        "abs_roi_model_delta",
        "full_x_area",
        "roi_x_area",
        "runtime_x_area",
        "roi_delta_x_area",
        "area_bin_small",
        "area_bin_medium",
        "area_bin_large",
        "area_bin_very_large",
    ),
    "logit_rich": (
        "logit_runtime_stack",
        "logit_full_blend",
        "logit_roi_blend",
        "logit_full_eff",
        "logit_full_conv",
        "logit_roi_eff",
        "logit_roi_conv",
        "area_ratio",
        "log_area",
        "sqrt_area",
        "abs_full_roi_delta",
        "abs_full_model_delta",
        "abs_roi_model_delta",
        "area_bin_small",
        "area_bin_medium",
        "area_bin_large",
        "area_bin_very_large",
    ),
    "compact_interaction": (
        "runtime_stack",
        "full_blend",
        "roi_blend",
        "area_ratio",
        "log_area",
        "full_roi_delta",
        "abs_full_roi_delta",
        "runtime_x_area",
        "roi_delta_x_area",
        "area_bin_small",
        "area_bin_medium",
        "area_bin_large",
        "area_bin_very_large",
    ),
}


def _feature_matrix(feature_map: dict[str, np.ndarray], feature_set: str) -> tuple[np.ndarray, list[str]]:
    names = list(FEATURE_SETS[feature_set])
    return np.vstack([feature_map[name] for name in names]).T.astype(np.float64), names


def _make_logistic(C: float, class_weight: str | None) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(C),
                    class_weight=class_weight,
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def _make_hgb(
    *,
    learning_rate: float,
    max_leaf_nodes: int,
    l2_regularization: float,
    max_iter: int,
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=float(learning_rate),
        max_leaf_nodes=int(max_leaf_nodes),
        l2_regularization=float(l2_regularization),
        max_iter=int(max_iter),
        min_samples_leaf=25,
        random_state=42,
        early_stopping=False,
    )


def _candidate_specs() -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for feature_set in FEATURE_SETS:
        for C in (0.03, 0.1, 0.3, 1.0, 3.0):
            for class_weight in (None, "balanced"):
                name = f"logistic_{feature_set}_C{C:g}_{class_weight or 'none'}"
                specs.append(
                    CandidateSpec(
                        name=name,
                        feature_set=feature_set,
                        model_family="logistic_regression",
                        build_model=lambda C=C, class_weight=class_weight: _make_logistic(C, class_weight),
                        params={"C": C, "class_weight": class_weight},
                    )
                )
    for feature_set in ("minimal", "compact_interaction", "probability_rich"):
        for learning_rate, max_leaf_nodes, l2_regularization, max_iter in (
            (0.03, 3, 0.1, 120),
            (0.03, 7, 0.1, 120),
            (0.05, 3, 1.0, 80),
            (0.05, 7, 1.0, 80),
            (0.02, 7, 1.0, 160),
        ):
            name = (
                f"hgb_{feature_set}_lr{learning_rate:g}_leaf{max_leaf_nodes}"
                f"_l2{l2_regularization:g}_iter{max_iter}"
            )
            specs.append(
                CandidateSpec(
                    name=name,
                    feature_set=feature_set,
                    model_family="hist_gradient_boosting",
                    build_model=(
                        lambda learning_rate=learning_rate,
                        max_leaf_nodes=max_leaf_nodes,
                        l2_regularization=l2_regularization,
                        max_iter=max_iter: _make_hgb(
                            learning_rate=learning_rate,
                            max_leaf_nodes=max_leaf_nodes,
                            l2_regularization=l2_regularization,
                            max_iter=max_iter,
                        )
                    ),
                    params={
                        "learning_rate": learning_rate,
                        "max_leaf_nodes": max_leaf_nodes,
                        "l2_regularization": l2_regularization,
                        "max_iter": max_iter,
                        "min_samples_leaf": 25,
                    },
                )
            )
    return specs


def _nested_oof_predict(
    spec: CandidateSpec,
    X: np.ndarray,
    y_true: np.ndarray,
    fold_ids: np.ndarray,
) -> np.ndarray:
    probabilities = np.zeros(len(y_true), dtype=np.float64)
    for fold_id in sorted(np.unique(fold_ids)):
        val_mask = fold_ids == fold_id
        train_mask = ~val_mask
        model = spec.build_model()
        model.fit(X[train_mask], y_true[train_mask])
        probabilities[val_mask] = model.predict_proba(X[val_mask])[:, 1]
    return probabilities


def _fit_final(spec: CandidateSpec, X: np.ndarray, y_true: np.ndarray) -> Any:
    model = spec.build_model()
    model.fit(X, y_true)
    return model


def _load_busi_full_views() -> tuple[list[dict[str, str]], np.ndarray, dict[str, list[dict[str, Any]]]]:
    reports = {
        "eff_identity": "artifacts/reports/busi_efficientnetv2_s_5fold_identity.json",
        "conv_crop_sweep": "artifacts/reports/busi_convnext_tiny_tta_crop_sweep.json",
    }
    reference: list[dict[str, str]] | None = None
    y_true: np.ndarray | None = None
    views: dict[str, list[dict[str, Any]]] = {}
    for view_name, path in reports.items():
        data = _load_json(path)
        rows = data["rows"]
        current = [
            {"sample_id": str(row["sample_id"]), "pathology_label": str(row["pathology_label"])}
            for row in rows
        ]
        if reference is None:
            reference = current
            y_true = np.asarray(
                [1 if row["pathology_label"] == "malignant" else 0 for row in rows],
                dtype=np.int32,
            )
        elif current != reference:
            raise ValueError(f"BUSI full report row order mismatch: {path}")
        views[view_name] = [
            {
                "sample_id": row["sample_id"],
                "pathology_label": row["pathology_label"],
                "malignant_probability": float(row["malignant_probability"]),
            }
            for row in rows
        ]
    if reference is None or y_true is None:
        raise ValueError("No BUSI full reports loaded.")
    return reference, y_true, views


def _format_metric(metric: dict[str, Any], key: str) -> str:
    return f"{float(metric[key]):.4f}"


def _confusion_text(metric: dict[str, Any]) -> str:
    c = metric["confusion"]
    return f"TN {c['tn']} / FP {c['fp']} / FN {c['fn']} / TP {c['tp']}"


def _metrics_row(name: str, metric: dict[str, Any]) -> str:
    return (
        "| "
        + " | ".join(
            [
                name,
                str(metric["sample_count"]),
                str(metric["positive_count"]),
                str(metric["negative_count"]),
                _format_metric(metric, "auc"),
                f"{metric['threshold']:.3f}",
                _format_metric(metric, "accuracy"),
                _format_metric(metric, "sensitivity"),
                _format_metric(metric, "recall"),
                _format_metric(metric, "specificity"),
                _format_metric(metric, "precision"),
                _format_metric(metric, "npv"),
                _format_metric(metric, "f1_score"),
                _format_metric(metric, "balanced_accuracy"),
                _format_metric(metric, "youden_j"),
                _format_metric(metric, "fpr"),
                _format_metric(metric, "fnr"),
                _confusion_text(metric),
            ]
        )
        + " |"
    )


def build_markdown(report: dict[str, Any]) -> list[str]:
    selected = report["selected_candidate"]
    lines = [
        "# OOF Meta-Learner ROI \u878d\u5408\u5b9e\u9a8c",
        "",
        "\u65e5\u671f\uff1a2026-04-26",
        "",
        "## \u5b9e\u9a8c\u8fb9\u754c",
        "",
        "- \u53ea\u4f7f\u7528 BUSBRA OOF \u9884\u6d4b\u3001ROI \u9762\u79ef\u548c\u6807\u7b7e\u8fdb\u884c\u5019\u9009\u9009\u62e9\u3002",
        "- \u4f7f\u7528\u539f 5 \u6298 fold_id \u505a\u5d4c\u5957 OOF\uff1a\u6bcf\u4e2a\u6837\u672c\u7684 meta \u9884\u6d4b\u6765\u81ea\u672a\u89c1\u8fc7\u8be5 fold \u7684 meta \u6a21\u578b\u3002",
        "- \u5916\u90e8\u6570\u636e\u53ea\u5728 OOF \u9009\u5b9a\u6a21\u578b\u3001\u9608\u503c\u548c\u7279\u5f81\u540e\u505a\u4e00\u6b21\u56fa\u5b9a\u590d\u6838\uff0c\u4e0d\u53c2\u4e0e\u8bad\u7ec3\u6216\u8c03\u53c2\u3002",
        "",
        "## \u5165\u9009\u65b9\u6848",
        "",
        f"- \u5019\u9009\u540d\u79f0\uff1a`{selected['name']}`",
        f"- \u6a21\u578b\u65cf\uff1a`{selected['model_family']}`",
        f"- \u7279\u5f81\u96c6\uff1a`{selected['feature_set']}`",
        f"- \u53c2\u6570\uff1a`{json.dumps(selected['params'], ensure_ascii=False)}`",
        f"- OOF \u9608\u503c\uff1a`{selected['metrics']['threshold']:.3f}`",
        "",
        "\u4f7f\u7528\u7279\u5f81\uff1a",
        "",
    ]
    for feature_name in selected["feature_names"]:
        lines.append(f"- `{feature_name}`")
    lines.extend(
        [
            "",
            "## \u5b8c\u6574\u6307\u6807\u5bf9\u6bd4",
            "",
            "| \u65b9\u6848 | \u6837\u672c\u6570 | \u9633\u6027 | \u9634\u6027 | AUC | \u9608\u503c | Accuracy | Sensitivity | Recall | Specificity | Precision | NPV | F1-Score | Balanced Acc | Youden J | FPR | FNR | Confusion |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            _metrics_row("OOF \u5f53\u524d\u4e3b\u7ebf", report["baseline_oof_metrics"]),
            _metrics_row("OOF meta-learner", selected["metrics"]),
            _metrics_row("\u5916\u90e8\u5f53\u524d\u4e3b\u7ebf", report["baseline_busi_metrics"]),
            _metrics_row("\u5916\u90e8 meta-learner \u56fa\u5b9a\u590d\u6838", report["busi_metrics"]),
            "",
            "## OOF Top \u5019\u9009",
            "",
            "| \u6392\u540d | \u5019\u9009 | \u6a21\u578b\u65cf | \u7279\u5f81\u96c6 | AUC | \u9608\u503c | Accuracy | Sensitivity | Specificity | Precision | F1-Score | FPR | FNR | Confusion |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for index, row in enumerate(report["top_candidates"][:10], start=1):
        metric = row["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`{row['name']}`",
                    f"`{row['model_family']}`",
                    f"`{row['feature_set']}`",
                    _format_metric(metric, "auc"),
                    f"{metric['threshold']:.3f}",
                    _format_metric(metric, "accuracy"),
                    _format_metric(metric, "sensitivity"),
                    _format_metric(metric, "specificity"),
                    _format_metric(metric, "precision"),
                    _format_metric(metric, "f1_score"),
                    _format_metric(metric, "fpr"),
                    _format_metric(metric, "fnr"),
                    _confusion_text(metric),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## \u7ed3\u8bba",
            "",
        ]
    )
    if report["recommendation"] == "keep":
        lines.append("- OOF meta-learner \u5728\u56fa\u5b9a\u5916\u90e8\u590d\u6838\u4e2d\u8d85\u8fc7\u5f53\u524d\u4e3b\u7ebf\uff0c\u5efa\u8bae\u8fdb\u5165\u4e0b\u4e00\u6b65 runtime \u96c6\u6210\u9a8c\u8bc1\u3002")
    else:
        lines.append("- OOF meta-learner \u672a\u5f62\u6210\u7a33\u5b9a\u7efc\u5408\u63d0\u5347\uff0c\u5efa\u8bae\u6682\u4e0d\u5408\u5e76\u4e3b\u7ebf\u3002")
    lines.append("- \u5f53\u524d\u5b9e\u9a8c\u5df2\u7ecf\u5217\u51fa AUC\u3001Accuracy\u3001Recall/Sensitivity\u3001Precision\u3001Specificity\u3001F1-Score\uff0c\u4ee5\u53ca NPV\u3001Balanced Accuracy\u3001Youden J\u3001FPR\u3001FNR \u548c\u6df7\u6dc6\u77e9\u9635\u3002")
    return [line.encode("utf-8").decode("unicode_escape") if "\\u" in line else line for line in lines]

def _normalise_baseline_metrics(metrics: dict[str, Any], y_true: np.ndarray | None = None) -> dict[str, Any]:
    if "npv" in metrics and "sample_count" in metrics:
        return metrics
    confusion = metrics["confusion"]
    sample_count = sum(int(confusion[key]) for key in ("tn", "fp", "fn", "tp"))
    if y_true is not None:
        positive_count = int(np.asarray(y_true).sum())
    else:
        positive_count = int(confusion["tp"] + confusion["fn"])
    negative_count = int(sample_count - positive_count)
    enriched = dict(metrics)
    enriched["sample_count"] = sample_count
    enriched["positive_count"] = positive_count
    enriched["negative_count"] = negative_count
    enriched["npv"] = float(confusion["tn"] / (confusion["tn"] + confusion["fn"])) if confusion["tn"] + confusion["fn"] else 0.0
    enriched["balanced_accuracy"] = float((metrics["sensitivity"] + metrics["specificity"]) / 2.0)
    enriched["youden_j"] = float(metrics["sensitivity"] + metrics["specificity"] - 1.0)
    enriched["fpr"] = float(confusion["fp"] / (confusion["fp"] + confusion["tn"])) if confusion["fp"] + confusion["tn"] else 0.0
    enriched["fnr"] = float(confusion["fn"] / (confusion["fn"] + confusion["tp"])) if confusion["fn"] + confusion["tp"] else 0.0
    return enriched


def run(args: argparse.Namespace) -> dict[str, Any]:
    runtime_config = yaml.safe_load(Path(args.runtime_config).read_text(encoding="utf-8"))
    full_oof = _load_json(args.full_oof_cache)
    roi_oof = _load_json(args.roi_oof_cache)
    area_cache = _load_json(args.area_cache)
    _validate_order(full_oof["views"], view_names=PAIR_VIEWS)
    _validate_order(roi_oof["views"], view_names=PAIR_VIEWS)

    y_true = np.asarray(
        [
            1 if str(row["pathology_label"]).lower() == "malignant" else 0
            for row in full_oof["views"]["eff_identity"]
        ],
        dtype=np.int32,
    )
    fold_ids = np.asarray(
        [int(row["fold_id"]) for row in full_oof["views"]["eff_identity"]],
        dtype=np.int32,
    )
    area_ratios = np.asarray(area_cache["oof_lcc_area_ratio"], dtype=np.float64)
    if len(area_ratios) != len(y_true):
        raise ValueError("OOF area ratio length mismatch.")

    full_blend = _pair_blend(full_oof["views"])
    roi_blend = _pair_blend(roi_oof["views"])
    baseline_oof_probabilities = _runtime_stack(
        full_blend,
        roi_blend,
        runtime_config=runtime_config,
    )
    baseline_oof_metrics = _normalise_baseline_metrics(
        _metrics(y_true, baseline_oof_probabilities, threshold=0.55),
        y_true,
    )
    feature_map = _base_feature_map(
        full_views=full_oof["views"],
        roi_views=roi_oof["views"],
        area_ratios=area_ratios,
        runtime_config=runtime_config,
    )

    candidate_rows: list[dict[str, Any]] = []
    for spec in _candidate_specs():
        X, feature_names = _feature_matrix(feature_map, spec.feature_set)
        probabilities = _nested_oof_predict(spec, X, y_true, fold_ids)
        metrics = _best_threshold(
            y_true,
            probabilities,
            min_sensitivity=float(args.min_sensitivity),
        )
        accepted = (
            metrics["auc"] >= baseline_oof_metrics["auc"] - float(args.max_auc_drop)
            and metrics["sensitivity"] >= float(args.min_sensitivity)
            and metrics["precision"] > baseline_oof_metrics["precision"]
            and metrics["f1_score"] > baseline_oof_metrics["f1_score"]
        )
        candidate_rows.append(
            {
                "name": spec.name,
                "model_family": spec.model_family,
                "feature_set": spec.feature_set,
                "feature_names": feature_names,
                "params": spec.params,
                "metrics": metrics,
                "accepted_by_oof_protocol": bool(accepted),
            }
        )
    candidate_rows.sort(
        key=lambda row: (
            row["accepted_by_oof_protocol"],
            row["metrics"]["f1_score"],
            row["metrics"]["precision"],
            row["metrics"]["auc"],
            row["metrics"]["sensitivity"],
        ),
        reverse=True,
    )
    selected = next((row for row in candidate_rows if row["accepted_by_oof_protocol"]), candidate_rows[0])
    selected_spec = next(spec for spec in _candidate_specs() if spec.name == selected["name"])
    selected_X, _ = _feature_matrix(feature_map, selected_spec.feature_set)
    final_model = _fit_final(selected_spec, selected_X, y_true)

    busi_reference, busi_y_true, busi_full_views = _load_busi_full_views()
    busi_roi = _load_json(args.busi_roi_cache)
    busi_roi_views = busi_roi["views"]
    if [str(row["sample_id"]) for row in busi_roi_views["eff_identity"]] != [
        row["sample_id"] for row in busi_reference
    ]:
        raise ValueError("BUSI ROI cache order does not match full reports.")
    busi_area = np.asarray(busi_roi["roi_area_ratios"], dtype=np.float64)
    busi_feature_map = _base_feature_map(
        full_views=busi_full_views,
        roi_views=busi_roi_views,
        area_ratios=busi_area,
        runtime_config=runtime_config,
    )
    busi_X, _ = _feature_matrix(busi_feature_map, selected_spec.feature_set)
    busi_probabilities = final_model.predict_proba(busi_X)[:, 1]
    busi_metrics = _normalise_baseline_metrics(
        _metrics(
            busi_y_true,
            busi_probabilities,
            threshold=float(selected["metrics"]["threshold"]),
        ),
        busi_y_true,
    )
    baseline_busi_metrics = _normalise_baseline_metrics(
        _load_json(args.baseline_busi_report)["metrics"],
        busi_y_true,
    )
    recommendation = (
        "keep"
        if (
            busi_metrics["auc"] >= baseline_busi_metrics["auc"]
            and busi_metrics["precision"] > baseline_busi_metrics["precision"]
            and busi_metrics["f1_score"] > baseline_busi_metrics["f1_score"]
            and busi_metrics["sensitivity"] >= baseline_busi_metrics["sensitivity"] - 0.02
        )
        else "abandon"
    )
    report = {
        "method": "nested OOF meta-learner selected on BUSBRA OOF only",
        "data_boundary": {
            "selection": "BUSBRA OOF predictions, labels, fold_id, and ROI area ratios only",
            "external_review": "BUSI used once after candidate and threshold are fixed",
        },
        "feature_sets": {name: list(values) for name, values in FEATURE_SETS.items()},
        "baseline_oof_metrics": baseline_oof_metrics,
        "selected_candidate": selected,
        "top_candidates": candidate_rows[:10],
        "baseline_busi_metrics": baseline_busi_metrics,
        "busi_metrics": busi_metrics,
        "recommendation": recommendation,
        "busi_rows": [
            {
                "sample_id": row["sample_id"],
                "pathology_label": row["pathology_label"],
                "malignant_probability": float(probability),
            }
            for row, probability in zip(busi_reference, busi_probabilities)
        ],
        "notes": [
            "Nested OOF is used to avoid evaluating the meta-learner on samples it has trained on.",
            "All listed external metrics use the OOF-selected threshold without BUSI threshold tuning.",
        ],
    }
    write_json_report(args.output, report)
    write_markdown_report(args.markdown, build_markdown(report))
    return report


def main() -> int:
    args = build_parser().parse_args()
    report = run(args)
    print(
        {
            "recommendation": report["recommendation"],
            "selected": report["selected_candidate"]["name"],
            "oof_auc": report["selected_candidate"]["metrics"]["auc"],
            "oof_precision": report["selected_candidate"]["metrics"]["precision"],
            "oof_f1": report["selected_candidate"]["metrics"]["f1_score"],
            "busi_auc": report["busi_metrics"]["auc"],
            "busi_precision": report["busi_metrics"]["precision"],
            "busi_f1": report["busi_metrics"]["f1_score"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
