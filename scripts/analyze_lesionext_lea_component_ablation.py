"""Build reproducible component-ablation evidence for the LesioNeXt-LEA manuscript.

The analysis only reads frozen BUSBRA out-of-fold predictions. It verifies that
all variants contain the same samples and case-level fold assignments, then
reports pooled metrics, fold summaries, and case-cluster paired bootstrap
intervals for the LesioNeXt-LEA versus comparator contrasts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_NAMES = ("auc", "accuracy", "sensitivity", "specificity", "precision", "f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "reports" / "paper_evidence",
        help="Directory for JSON, CSV, and Markdown outputs.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=10_000,
        help="Number of case-cluster paired bootstrap resamples.",
    )
    parser.add_argument("--seed", type=int, default=20_260_811)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected non-empty OOF rows in {path}.")
    return rows


def labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["pathology_label"] == "malignant") for row in rows], dtype=np.int8)


def probabilities(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["malignant_probability"]) for row in rows], dtype=np.float64)


def metric_values(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability >= 0.5
    positive = y_true == 1
    negative = ~positive
    true_negative = int(np.sum(~prediction & negative))
    false_positive = int(np.sum(prediction & negative))
    return {
        "auc": float(roc_auc_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(true_negative / max(1, true_negative + false_positive)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
    }


def sample_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["sample_id"]),
        str(row["case_id"]),
        str(row["pathology_label"]),
        int(row["fold_id"]),
    )


def validate_alignment(reference: list[dict[str, Any]], candidate: list[dict[str, Any]], name: str) -> None:
    reference_keys = {sample_key(row) for row in reference}
    candidate_keys = {sample_key(row) for row in candidate}
    if reference_keys != candidate_keys:
        missing = sorted(reference_keys - candidate_keys)[:3]
        extra = sorted(candidate_keys - reference_keys)[:3]
        raise ValueError(
            f"OOF sample or fold mismatch for {name}. Missing examples: {missing}; extra examples: {extra}."
        )


def order_like_reference(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_by_key = {sample_key(row): row for row in candidate}
    return [candidate_by_key[sample_key(row)] for row in reference]


def fold_summary(rows: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    values: list[dict[str, float | int]] = []
    for fold_id in range(1, 6):
        subset = [row for row in rows if int(row["fold_id"]) == fold_id]
        fold_metrics = metric_values(labels(subset), probabilities(subset))
        values.append(
            {
                "fold": fold_id,
                "image_count": len(subset),
                "case_count": len({str(row["case_id"]) for row in subset}),
                **fold_metrics,
            }
        )
    return values


def mean_sd(rows: list[dict[str, float | int]]) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": float(np.mean([float(row[metric]) for row in rows])),
            "sd": float(np.std([float(row[metric]) for row in rows], ddof=1)),
        }
        for metric in METRIC_NAMES
    }


def case_groups(rows: list[dict[str, Any]]) -> list[np.ndarray]:
    mapping: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        mapping.setdefault(str(row["case_id"]), []).append(index)
    return [np.asarray(mapping[case_id], dtype=np.int64) for case_id in sorted(mapping)]


def bootstrap_difference(
    y_true: np.ndarray,
    method_probability: np.ndarray,
    comparator_probability: np.ndarray,
    groups: list[np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    observed_method = metric_values(y_true, method_probability)
    observed_comparator = metric_values(y_true, comparator_probability)
    observed = {metric: observed_method[metric] - observed_comparator[metric] for metric in METRIC_NAMES}
    random = np.random.default_rng(seed)
    draws = {metric: np.empty(replicates, dtype=np.float64) for metric in METRIC_NAMES}

    for replicate in range(replicates):
        sampled_group_indices = random.integers(0, len(groups), size=len(groups))
        sampled_indices = np.concatenate([groups[index] for index in sampled_group_indices])
        method_metrics = metric_values(y_true[sampled_indices], method_probability[sampled_indices])
        comparator_metrics = metric_values(y_true[sampled_indices], comparator_probability[sampled_indices])
        for metric in METRIC_NAMES:
            draws[metric][replicate] = method_metrics[metric] - comparator_metrics[metric]

    return {
        metric: {
            "observed_delta": float(observed[metric]),
            "ci95_lower": float(np.quantile(draws[metric], 0.025)),
            "ci95_upper": float(np.quantile(draws[metric], 0.975)),
        }
        for metric in METRIC_NAMES
    }


def write_csv(path: Path, metrics_by_method: dict[str, dict[str, float]], fold_stats: dict[str, dict[str, dict[str, float]]]) -> None:
    fields = ["method", "metric", "pooled_oof", "fold_mean", "fold_sd"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, metrics in metrics_by_method.items():
            for metric in METRIC_NAMES:
                writer.writerow(
                    {
                        "method": method,
                        "metric": metric,
                        "pooled_oof": metrics[metric],
                        "fold_mean": fold_stats[method][metric]["mean"],
                        "fold_sd": fold_stats[method][metric]["sd"],
                    }
                )


def write_differences_csv(path: Path, differences: dict[str, dict[str, dict[str, float]]]) -> None:
    fields = ["comparison", "metric", "observed_delta", "ci95_lower", "ci95_upper"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for comparison, comparison_metrics in differences.items():
            for metric, values in comparison_metrics.items():
                writer.writerow({"comparison": comparison, "metric": metric, **values})


def markdown_report(report: dict[str, Any]) -> str:
    methods = report["component_definitions"]
    pooled = report["pooled_metrics"]
    difference = report["case_cluster_bootstrap_differences"]
    lines = [
        "# LesioNeXt-LEA Component Ablation",
        "",
        "## Scope",
        "",
        "- Data: frozen BUSBRA image-level pooled OOF predictions from the fixed case-level five-fold split.",
        "- Threshold: 0.50 for Accuracy, Sensitivity, Specificity, Precision, and F1.",
        "- Uncertainty: paired case-cluster percentile bootstrap. Each resampled case retained all of its images.",
        f"- Cases: `{report['case_count']}`; images: `{report['image_count']}`; bootstrap replicates: `{report['bootstrap_replicates']}`.",
        "- External cohorts were not read.",
        "",
        "## Component Definitions",
        "",
        "| Variant | Evidence head | BBOX alignment loss | Local feature interpolation |",
        "| --- | --- | --- | --- |",
    ]
    for method, definition in methods.items():
        lines.append(
            f"| {method} | {definition['evidence_head']} | {definition['bbox_alignment']} | {definition['local_interpolation']} |"
        )
    lines.extend(
        [
            "",
            "## Pooled OOF Metrics",
            "",
            "| Variant | AUC | Accuracy | Sensitivity | Specificity | Precision | F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method, values in pooled.items():
        lines.append(
            "| "
            + method
            + " | "
            + " | ".join(f"{values[metric]:.4f}" for metric in METRIC_NAMES)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Paired Case-Cluster Bootstrap Differences",
            "",
            "All contrasts are LesioNeXt-LEA minus the listed comparator. Intervals quantify precision and do not define model selection.",
            "",
            "| Comparator | Metric | Delta | 95% CI |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for comparison, values_by_metric in difference.items():
        for metric in METRIC_NAMES:
            values = values_by_metric[metric]
            lines.append(
                f"| {comparison} | {metric} | {values['observed_delta']:+.4f} | [{values['ci95_lower']:+.4f}, {values['ci95_upper']:+.4f}] |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "ConvNeXt-Tiny": PROJECT_ROOT / "artifacts" / "reports" / "fixed_classification_benchmark" / "oof" / "convnext_tiny.json",
        "Evidence head only": PROJECT_ROOT / "artifacts" / "reports" / "lesionext_lens_v1a_no_alignment_control_5fold_oof.json",
        "LesioNeXt-LEA": PROJECT_ROOT / "artifacts" / "reports" / "lesionext_lens_v1a_evidence_only_5fold_oof.json",
        "LEA + local interpolation": PROJECT_ROOT / "artifacts" / "reports" / "lesionext_lens_v1_5fold_oof.json",
    }
    component_definitions = {
        "ConvNeXt-Tiny": {
            "evidence_head": "No",
            "bbox_alignment": "No",
            "local_interpolation": "No",
        },
        "Evidence head only": {
            "evidence_head": "Yes",
            "bbox_alignment": "No",
            "local_interpolation": "No",
        },
        "LesioNeXt-LEA": {
            "evidence_head": "Yes",
            "bbox_alignment": "Yes",
            "local_interpolation": "No (alpha = 0)",
        },
        "LEA + local interpolation": {
            "evidence_head": "Yes",
            "bbox_alignment": "Yes",
            "local_interpolation": "Yes (learned static alpha)",
        },
    }
    rows_by_method = {name: load_rows(path) for name, path in source_paths.items()}
    reference_rows = rows_by_method["LesioNeXt-LEA"]
    for name, rows in rows_by_method.items():
        validate_alignment(reference_rows, rows, name)
        rows_by_method[name] = order_like_reference(reference_rows, rows)

    y_true = labels(reference_rows)
    probabilities_by_method = {name: probabilities(rows) for name, rows in rows_by_method.items()}
    pooled_metrics = {
        name: metric_values(y_true, probability)
        for name, probability in probabilities_by_method.items()
    }
    fold_metrics = {name: fold_summary(rows) for name, rows in rows_by_method.items()}
    fold_mean_sd = {name: mean_sd(values) for name, values in fold_metrics.items()}
    groups = case_groups(reference_rows)
    comparators = ("ConvNeXt-Tiny", "Evidence head only", "LEA + local interpolation")
    differences = {
        comparator: bootstrap_difference(
            y_true,
            probabilities_by_method["LesioNeXt-LEA"],
            probabilities_by_method[comparator],
            groups,
            replicates=args.bootstrap_replicates,
            seed=args.seed + index,
        )
        for index, comparator in enumerate(comparators)
    }
    report = {
        "scope": "Frozen BUSBRA five-fold component ablation for the LesioNeXt-LEA manuscript",
        "analysis_unit": {
            "primary_metrics": "image-level pooled OOF predictions",
            "split_unit": "case-level five-fold assignment",
            "threshold": 0.5,
            "uncertainty": "paired case-cluster percentile bootstrap retaining all images from every sampled case",
        },
        "component_definitions": component_definitions,
        "source_files": {name: str(path.relative_to(PROJECT_ROOT)) for name, path in source_paths.items()},
        "image_count": len(reference_rows),
        "case_count": len(groups),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "pooled_metrics": pooled_metrics,
        "fold_metrics": fold_metrics,
        "fold_mean_sd": fold_mean_sd,
        "case_cluster_bootstrap_differences": differences,
    }
    json_path = output_dir / "lesionext_lea_component_ablation.json"
    metrics_csv_path = output_dir / "lesionext_lea_component_ablation_metrics.csv"
    differences_csv_path = output_dir / "lesionext_lea_component_ablation_differences.csv"
    markdown_path = output_dir / "lesionext_lea_component_ablation.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(metrics_csv_path, pooled_metrics, fold_mean_sd)
    write_differences_csv(differences_csv_path, differences)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "metrics_csv": str(metrics_csv_path), "differences_csv": str(differences_csv_path), "markdown": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
