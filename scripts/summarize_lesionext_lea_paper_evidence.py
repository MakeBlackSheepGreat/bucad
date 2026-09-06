"""Summarize frozen OOF and evidence-alignment results for the manuscript."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "reports" / "paper_evidence",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_810)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected non-empty rows in {path}")
    return rows


def label_vector(rows: list[dict]) -> np.ndarray:
    return np.array([int(row["pathology_label"] == "malignant") for row in rows], dtype=int)


def probability_vector(rows: list[dict]) -> np.ndarray:
    return np.array([float(row["malignant_probability"]) for row in rows], dtype=float)


def classification_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability >= 0.5
    positive = y_true == 1
    negative = ~positive
    true_positive = int(np.sum(prediction & positive))
    true_negative = int(np.sum(~prediction & negative))
    false_positive = int(np.sum(prediction & negative))
    false_negative = int(np.sum(~prediction & positive))
    return {
        "auc": float(roc_auc_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(true_negative / (true_negative + false_positive)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
    }


def validate_alignment(reference_rows: list[dict], candidate_rows: list[dict], name: str) -> None:
    reference_index = {(row["sample_id"], row["case_id"]): row for row in reference_rows}
    candidate_index = {(row["sample_id"], row["case_id"]): row for row in candidate_rows}
    if reference_index.keys() != candidate_index.keys():
        raise ValueError(f"OOF sample identity mismatch for {name}")
    for key, reference in reference_index.items():
        candidate = candidate_index[key]
        for field in ("pathology_label", "fold_id"):
            if reference[field] != candidate[field]:
                raise ValueError(f"OOF {field} mismatch for {name}: {key}")


def fold_metrics(rows: list[dict]) -> list[dict[str, float | int]]:
    output: list[dict[str, float | int]] = []
    for fold in range(1, 6):
        subset = [row for row in rows if int(row["fold_id"]) == fold]
        values = classification_metrics(label_vector(subset), probability_vector(subset))
        output.append(
            {
                "fold": fold,
                "image_count": len(subset),
                "case_count": len({str(row["case_id"]) for row in subset}),
                **values,
            }
        )
    return output


def mean_and_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(array)), "sd": float(np.std(array, ddof=1))}


def cluster_bootstrap_auc_difference(
    rows: list[dict],
    method_probability: np.ndarray,
    comparator_probability: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    y_true = label_vector(rows)
    case_to_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        case_to_indices.setdefault(str(row["case_id"]), []).append(index)
    case_ids = np.array(sorted(case_to_indices), dtype=object)
    index_groups = [np.asarray(case_to_indices[case_id], dtype=int) for case_id in case_ids]
    observed = float(roc_auc_score(y_true, method_probability) - roc_auc_score(y_true, comparator_probability))
    random = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        group_indices = random.integers(0, len(index_groups), size=len(index_groups))
        sampled = np.concatenate([index_groups[index] for index in group_indices])
        differences[replicate] = roc_auc_score(y_true[sampled], method_probability[sampled]) - roc_auc_score(
            y_true[sampled], comparator_probability[sampled]
        )
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "observed_delta_auc": observed,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "case_count": int(len(case_ids)),
        "image_count": int(len(rows)),
        "bootstrap_replicates": int(replicates),
        "seed": int(seed),
    }


def diagnostics_summary(output_dir: Path, prefix: str) -> dict[str, object]:
    reports = []
    for fold in range(1, 6):
        path = output_dir / f"{prefix}_fold{fold}_diagnostics.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        reports.append(report)
    mass = [float(report["mean_bbox_evidence_mass"]) for report in reports]
    p95_latency = [float(report["single_image_latency_ms_p95"]) for report in reports]
    return {
        "fold_count": len(reports),
        "bbox_valid_ratio": mean_and_sd([float(report["bbox_valid_ratio"]) for report in reports]),
        "bbox_evidence_mass": mean_and_sd(mass),
        "background_evidence_mass": mean_and_sd(
            [float(report["mean_background_evidence_mass"]) for report in reports]
        ),
        "evidence_alpha": mean_and_sd([float(report["mean_evidence_alpha"]) for report in reports]),
        "parameter_count": int(reports[0]["parameter_count"]),
        "batch_normalized_p95_latency_ms": mean_and_sd(p95_latency),
        "per_fold": [
            {
                "fold": int(report["fold"]),
                "bbox_evidence_mass": float(report["mean_bbox_evidence_mass"]),
                "batch_normalized_p95_latency_ms": float(report["single_image_latency_ms_p95"]),
            }
            for report in reports
        ],
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "LesioNeXt-LEA": PROJECT_ROOT
        / "artifacts"
        / "reports"
        / "lesionext_lens_v1a_evidence_only_5fold_oof.json",
        "ConvNeXt-Tiny": PROJECT_ROOT
        / "artifacts"
        / "reports"
        / "fixed_classification_benchmark"
        / "oof"
        / "convnext_tiny.json",
        "Without evidence alignment": PROJECT_ROOT
        / "artifacts"
        / "reports"
        / "lesionext_lens_v1a_no_alignment_control_5fold_oof.json",
    }
    rows_by_model = {name: load_rows(path) for name, path in sources.items()}
    reference_rows = rows_by_model["LesioNeXt-LEA"]
    for name, rows in rows_by_model.items():
        validate_alignment(reference_rows, rows, name)

    fold_results = {name: fold_metrics(rows) for name, rows in rows_by_model.items()}
    fold_summary = {}
    for name, rows in fold_results.items():
        fold_summary[name] = {
            metric: mean_and_sd([float(row[metric]) for row in rows])
            for metric in ("auc", "accuracy", "sensitivity", "specificity", "precision", "f1")
        }
    probabilities = {name: probability_vector(rows) for name, rows in rows_by_model.items()}
    bootstrap = {
        "vs_convnext_tiny": cluster_bootstrap_auc_difference(
            reference_rows,
            probabilities["LesioNeXt-LEA"],
            probabilities["ConvNeXt-Tiny"],
            args.bootstrap_replicates,
            args.seed,
        ),
        "vs_without_evidence_alignment": cluster_bootstrap_auc_difference(
            reference_rows,
            probabilities["LesioNeXt-LEA"],
            probabilities["Without evidence alignment"],
            args.bootstrap_replicates,
            args.seed + 1,
        ),
    }
    aligned_diagnostics = diagnostics_summary(output_dir, "lesionext_lea")
    unaligned_diagnostics = diagnostics_summary(output_dir, "unaligned_control")
    report = {
        "scope": "Frozen BUSBRA OOF manuscript evidence summary",
        "analysis_unit": {
            "primary_metrics": "image-level pooled OOF predictions",
            "split_unit": "case-level five-fold assignment",
            "uncertainty": "case-cluster bootstrap retaining all images from each sampled case",
            "threshold": 0.5,
        },
        "source_files": {name: str(path.relative_to(PROJECT_ROOT)) for name, path in sources.items()},
        "fold_metrics": fold_results,
        "fold_mean_sd": fold_summary,
        "case_cluster_bootstrap_auc_difference": bootstrap,
        "evidence_alignment_diagnostics": {
            "LesioNeXt-LEA": aligned_diagnostics,
            "Without evidence alignment": unaligned_diagnostics,
            "mean_foldwise_evidence_mass_difference": float(
                aligned_diagnostics["bbox_evidence_mass"]["mean"]
                - unaligned_diagnostics["bbox_evidence_mass"]["mean"]
            ),
        },
    }
    json_path = output_dir / "lesionext_lea_paper_evidence_summary.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = output_dir / "lesionext_lea_fold_metrics.csv"
    fields = [
        "model",
        "fold",
        "image_count",
        "case_count",
        "auc",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_name, rows in fold_results.items():
            for row in rows:
                writer.writerow({"model": model_name, **row})
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
