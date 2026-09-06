"""Create the LesioNeXt-LEA component-ablation figure from frozen analysis outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGURES = PROJECT_ROOT.parent / "Paper" / "LesioNeXt_LEA_LaTeX" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-json",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "reports" / "paper_evidence" / "lesionext_lea_component_ablation.json",
    )
    parser.add_argument(
        "--evidence-summary-json",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "reports" / "paper_evidence" / "lesionext_lea_paper_evidence_summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=PAPER_FIGURES)
    return parser.parse_args()


def add_panel_label(axis, label: str) -> None:
    axis.text(
        -0.16,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=8.8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def write_source_data(path: Path, analysis: dict, evidence: dict) -> None:
    fields = [
        "panel",
        "label",
        "metric",
        "value",
        "ci95_lower",
        "ci95_upper",
        "fold",
    ]
    rows: list[dict[str, object]] = []
    for method, values in analysis["pooled_metrics"].items():
        rows.append(
            {
                "panel": "a",
                "label": method,
                "metric": "pooled_oof_auc",
                "value": values["auc"],
                "ci95_lower": "",
                "ci95_upper": "",
                "fold": "",
            }
        )
    for comparator, values_by_metric in analysis["case_cluster_bootstrap_differences"].items():
        values = values_by_metric["auc"]
        rows.append(
            {
                "panel": "b",
                "label": comparator,
                "metric": "lea_minus_comparator_auc",
                "value": values["observed_delta"],
                "ci95_lower": values["ci95_lower"],
                "ci95_upper": values["ci95_upper"],
                "fold": "",
            }
        )
    evidence_labels = (("LesioNeXt-LEA", "LesioNeXt-LEA"), ("Evidence head only", "Without evidence alignment"))
    for display_name, evidence_name in evidence_labels:
        for fold in evidence["evidence_alignment_diagnostics"][evidence_name]["per_fold"]:
            rows.append(
                {
                    "panel": "c",
                    "label": display_name,
                    "metric": "bbox_evidence_mass",
                    "value": fold["bbox_evidence_mass"],
                    "ci95_lower": "",
                    "ci95_upper": "",
                    "fold": fold["fold"],
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    analysis = json.loads(args.analysis_json.read_text(encoding="utf-8"))
    evidence = json.loads(args.evidence_summary_json.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    plt.rcParams["font.size"] = 7.2
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    colors = {
        "ConvNeXt-Tiny": "#6F7782",
        "Evidence head only": "#8BA6C8",
        "LesioNeXt-LEA": "#0F4D92",
        "LEA + local interpolation": "#C98583",
    }
    labels = {
        "ConvNeXt-Tiny": "ConvNeXt\nTiny",
        "Evidence head only": "Evidence head\nonly",
        "LesioNeXt-LEA": "LesioNeXt\nLEA",
        "LEA + local interpolation": "LEA + local\ninterpolation",
    }
    order = list(labels)

    figure, axes = plt.subplots(3, 1, figsize=(3.45, 6.15), gridspec_kw={"height_ratios": [1.15, 1.0, 1.05]})
    figure.subplots_adjust(left=0.25, right=0.97, top=0.97, bottom=0.08, hspace=0.8)

    axis = axes[0]
    auc_values = [analysis["pooled_metrics"][method]["auc"] for method in order]
    bars = axis.bar(np.arange(len(order)), auc_values, color=[colors[method] for method in order], width=0.68)
    axis.set_ylim(0.84, 0.95)
    axis.set_ylabel("Pooled OOF AUC")
    axis.set_xticks(np.arange(len(order)), [labels[method] for method in order])
    axis.tick_params(axis="x", labelsize=6.2, pad=2)
    axis.grid(axis="y", color="#E1E4E8", linewidth=0.55)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, auc_values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.4f}", ha="center", va="bottom", fontsize=6.2)
    add_panel_label(axis, "a")

    axis = axes[1]
    comparator_order = ["ConvNeXt-Tiny", "Evidence head only", "LEA + local interpolation"]
    y_positions = np.arange(len(comparator_order))[::-1]
    for y_position, comparator in zip(y_positions, comparator_order):
        values = analysis["case_cluster_bootstrap_differences"][comparator]["auc"]
        lower = values["ci95_lower"]
        upper = values["ci95_upper"]
        estimate = values["observed_delta"]
        axis.hlines(y_position, lower, upper, color="#0F4D92", linewidth=1.4)
        axis.plot(estimate, y_position, "o", color="#0F4D92", markersize=4.6)
        axis.text(upper + 0.0018, y_position, f"{estimate:+.4f}", va="center", ha="left", fontsize=6.3)
    axis.axvline(0.0, color="#767676", linewidth=0.8, linestyle="--")
    axis.set_yticks(y_positions, ["ConvNeXt-Tiny", "Evidence head only", "Local interpolation"])
    axis.set_xlabel("LEA minus comparator AUC")
    axis.set_xlim(-0.035, 0.060)
    axis.tick_params(axis="y", labelsize=6.4)
    axis.grid(axis="x", color="#E1E4E8", linewidth=0.55)
    axis.set_axisbelow(True)
    add_panel_label(axis, "b")

    axis = axes[2]
    fold_ids = np.arange(1, 6)
    aligned = evidence["evidence_alignment_diagnostics"]["LesioNeXt-LEA"]["per_fold"]
    unaligned = evidence["evidence_alignment_diagnostics"]["Without evidence alignment"]["per_fold"]
    aligned_values = [item["bbox_evidence_mass"] for item in aligned]
    unaligned_values = [item["bbox_evidence_mass"] for item in unaligned]
    axis.plot(fold_ids, aligned_values, marker="o", markersize=4.0, linewidth=1.5, color="#0F4D92", label="LesioNeXt-LEA")
    axis.plot(fold_ids, unaligned_values, marker="o", markersize=4.0, linewidth=1.4, color="#8BA6C8", label="Evidence head only")
    axis.set_xlim(0.75, 5.25)
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(fold_ids)
    axis.set_xlabel("Validation fold")
    axis.set_ylabel("BBOX evidence mass")
    axis.grid(axis="y", color="#E1E4E8", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", fontsize=6.0, handlelength=1.8)
    add_panel_label(axis, "c")

    stem = output_dir / "lesionext_lea_component_ablation"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    source_data = output_dir / "lesionext_lea_component_ablation_source_data.csv"
    write_source_data(source_data, analysis, evidence)
    print(json.dumps({"figure": str(stem), "source_data": str(source_data)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
