"""Analyze calibration and threshold behavior from a frozen BUSBRA OOF report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import best_threshold_by_youden, classification_metrics, threshold_sweep
from src.utils.reporting import write_json_report


def _expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    """Return equal-width ECE and the reliability-bin rows."""
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(int(bins)):
        lower = index / float(bins)
        upper = (index + 1) / float(bins)
        active = (probabilities >= lower) & (
            (probabilities < upper) if index < bins - 1 else (probabilities <= upper)
        )
        count = int(active.sum())
        if count == 0:
            rows.append({"bin": index, "lower": lower, "upper": upper, "count": 0, "confidence": None, "accuracy": None})
            continue
        confidence = float(probabilities[active].mean())
        accuracy = float(y_true[active].mean())
        gap = abs(confidence - accuracy)
        ece += (count / max(1, len(y_true))) * gap
        rows.append({"bin": index, "lower": lower, "upper": upper, "count": count, "confidence": confidence, "accuracy": accuracy, "gap": gap})
    return float(ece), rows


def analyze_oof_report(report_path: str | Path, *, bins: int = 10) -> dict[str, Any]:
    """Build calibration statistics from the rows emitted by evaluate_classifier_oof."""
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not rows:
        raise ValueError("OOF report does not contain prediction rows.")
    label_map = {"benign": 0, "malignant": 1}
    y_true = np.asarray([label_map[str(row["pathology_label"]).lower()] for row in rows], dtype=np.int32)
    probabilities = np.asarray([float(row["malignant_probability"]) for row in rows], dtype=np.float64)
    ece, reliability = _expected_calibration_error(y_true, probabilities, bins=bins)
    fixed = classification_metrics(y_true, probabilities, threshold=0.50)
    youden = best_threshold_by_youden(y_true, probabilities)
    threshold_rows = threshold_sweep(y_true, probabilities)
    return {
        "source_report": str(report_path),
        "model_name": payload.get("model_name"),
        "sample_count": int(len(rows)),
        "fixed_threshold": 0.50,
        "fixed_threshold_metrics": fixed,
        "youden_threshold_metrics": youden,
        "brier_score": float(np.mean((probabilities - y_true) ** 2)),
        "expected_calibration_error": ece,
        "reliability_bins": reliability,
        "threshold_sweep": threshold_rows,
    }


def _render_markdown(result: dict[str, Any], *, language: str) -> str:
    """Render a compact bilingual calibration report."""
    fixed = result["fixed_threshold_metrics"]
    youden = result["youden_threshold_metrics"]
    if language == "zh":
        title = "# BUSBRA OOF 校准与阈值分析"
        labels = ("模型", "样本数", "固定阈值", "Youden 阈值", "ECE", "Brier score", "说明")
        note = "阈值仅用于 BUSBRA OOF 分析；外部数据集继续使用冻结的 0.50。"
    else:
        title = "# BUSBRA OOF Calibration and Threshold Analysis"
        labels = ("Model", "Samples", "Fixed threshold", "Youden threshold", "ECE", "Brier score", "Note")
        note = "Threshold analysis is restricted to BUSBRA OOF; external evaluation remains frozen at 0.50."
    return "\n".join(
        [
            title,
            "",
            f"- {labels[0]}: `{result.get('model_name')}`",
            f"- {labels[1]}: `{result['sample_count']}`",
            f"- {labels[2]} AUC / Accuracy / Sensitivity / Specificity / F1: `"
            f"{fixed['auc']:.6f}` / `{fixed['accuracy']:.6f}` / `{fixed['sensitivity']:.6f}` / "
            f"`{fixed['specificity']:.6f}` / `{fixed['f1_score']:.6f}`",
            f"- {labels[3]}: `{youden['threshold']:.2f}`; Accuracy `{youden['accuracy']:.6f}`; "
            f"Sensitivity `{youden['sensitivity']:.6f}`; Specificity `{youden['specificity']:.6f}`",
            f"- {labels[4]}: `{result['expected_calibration_error']:.6f}`",
            f"- {labels[5]}: `{result['brier_score']:.6f}`",
            "",
            f"> {note}",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Analyze calibration from a BUSBRA OOF JSON report.")
    parser.add_argument("--oof-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--markdown-zh", required=True)
    parser.add_argument("--markdown-en", required=True)
    parser.add_argument("--bins", type=int, default=10)
    return parser


def main() -> int:
    """Run calibration analysis and persist JSON, CSV, and bilingual Markdown."""
    args = build_parser().parse_args()
    result = analyze_oof_report(args.oof_report, bins=args.bins)
    write_json_report(Path(args.output), result)
    pd.DataFrame(result["threshold_sweep"]).to_csv(args.csv, index=False)
    Path(args.markdown_zh).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown_en).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown_zh).write_text(_render_markdown(result, language="zh") + "\n", encoding="utf-8")
    Path(args.markdown_en).write_text(_render_markdown(result, language="en") + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output, "ece": result["expected_calibration_error"], "brier": result["brier_score"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
