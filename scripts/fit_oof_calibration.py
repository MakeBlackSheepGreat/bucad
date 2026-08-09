"""Fit BUSBRA-only nested logit calibration and apply it to locked predictions."""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import classification_metrics


FIXED_THRESHOLD = 0.50
LABEL_MAP = {"benign": 0, "malignant": 1}


def _logit(probabilities: np.ndarray) -> np.ndarray:
    """Convert probabilities to numerically bounded logits."""
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(values) - np.log1p(-values)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Convert logits to numerically bounded probabilities."""
    values = np.clip(np.asarray(logits, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def _nll(parameters: np.ndarray, logits: np.ndarray, labels: np.ndarray, *, mode: str) -> float:
    """Return the binary negative log likelihood for a calibration parameterization."""
    if mode == "bias":
        temperature, bias = 1.0, float(parameters[0])
    else:
        temperature, bias = math.exp(float(parameters[0])), float(parameters[1])
    calibrated = _sigmoid(logits / temperature + bias)
    return float(-np.mean(labels * np.log(np.clip(calibrated, 1e-7, 1.0)) + (1 - labels) * np.log(np.clip(1 - calibrated, 1e-7, 1.0))))


def fit_calibrator(probabilities: np.ndarray, labels: np.ndarray, *, mode: str) -> dict[str, float]:
    """Fit a bounded bias or temperature-plus-bias calibrator."""
    logits = _logit(probabilities)
    if mode == "bias":
        objective = partial(_nll, logits=logits, labels=labels, mode=mode)
        result = minimize(objective, np.zeros(1), bounds=[(-2.0, 2.0)], method="L-BFGS-B")
        return {"temperature": 1.0, "bias": float(result.x[0]), "nll": float(result.fun)}
    if mode == "tempbias":
        objective = partial(_nll, logits=logits, labels=labels, mode=mode)
        result = minimize(objective, np.zeros(2), bounds=[(math.log(0.5), math.log(3.0)), (-2.0, 2.0)], method="L-BFGS-B")
        return {"temperature": float(math.exp(result.x[0])), "bias": float(result.x[1]), "nll": float(result.fun)}
    raise ValueError(f"Unknown calibration mode: {mode}")


def apply_calibrator(probabilities: np.ndarray, calibrator: dict[str, float]) -> np.ndarray:
    """Apply a fitted calibrator to malignant probabilities."""
    return _sigmoid(_logit(probabilities) / float(calibrator["temperature"]) + float(calibrator["bias"]))


def _rows_from_oof(report_path: Path) -> pd.DataFrame:
    """Load OOF rows and map benign/malignant labels to binary targets."""
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not rows:
        raise ValueError(f"OOF report has no rows: {report_path}")
    frame = pd.DataFrame(rows)
    frame["y_true"] = frame["pathology_label"].str.lower().map(LABEL_MAP)
    if frame["y_true"].isna().any() or frame["fold_id"].isna().any():
        raise ValueError("OOF rows require benign/malignant labels and fold_id.")
    return frame


def nested_oof_calibration(report_path: Path, *, mode: str) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    """Fit each fold calibrator without using that fold's validation rows."""
    frame = _rows_from_oof(report_path)
    calibrated = np.zeros(len(frame), dtype=np.float64)
    fold_calibrators: list[dict[str, float]] = []
    for fold in sorted(frame["fold_id"].astype(int).unique()):
        train = frame[frame["fold_id"].astype(int) != fold]
        valid = frame[frame["fold_id"].astype(int) == fold]
        calibrator = fit_calibrator(train["malignant_probability"].to_numpy(), train["y_true"].to_numpy(), mode=mode)
        calibrated[valid.index.to_numpy()] = apply_calibrator(valid["malignant_probability"].to_numpy(), calibrator)
        fold_calibrators.append({"fold_id": int(fold), **calibrator})
    frame["calibrated_probability"] = calibrated
    return frame, fold_calibrators


def summarize(frame: pd.DataFrame, probabilities: str) -> dict[str, Any]:
    """Summarize calibrated probabilities at the frozen decision threshold."""
    y_true = frame["y_true"].to_numpy(dtype=np.int32)
    values = frame[probabilities].to_numpy(dtype=np.float64)
    metrics = classification_metrics(y_true, values, threshold=FIXED_THRESHOLD)
    metrics["auc"] = float(roc_auc_score(y_true, values))
    metrics["threshold"] = FIXED_THRESHOLD
    metrics["brier_score"] = float(np.mean((values - y_true) ** 2))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit BUSBRA nested logit calibration.")
    parser.add_argument("--oof-report", required=True)
    parser.add_argument("--mode", choices=("bias", "tempbias"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-csv", required=True)
    return parser


def main() -> int:
    """Run nested calibration and write JSON plus prediction CSV artifacts."""
    args = build_parser().parse_args()
    report_path = Path(args.oof_report)
    output_path = Path(args.output)
    predictions_path = Path(args.predictions_csv)
    frame, fold_calibrators = nested_oof_calibration(report_path, mode=args.mode)
    mean_calibrator = {"temperature": float(np.mean([x["temperature"] for x in fold_calibrators])), "bias": float(np.mean([x["bias"] for x in fold_calibrators]))}
    result = {
        "method": f"LesioNeXt-LENS v1a BUSBRA-nested {args.mode} calibration",
        "source_oof": str(report_path),
        "mode": args.mode,
        "fixed_threshold": FIXED_THRESHOLD,
        "fold_calibrators": fold_calibrators,
        "mean_calibrator_for_external": mean_calibrator,
        "raw_oof_metrics": summarize(frame, "malignant_probability"),
        "calibrated_oof_metrics": summarize(frame, "calibrated_probability"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    frame.to_csv(predictions_path, index=False)
    print(json.dumps({"output": str(output_path), **result["calibrated_oof_metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
