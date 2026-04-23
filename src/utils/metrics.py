from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def normalize_binary_probs(benign: float, malignant: float) -> tuple[float, float]:
    total = benign + malignant
    if total <= 0:
        return 0.5, 0.5
    benign /= total
    malignant /= total
    return float(benign), float(malignant)


def confusion_summary(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def classification_metrics(
    y_true: list[int] | np.ndarray,
    malignant_probabilities: list[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    prob_arr = np.asarray(malignant_probabilities, dtype=np.float32)
    preds = (prob_arr >= threshold).astype(np.int32)
    confusion = confusion_summary(y_true_arr, preds)
    auc = None
    if len(np.unique(y_true_arr)) > 1:
        auc = float(roc_auc_score(y_true_arr, prob_arr))
    sensitivity = safe_divide(confusion["tp"], confusion["tp"] + confusion["fn"])
    specificity = safe_divide(confusion["tn"], confusion["tn"] + confusion["fp"])
    accuracy = safe_divide(
        confusion["tp"] + confusion["tn"],
        confusion["tp"] + confusion["tn"] + confusion["fp"] + confusion["fn"],
    )
    return {
        "auc": auc,
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "confusion": confusion,
    }


def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred_mask.astype(np.float32) > 0.5
    true = true_mask.astype(np.float32) > 0.5
    intersection = float(np.logical_and(pred, true).sum())
    return (2.0 * intersection + eps) / (float(pred.sum()) + float(true.sum()) + eps)
