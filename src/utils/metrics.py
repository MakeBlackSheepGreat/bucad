"""Classification and segmentation metrics used by training and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.utils.runtime import optional_import


cv2 = optional_import("cv2")
scipy_ndimage = optional_import("scipy.ndimage")


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
    precision = safe_divide(confusion["tp"], confusion["tp"] + confusion["fp"])
    specificity = safe_divide(confusion["tn"], confusion["tn"] + confusion["fp"])
    accuracy = safe_divide(
        confusion["tp"] + confusion["tn"],
        confusion["tp"] + confusion["tn"] + confusion["fp"] + confusion["fn"],
    )
    f1_score = safe_divide(2.0 * precision * sensitivity, precision + sensitivity)
    return {
        "auc": auc,
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "precision": precision,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "confusion": confusion,
    }


def threshold_sweep(
    y_true: list[int] | np.ndarray,
    malignant_probabilities: list[float] | np.ndarray,
    *,
    thresholds: list[float] | np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if thresholds is None:
        thresholds = np.round(np.arange(0.1, 0.9001, 0.01), 2)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = classification_metrics(
            y_true,
            malignant_probabilities,
            threshold=float(threshold),
        )
        rows.append(
            {
                **metrics,
                "youden_j": float(metrics["sensitivity"] + metrics["specificity"] - 1.0),
            }
        )
    return rows


def best_threshold_by_youden(
    y_true: list[int] | np.ndarray,
    malignant_probabilities: list[float] | np.ndarray,
    *,
    thresholds: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    rows = threshold_sweep(y_true, malignant_probabilities, thresholds=thresholds)
    if not rows:
        return {}
    return max(rows, key=lambda row: (row["youden_j"], row["sensitivity"], row["specificity"]))


def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred_mask.astype(np.float32) > 0.5
    true = true_mask.astype(np.float32) > 0.5
    intersection = float(np.logical_and(pred, true).sum())
    return (2.0 * intersection + eps) / (float(pred.sum()) + float(true.sum()) + eps)


def iou_score(pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred_mask.astype(np.float32) > 0.5
    true = true_mask.astype(np.float32) > 0.5
    intersection = float(np.logical_and(pred, true).sum())
    union = float(np.logical_or(pred, true).sum())
    return (intersection + eps) / (union + eps)


def _binary_erosion(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(bool)
    if cv2 is not None:
        kernel = np.ones((3, 3), dtype=np.uint8)
        return cv2.erode(binary.astype(np.uint8), kernel, iterations=1).astype(bool)
    if scipy_ndimage is not None:
        return scipy_ndimage.binary_erosion(binary, structure=np.ones((3, 3), dtype=bool))
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    eroded = np.ones(binary.shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy:dy + binary.shape[0], dx:dx + binary.shape[1]]
    return eroded


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.float32) > 0.5
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    return np.logical_xor(binary, _binary_erosion(binary))


def boundary_f1_score(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    *,
    tolerance: int = 2,
    eps: float = 1e-6,
) -> float:
    pred_boundary = mask_boundary(pred_mask)
    true_boundary = mask_boundary(true_mask)
    if not pred_boundary.any() and not true_boundary.any():
        return 1.0
    if not pred_boundary.any() or not true_boundary.any():
        return 0.0
    if cv2 is not None:
        kernel_size = max(1, int(tolerance) * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        pred_dilated = cv2.dilate(pred_boundary.astype(np.uint8), kernel, iterations=1).astype(bool)
        true_dilated = cv2.dilate(true_boundary.astype(np.uint8), kernel, iterations=1).astype(bool)
    elif scipy_ndimage is not None:
        structure = np.ones((max(1, int(tolerance) * 2 + 1),) * 2, dtype=bool)
        pred_dilated = scipy_ndimage.binary_dilation(pred_boundary, structure=structure)
        true_dilated = scipy_ndimage.binary_dilation(true_boundary, structure=structure)
    else:
        pred_dilated = pred_boundary
        true_dilated = true_boundary
    precision = safe_divide(float(np.logical_and(pred_boundary, true_dilated).sum()), float(pred_boundary.sum()))
    recall = safe_divide(float(np.logical_and(true_boundary, pred_dilated).sum()), float(true_boundary.sum()))
    return safe_divide(2.0 * precision * recall + eps, precision + recall + eps)


def _surface_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_boundary = mask_boundary(source)
    target_boundary = mask_boundary(target)
    if not source_boundary.any() or not target_boundary.any():
        return np.asarray([], dtype=np.float64)
    if scipy_ndimage is not None:
        distance_map = scipy_ndimage.distance_transform_edt(~target_boundary)
        return distance_map[source_boundary].astype(np.float64)
    source_points = np.argwhere(source_boundary)
    target_points = np.argwhere(target_boundary)
    max_points = 5000
    if len(source_points) > max_points:
        source_points = source_points[np.linspace(0, len(source_points) - 1, max_points).astype(int)]
    if len(target_points) > max_points:
        target_points = target_points[np.linspace(0, len(target_points) - 1, max_points).astype(int)]
    distances = []
    for point in source_points:
        delta = target_points - point
        distances.append(float(np.sqrt((delta * delta).sum(axis=1)).min()))
    return np.asarray(distances, dtype=np.float64)


def hd95_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    pred = pred_mask.astype(np.float32) > 0.5
    true = true_mask.astype(np.float32) > 0.5
    if not pred.any() and not true.any():
        return 0.0
    if not pred.any() or not true.any():
        return float("inf")
    distances = np.concatenate([_surface_distances(pred, true), _surface_distances(true, pred)])
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))
