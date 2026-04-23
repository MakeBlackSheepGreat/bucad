from __future__ import annotations

import numpy as np

from src.utils.metrics import best_threshold_by_youden, classification_metrics, dice_score, threshold_sweep


def test_classification_metrics_exposes_auc_and_confusion() -> None:
    metrics = classification_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert metrics["auc"] is not None
    assert metrics["confusion"]["tp"] == 2


def test_dice_score_is_one_for_identical_masks() -> None:
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[4:8, 4:8] = 1.0
    assert abs(dice_score(mask, mask) - 1.0) < 1e-6


def test_threshold_sweep_reports_youden_scores() -> None:
    rows = threshold_sweep([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9], thresholds=[0.3, 0.5, 0.7])
    best = best_threshold_by_youden([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9], thresholds=[0.3, 0.5, 0.7])

    assert len(rows) == 3
    assert all("youden_j" in row for row in rows)
    assert best["threshold"] == 0.5
