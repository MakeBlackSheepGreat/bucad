from __future__ import annotations

import numpy as np

from src.utils.metrics import classification_metrics, dice_score


def test_classification_metrics_exposes_auc_and_confusion() -> None:
    metrics = classification_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert metrics["auc"] is not None
    assert metrics["confusion"]["tp"] == 2


def test_dice_score_is_one_for_identical_masks() -> None:
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[4:8, 4:8] = 1.0
    assert abs(dice_score(mask, mask) - 1.0) < 1e-6
