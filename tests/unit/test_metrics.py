"""Unit tests for metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils import metrics as metrics_module
from src.utils.metrics import best_threshold_by_youden, classification_metrics, dice_score, threshold_sweep


def test_classification_metrics_exposes_auc_and_confusion() -> None:
    """Verify classification metrics exposes auc and confusion."""
    metrics = classification_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert metrics["auc"] is not None
    assert metrics["confusion"]["tp"] == 2
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == metrics["sensitivity"]
    assert metrics["f1_score"] == 1.0


def test_dice_score_is_one_for_identical_masks() -> None:
    """Verify dice score is one for identical masks."""
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[4:8, 4:8] = 1.0
    assert abs(dice_score(mask, mask) - 1.0) < 1e-6


def test_threshold_sweep_reports_youden_scores() -> None:
    """Verify threshold sweep reports youden scores."""
    rows = threshold_sweep([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9], thresholds=[0.3, 0.5, 0.7])
    best = best_threshold_by_youden([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9], thresholds=[0.3, 0.5, 0.7])

    assert len(rows) == 3
    assert all("youden_j" in row for row in rows)
    assert all("precision" in row for row in rows)
    assert all("f1_score" in row for row in rows)
    assert best["threshold"] == 0.5


def test_default_threshold_sweep_uses_one_percent_steps() -> None:
    """Verify default threshold sweep uses one percent steps."""
    rows = threshold_sweep([0, 1], [0.2, 0.8])

    assert rows[0]["threshold"] == 0.1
    assert rows[1]["threshold"] == 0.11
    assert rows[-1]["threshold"] == 0.9
    assert len(rows) == 81


def test_threshold_sweep_matches_direct_metrics_with_tied_probabilities() -> None:
    """Verify prefix-sum threshold results match direct classification metrics."""
    labels = [0, 1, 0, 1, 1, 0]
    probabilities = [0.2, 0.5, 0.5, 0.7, 0.7, 0.9]
    thresholds = [0.2, 0.5, 0.7, 0.9]

    rows = threshold_sweep(labels, probabilities, thresholds=thresholds)

    for threshold, row in zip(thresholds, rows, strict=True):
        direct = classification_metrics(labels, probabilities, threshold=threshold)
        assert row["confusion"] == direct["confusion"]
        assert row["sensitivity"] == pytest.approx(direct["sensitivity"])
        assert row["specificity"] == pytest.approx(direct["specificity"])
        assert row["precision"] == pytest.approx(direct["precision"])
        assert row["f1_score"] == pytest.approx(direct["f1_score"])
        assert row["auc"] == pytest.approx(direct["auc"])


def test_threshold_sweep_calculates_auc_once(monkeypatch) -> None:
    """Verify a multi-threshold sweep avoids repeated AUC sorting work."""
    calls = 0
    real_roc_auc_score = metrics_module.roc_auc_score

    def counted_roc_auc_score(*args, **kwargs):
        """Count the AUC calls delegated by the threshold sweep."""
        nonlocal calls
        calls += 1
        return real_roc_auc_score(*args, **kwargs)

    monkeypatch.setattr(metrics_module, "roc_auc_score", counted_roc_auc_score)

    metrics_module.threshold_sweep([0, 1, 0, 1], [0.1, 0.4, 0.6, 0.9], thresholds=[0.2, 0.4, 0.6])

    assert calls == 1
