"""Unit tests for train cls config."""

from __future__ import annotations

from pathlib import Path

from src.engine.train_cls import (
    _atomic_torch_save,
    _extra_model_kwargs,
    _score_checkpoint_candidate,
)


def test_extra_model_kwargs_excludes_standard_classifier_fields() -> None:
    assert _extra_model_kwargs(
        {
            "name": "convnext_small",
            "pretrained": True,
            "in_chans": 3,
            "num_classes": 2,
            "drop_path_rate": 0.2,
        }
    ) == {"drop_path_rate": 0.2}


def test_checkpoint_scoring_reuses_validation_probabilities() -> None:
    metrics, score = _score_checkpoint_candidate(
        epoch_metrics={"auc": 0.9, "sensitivity": 0.5, "specificity": 0.5},
        y_true=[0, 1],
        malignant_probabilities=[0.2, 0.8],
        checkpoint_strategy="selected_threshold",
        selection_threshold=0.5,
        sensitivity_weight=0.0,
        min_specificity=0.0,
    )

    assert metrics["auc"] == 0.9
    assert metrics["sensitivity"] == 1.0
    assert score > 0.0


def test_atomic_torch_save_replaces_destination(tmp_path: Path) -> None:
    destination = tmp_path / "model.pt"

    _atomic_torch_save({"value": 1}, destination)
    _atomic_torch_save({"value": 2}, destination)

    assert destination.exists()
    assert not (tmp_path / ".model.pt.tmp").exists()
