"""Unit tests for train cls config."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.engine.train_cls import (
    _atomic_torch_save,
    _extra_model_kwargs,
    _final_classifier_metrics,
    _score_checkpoint_candidate,
    run_classifier_training,
)
from src.engine import train_cls


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


def test_last_checkpoint_metrics_reuse_final_epoch_report(monkeypatch) -> None:
    def _fail_evaluate_model(*_args, **_kwargs):
        raise AssertionError("last checkpoint should reuse final epoch validation metrics")

    monkeypatch.setattr(train_cls, "_evaluate_model", _fail_evaluate_model)

    metrics = _final_classifier_metrics(
        model=object(),
        val_loader=object(),
        device="cpu",
        checkpoint_strategy="last",
        epoch_reports=[{"metrics": {"auc": 0.75, "threshold": 0.5}}],
    )

    assert metrics == {"auc": 0.75, "threshold": 0.5}


def test_non_last_checkpoint_metrics_revalidate_selected_state(monkeypatch) -> None:
    monkeypatch.setattr(
        train_cls,
        "_evaluate_model",
        lambda *_args, **_kwargs: {"auc": 0.91, "threshold": 0.5},
    )

    metrics = _final_classifier_metrics(
        model=object(),
        val_loader=object(),
        device="cpu",
        checkpoint_strategy="youden",
        epoch_reports=[{"metrics": {"auc": 0.75, "threshold": 0.5}}],
    )

    assert metrics == {"auc": 0.91, "threshold": 0.5}


def test_atomic_torch_save_replaces_destination(tmp_path: Path) -> None:
    destination = tmp_path / "model.pt"

    _atomic_torch_save({"value": 1}, destination)
    _atomic_torch_save({"value": 2}, destination)

    assert destination.exists()
    assert not (tmp_path / ".model.pt.tmp").exists()


def test_run_classifier_training_smoke_uses_loop_config_outputs(tmp_path: Path, monkeypatch) -> None:
    if train_cls.torch is None:
        return
    torch = train_cls.torch
    manifest = pd.DataFrame(
        [
            {"sample_id": "a", "case_id": "c1", "pathology_label": "benign"},
            {"sample_id": "b", "case_id": "c2", "pathology_label": "malignant"},
            {"sample_id": "c", "case_id": "c3", "pathology_label": "benign"},
            {"sample_id": "d", "case_id": "c4", "pathology_label": "malignant"},
        ]
    )

    class _Dataset(torch.utils.data.Dataset):
        def __init__(self, frame, **_kwargs) -> None:
            self.frame = frame.reset_index(drop=True)

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, index: int):
            row = self.frame.iloc[index]
            label = 1 if row["pathology_label"] == "malignant" else 0
            return {
                "image": torch.full((3, 8, 8), float(label)),
                "label": torch.tensor(label, dtype=torch.long),
                "sample_id": row["sample_id"],
            }

    class _TinyClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(3 * 8 * 8, 2)

        def forward(self, images):
            return self.fc(images.flatten(1))

    class _Paths:
        project_root = tmp_path
        busbra_root = tmp_path / "unused"
        checkpoints_root = tmp_path / "checkpoints"
        reports_root = tmp_path / "reports"

    config_path = tmp_path / "config.yml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    monkeypatch.setattr(train_cls, "load_project_config", lambda _path: ({"device": "cpu", "training": {"epochs": 1}, "output": {}}, _Paths()))
    monkeypatch.setattr(train_cls, "load_busbra_manifest", lambda _root: manifest)
    monkeypatch.setattr(train_cls, "_prepare_fold_manifests", lambda **_kwargs: (manifest.iloc[:2], manifest.iloc[2:]))
    monkeypatch.setattr(train_cls, "BUSBRAClassificationDataset", _Dataset)
    monkeypatch.setattr(train_cls, "_build_classifier_model_and_transforms", lambda **_kwargs: (_TinyClassifier(), 8, None, None, {"image_size": 8}))

    report = run_classifier_training(config_path, fold=1)

    assert report["checkpoint_strategy"] == "last"
    assert report["scheduler"] == {}
    assert report["min_specificity"] == 0.0
    assert len(report["epoch_reports"]) == 1
