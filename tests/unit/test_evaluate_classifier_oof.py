"""Regression checks for synthesized BUSBRA OOF runtime configuration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_oof_script_module():
    """Load the standalone OOF script as a module for focused unit tests."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_classifier_oof.py"
    spec = importlib.util.spec_from_file_location("evaluate_classifier_oof", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthesized_runtime_resolves_checkpoint_under_artifacts_root(tmp_path: Path) -> None:
    """Ensure a classifier YAML checkpoint filename never falls back to random weights."""
    module = _load_oof_script_module()
    runtime = module._runtime_from_classifier_config(
        {
            "model": {"name": "lesionext_lens_tiny"},
            "output": {"checkpoint_name": "lens_fold{fold}.pt"},
        },
        project_root=tmp_path,
        fold=3,
    )

    assert runtime["classifier_checkpoint"] == str(
        tmp_path / "artifacts" / "checkpoints" / "lens_fold3.pt"
    )


def test_missing_synthesized_checkpoint_is_rejected_before_inference(tmp_path: Path, monkeypatch) -> None:
    """Protect OOF metrics from silently evaluating randomly initialized models."""
    module = _load_oof_script_module()
    config = tmp_path / "classifier.yml"
    config.write_text("paths_config: configs/paths.local.yml\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "load_project_config",
        lambda _path: (
            {
                "model": {"name": "lesionext_lens_tiny"},
                "training": {"split_path": "splits.csv"},
                "output": {"checkpoint_name": "missing_fold{fold}.pt"},
            },
            type("Paths", (), {"project_root": tmp_path, "busbra_root": tmp_path})(),
        ),
    )
    monkeypatch.setattr(module, "load_busbra_manifest", lambda _root: __import__("pandas").DataFrame())
    monkeypatch.setattr(module.pd, "read_csv", lambda _path: __import__("pandas").DataFrame())
    monkeypatch.setattr(module, "select_device", lambda _value: "cpu")
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_classifier_oof.py",
            "--classifier-config",
            str(config),
            "--fold-count",
            "1",
            "--output",
            str(tmp_path / "oof.json"),
        ],
    )

    with pytest.raises(FileNotFoundError, match="missing_fold1.pt"):
        module.main()
