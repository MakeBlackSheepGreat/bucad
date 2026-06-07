"""Smoke tests for comparison config."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.engine.compare_cls import run_classifier_comparison
from src.utils.config import load_yaml


def test_comparison_config_contains_expected_models() -> None:
    config = load_yaml("configs/classifier/comparison.yml")
    models = [entry["name"] for entry in config["comparison"]["models"]]

    assert "basic_cnn" in models
    assert "tf_efficientnetv2_s" in models
    assert len(models) == len(set(models))


def test_comparison_runner_dry_run_writes_report(tmp_path) -> None:
    config = load_yaml("configs/classifier/comparison.yml")
    config["comparison"]["output_path"] = str(tmp_path / "comparison_results.json")
    config_path = tmp_path / "comparison.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    report = run_classifier_comparison(
        config_path,
        model_limit=1,
        dry_run=True,
    )

    assert report["dry_run"] is True
    assert report["model_count"] == 1
    assert report["results"][0]["dataset_boundary"]["busi_used_for_training"] is False
    assert Path(config["comparison"]["output_path"]).exists()
