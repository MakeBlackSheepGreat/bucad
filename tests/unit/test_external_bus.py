"""Tests for external breast-ultrasound dataset manifests and frozen evaluation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.datasets.external_bus import load_external_manifest
from src.engine import inference
from src.preprocess.io import save_image
from src.utils.results import build_diagnostic_result


def test_load_external_manifest_from_label_folders(tmp_path: Path) -> None:
    for label, value in (("benign", 40), ("malignant", 220)):
        folder = tmp_path / label
        folder.mkdir()
        image_path = folder / f"{label}_01.png"
        mask_path = folder / f"{label}_01_mask.png"
        save_image(image_path, np.full((64, 64), value, dtype=np.uint8))
        save_image(mask_path, np.pad(np.ones((20, 20), dtype=np.uint8), 22))
    manifest = load_external_manifest("bus_uclm", tmp_path)
    assert list(manifest["pathology_label"]) == ["benign", "malignant"]
    assert manifest["mask_path"].notna().all()


def test_external_evaluation_locks_threshold_and_writes_predictions(tmp_path: Path, monkeypatch) -> None:
    for label, value in (("benign", 40), ("malignant", 220)):
        folder = tmp_path / label
        folder.mkdir()
        save_image(folder / f"{label}_01.png", np.full((64, 64), value, dtype=np.uint8))
    config_path = tmp_path / "frozen.yml"
    config_path.write_text("runtime:\n  default_threshold: 0.7\n", encoding="utf-8")

    class FakeService:
        runtime_config = {"default_threshold": 0.7}

        @staticmethod
        def _model_identifier() -> str:
            return "fake-frozen-model"

        @staticmethod
        def diagnose(image_path, **kwargs):
            probability = 0.2 if "benign" in str(image_path) else 0.9
            result = build_diagnostic_result(1.0 - probability, probability, threshold=0.7)
            return SimpleNamespace(result=result, status="completed")

    monkeypatch.setattr(inference, "_inference_service_from_project_config", lambda path: (FakeService(), SimpleNamespace(reports_root=tmp_path)))
    output_path = tmp_path / "external.json"
    report = inference.evaluate_external_bus_dataset("bus_uclm", tmp_path, config_path=config_path, output_path=output_path, bootstrap_replicates=20)
    assert report["metrics"]["threshold"] == 0.7
    assert report["threshold_source"].startswith("frozen_runtime_default")
    assert output_path.exists()
    assert output_path.with_name("external_predictions.csv").exists()
