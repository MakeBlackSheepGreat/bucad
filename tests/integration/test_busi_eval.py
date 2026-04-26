from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from src.engine.inference import evaluate_busi_dataset
from src.preprocess.io import save_image


TEST_ROOT = Path("artifacts/test-workspace/test_busi_eval")


def _write_busi_fixture(root: Path) -> Path:
    busi_root = root / "Dataset_BUSI_with_GT"
    for label in ("benign", "malignant"):
        label_dir = busi_root / label
        label_dir.mkdir(parents=True)
        for index in range(2):
            image_name = f"{label} ({index + 1}).png"
            mask_name = f"{label} ({index + 1})_mask.png"
            image = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[20:40, 20:40] = 255
            save_image(label_dir / image_name, image)
            save_image(label_dir / mask_name, mask)
    return busi_root


def test_busi_evaluation_writes_metrics_report() -> None:
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    busi_root = _write_busi_fixture(TEST_ROOT)
    config_path = TEST_ROOT / "demo.yml"
    paths_path = TEST_ROOT / "paths.local.yml"
    paths_path.write_text(
        "\n".join(
                [
                    "project_root: .",
                    "datasets:",
                    "  busbra_root: ./unused",
                    f"  busi_root: {busi_root.resolve().as_posix()}",
                "artifacts:",
                "  root: ./artifacts",
                "  checkpoints: ./artifacts/checkpoints",
                "  logs: ./artifacts/logs",
                "  reports: ./artifacts/reports",
                "runtime:",
                "  default_classifier_ckpt: ./artifacts/checkpoints/classifier.pt",
                "  default_segmenter_ckpt: ./artifacts/checkpoints/segmenter.pt",
            ]
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                f"paths_config: {paths_path.name}",
                "runtime:",
                "  default_threshold: 0.5",
                "  borderline_margin: 0.05",
            ]
        ),
        encoding="utf-8",
    )

    def predictor(image: np.ndarray) -> tuple[float, float]:
        return (0.2, 0.8) if image.mean() > 120 else (0.8, 0.2)

    report = evaluate_busi_dataset(config_path, classifier_predictor=predictor)

    assert report["sample_count"] == 4
    assert "metrics" in report
    assert "sensitivity" in report["metrics"]
    assert "precision" in report["metrics"]
    assert "f1_score" in report["metrics"]
    assert "threshold_analysis" in report
    assert "best_by_youden" in report["threshold_analysis"]
    threshold_report = TEST_ROOT / "artifacts" / "reports" / "threshold_analysis.md"
    assert threshold_report.exists()
    threshold_text = threshold_report.read_text(encoding="utf-8")
    assert "Precision" in threshold_text
    assert "F1-Score" in threshold_text


def test_busi_evaluation_uses_runtime_default_threshold() -> None:
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    busi_root = TEST_ROOT / "Dataset_BUSI_with_GT"
    for label, value in (("benign", 40), ("malignant", 220)):
        label_dir = busi_root / label
        label_dir.mkdir(parents=True)
        for index in range(2):
            image_name = f"{label} ({index + 1}).png"
            mask_name = f"{label} ({index + 1})_mask.png"
            image = np.full((64, 64), value, dtype=np.uint8)
            image[:, ::2] = max(0, value - 20)
            image[:, 1::2] = min(255, value + 20)
            mask = np.zeros((64, 64), dtype=np.uint8)
            save_image(label_dir / image_name, image)
            save_image(label_dir / mask_name, mask)

    config_path = TEST_ROOT / "demo.yml"
    paths_path = TEST_ROOT / "paths.local.yml"
    paths_path.write_text(
        "\n".join(
            [
                "project_root: .",
                "datasets:",
                "  busbra_root: ./unused",
                f"  busi_root: {busi_root.resolve().as_posix()}",
                "artifacts:",
                "  root: ./artifacts",
                "  checkpoints: ./artifacts/checkpoints",
                "  logs: ./artifacts/logs",
                "  reports: ./artifacts/reports",
            ]
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                f"paths_config: {paths_path.name}",
                "runtime:",
                "  default_threshold: 0.7",
                "  borderline_margin: 0.05",
            ]
        ),
        encoding="utf-8",
    )

    def predictor(image: np.ndarray) -> tuple[float, float]:
        return (0.4, 0.6) if image.mean() > 120 else (0.6, 0.4)

    report = evaluate_busi_dataset(config_path, classifier_predictor=predictor)

    assert report["metrics"]["threshold"] == 0.7
    assert report["metrics"]["sensitivity"] == 0.0
    assert report["metrics"]["specificity"] == 1.0
