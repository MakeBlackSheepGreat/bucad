from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from scripts.export_visual_evidence import export_visual_evidence
from src.explain.overlay import render_heatmap_overlay, render_mask_overlay
from src.preprocess.io import save_image


TEST_ROOT = Path("artifacts/test-workspace/test_visual_exports")


def test_visual_export_helpers_write_png_outputs() -> None:
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    image = np.random.randint(0, 255, size=(96, 96), dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.float32)
    mask[20:60, 20:60] = 1.0
    heatmap = np.linspace(0.0, 1.0, num=96 * 96, dtype=np.float32).reshape(96, 96)

    overlay = render_mask_overlay(image, mask)
    explanation = render_heatmap_overlay(image, heatmap)
    overlay_path = save_image(TEST_ROOT / "overlay.png", overlay)
    explanation_path = save_image(TEST_ROOT / "explanation.png", explanation)

    assert Path(overlay_path).exists()
    assert Path(explanation_path).exists()


def test_visual_evidence_export_writes_review_pack() -> None:
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    busi_root = TEST_ROOT / "Dataset_BUSI_with_GT"
    label_dir = busi_root / "benign"
    label_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((64, 64), 128, dtype=np.uint8)
    save_image(label_dir / "benign (1).png", image)
    save_image(label_dir / "benign (1)_mask.png", np.ones((64, 64), dtype=np.uint8) * 255)
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
    config_path = TEST_ROOT / "demo.yml"
    config_path.write_text(f"paths_config: {paths_path.name}\nruntime:\n  default_threshold: 0.5\n", encoding="utf-8")

    result = export_visual_evidence(
        config_path,
        (TEST_ROOT / "visual_pack").resolve(),
        limit=1,
        classifier_predictor=lambda _: (0.8, 0.2),
        segmenter_predictor=lambda _: np.ones((64, 64), dtype=np.float32),
        explanation_generator=lambda _: np.ones((64, 64), dtype=np.float32),
    )

    assert result["sample_count"] == 1
    assert (TEST_ROOT / "visual_pack").exists()
    assert (TEST_ROOT / "artifacts" / "reports" / "visual_evidence_review.md").exists()
