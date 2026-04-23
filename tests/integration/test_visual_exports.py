from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

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
