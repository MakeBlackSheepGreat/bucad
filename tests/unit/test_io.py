"""Unit tests for io."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from src.preprocess.io import read_image, save_image


TEST_ROOT = Path("artifacts/test-workspace/test_unit_io")


def test_read_and_save_image_support_unicode_paths() -> None:
    """Verify read and save image support unicode paths."""
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    image = np.random.randint(0, 255, size=(32, 32), dtype=np.uint8)
    image_path = TEST_ROOT / "训练集" / "样例.png"

    save_image(image_path, image)
    restored = read_image(image_path, grayscale=True)

    assert np.array_equal(restored, image)
