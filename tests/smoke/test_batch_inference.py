from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np

from scripts.batch_infer import batch_infer
from src.preprocess.io import save_image


TEST_ROOT = Path("artifacts/test-workspace/test_batch_inference")


def test_batch_inference_writes_csv_rows() -> None:
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    image_dir = TEST_ROOT / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    save_image(image_dir / "case1.png", np.tile(np.arange(64, dtype=np.uint8), (64, 1)))
    save_image(image_dir / "case2.png", np.tile(np.arange(64, dtype=np.uint8) + 190, (64, 1)))

    paths_path = TEST_ROOT / "paths.local.yml"
    paths_path.write_text(
        "\n".join(
            [
                "project_root: .",
                "datasets:",
                "  busbra_root: ./unused",
                "  busi_root: ./unused",
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

    output_csv = batch_infer(
        config_path,
        image_dir.resolve(),
        (TEST_ROOT / "batch.csv").resolve(),
        classifier_predictor=lambda image: (0.3, 0.7) if image.mean() > 200 else (0.7, 0.3),
    )

    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8")))
    assert [row["filename"] for row in rows] == ["case1.png", "case2.png"]
    assert rows[1]["final_label"] == "malignant"
