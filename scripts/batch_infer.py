"""Batch inference CLI for folders of ultrasound images."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.inference import BreastUltrasoundInferenceService
from src.preprocess.io import SUPPORTED_SUFFIXES
from src.utils.config import load_project_config


def batch_infer(
    config_path: str | Path,
    input_dir: str | Path,
    output_csv: str | Path,
    *,
    classifier_predictor: Callable[[np.ndarray], tuple[float, float]] | None = None,
) -> Path:
    """Run folder-level diagnosis and write one CSV row per supported image."""
    config, paths = load_project_config(config_path)
    source_dir = Path(input_dir)
    if not source_dir.is_absolute():
        source_dir = (paths.project_root / source_dir).resolve()
    output_path = Path(output_csv)
    if not output_path.is_absolute():
        output_path = (paths.project_root / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    service = BreastUltrasoundInferenceService(
        config.get("runtime", {}),
        paths=paths,
        classifier_predictor=classifier_predictor,
    )
    image_paths = [
        path
        for path in sorted(source_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and "_mask" not in path.stem
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "status",
                "benign_probability",
                "malignant_probability",
                "final_label",
                "warnings",
            ],
        )
        writer.writeheader()
        for image_path in image_paths:
            response = service.diagnose(
                image_path,
                need_segmentation=False,
                need_explanation=False,
            )
            writer.writerow(
                {
                    "filename": image_path.name,
                    "status": response.status,
                    "benign_probability": response.result.benign_probability if response.result else "",
                    "malignant_probability": response.result.malignant_probability if response.result else "",
                    "final_label": response.result.final_label if response.result else "",
                    "warnings": "; ".join(response.warnings),
                }
            )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build CLI options for batch inference export."""
    parser = argparse.ArgumentParser(description="Run batch inference for a folder of ultrasound images.")
    parser.add_argument("--config", default="configs/inference/demo.yml")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default="artifacts/reports/batch_inference.csv")
    return parser


def main() -> int:
    """Run batch inference from CLI arguments and print the CSV path."""
    args = build_parser().parse_args()
    output = batch_infer(args.config, args.input_dir, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
