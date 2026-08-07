"""CLI wrapper for one LesioNeXt-BUS joint training fold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.train_seg import run_segmentation_training


def build_parser() -> argparse.ArgumentParser:
    """Build CLI options for one LesioNeXt fold."""
    parser = argparse.ArgumentParser(description="Train one LesioNeXt-BUS fold.")
    parser.add_argument("--config", default="configs/segmenter/lesionext.yml")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    return parser


def main() -> int:
    """Run joint LesioNeXt training and print the checkpoint path."""
    args = build_parser().parse_args()
    report = run_segmentation_training(args.config, fold=args.fold, epochs_override=args.epochs)
    print(report["checkpoint_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
