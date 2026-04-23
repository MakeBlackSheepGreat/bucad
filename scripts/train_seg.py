from __future__ import annotations

import argparse

from src.engine.train_seg import run_segmentation_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the baseline segmenter.")
    parser.add_argument("--config", required=True, help="Path to segmenter config YAML")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_segmentation_training(args.config, fold=args.fold, epochs_override=args.epochs)
    print(report["checkpoint_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
