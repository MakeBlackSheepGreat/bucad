from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.train_cls import run_classifier_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the baseline classifier.")
    parser.add_argument("--config", required=True, help="Path to classifier config YAML")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_classifier_training(args.config, fold=args.fold, epochs_override=args.epochs)
    print(report["checkpoint_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
