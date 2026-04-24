from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.compare_cls import run_classifier_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run classifier comparison experiments.")
    parser.add_argument("--config", default="configs/classifier/comparison.yml")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--model-limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate configs/models without training.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_classifier_comparison(
        args.config,
        fold=args.fold,
        epochs_override=args.epochs,
        model_limit=args.model_limit,
        dry_run=args.dry_run,
    )
    print({"model_count": report["model_count"], "dry_run": report["dry_run"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
