"""CLI wrapper for BUSI external evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.inference import evaluate_busi_dataset


def build_parser() -> argparse.ArgumentParser:
    """Build CLI options for BUSI external evaluation."""
    parser = argparse.ArgumentParser(description="Run BUSI external evaluation.")
    parser.add_argument("--config", required=True, help="Path to inference config YAML")
    parser.add_argument("--output", default=None, help="Optional output path for the report")
    return parser


def main() -> int:
    """Evaluate the frozen inference config and print headline metrics."""
    args = build_parser().parse_args()
    report = evaluate_busi_dataset(args.config, output_path=args.output)
    print(report["metrics"])
    best = report.get("threshold_analysis", {}).get("best_by_youden", {})
    if best:
        print({"best_threshold_by_youden": best.get("threshold")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
