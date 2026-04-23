from __future__ import annotations

import argparse

from src.engine.inference import evaluate_busi_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BUSI external evaluation.")
    parser.add_argument("--config", required=True, help="Path to inference config YAML")
    parser.add_argument("--output", default=None, help="Optional output path for the report")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = evaluate_busi_dataset(args.config, output_path=args.output)
    print(report["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
