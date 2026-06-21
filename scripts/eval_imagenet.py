"""Evaluate one configured classifier on ImageNet-1K."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.imagenet import build_eval_parser, evaluate_imagenet_config


def main() -> int:
    """Parse CLI args and run one ImageNet evaluation."""
    args = build_eval_parser().parse_args()
    report = evaluate_imagenet_config(
        args.config,
        root=args.root,
        split=args.split,
        output=args.output,
        checkpoint=args.checkpoint,
        max_batches=args.max_batches,
    )
    print(json.dumps({
        "report_path": report.get("report_path"),
        "metrics": report.get("metrics"),
        "parameter_count": report.get("parameter_count"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
