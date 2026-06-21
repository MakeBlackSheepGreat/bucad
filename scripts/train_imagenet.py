"""Train one configured classifier on ImageNet-1K."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.imagenet import build_train_parser, train_imagenet_config


def main() -> int:
    """Parse CLI args and run ImageNet training."""
    args = build_train_parser().parse_args()
    report = train_imagenet_config(
        args.config,
        root=args.root,
        output=args.output,
        checkpoint=args.checkpoint,
        epochs_override=args.epochs_override,
        max_batches=args.max_batches,
        max_val_batches=args.max_val_batches,
        skip_val=bool(args.skip_val),
    )
    print(
        json.dumps(
            {
                "report_path": report.get("report_path"),
                "checkpoint_path": report.get("checkpoint_path"),
                "best_epoch": report.get("best_epoch"),
                "best_metrics": report.get("best_metrics"),
                "parameter_count": report.get("parameter_count"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
