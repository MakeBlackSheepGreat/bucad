"""Prepare authorized ImageNet-1K archives for torchvision evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.imagenet import build_prepare_parser, prepare_imagenet_archives


def main() -> int:
    """Parse CLI args and prepare ImageNet."""
    args = build_prepare_parser().parse_args()
    report = prepare_imagenet_archives(
        args.config,
        root=args.root,
        skip_md5=bool(args.skip_md5),
        prepare_train=bool(args.prepare_train),
        prepare_val=bool(args.prepare_val),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
