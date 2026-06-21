"""Run ImageNet-1K ablations for SonoGloRe-ConvNeXt V1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.imagenet import build_ablation_parser, run_imagenet_ablation


def main() -> int:
    """Parse CLI args and run the ImageNet ablation suite."""
    args = build_ablation_parser().parse_args()
    report = run_imagenet_ablation(
        args.config,
        root=args.root,
        output=args.output,
        markdown=args.markdown,
        max_batches=args.max_batches,
    )
    print(json.dumps({
        "report_path": report.get("report_path"),
        "markdown_path": report.get("markdown_path"),
        "rows": [
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "metrics": row.get("metrics"),
                "parameter_count": row.get("parameter_count"),
            }
            for row in report.get("rows", [])
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
