"""CLI entry point for the SonoGloReNet ablation matrix workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.sonoglore_ablation import build_parser, dry_run, run


def main() -> int:
    """Parse CLI arguments and run the ablation workflow."""
    args = build_parser().parse_args()
    report = dry_run(args) if args.dry_run else run(args)
    print(
        json.dumps(
            {
                "dry_run": bool(args.dry_run),
                "final_candidate": (
                    {
                        "candidate_id": report.get("final_candidate", {}).get("candidate_id"),
                        "candidate_name": report.get("final_candidate", {}).get("candidate_name"),
                    }
                    if isinstance(report.get("final_candidate"), dict)
                    else None
                ),
                "promotion_gate": report.get("promotion_gate"),
                "report_paths": report.get("report_paths"),
                "summary_json": report.get("summary_json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

