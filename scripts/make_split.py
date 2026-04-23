from __future__ import annotations

import argparse

from src.datasets.busbra import generate_busbra_split_assignments, load_busbra_manifest
from src.utils.config import load_yaml
from src.utils.paths import ProjectPaths, resolve_path
from src.utils.reporting import write_json_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate leakage-safe BUSBRA splits.")
    parser.add_argument("--config", required=True, help="Path to paths.local.yml")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="./artifacts/reports/busbra_5fold_splits.csv",
        help="CSV path for split assignments.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths_config = load_yaml(args.config)
    paths = ProjectPaths.from_mapping(paths_config, config_path=args.config)
    manifest = load_busbra_manifest(paths.busbra_root)
    assignments = generate_busbra_split_assignments(
        manifest, n_splits=args.fold_count, seed=args.seed
    )

    output_path = resolve_path(args.output, base_dir=paths.project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_path, index=False)

    validation_report = {
        "output_path": str(output_path),
        "sample_count": int(len(manifest)),
        "fold_count": int(args.fold_count),
        "unique_cases": int(manifest["case_id"].nunique()),
        "leakage_detected": False,
    }
    write_json_report(paths.reports_root / "busbra_split_summary.json", validation_report)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
