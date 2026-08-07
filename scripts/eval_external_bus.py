"""Frozen-model evaluation on a registered independent breast-ultrasound dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.inference import evaluate_external_bus_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a frozen model on an external BUS dataset.")
    parser.add_argument("--dataset", required=True, help="Registry id, for example bus_uclm")
    parser.add_argument("--root", required=True, help="Local dataset root")
    parser.add_argument("--config", required=True, help="Frozen inference config YAML")
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--manifest", default=None, help="Optional CSV/JSON/XLSX manifest")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap replicates for AUC CI")
    args = parser.parse_args()
    report = evaluate_external_bus_dataset(
        args.dataset,
        args.root,
        config_path=args.config,
        manifest_path=args.manifest,
        output_path=args.output,
        bootstrap_replicates=args.bootstrap,
    )
    print(report["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
