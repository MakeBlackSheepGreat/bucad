"""List external breast-ultrasound dataset cards and local readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.external_bus import DATASET_REGISTRY, dataset_cards, load_external_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="List external breast ultrasound datasets.")
    parser.add_argument("--root", default=None, help="Optional root containing dataset subdirectories")
    args = parser.parse_args()
    payload = []
    for card in dataset_cards():
        item = dict(card)
        if args.root:
            root = Path(args.root) / card["dataset_id"]
            item["local_root"] = str(root)
            item["ready"] = False
            if root.exists():
                manifest_path = root / str(card.get("manifest_relpath", "manifest.csv"))
                try:
                    manifest = load_external_manifest(
                        card["dataset_id"],
                        root,
                        manifest_path=manifest_path if manifest_path.exists() else None,
                    )
                    item["ready"] = True
                    item["sample_count"] = len(manifest)
                except (OSError, ValueError, KeyError):
                    pass
        payload.append(item)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
