"""Convert BUS-UCLM RGB masks into the shared classification manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare BUS-UCLM manifest.")
    parser.add_argument("--root", default="data/external/bus_uclm")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = []
    for image_path in sorted((root / "images").glob("*.png")):
        mask_path = root / "masks" / image_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if mask is None:
            continue
        colors = mask.reshape(-1, 3)
        has_malignant = bool(np.any(np.all(colors == np.asarray([0, 0, 255]), axis=1)))
        has_benign = bool(np.any(np.all(colors == np.asarray([0, 255, 0]), axis=1)))
        label = "malignant" if has_malignant else "benign" if has_benign else "normal"
        rows.append({
            "sample_id": image_path.stem,
            "case_id": image_path.stem,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "pathology_label": label,
            "dataset_name": "bus_uclm",
        })
    output = root / "manifest.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {label: sum(row["pathology_label"] == label for row in rows) for label in ("benign", "malignant", "normal")}
    print(f"manifest={output} total={len(rows)} counts={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
