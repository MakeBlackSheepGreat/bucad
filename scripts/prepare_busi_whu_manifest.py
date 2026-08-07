"""Build a labeled BUSI-WHU manifest from split/image/mask folders."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare BUSI-WHU manifest.")
    parser.add_argument("--root", default="data/external/busi_whu/label_reference_hf")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = []
    for split in ("train", "validation", "test"):
        image_dir = root / split / "images"
        mask_dir = root / split / "masks"
        for image_path in sorted(image_dir.glob("*")):
            if image_path.suffix.lower() not in {".bmp", ".png", ".jpg", ".jpeg"}:
                continue
            stem = image_path.stem
            label = "malignant" if stem.lower().startswith("malignant_") else "benign" if stem.lower().startswith("benign_") else None
            if label is None:
                continue
            mask_candidates = [mask_dir / f"{stem}{image_path.suffix}", mask_dir / f"{stem}.bmp", mask_dir / f"{stem}.png"]
            mask_path = next((path for path in mask_candidates if path.exists()), None)
            rows.append({
                "sample_id": f"{split}:{stem}",
                "case_id": stem,
                "dataset_name": "busi_whu",
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path else "",
                "pathology_label": label,
                "split": split,
                "quality_flag": "valid" if mask_path else "missing_mask",
            })
    if not rows:
        raise RuntimeError(f"No labeled images found under {root}")
    output = root / "manifest_hf_partial.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {label: sum(row["pathology_label"] == label for row in rows) for label in ("benign", "malignant")}
    print(f"manifest={output} total={len(rows)} counts={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
