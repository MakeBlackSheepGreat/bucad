"""Map verified BUSI-WHU class labels onto the original public image/mask archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_mask_path(image_path: Path) -> Path:
    if image_path.parent.name == "ori":
        split_root = image_path.parents[2]
        return split_root / "gt" / "ori" / f"{image_path.stem}_anno{image_path.suffix}"
    split_root = image_path.parents[1]
    return split_root / "gt" / f"{image_path.stem}_anno{image_path.suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare mapped BUSI-WHU external manifest.")
    parser.add_argument("--root", default="data/external/busi_whu")
    parser.add_argument("--raw-root", default="data/external/_staging/busi_whu_kaggle")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    raw_root = Path(args.raw_root).resolve()
    labels: dict[str, str] = {}
    reference_root = root / "label_reference_hf"
    for image_path in reference_root.rglob("*.bmp"):
        if "/masks/" in image_path.as_posix():
            continue
        name = image_path.stem.lower()
        if name.startswith("benign_"):
            labels[_sha256(image_path)] = "benign"
        elif name.startswith("malignant_"):
            labels[_sha256(image_path)] = "malignant"
    rows = []
    for image_path in raw_root.rglob("*.bmp"):
        if "/gt/" in image_path.as_posix():
            continue
        label = labels.get(_sha256(image_path))
        if label is None:
            continue
        mask_path = _raw_mask_path(image_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing BUSI-WHU mask: {mask_path}")
        rows.append({
            "sample_id": f"busi_whu:{image_path.stem}",
            "case_id": f"busi_whu:{image_path.stem}",
            "dataset_name": "busi_whu",
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "pathology_label": label,
            "quality_flag": "valid",
        })
    if not rows:
        raise RuntimeError("No BUSI-WHU labels could be mapped to original files.")
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output = manifest_dir / "manifest_external_788.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {label: sum(row["pathology_label"] == label for row in rows) for label in ("benign", "malignant")}
    print(f"manifest={output} total={len(rows)} counts={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
