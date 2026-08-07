"""Extract TCIA BrEaST assets and create a pathology-aligned external manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare TCIA BrEaST external-evaluation files.")
    parser.add_argument("--root", default="data/external/tcia_breast_us")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    downloads = root / "downloads"
    archive = downloads / "BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip"
    clinical = downloads / "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
    extracted = root / "images_and_masks"
    if not extracted.exists():
        with ZipFile(archive) as handle:
            handle.extractall(root)
        source = root / "BrEaST-Lesions_USG-images_and_masks"
        source.rename(extracted)
    table = pd.read_excel(clinical, sheet_name="BrEaST-Lesions-USG clinical dat")
    rows = []
    for entry in table.itertuples(index=False):
        label = str(entry.Classification).strip().lower()
        if label not in {"benign", "malignant"}:
            continue
        image_path = extracted / str(entry.Image_filename)
        mask_value = entry.Mask_tumor_filename
        mask_path = extracted / str(mask_value) if pd.notna(mask_value) else None
        if not image_path.exists():
            raise FileNotFoundError(f"Missing TCIA image: {image_path}")
        if mask_path is not None and not mask_path.exists():
            raise FileNotFoundError(f"Missing TCIA tumor mask: {mask_path}")
        rows.append({
            "sample_id": f"tcia:{image_path.stem}",
            "case_id": f"tcia:{int(entry.CaseID):03d}",
            "dataset_name": "tcia_breast_us",
            "image_path": str(image_path),
            "mask_path": str(mask_path) if mask_path is not None else "",
            "pathology_label": label,
            "verification": str(entry.Verification),
            "diagnosis": str(entry.Diagnosis),
            "birads": str(entry.BIRADS),
            "quality_flag": "valid",
        })
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output = manifest_dir / "manifest.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {label: sum(row["pathology_label"] == label for row in rows) for label in ("benign", "malignant")}
    print(f"manifest={output} total={len(rows)} counts={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
