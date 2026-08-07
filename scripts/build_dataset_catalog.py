"""Create a local catalog for training and external breast-ultrasound cohorts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.busi import load_busi_manifest
from src.datasets.busbra import load_busbra_manifest
from src.datasets.external_bus import load_external_manifest


ROOT = Path(__file__).resolve().parents[1]


def _summary(dataset_id: str, role: str, manifest: pd.DataFrame) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "role": role,
        "sample_count": int(len(manifest)),
        "case_count": int(manifest["case_id"].nunique()),
        "label_counts": {str(label): int(count) for label, count in manifest["pathology_label"].value_counts().to_dict().items()},
        "mask_count": int(manifest["mask_path"].notna().sum()),
        "image_root": str(Path(manifest.iloc[0]["image_path"]).parent) if not manifest.empty else "",
    }


def main() -> int:
    catalog_root = ROOT / "data" / "catalog"
    catalog_root.mkdir(parents=True, exist_ok=True)
    sources = [
        ("busbra", "training_internal_oof", load_busbra_manifest(ROOT / "训练集" / "BUSBRA")),
        ("busi", "external_locked", load_busi_manifest(ROOT / "测试集" / "Dataset_BUSI_with_GT", include_normal=False)),
        ("bus_uclm", "external_locked", load_external_manifest("bus_uclm", ROOT / "data" / "external" / "bus_uclm", manifest_path=ROOT / "data" / "external" / "bus_uclm" / "manifest.csv")),
        ("busi_whu", "external_locked_label_mapped", load_external_manifest("busi_whu", ROOT / "data" / "external" / "busi_whu", manifest_path=ROOT / "data" / "external" / "busi_whu" / "manifests" / "manifest_external_788.csv")),
        ("tcia_breast_us", "external_locked", load_external_manifest("tcia_breast_us", ROOT / "data" / "external" / "tcia_breast_us", manifest_path=ROOT / "data" / "external" / "tcia_breast_us" / "manifests" / "manifest.csv")),
    ]
    summaries = []
    for dataset_id, role, manifest in sources:
        manifest.to_csv(catalog_root / f"{dataset_id}_manifest.csv", index=False, encoding="utf-8-sig")
        summaries.append(_summary(dataset_id, role, manifest))
    (catalog_root / "dataset_catalog.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame.from_records(summaries).to_csv(catalog_root / "dataset_catalog.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
