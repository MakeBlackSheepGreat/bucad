from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.datasets.busbra import generate_busbra_split_assignments, load_busbra_manifest


TEST_ROOT = Path("artifacts/test-workspace/test_case_split")


def _write_busbra_fixture(root: Path) -> Path:
    busbra_root = root / "BUSBRA"
    images = busbra_root / "Images"
    masks = busbra_root / "Masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)

    records = [
        ["bus_0001-l", 1, "stub", "malignant", 4, "device", 10, 10, "left", "[0,0,1,1]"],
        ["bus_0001-r", 1, "stub", "malignant", 4, "device", 10, 10, "right", "[0,0,1,1]"],
        ["bus_0002-l", 2, "stub", "benign", 4, "device", 10, 10, "left", "[0,0,1,1]"],
        ["bus_0002-r", 2, "stub", "benign", 4, "device", 10, 10, "right", "[0,0,1,1]"],
        ["bus_0003-s", 3, "stub", "malignant", 4, "device", 10, 10, "single", "[0,0,1,1]"],
        ["bus_0004-s", 4, "stub", "benign", 4, "device", 10, 10, "single", "[0,0,1,1]"],
    ]
    frame = pd.DataFrame(
        records,
        columns=["ID", "Case", "Histology", "Pathology", "BIRADS", "Device", "Width", "Height", "Side", "BBOX"],
    )
    frame.to_csv(busbra_root / "bus_data.csv", index=False)
    for sample_id in frame["ID"]:
        (images / f"{sample_id}.png").write_bytes(b"placeholder")
        (masks / f"{sample_id.replace('bus_', 'mask_')}.png").write_bytes(b"placeholder")
    return busbra_root


def test_case_split_prevents_case_leakage() -> None:
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    root = _write_busbra_fixture(TEST_ROOT)
    manifest = load_busbra_manifest(root)
    assignments = generate_busbra_split_assignments(manifest, n_splits=2, seed=7)

    assert {"sample_id", "case_id", "fold_id", "stage"}.issubset(assignments.columns)
    grouped = assignments.groupby(["fold_id", "case_id"])["stage"].nunique()
    assert grouped.max() == 1
