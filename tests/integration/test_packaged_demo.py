"""Integration tests for packaged demo."""

from __future__ import annotations

import shutil
from pathlib import Path

from scripts.export_demo_assets import export_demo_assets


def test_demo_asset_export_creates_expected_layout() -> None:
    root = Path("artifacts/test-workspace/test_packaged_demo")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / "configs" / "inference").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "inference" / "demo.yml").write_text("paths_config: paths.local.yml\n", encoding="utf-8")
    checkpoints_dir = root / "artifacts" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / "classifier.pt").write_bytes(b"classifier")
    (checkpoints_dir / "segmenter.pt").write_bytes(b"segmenter")
    config_path = root / "paths.local.yml"
    config_path.write_text(
        "\n".join(
            [
                "project_root: .",
                "datasets:",
                "  busbra_root: ./训练集/BUSBRA",
                "  busi_root: ./测试集/Dataset_BUSI_with_GT",
                "artifacts:",
                "  root: ./artifacts",
                "  checkpoints: ./artifacts/checkpoints",
                "  logs: ./artifacts/logs",
                "  reports: ./artifacts/reports",
                "runtime:",
                "  default_classifier_ckpt: ./artifacts/checkpoints/classifier.pt",
                "  default_segmenter_ckpt: ./artifacts/checkpoints/segmenter.pt",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = export_demo_assets(config_path, root / "bundle")

    assert (output_dir / "configs" / "paths.local.yml").exists()
    assert (output_dir / "configs" / "inference" / "demo.yml").exists()
    assert (output_dir / "artifacts" / "checkpoints" / "classifier.pt").exists()
    assert (output_dir / "artifacts" / "checkpoints" / "segmenter.pt").exists()
    assert (output_dir / "release_v1.sha256").exists()
    assert (root / "artifacts" / "reports" / "release_v1_manifest.md").exists()
    assert Path("packaging/demo.spec").exists()
