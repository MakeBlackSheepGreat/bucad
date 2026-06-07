"""Unit tests for mainline config."""

from __future__ import annotations

import subprocess
from collections import Counter
from pathlib import Path

from src.utils.config import load_project_config


ROOT = Path(__file__).resolve().parents[2]


def _tracked_mainline_checkpoints() -> set[str]:
    """Return tracked mainline checkpoints."""
    completed = subprocess.run(
        ["git", "ls-files", "artifacts/checkpoints/*.pt"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        str((ROOT / line.strip()).resolve())
        for line in completed.stdout.splitlines()
        if line.strip()
    }


def test_demo_config_keeps_frozen_mainline_members() -> None:
    """Verify demo config keeps frozen mainline members."""
    config, paths = load_project_config(ROOT / "configs/inference/demo.yml")
    runtime = config["runtime"]
    members = runtime["classifier_members"]
    member_counts = Counter(str(member["model"]) for member in members)

    assert len(members) == 10
    assert member_counts == {
        "convnext_tiny": 5,
        "tf_efficientnetv2_s": 5,
    }
    assert runtime["segmenter_checkpoint"] == "./artifacts/checkpoints/segmenter_fold1.pt"
    assert float(runtime["default_threshold"]) == 0.51

    expected_checkpoints = {
        *(f"convnext_tiny_timm_recipe_fold{fold}.pt" for fold in range(1, 6)),
        *(f"efficientnetv2_s_fold{fold}.pt" for fold in range(1, 6)),
        "segmenter_fold1.pt",
    }
    configured_checkpoints = {
        Path(member["checkpoint"]).name for member in members
    } | {Path(runtime["segmenter_checkpoint"]).name}
    assert configured_checkpoints == expected_checkpoints

    tracked = _tracked_mainline_checkpoints()
    configured_abs = {
        str((paths.project_root / member["checkpoint"]).resolve())
        for member in members
    }
    configured_abs.add(str((paths.project_root / runtime["segmenter_checkpoint"]).resolve()))
    assert configured_abs == tracked
