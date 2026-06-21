"""Unit tests for config."""

from __future__ import annotations

import shutil
from pathlib import Path

from src.utils.config import deep_merge, load_project_config, load_yaml, save_yaml


TEST_ROOT = Path("artifacts/test-workspace/test_unit_config")


def test_yaml_round_trip() -> None:
    """Verify yaml round trip."""
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_ROOT / "config.yml"
    payload = {"a": 1, "nested": {"b": 2}}
    save_yaml(path, payload)

    assert load_yaml(path) == payload


def test_deep_merge_preserves_nested_values() -> None:
    """Verify deep merge preserves nested values."""
    merged = deep_merge({"a": 1, "nested": {"b": 2}}, {"nested": {"c": 3}})
    assert merged == {"a": 1, "nested": {"b": 2, "c": 3}}


def test_load_project_config_resolves_paths_relative_to_paths_file() -> None:
    """Verify load project config resolves paths relative to paths file."""
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    config_root = TEST_ROOT / "demo-project"
    (config_root / "configs" / "classifier").mkdir(parents=True, exist_ok=True)
    (config_root / "datasets" / "BUSBRA").mkdir(parents=True, exist_ok=True)
    (config_root / "datasets" / "BUSI").mkdir(parents=True, exist_ok=True)
    (config_root / "datasets" / "imagenet").mkdir(parents=True, exist_ok=True)

    paths_path = config_root / "configs" / "paths.local.yml"
    save_yaml(
        paths_path,
        {
            "project_root": "..",
            "datasets": {
                "busbra_root": "./datasets/BUSBRA",
                "busi_root": "./datasets/BUSI",
                "imagenet_root": "./datasets/imagenet",
            },
        },
    )
    config_path = config_root / "configs" / "classifier" / "baseline.yml"
    save_yaml(config_path, {"paths_config": "../paths.local.yml"})

    _, paths = load_project_config(config_path)

    assert paths.project_root == config_root.resolve()
    assert paths.busbra_root == (config_root / "datasets" / "BUSBRA").resolve()
    assert paths.busi_root == (config_root / "datasets" / "BUSI").resolve()
    assert paths.imagenet_root == (config_root / "datasets" / "imagenet").resolve()
