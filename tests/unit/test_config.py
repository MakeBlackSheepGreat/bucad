from __future__ import annotations

import shutil
from pathlib import Path

from src.utils.config import deep_merge, load_yaml, save_yaml


TEST_ROOT = Path("artifacts/test-workspace/test_unit_config")


def test_yaml_round_trip() -> None:
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_ROOT / "config.yml"
    payload = {"a": 1, "nested": {"b": 2}}
    save_yaml(path, payload)

    assert load_yaml(path) == payload


def test_deep_merge_preserves_nested_values() -> None:
    merged = deep_merge({"a": 1, "nested": {"b": 2}}, {"nested": {"c": 3}})
    assert merged == {"a": 1, "nested": {"b": 2, "c": 3}}
