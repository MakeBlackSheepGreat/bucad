from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from src.utils.paths import ProjectPaths


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}, got {type(data)!r}")
    return data


def save_yaml(path: str | Path, data: Mapping[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False, allow_unicode=True)


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_project_config(config_path: str | Path) -> tuple[dict[str, Any], ProjectPaths]:
    config = load_yaml(config_path)
    paths_config = config.get("paths_config")
    if paths_config:
        paths_mapping = load_yaml(Path(config_path).resolve().parent / paths_config)
    else:
        paths_mapping = {}
    paths = ProjectPaths.from_mapping(paths_mapping, config_path=config_path)
    return config, paths
