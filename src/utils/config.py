"""YAML config loading, merging, and project path resolution helpers."""

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


def resolve_config_reference(config_path: str | Path, reference: str | Path) -> Path:
    """Resolve referenced config files near the caller, then near the project root."""
    config_file = Path(config_path).resolve()
    reference_path = Path(reference)
    candidates = []
    if reference_path.is_absolute():
        candidates.append(reference_path)
    else:
        candidates.append((config_file.parent / reference_path).resolve())
        candidates.append((Path(__file__).resolve().parents[2] / reference_path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if reference_path.name == "paths.local.yml":
        example_reference = reference_path.with_name("paths.example.yml")
        example_candidates = []
        if example_reference.is_absolute():
            example_candidates.append(example_reference)
        else:
            example_candidates.append((config_file.parent / example_reference).resolve())
            example_candidates.append((Path(__file__).resolve().parents[2] / example_reference).resolve())
        for candidate in example_candidates:
            if candidate.exists():
                return candidate

    return candidates[0]


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
        resolved_paths_config = resolve_config_reference(config_path, paths_config)
        paths_mapping = load_yaml(resolved_paths_config)
    else:
        resolved_paths_config = config_path
        paths_mapping = {}
    paths = ProjectPaths.from_mapping(paths_mapping, config_path=resolved_paths_config)
    return config, paths
