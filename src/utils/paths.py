"""Project path resolution primitives for configs, data, artifacts, and reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def project_root() -> Path:
    """Return the repository root inferred from this file's location."""
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path, *, base_dir: str | Path | None = None) -> Path:
    """Expand *value* to an absolute path, resolving relative paths against *base_dir*."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    anchor = Path(base_dir) if base_dir is not None else project_root()
    return (anchor / path).resolve()


@dataclass(slots=True)
class ProjectPaths:
    """Resolved filesystem roots shared by training, inference, and reports."""

    project_root: Path
    busbra_root: Path
    busi_root: Path
    artifacts_root: Path
    checkpoints_root: Path
    logs_root: Path
    reports_root: Path
    default_classifier_ckpt: Path
    default_segmenter_ckpt: Path

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any], *, config_path: str | Path | None = None
    ) -> "ProjectPaths":
        """Resolve relative config paths against the config file then project root."""
        config_dir = (
            Path(config_path).resolve().parent if config_path is not None else project_root()
        )
        root = resolve_path(mapping.get("project_root", "."), base_dir=config_dir)
        datasets = mapping.get("datasets", {})
        artifacts = mapping.get("artifacts", {})
        runtime = mapping.get("runtime", {})
        artifacts_root = resolve_path(artifacts.get("root", "./artifacts"), base_dir=root)
        return cls(
            project_root=root,
            busbra_root=resolve_path(datasets.get("busbra_root", "./训练集/BUSBRA"), base_dir=root),
            busi_root=resolve_path(datasets.get("busi_root", "./测试集/Dataset_BUSI_with_GT"), base_dir=root),
            artifacts_root=artifacts_root,
            checkpoints_root=resolve_path(
                artifacts.get("checkpoints", artifacts_root / "checkpoints"),
                base_dir=root,
            ),
            logs_root=resolve_path(
                artifacts.get("logs", artifacts_root / "logs"),
                base_dir=root,
            ),
            reports_root=resolve_path(
                artifacts.get("reports", artifacts_root / "reports"),
                base_dir=root,
            ),
            default_classifier_ckpt=resolve_path(
                runtime.get(
                    "default_classifier_ckpt",
                    artifacts_root / "checkpoints" / "classifier_fold1.pt",
                ),
                base_dir=root,
            ),
            default_segmenter_ckpt=resolve_path(
                runtime.get(
                    "default_segmenter_ckpt",
                    artifacts_root / "checkpoints" / "segmenter_fold1.pt",
                ),
                base_dir=root,
            ),
        )
