"""CLI for exporting packaged demo assets."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_yaml
from src.utils.paths import ProjectPaths, resolve_path
from src.utils.reporting import write_markdown_report


def _sha256(path: Path) -> str:
    """Calculate a file digest for the release manifest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_rows(destination: Path) -> list[tuple[str, str, int]]:
    """Collect relative file paths, hashes, and sizes under the release folder."""
    rows: list[tuple[str, str, int]] = []
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path.name in {"release_v1.sha256"}:
            continue
        relative = path.relative_to(destination).as_posix()
        rows.append((relative, _sha256(path), path.stat().st_size))
    return rows


def _write_sha256_manifest(destination: Path, rows: list[tuple[str, str, int]]) -> Path:
    """Write the plain SHA-256 checksum file consumed by release reviewers."""
    manifest_path = destination / "release_v1.sha256"
    manifest_path.write_text(
        "\n".join(f"{sha256}  {relative}" for relative, sha256, _ in rows) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _release_manifest_lines(destination: Path, rows: list[tuple[str, str, int]]) -> list[str]:
    """Format the Markdown release manifest from collected file metadata."""
    lines = [
        "# Release v1 Manifest",
        "",
        f"Release directory: `{destination}`",
        "",
        "| File | Size Bytes | SHA-256 |",
        "| --- | ---: | --- |",
    ]
    for relative, sha256, size in rows:
        lines.append(f"| `{relative}` | {size} | `{sha256}` |")
    return lines


def export_demo_assets(config_path: str | Path, output_dir: str | Path) -> Path:
    """Copy demo configs and default checkpoints into a release asset folder."""
    paths_config = load_yaml(config_path)
    paths = ProjectPaths.from_mapping(paths_config, config_path=config_path)
    destination = resolve_path(output_dir, base_dir=paths.project_root)
    destination.mkdir(parents=True, exist_ok=True)

    configs_dir = destination / "configs"
    source_configs_dir = paths.project_root / "configs"
    if source_configs_dir.exists():
        shutil.copytree(source_configs_dir, configs_dir, dirs_exist_ok=True)
    else:
        configs_dir.mkdir(exist_ok=True)
    shutil.copy2(config_path, configs_dir / Path(config_path).name)

    checkpoints_dir = destination / "artifacts" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    for source in (paths.default_classifier_ckpt, paths.default_segmenter_ckpt):
        if source.exists():
            shutil.copy2(source, checkpoints_dir / source.name)
    release_rows = _manifest_rows(destination)
    _write_sha256_manifest(destination, release_rows)
    write_markdown_report(
        paths.reports_root / "release_v1_manifest.md",
        _release_manifest_lines(destination, release_rows),
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    """Build CLI options for packaged demo asset export."""
    parser = argparse.ArgumentParser(description="Export assets required for the demo bundle.")
    parser.add_argument("--config", required=True, help="Path to paths.local.yml")
    parser.add_argument("--output-dir", default="./artifacts/demo_assets")
    return parser


def main() -> int:
    """Export release assets and print the output directory."""
    args = build_parser().parse_args()
    output = export_demo_assets(args.config, args.output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
