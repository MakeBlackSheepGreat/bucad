from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_yaml
from src.utils.paths import ProjectPaths, resolve_path


def export_demo_assets(config_path: str | Path, output_dir: str | Path) -> Path:
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
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export assets required for the demo bundle.")
    parser.add_argument("--config", required=True, help="Path to paths.local.yml")
    parser.add_argument("--output-dir", default="./artifacts/demo_assets")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = export_demo_assets(args.config, args.output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
