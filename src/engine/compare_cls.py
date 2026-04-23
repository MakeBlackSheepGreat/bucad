from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from src.engine.train_cls import run_classifier_training
from src.models.classifier import create_classifier
from src.utils.config import deep_merge, load_project_config, load_yaml, save_yaml
from src.utils.reporting import write_json_report


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _resolve_project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _load_base_config(comparison_config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    base_config_ref = comparison_config.get("base_config", "configs/classifier/baseline.yml")
    base_config_path = Path(base_config_ref)
    if not base_config_path.is_absolute():
        candidates = [
            (config_file.parent / base_config_path).resolve(),
            (Path(__file__).resolve().parents[2] / base_config_path).resolve(),
        ]
        base_config_path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
    return load_yaml(base_config_path)


def _build_model_run_config(
    base_config: dict[str, Any],
    comparison_config: dict[str, Any],
    model_entry: dict[str, Any],
) -> dict[str, Any]:
    model_name = str(model_entry["name"])
    safe_name = _safe_model_name(model_name)
    overrides = {
        "paths_config": comparison_config.get("paths_config", base_config.get("paths_config")),
        "seed": comparison_config.get("seed", base_config.get("seed", 42)),
        "device": comparison_config.get("device", base_config.get("device", "auto")),
        "model": {
            "name": model_name,
            "pretrained": bool(model_entry.get("pretrained", True)),
            "in_chans": int(model_entry.get("in_chans", 3)),
            "num_classes": int(model_entry.get("num_classes", 2)),
        },
        "data": comparison_config.get("data", base_config.get("data", {})),
        "training": comparison_config.get("training", base_config.get("training", {})),
        "output": {
            "checkpoint_name": f"{safe_name}_fold{{fold}}.pt",
            "report_name": f"comparison_{safe_name}_fold{{fold}}.json",
        },
    }
    return deep_merge(base_config, overrides)


def run_classifier_comparison(
    config_path: str | Path,
    *,
    fold: int = 1,
    epochs_override: int | None = None,
    model_limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config, paths = load_project_config(config_path)
    base_config = _load_base_config(config, config_path)
    entries = list(config.get("comparison", {}).get("models", []))
    if model_limit is not None:
        entries = entries[: int(model_limit)]

    results: list[dict[str, Any]] = []
    temp_root = Path(tempfile.gettempdir()) / "bucad_comparison_configs"
    temp_root.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        model_name = str(entry["name"])
        started = time.perf_counter()
        run_config = _build_model_run_config(base_config, config, entry)
        config_file = temp_root / f"{_safe_model_name(model_name)}_fold{fold}.yml"
        save_yaml(config_file, run_config)
        result: dict[str, Any] = {
            "model_name": model_name,
            "description": entry.get("description", model_name),
            "fold": int(fold),
            "config_path": str(config_file),
            "status": "planned" if dry_run else "running",
            "dataset_boundary": {
                "training_dataset": "BUSBRA",
                "external_evaluation_dataset": "BUSI",
                "busi_used_for_training": False,
            },
        }
        try:
            if dry_run:
                create_classifier(
                    model_name=model_name,
                    pretrained=False,
                    in_chans=int(entry.get("in_chans", 3)),
                    num_classes=int(entry.get("num_classes", 2)),
                )
                result["status"] = "validated"
                result["metrics"] = {}
            else:
                training_report = run_classifier_training(
                    config_file,
                    fold=fold,
                    epochs_override=epochs_override,
                )
                result["status"] = "completed"
                result["metrics"] = training_report.get("metrics", {})
                result["checkpoint_path"] = training_report.get("checkpoint_path")
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)
        result["runtime_seconds"] = round(time.perf_counter() - started, 3)
        results.append(result)

    output_ref = config.get("comparison", {}).get(
        "output_path", "./artifacts/reports/comparison_results.json"
    )
    output_path = _resolve_project_path(output_ref, paths.project_root)
    report = {
        "fold": int(fold),
        "dry_run": bool(dry_run),
        "model_count": len(results),
        "results": results,
    }
    write_json_report(output_path, report)
    return report
