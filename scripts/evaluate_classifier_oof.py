"""Evaluate one classifier config across BUSBRA folds and report aggregated OOF metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.busbra import LABEL_TO_INDEX, load_busbra_manifest
from src.engine.classifier_ensemble import ClassifierEnsemble
from src.engine.runtime_config import RuntimeConfig
from src.preprocess.io import read_image
from src.utils.config import load_project_config, load_yaml
from src.utils.metrics import best_threshold_by_youden, classification_metrics
from src.utils.reporting import write_json_report, write_markdown_report
from src.utils.runtime import select_device


def build_parser() -> argparse.ArgumentParser:
    """Build CLI args for OOF evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate BUSBRA OOF metrics for one classifier config.")
    parser.add_argument("--classifier-config", required=True, help="Classifier YAML config path.")
    parser.add_argument(
        "--inference-config",
        default=None,
        help="Optional inference YAML config path. When omitted, one is synthesized from the classifier config.",
    )
    parser.add_argument("--fold-count", type=int, default=5, help="Number of BUSBRA folds to aggregate.")
    parser.add_argument("--device", default="auto", help="Torch device selector.")
    parser.add_argument(
        "--inference-config-template",
        default=None,
        help="Fold-specific inference config template such as configs/inference/model_fold{fold}.yml.",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        default=None,
        help="Optional checkpoint filename pattern such as model_fold{fold}.pt.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--markdown",
        default=None,
        help="Optional Markdown summary path.",
    )
    return parser


def _resolve_project_path(project_root: Path, value: str | Path) -> Path:
    """Resolve one possibly relative project path."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _runtime_from_classifier_config(
    classifier_cfg: dict[str, Any],
    *,
    project_root: Path,
    fold: int,
) -> dict[str, Any]:
    """Build one runtime config compatible with the inference stack."""
    model_cfg = dict(classifier_cfg["model"])
    data_cfg = dict(classifier_cfg.get("data", {}))
    preprocess_cfg = dict(data_cfg.get("preprocess", {}))
    output_cfg = dict(classifier_cfg.get("output", {}))

    checkpoint_name = str(output_cfg["checkpoint_name"]).format(fold=fold)
    checkpoint_path = Path(checkpoint_name)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / "artifacts" / "checkpoints" / checkpoint_path
    runtime = {
        "classifier_model": str(model_cfg["name"]),
        "classifier_pretrained": False,
        "classifier_image_size": int(data_cfg.get("image_size", 224)),
        "classifier_apply_clahe": bool(preprocess_cfg.get("clahe", False)),
        "classifier_checkpoint": str(checkpoint_path),
        "classifier_checkpoints": [],
        "classifier_members": [],
        "default_threshold": 0.5,
        "borderline_margin": 0.08,
        "gradcam_enabled": False,
        "segmentation_enabled": False,
        "allow_missing_visuals": True,
    }
    mean = preprocess_cfg.get("mean")
    std = preprocess_cfg.get("std")
    interpolation = preprocess_cfg.get("interpolation")
    crop_pct = preprocess_cfg.get("crop_pct")
    if mean is not None:
        runtime["classifier_mean"] = list(mean)
    if std is not None:
        runtime["classifier_std"] = list(std)
    if interpolation is not None:
        runtime["classifier_interpolation"] = str(interpolation)
    if crop_pct is not None:
        runtime["classifier_crop_pct"] = float(crop_pct)
    runtime["classifier_tta_horizontal_flip"] = False
    runtime["classifier_model_kwargs"] = {
        key: value
        for key, value in model_cfg.items()
        if key not in {"name", "pretrained", "in_chans", "num_classes"}
    }
    return runtime


def _runtime_from_inference_config(
    inference_path: Path,
) -> dict[str, Any]:
    """Load and resolve a fold-specific runtime config from an inference YAML."""
    payload = load_yaml(inference_path)
    runtime = dict(payload.get("runtime", {}))
    return runtime


def _checkpoint_name_from_classifier_config(classifier_cfg: dict[str, Any], *, fold: int) -> str:
    """Return the expected fold-specific checkpoint filename from a classifier config."""
    output_cfg = dict(classifier_cfg.get("output", {}))
    return str(output_cfg["checkpoint_name"]).format(fold=fold)


def _override_runtime_checkpoint_for_fold(
    runtime: dict[str, Any],
    *,
    project_root: Path,
    checkpoint_name: str,
) -> dict[str, Any]:
    """Return a runtime copy with the primary classifier checkpoint replaced for one fold."""
    resolved = dict(runtime)
    original = resolved.get("classifier_checkpoint")
    if isinstance(original, str) and original:
        original_path = _resolve_project_path(project_root, original)
        replaced = original_path.with_name(checkpoint_name)
    else:
        replaced = project_root / "artifacts" / "checkpoints" / checkpoint_name
    resolved["classifier_checkpoint"] = str(replaced)
    resolved["classifier_checkpoints"] = []
    resolved["classifier_members"] = []
    return resolved


def _manifest_for_fold(
    manifest: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    fold: int,
) -> pd.DataFrame:
    """Return the validation manifest subset for one fold."""
    fold_rows = assignments[(assignments["fold_id"] == int(fold)) & (assignments["stage"] == "val")]
    sample_ids = set(fold_rows["sample_id"].astype(str))
    subset = manifest[manifest["sample_id"].astype(str).isin(sample_ids)].copy()
    return subset.reset_index(drop=True)


def _predict_fold_rows(
    manifest: pd.DataFrame,
    runtime_config: dict[str, Any],
    *,
    project_root: Path,
    device: str,
) -> list[dict[str, Any]]:
    """Predict malignant probabilities for one fold validation subset."""
    runtime_payload = dict(runtime_config)
    runtime_payload["device"] = device
    ensemble = ClassifierEnsemble(RuntimeConfig.from_mapping(runtime_payload))
    rows: list[dict[str, Any]] = []
    for item in manifest.itertuples(index=False):
        image = read_image(item.image_path, grayscale=True)
        benign_probability, malignant_probability = ensemble.predict_on_image(image)
        rows.append(
            {
                "sample_id": str(item.sample_id),
                "case_id": str(item.case_id),
                "pathology_label": str(item.pathology_label),
                "malignant_probability": float(malignant_probability),
            }
        )
    return rows


def _markdown_lines(report: dict[str, Any]) -> list[str]:
    """Render a short Markdown summary."""
    metrics = report["metrics"]
    best = report["best_by_youden"]
    lines = [
        f"# {report['model_name']} BUSBRA OOF Summary",
        "",
        f"- folds: `{report['fold_count']}`",
        f"- sample count: `{report['sample_count']}`",
        f"- AUC@0.50: `{metrics['auc']:.6f}`",
        f"- Sensitivity@0.50: `{metrics['sensitivity']:.6f}`",
        f"- Specificity@0.50: `{metrics['specificity']:.6f}`",
        f"- F1@0.50: `{metrics['f1_score']:.6f}`",
        "",
        "## Best Threshold By Youden",
        "",
        f"- threshold: `{best['threshold']:.2f}`",
        f"- AUC: `{best['auc']:.6f}`",
        f"- Sensitivity: `{best['sensitivity']:.6f}`",
        f"- Specificity: `{best['specificity']:.6f}`",
        f"- F1: `{best['f1_score']:.6f}`",
    ]
    return lines


def main() -> int:
    """Run BUSBRA OOF evaluation and write reports."""
    args = build_parser().parse_args()
    classifier_config_path = Path(args.classifier_config)
    classifier_cfg, project_paths = load_project_config(classifier_config_path)
    project_root = project_paths.project_root
    manifest = load_busbra_manifest(project_paths.busbra_root)

    training_cfg = dict(classifier_cfg.get("training", {}))
    split_path = _resolve_project_path(project_root, training_cfg["split_path"])
    assignments = pd.read_csv(split_path)

    runtime_config_by_fold: dict[int, dict[str, Any]] = {}
    inference_path = Path(args.inference_config) if args.inference_config else None
    inference_template = str(args.inference_config_template) if args.inference_config_template else None
    for fold in range(1, int(args.fold_count) + 1):
        fold_inference_path: Path | None = None
        if inference_template:
            fold_inference_path = Path(inference_template.format(fold=fold))
        elif inference_path is not None:
            fold_inference_path = inference_path
        if fold_inference_path is not None:
            runtime = _runtime_from_inference_config(fold_inference_path)
            checkpoint_name = (
                str(args.checkpoint_pattern).format(fold=fold)
                if args.checkpoint_pattern
                else _checkpoint_name_from_classifier_config(classifier_cfg, fold=fold)
            )
            runtime = _override_runtime_checkpoint_for_fold(
                runtime,
                project_root=project_root,
                checkpoint_name=checkpoint_name,
            )
        else:
            runtime = _runtime_from_classifier_config(
                classifier_cfg,
                project_root=project_root,
                fold=fold,
            )
        checkpoint = runtime.get("classifier_checkpoint")
        if isinstance(checkpoint, str):
            runtime["classifier_checkpoint"] = str(_resolve_project_path(project_root, checkpoint))
            if not Path(runtime["classifier_checkpoint"]).exists():
                raise FileNotFoundError(
                    f"Classifier checkpoint for fold {fold} was not found: "
                    f"{runtime['classifier_checkpoint']}"
                )
        members = runtime.get("classifier_members")
        if isinstance(members, list):
            resolved_members = []
            for member in members:
                resolved = dict(member)
                if "checkpoint" in resolved:
                    resolved["checkpoint"] = str(_resolve_project_path(project_root, resolved["checkpoint"]))
                resolved_members.append(resolved)
            runtime["classifier_members"] = resolved_members
        runtime_config_by_fold[fold] = runtime

    device = select_device(str(args.device))
    all_rows: list[dict[str, Any]] = []
    for fold in range(1, int(args.fold_count) + 1):
        fold_manifest = _manifest_for_fold(manifest, assignments, fold=fold)
        fold_rows = _predict_fold_rows(
            fold_manifest,
            runtime_config_by_fold[fold],
            project_root=project_root,
            device=device,
        )
        for row in fold_rows:
            row["fold_id"] = fold
        all_rows.extend(fold_rows)

    y_true = [LABEL_TO_INDEX[str(row["pathology_label"]).lower()] for row in all_rows]
    probabilities = [float(row["malignant_probability"]) for row in all_rows]
    metrics = classification_metrics(y_true, probabilities, threshold=0.5)
    best = best_threshold_by_youden(y_true, probabilities)
    report = {
        "method": "single-model BUSBRA OOF evaluation",
        "model_name": str(classifier_cfg["model"]["name"]),
        "classifier_config": str(classifier_config_path),
        "inference_config": str(inference_path) if inference_path is not None else None,
        "inference_config_template": inference_template,
        "fold_count": int(args.fold_count),
        "sample_count": int(len(all_rows)),
        "metrics": metrics,
        "best_by_youden": best,
        "rows": all_rows,
    }
    output_path = Path(args.output)
    write_json_report(output_path, report)
    if args.markdown:
        write_markdown_report(Path(args.markdown), _markdown_lines(report))
    print(json.dumps({"output": str(output_path), "metrics": metrics, "best_by_youden": best}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
