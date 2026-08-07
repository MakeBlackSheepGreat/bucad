"""Diagnose LesioNeXt-LENS evidence alignment on one BUSBRA fold."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.busbra import BUSBRAClassificationLENSDataSet, load_busbra_manifest
from src.models.classifier import load_classifier
from src.preprocess.transforms import build_classifier_transform
from src.utils.config import load_project_config
from src.utils.runtime import optional_import


torch = optional_import("torch")


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    """Resolve a path relative to the project root when needed."""
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _bbox_mask(bbox, height: int, width: int, margin: float) -> np.ndarray:
    """Create a padded rectangular target mask from normalized BBOX coordinates."""
    mask = np.zeros((height, width), dtype=np.float32)
    x1, y1, x2, y2 = [float(item) for item in bbox]
    pad_x = (x2 - x1) * float(margin)
    pad_y = (y2 - y1) * float(margin)
    x1 = max(0.0, x1 - pad_x)
    y1 = max(0.0, y1 - pad_y)
    x2 = min(1.0, x2 + pad_x)
    y2 = min(1.0, y2 + pad_y)
    left = min(width - 1, max(0, int(np.floor(x1 * width))))
    top = min(height - 1, max(0, int(np.floor(y1 * height))))
    right = min(width, max(left + 1, int(np.ceil(x2 * width))))
    bottom = min(height, max(top + 1, int(np.ceil(y2 * height))))
    mask[top:bottom, left:right] = 1.0
    return mask


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze LENS evidence maps and latency.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    if torch is None:
        raise RuntimeError("Torch is required for LENS diagnostics.")
    config, paths = load_project_config(args.config)
    data_cfg = config.get("data", {})
    preprocess_cfg = data_cfg.get("preprocess", {}) or {}
    transform = build_classifier_transform(
        image_size=int(data_cfg.get("image_size", 224)),
        apply_clahe_enabled=bool(preprocess_cfg.get("clahe", False)),
        mean=preprocess_cfg.get("mean"),
        std=preprocess_cfg.get("std"),
        interpolation=str(preprocess_cfg.get("interpolation", "area")),
        crop_pct=float(preprocess_cfg.get("crop_pct", 1.0)),
    )
    manifest = load_busbra_manifest(paths.busbra_root)
    split_path = _resolve_path(config["training"]["split_path"], paths.project_root)
    split = pd.read_csv(split_path)
    val_ids = set(split.loc[(split["fold_id"] == int(args.fold)) & (split["stage"] == "val"), "sample_id"].astype(str))
    val_manifest = manifest[manifest["sample_id"].astype(str).isin(val_ids)].copy()
    if args.max_samples > 0:
        val_manifest = val_manifest.head(int(args.max_samples))
    dataset = BUSBRAClassificationLENSDataSet(
        val_manifest,
        image_size=int(data_cfg.get("image_size", 224)),
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(data_cfg.get("batch_size", 8)), shuffle=False, num_workers=0)
    model_cfg = dict(config.get("model", {}))
    model = load_classifier(model_cfg, checkpoint_path=_resolve_path(args.checkpoint, paths.project_root), map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() and str(config.get("device", "auto")).lower() != "cpu" else "cpu")
    model.to(device).eval()
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    warmup = max(0, int(args.warmup))
    alpha_values = []
    concentration_values = []
    agreement_values = []
    inside_values = []
    background_values = []
    valid_count = 0
    sample_count = 0
    timings = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(device=device, dtype=torch.float32)
            start = time.perf_counter()
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(1, images.shape[0])
            if batch_index >= warmup:
                timings.append(elapsed_ms)
            evidence = getattr(model, "last_evidence_map", None)
            if evidence is None:
                continue
            evidence = torch.softmax(evidence.flatten(1), dim=1).reshape_as(evidence).detach().cpu().numpy()[:, 0]
            bbox_values = batch["bbox"].detach().cpu().numpy()
            bbox_valid = batch["bbox_valid"].detach().cpu().numpy() > 0.5
            alpha = getattr(model, "last_evidence_alpha", None)
            if alpha is not None:
                alpha_values.extend(alpha.detach().cpu().reshape(-1).tolist())
            concentration = getattr(model, "last_evidence_concentration", None)
            if concentration is not None:
                concentration_values.extend(concentration.detach().cpu().reshape(-1).tolist())
            agreement = getattr(model, "last_evidence_agreement", None)
            if agreement is not None:
                agreement_values.extend(agreement.detach().cpu().reshape(-1).tolist())
            for index in range(evidence.shape[0]):
                sample_count += 1
                if not bbox_valid[index]:
                    continue
                valid_count += 1
                target = _bbox_mask(bbox_values[index], evidence.shape[1], evidence.shape[2], float(config.get("training", {}).get("lesion_evidence", {}).get("bbox_margin", 0.08)))
                mass_inside = float(evidence[index][target > 0].sum())
                mass_background = float(evidence[index][target <= 0].sum())
                inside_values.append(mass_inside)
                background_values.append(mass_background)
    report = {
        "model": model_cfg,
        "checkpoint": str(_resolve_path(args.checkpoint, paths.project_root)),
        "fold": int(args.fold),
        "device": str(device),
        "sample_count": sample_count,
        "bbox_valid_count": valid_count,
        "bbox_valid_ratio": float(valid_count / sample_count) if sample_count else 0.0,
        "mean_bbox_evidence_mass": float(np.mean(inside_values)) if inside_values else None,
        "mean_background_evidence_mass": float(np.mean(background_values)) if background_values else None,
        "mean_evidence_alpha": float(np.mean(alpha_values)) if alpha_values else None,
        "evidence_alpha_std": float(np.std(alpha_values)) if alpha_values else None,
        "mean_evidence_concentration": float(np.mean(concentration_values)) if concentration_values else None,
        "mean_local_global_agreement": float(np.mean(agreement_values)) if agreement_values else None,
        "parameter_count": parameter_count,
        "single_image_latency_ms_mean": float(np.mean(timings)) if timings else None,
        "single_image_latency_ms_p95": float(np.percentile(timings, 95)) if timings else None,
        "timed_batches": len(timings),
    }
    output = _resolve_path(args.output, paths.project_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
