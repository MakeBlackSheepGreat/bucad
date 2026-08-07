"""Summarize Block Attention Residual behavior on one BUSBRA validation fold."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.busbra import BUSBRAClassificationDataset, load_busbra_manifest
from src.models.classifier import create_classifier
from src.preprocess.transforms import build_classifier_transform
from src.utils.config import load_project_config
from src.utils.runtime import require_dependency, optional_import


torch = optional_import("torch")
torch_utils_data = optional_import("torch.utils.data")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a LesioNeXt AttnRes validation fold.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--output", required=True)
    return parser


def _device_from_config(value: str) -> str:
    if str(value).lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(value)


def main() -> int:
    require_dependency("torch", torch)
    require_dependency("torch.utils.data", torch_utils_data)
    args = _parser().parse_args()
    config, paths = load_project_config(args.config)
    model_cfg = dict(config["model"])
    device = _device_from_config(str(config.get("device", "auto")))
    model_cfg.pop("pretrained", None)
    model = create_classifier(
        model_name=model_cfg.pop("name"),
        pretrained=False,
        in_chans=int(model_cfg.pop("in_chans", 3)),
        num_classes=int(model_cfg.pop("num_classes", 2)),
        **model_cfg,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    data_cfg = config["data"]
    preprocess_cfg = data_cfg.get("preprocess", {}) or {}
    pretrained_cfg = getattr(model, "pretrained_cfg", {}) or {}
    image_size = int(pretrained_cfg.get("input_size", (3, data_cfg["image_size"], data_cfg["image_size"]))[-1])
    transform = build_classifier_transform(
        image_size=image_size,
        apply_clahe_enabled=bool(preprocess_cfg.get("clahe", False)),
        mean=list(pretrained_cfg.get("mean", (0.485, 0.456, 0.406))),
        std=list(pretrained_cfg.get("std", (0.229, 0.224, 0.225))),
        interpolation=str(pretrained_cfg.get("interpolation", "bicubic")),
        crop_pct=float(pretrained_cfg.get("crop_pct", 0.95)),
    )
    manifest = load_busbra_manifest(paths.busbra_root)
    assignments = pd.read_csv(paths.project_root / config["training"]["split_path"])
    val_ids = set(
        assignments.loc[
            (assignments["fold_id"] == int(args.fold)) & (assignments["stage"] == "val"),
            "sample_id",
        ].astype(str)
    )
    val_manifest = manifest.loc[manifest["sample_id"].astype(str).isin(val_ids)].copy()
    loader = torch_utils_data.DataLoader(
        BUSBRAClassificationDataset(val_manifest, image_size=image_size, transform=transform),
        batch_size=int(data_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=0,
    )

    totals: dict[str, list[np.ndarray]] = {}
    batch_count = 0
    first_image = None
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            if first_image is None:
                first_image = images[:1]
            model(images)
            for stage, rows in model.attention_summary().items():
                if stage not in totals:
                    totals[stage] = [np.zeros(len(row), dtype=np.float64) for row in rows]
                for index, row in enumerate(rows):
                    totals[stage][index] += np.asarray(row, dtype=np.float64)
            batch_count += 1

    if first_image is None:
        raise RuntimeError("Validation fold is empty.")
    for _ in range(10):
        model(first_image)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(30):
        model(first_image)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000.0 / 30.0

    gates = {
        f"stage_{stage}": [float(value) for value in torch.tanh(module.residual_gates).detach().cpu().tolist()]
        for stage, module in model.attnres.items()
    }
    attention = {
        stage: [[float(value) for value in row / float(batch_count)] for row in rows]
        for stage, rows in totals.items()
    }
    report = {
        "fold": int(args.fold),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "validation_sample_count": int(len(val_manifest)),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "single_image_latency_ms": latency_ms,
        "residual_gates_tanh": gates,
        "mean_depth_attention": attention,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
