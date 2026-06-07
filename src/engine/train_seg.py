"""Segmenter training loop and metric report writer for BUSBRA folds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.datasets.busbra import BUSBRASegmentationDataset, generate_busbra_split_assignments, load_busbra_manifest
from src.engine.checkpoints import atomic_torch_save
from src.engine.segmentation_losses import segmentation_loss
from src.models.segmenter import create_segmenter
from src.utils.config import load_project_config
from src.utils.logging import get_logger
from src.utils.metrics import boundary_f1_score, dice_score, hd95_score, iou_score
from src.utils.reporting import write_json_report
from src.utils.runtime import ensure_dir, optional_import, require_dependency, seed_everything, select_device


torch = optional_import("torch")
optim = optional_import("torch.optim")
torch_utils_data = optional_import("torch.utils.data")


def _model_forward_outputs(model, images):
    if hasattr(model, "forward_with_aux"):
        return model.forward_with_aux(images)
    return model(images)


def _mask_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["mask"]
    return outputs


def _load_or_create_splits(
    manifest: pd.DataFrame, split_path: Path, *, fold_count: int, seed: int
) -> pd.DataFrame:
    if split_path.exists():
        return pd.read_csv(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    assignments = generate_busbra_split_assignments(
        manifest, n_splits=fold_count, seed=seed
    )
    assignments.to_csv(split_path, index=False)
    return assignments


def _split_manifest_for_fold(
    manifest: pd.DataFrame, assignments: pd.DataFrame, fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_assignments = assignments[assignments["fold_id"] == fold]
    train_ids = set(fold_assignments.loc[fold_assignments["stage"] == "train", "sample_id"])
    val_ids = set(fold_assignments.loc[fold_assignments["stage"] == "val", "sample_id"])
    train_manifest = manifest[manifest["sample_id"].isin(train_ids)].reset_index(drop=True)
    val_manifest = manifest[manifest["sample_id"].isin(val_ids)].reset_index(drop=True)
    return train_manifest, val_manifest


def _prepare_fold_manifests(
    *,
    manifest: pd.DataFrame,
    paths,
    training_cfg: dict[str, Any],
    fold: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_path = Path(training_cfg.get("split_path", paths.reports_root / "busbra_5fold_splits.csv"))
    if not split_path.is_absolute():
        split_path = (paths.project_root / split_path).resolve()
    assignments = _load_or_create_splits(
        manifest,
        split_path,
        fold_count=int(training_cfg.get("fold_count", 5)),
        seed=seed,
    )
    train_manifest, val_manifest = _split_manifest_for_fold(manifest, assignments, fold)
    if train_manifest.empty or val_manifest.empty:
        raise ValueError(f"Fold {fold} produced an empty train or validation split.")
    return train_manifest, val_manifest


def _build_segmenter(model_cfg: dict[str, Any], *, device: str):
    model_kwargs = {
        key: value
        for key, value in model_cfg.items()
        if key
        not in {
            "architecture",
            "encoder_name",
            "encoder_weights",
            "in_channels",
            "classes",
        }
    }
    return create_segmenter(
        architecture=model_cfg.get("architecture", "unet"),
        encoder_name=model_cfg.get("encoder_name", "resnet18"),
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        in_channels=int(model_cfg.get("in_channels", 3)),
        classes=int(model_cfg.get("classes", 1)),
        **model_kwargs,
    ).to(device)


def _build_segmentation_loaders(
    *,
    train_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
    data_cfg: dict[str, Any],
    device: str,
):
    image_size = int(data_cfg.get("image_size", 256))
    train_dataset = BUSBRASegmentationDataset(train_manifest, image_size=image_size)
    val_dataset = BUSBRASegmentationDataset(val_manifest, image_size=image_size)
    num_workers = int(data_cfg.get("num_workers", 0))
    loader_kwargs = {
        "batch_size": int(data_cfg.get("batch_size", 4)),
        "num_workers": num_workers,
        "pin_memory": device.startswith("cuda"),
    }
    # Avoid enabling persistent workers for the default Windows smoke path.
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = torch_utils_data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = torch_utils_data.DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def _evaluate_model(model, loader, device: str, *, max_batches: int | None = None) -> dict[str, Any]:
    require_dependency("torch", torch)
    model.eval()
    dice_scores: list[float] = []
    iou_scores: list[float] = []
    boundary_scores: list[float] = []
    hd95_scores: list[float] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            images = batch["image"].to(device=device, dtype=torch.float32)
            masks = batch["mask"].to(device=device, dtype=torch.float32)
            outputs = _model_forward_outputs(model, images)
            logits = _mask_logits(outputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = masks.cpu().numpy()
            for pred, target in zip(probs, targets):
                dice_scores.append(dice_score(pred[0], target[0]))
                iou_scores.append(iou_score(pred[0], target[0]))
                boundary_scores.append(boundary_f1_score(pred[0], target[0]))
                hd95_scores.append(hd95_score(pred[0], target[0]))
    finite_hd95 = [value for value in hd95_scores if np.isfinite(value)]
    return {
        "dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
        "boundary_f1": float(np.mean(boundary_scores)) if boundary_scores else 0.0,
        "hd95": float(np.mean(finite_hd95)) if finite_hd95 else float("inf"),
    }


def _train_segmentation_epoch(
    model,
    train_loader,
    optimizer,
    *,
    device: str,
    loss_cfg: dict[str, Any],
    max_train_batches: int | None,
) -> tuple[float, dict[str, float]]:
    model.train()
    losses: list[float] = []
    component_losses: dict[str, list[float]] = {}
    for batch_index, batch in enumerate(train_loader, start=1):
        if max_train_batches is not None and batch_index > max_train_batches:
            break
        images = batch["image"].to(device=device, dtype=torch.float32)
        masks = batch["mask"].to(device=device, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = _model_forward_outputs(model, images)
        loss, components = segmentation_loss(outputs, masks, loss_cfg)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        for name, value in components.items():
            component_losses.setdefault(name, []).append(float(value.detach().item()))
    component_summary = {
        name: float(np.mean(values))
        for name, values in sorted(component_losses.items())
        if values
    }
    return float(np.mean(losses)) if losses else 0.0, component_summary


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    atomic_torch_save(payload, destination)


def _write_segmentation_training_outputs(
    *,
    config: dict[str, Any],
    paths,
    output_cfg: dict[str, Any],
    fold: int,
    device: str,
    model,
    metrics: dict[str, Any],
    loss_cfg: dict[str, Any],
    train_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
    max_train_batches: int | None,
    max_val_batches: int | None,
) -> dict[str, Any]:
    checkpoints_dir = ensure_dir(paths.checkpoints_root)
    reports_dir = ensure_dir(paths.reports_root)
    checkpoint_path = checkpoints_dir / output_cfg.get("checkpoint_name", "segmenter_fold{fold}.pt").format(fold=fold)
    report_path = reports_dir / output_cfg.get("report_name", "train_seg_fold{fold}.json").format(fold=fold)

    _atomic_torch_save(
        {
            "state_dict": model.state_dict(),
            "model_config": config.get("model", {}),
            "loss_config": loss_cfg,
            "fold": fold,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    report = {
        "fold": fold,
        "device": device,
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics,
        "loss_config": loss_cfg,
        "train_size": int(len(train_manifest)),
        "val_size": int(len(val_manifest)),
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
    }
    write_json_report(report_path, report)
    return report


def run_segmentation_training(
    config_path: str | Path,
    *,
    fold: int = 1,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    """Train one BUSBRA segmentation fold and write checkpoint plus JSON report."""
    require_dependency("torch", torch)
    require_dependency("torch.optim", optim)
    require_dependency("torch.utils.data", torch_utils_data)

    config, paths = load_project_config(config_path)
    logger = get_logger("train_seg")
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = select_device(str(config.get("device", "auto")))

    manifest = load_busbra_manifest(paths.busbra_root)
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})
    loss_cfg = config.get("loss", {"name": "bce"})
    max_train_batches = training_cfg.get("max_train_batches")
    max_val_batches = training_cfg.get("max_val_batches")
    max_train_batches = int(max_train_batches) if max_train_batches is not None else None
    max_val_batches = int(max_val_batches) if max_val_batches is not None else None

    train_manifest, val_manifest = _prepare_fold_manifests(
        manifest=manifest,
        paths=paths,
        training_cfg=training_cfg,
        fold=fold,
        seed=seed,
    )
    train_loader, val_loader = _build_segmentation_loaders(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        data_cfg=data_cfg,
        device=device,
    )
    model = _build_segmenter(config.get("model", {}), device=device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(epochs_override or training_cfg.get("epochs", 5))
    for epoch in range(epochs):
        mean_loss, component_summary = _train_segmentation_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            loss_cfg=loss_cfg,
            max_train_batches=max_train_batches,
        )
        logger.info(
            "fold=%s epoch=%s loss=%.4f components=%s",
            fold,
            epoch + 1,
            mean_loss,
            component_summary,
        )

    metrics = _evaluate_model(model, val_loader, device, max_batches=max_val_batches)
    return _write_segmentation_training_outputs(
        config=config,
        paths=paths,
        output_cfg=output_cfg,
        fold=fold,
        device=device,
        model=model,
        metrics=metrics,
        loss_cfg=loss_cfg,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )
