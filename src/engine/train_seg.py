"""Segmenter training loop and metric report writer for BUSBRA folds."""

from __future__ import annotations

import dataclasses
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
from src.utils.metrics import (
    boundary_f1_score,
    classification_metrics,
    dice_score,
    hd95_score,
    iou_score,
)
from src.utils.reporting import write_json_report
from src.utils.runtime import ensure_dir, optional_import, require_dependency, seed_everything, select_device


torch = optional_import("torch")
optim = optional_import("torch.optim")
torch_utils_data = optional_import("torch.utils.data")


def _model_forward_outputs(model, images):
    """Return model outputs, using auxiliary outputs when the model exposes them."""
    if hasattr(model, "forward_with_aux"):
        return model.forward_with_aux(images)
    return model(images)


def _mask_logits(outputs):
    """Extract mask logits from dict outputs or raw tensor outputs."""
    if isinstance(outputs, dict):
        return outputs["mask"]
    return outputs


def _load_or_create_splits(
    manifest: pd.DataFrame, split_path: Path, *, fold_count: int, seed: int
) -> pd.DataFrame:
    """Load existing fold assignments or create BUSBRA split assignments."""
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
    """Split a manifest into train/validation rows for one segmentation fold."""
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
    """Prepare non-empty train/validation manifests for one segmentation fold."""
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
    """Construct the configured segmentation model and move it to device."""
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
    """Create segmentation train/validation DataLoaders with Windows-safe defaults."""
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
    """Evaluate segmentation metrics over a validation loader."""
    require_dependency("torch", torch)
    model.eval()
    dice_scores: list[float] = []
    iou_scores: list[float] = []
    boundary_scores: list[float] = []
    hd95_scores: list[float] = []
    classification_labels: list[int] = []
    malignant_probabilities: list[float] = []
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
            if isinstance(outputs, dict) and outputs.get("class_logits") is not None and "label" in batch:
                labels = batch["label"].to(device=device, dtype=torch.long)
                probabilities = torch.softmax(outputs["class_logits"], dim=1)[:, 1]
                classification_labels.extend(labels.cpu().tolist())
                malignant_probabilities.extend(probabilities.cpu().tolist())
    finite_hd95 = [value for value in hd95_scores if np.isfinite(value)]
    metrics = {
        "dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
        "boundary_f1": float(np.mean(boundary_scores)) if boundary_scores else 0.0,
        "hd95": float(np.mean(finite_hd95)) if finite_hd95 else float("inf"),
    }
    if classification_labels:
        classification = classification_metrics(
            classification_labels,
            malignant_probabilities,
        )
        metrics["classification"] = classification
        metrics["classification_accuracy"] = classification["accuracy"]
    return metrics


def _train_segmentation_epoch(
    model,
    train_loader,
    optimizer,
    *,
    device: str,
    loss_cfg: dict[str, Any],
    max_train_batches: int | None,
) -> tuple[float, dict[str, float]]:
    """Run one segmenter epoch and summarize total/component losses."""
    model.train()
    losses: list[float] = []
    component_losses: dict[str, list[float]] = {}
    classification_weight = float(loss_cfg.get("classification_weight", 0.0))
    for batch_index, batch in enumerate(train_loader, start=1):
        if max_train_batches is not None and batch_index > max_train_batches:
            break
        images = batch["image"].to(device=device, dtype=torch.float32)
        masks = batch["mask"].to(device=device, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = _model_forward_outputs(model, images)
        loss, components = segmentation_loss(outputs, masks, loss_cfg)
        if classification_weight and isinstance(outputs, dict) and outputs.get("class_logits") is not None and "label" in batch:
            labels = batch["label"].to(device=device, dtype=torch.long)
            classification_loss = torch.nn.functional.cross_entropy(outputs["class_logits"], labels)
            components["classification"] = classification_loss
            loss = loss + classification_weight * classification_loss
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
    """Compatibility wrapper around the shared atomic checkpoint writer."""
    atomic_torch_save(payload, destination)


@dataclasses.dataclass(slots=True)
class _SegmentationBatchLimits:
    """Optional smoke-test caps for train and validation dataloaders."""

    max_train_batches: int | None
    max_val_batches: int | None


@dataclasses.dataclass(slots=True)
class _SegmentationLoopConfig:
    """Settings needed by the segmentation epoch loop."""

    epochs: int
    loss_cfg: dict[str, Any]
    batch_limits: _SegmentationBatchLimits


@dataclasses.dataclass(slots=True)
class _PreparedSegmentationRun:
    """Objects prepared before segmentation epochs start."""

    model: Any
    train_loader: Any
    val_loader: Any
    train_manifest: pd.DataFrame
    val_manifest: pd.DataFrame
    optimizer: Any
    loop_config: _SegmentationLoopConfig


def _optional_int(value: Any) -> int | None:
    """Convert optional numeric config values to int."""
    return int(value) if value is not None else None


def _parse_segmentation_batch_limits(training_cfg: dict[str, Any]) -> _SegmentationBatchLimits:
    """Parse optional smoke-test batch caps from training config."""
    return _SegmentationBatchLimits(
        max_train_batches=_optional_int(training_cfg.get("max_train_batches")),
        max_val_batches=_optional_int(training_cfg.get("max_val_batches")),
    )


def _build_segmentation_optimizer(model, training_cfg: dict[str, Any]):
    """Build the AdamW optimizer used by segmentation training."""
    return optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )


def _prepare_segmentation_training_run(
    *,
    config: dict[str, Any],
    paths,
    fold: int,
    seed: int,
    device: str,
    epochs_override: int | None,
) -> _PreparedSegmentationRun:
    """Prepare fold data, model, optimizer, and loop settings."""
    manifest = load_busbra_manifest(paths.busbra_root)
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    loss_cfg = config.get("loss", {"name": "bce"})

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
    optimizer = _build_segmentation_optimizer(model, training_cfg)
    loop_config = _SegmentationLoopConfig(
        epochs=int(epochs_override or training_cfg.get("epochs", 5)),
        loss_cfg=loss_cfg,
        batch_limits=_parse_segmentation_batch_limits(training_cfg),
    )
    return _PreparedSegmentationRun(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        optimizer=optimizer,
        loop_config=loop_config,
    )


def _run_segmentation_loop(
    *,
    model,
    train_loader,
    optimizer,
    device: str,
    fold: int,
    loop_config: _SegmentationLoopConfig,
    logger,
) -> None:
    """Run all configured segmentation epochs and log their loss summaries."""
    cfg = loop_config
    for epoch in range(cfg.epochs):
        mean_loss, component_summary = _train_segmentation_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            loss_cfg=cfg.loss_cfg,
            max_train_batches=cfg.batch_limits.max_train_batches,
        )
        logger.info(
            "fold=%s epoch=%s loss=%.4f components=%s",
            fold,
            epoch + 1,
            mean_loss,
            component_summary,
        )


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
    """Persist the trained segmenter checkpoint and fold report."""
    checkpoints_dir = ensure_dir(paths.checkpoints_root)
    reports_dir = ensure_dir(paths.reports_root)
    checkpoint_path = checkpoints_dir / output_cfg.get("checkpoint_name", "segmenter_fold{fold}.pt").format(fold=fold)
    report_path = reports_dir / output_cfg.get("report_name", "train_seg_fold{fold}.json").format(fold=fold)

    _atomic_torch_save(
        _segmentation_checkpoint_payload(
            config=config,
            model=model,
            loss_cfg=loss_cfg,
            fold=fold,
            metrics=metrics,
        ),
        checkpoint_path,
    )
    report = _segmentation_training_report(
        fold=fold,
        device=device,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
        loss_cfg=loss_cfg,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )
    write_json_report(report_path, report)
    return report


def _segmentation_checkpoint_payload(
    *,
    config: dict[str, Any],
    model,
    loss_cfg: dict[str, Any],
    fold: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Keep the segmenter checkpoint schema next to the report schema."""
    return {
        "state_dict": model.state_dict(),
        "model_config": config.get("model", {}),
        "loss_config": loss_cfg,
        "fold": fold,
        "metrics": metrics,
    }


def _segmentation_training_report(
    *,
    fold: int,
    device: str,
    checkpoint_path: Path,
    metrics: dict[str, Any],
    loss_cfg: dict[str, Any],
    train_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
    max_train_batches: int | None,
    max_val_batches: int | None,
) -> dict[str, Any]:
    """Build the JSON report for one segmenter fold without writing files."""
    return {
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
    output_cfg = config.get("output", {})
    prepared = _prepare_segmentation_training_run(
        config=config,
        paths=paths,
        fold=fold,
        seed=seed,
        device=device,
        epochs_override=epochs_override,
    )
    _run_segmentation_loop(
        model=prepared.model,
        train_loader=prepared.train_loader,
        optimizer=prepared.optimizer,
        device=device,
        fold=fold,
        loop_config=prepared.loop_config,
        logger=logger,
    )

    batch_limits = prepared.loop_config.batch_limits
    metrics = _evaluate_model(
        prepared.model,
        prepared.val_loader,
        device,
        max_batches=batch_limits.max_val_batches,
    )
    return _write_segmentation_training_outputs(
        config=config,
        paths=paths,
        output_cfg=output_cfg,
        fold=fold,
        device=device,
        model=prepared.model,
        metrics=metrics,
        loss_cfg=prepared.loop_config.loss_cfg,
        train_manifest=prepared.train_manifest,
        val_manifest=prepared.val_manifest,
        max_train_batches=batch_limits.max_train_batches,
        max_val_batches=batch_limits.max_val_batches,
    )
