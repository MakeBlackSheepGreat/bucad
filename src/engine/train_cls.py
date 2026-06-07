"""Classifier training loop and report writer for BUSBRA folds."""

from __future__ import annotations

import math
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.datasets.busbra import BUSBRAClassificationDataset, generate_busbra_split_assignments, load_busbra_manifest
from src.engine.checkpoints import atomic_torch_save
from src.models.classifier import create_classifier
from src.preprocess.transforms import build_classifier_transform
from src.utils.config import load_project_config
from src.utils.logging import get_logger
from src.utils.metrics import best_threshold_by_youden, classification_metrics
from src.utils.reporting import write_json_report
from src.utils.runtime import ensure_dir, optional_import, require_dependency, seed_everything, select_device


torch = optional_import("torch")
optim = optional_import("torch.optim")
torch_utils_data = optional_import("torch.utils.data")


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


def _collect_validation_probabilities(model, loader, device: str) -> tuple[list[int], list[float]]:
    require_dependency("torch", torch)
    y_true: list[int] = []
    malignant_probabilities: list[float] = []
    model.eval()
    inference_context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with inference_context():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            malignant_probabilities.extend(probs[:, 1].cpu().tolist())
    return y_true, malignant_probabilities


def _evaluate_model(model, loader, device: str) -> dict[str, Any]:
    y_true, malignant_probabilities = _collect_validation_probabilities(model, loader, device)
    return classification_metrics(y_true, malignant_probabilities)


def _weighted_score(metrics: dict[str, Any], *, sensitivity_weight: float) -> float:
    auc = float(metrics.get("auc") or 0.0)
    sensitivity = float(metrics.get("sensitivity", 0.0))
    specificity = float(metrics.get("specificity", 0.0))
    return auc + sensitivity_weight * sensitivity + 0.1 * specificity


def _constrained_score(
    metrics: dict[str, Any],
    *,
    sensitivity_weight: float,
    min_specificity: float,
) -> float:
    score = _weighted_score(metrics, sensitivity_weight=sensitivity_weight)
    specificity = float(metrics.get("specificity", 0.0))
    if min_specificity > 0 and specificity < min_specificity:
        score -= (min_specificity - specificity) * 2.0
    return score


def _class_weights(train_manifest: pd.DataFrame, positive_weight: float | None):
    require_dependency("torch", torch)
    labels = train_manifest["pathology_label"].astype(str).str.lower()
    benign_count = int((labels == "benign").sum())
    malignant_count = int((labels == "malignant").sum())
    if benign_count <= 0 or malignant_count <= 0:
        return None
    if positive_weight is None:
        positive_weight = benign_count / malignant_count
    return torch.tensor([1.0, float(positive_weight)], dtype=torch.float32)


def _load_sample_weights(config_path: str | Path | None, *, weight_column: str) -> dict[str, float]:
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Sample weight file not found: {path}")
    frame = pd.read_csv(path)
    if "sample_id" not in frame.columns or weight_column not in frame.columns:
        raise ValueError(
            f"Sample weight file must contain sample_id and {weight_column!r} columns: {path}"
        )
    weights: dict[str, float] = {}
    for row in frame.itertuples(index=False):
        value = float(getattr(row, weight_column))
        weights[str(getattr(row, "sample_id"))] = max(0.0, value)
    return weights


def _batch_sample_weights(
    sample_ids: list[str],
    sample_weights: dict[str, float],
    *,
    device: str,
):
    require_dependency("torch", torch)
    if not sample_weights:
        return None
    values = [float(sample_weights.get(str(sample_id), 1.0)) for sample_id in sample_ids]
    return torch.tensor(values, dtype=torch.float32, device=device)


def _classification_loss(
    logits,
    labels,
    *,
    class_weights,
    sample_weights,
    label_smoothing: float,
    loss_name: str,
    focal_gamma: float,
):
    if loss_name == "focal":
        per_sample_loss = torch.nn.functional.cross_entropy(
            logits,
            labels,
            weight=class_weights,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        probabilities = torch.softmax(logits, dim=1)
        pt = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
        per_sample_loss = ((1.0 - pt) ** float(focal_gamma)) * per_sample_loss
    else:
        per_sample_loss = torch.nn.functional.cross_entropy(
            logits,
            labels,
            weight=class_weights,
            label_smoothing=label_smoothing,
            reduction="none",
        )
    if sample_weights is not None:
        per_sample_loss = per_sample_loss * sample_weights
        denominator = sample_weights.sum().clamp_min(1e-6)
        return per_sample_loss.sum() / denominator
    return per_sample_loss.mean()


def _rand_bbox(width: int, height: int, lam: float) -> tuple[int, int, int, int]:
    cut_ratio = math.sqrt(max(0.0, 1.0 - float(lam)))
    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)
    center_x = int(np.random.randint(width))
    center_y = int(np.random.randint(height))
    x1 = int(np.clip(center_x - cut_width // 2, 0, width))
    y1 = int(np.clip(center_y - cut_height // 2, 0, height))
    x2 = int(np.clip(center_x + cut_width // 2, 0, width))
    y2 = int(np.clip(center_y + cut_height // 2, 0, height))
    return x1, y1, x2, y2


def _maybe_apply_mix_augmentation(
    images,
    labels,
    sample_weights,
    *,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_probability: float,
):
    if max(float(mixup_alpha), float(cutmix_alpha)) <= 0.0:
        return images, labels, None, 1.0, sample_weights, None
    if images.shape[0] < 2 or np.random.random() >= float(mix_probability):
        return images, labels, None, 1.0, sample_weights, None

    use_cutmix = False
    if cutmix_alpha > 0.0 and mixup_alpha > 0.0:
        use_cutmix = bool(np.random.random() < 0.5)
    elif cutmix_alpha > 0.0:
        use_cutmix = True
    alpha = float(cutmix_alpha if use_cutmix else mixup_alpha)
    lam = float(np.random.beta(alpha, alpha))
    indices = torch.randperm(images.shape[0], device=images.device)
    labels_b = labels[indices]
    weights_b = sample_weights[indices] if sample_weights is not None else None

    if use_cutmix:
        mixed = images.clone()
        _, _, height, width = mixed.shape
        x1, y1, x2, y2 = _rand_bbox(width, height, lam)
        mixed[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
        patch_area = max(0, x2 - x1) * max(0, y2 - y1)
        lam = 1.0 - float(patch_area) / float(max(1, width * height))
        return mixed, labels, labels_b, lam, sample_weights, weights_b

    mixed = lam * images + (1.0 - lam) * images[indices]
    return mixed, labels, labels_b, lam, sample_weights, weights_b


def _pretrained_data_settings(model, preprocess_cfg: dict[str, Any]) -> dict[str, Any]:
    if not bool(preprocess_cfg.get("use_timm_data_config", False)):
        return {}
    pretrained_cfg = getattr(model, "pretrained_cfg", {}) or {}
    input_size = pretrained_cfg.get("input_size")
    return {
        "image_size": int(input_size[-1]) if input_size else None,
        "mean": list(pretrained_cfg.get("mean") or []),
        "std": list(pretrained_cfg.get("std") or []),
        "interpolation": pretrained_cfg.get("interpolation"),
        "crop_pct": pretrained_cfg.get("crop_pct"),
    }


def _extra_model_kwargs(model_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in model_cfg.items()
        if key not in {"name", "pretrained", "in_chans", "num_classes"}
    }


def _resolve_preprocess_settings(
    data_cfg: dict[str, Any],
    preprocess_cfg: dict[str, Any],
    model_data_settings: dict[str, Any],
) -> dict[str, Any]:
    image_size = int(
        preprocess_cfg.get(
            "image_size",
            model_data_settings.get("image_size") or data_cfg.get("image_size", 224),
        )
    )
    mean = preprocess_cfg.get("mean")
    if mean is None:
        mean = model_data_settings.get("mean") or None
    std = preprocess_cfg.get("std")
    if std is None:
        std = model_data_settings.get("std") or None
    interpolation = str(
        preprocess_cfg.get(
            "interpolation",
            model_data_settings.get("interpolation") or "area",
        )
    )
    crop_pct = float(
        preprocess_cfg.get(
            "crop_pct",
            model_data_settings.get("crop_pct") or 1.0,
        )
    )
    return {
        "image_size": image_size,
        "mean": mean,
        "std": std,
        "interpolation": interpolation,
        "crop_pct": crop_pct,
    }


def _lr_for_epoch(
    epoch_index: int,
    *,
    base_lr: float,
    epochs: int,
    scheduler_cfg: dict[str, Any],
) -> float:
    scheduler_name = str(scheduler_cfg.get("name", "none")).lower()
    if scheduler_name not in {"cosine", "warmup_cosine"}:
        return base_lr
    warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))
    min_lr = float(scheduler_cfg.get("min_lr", 0.0))
    if warmup_epochs > 0 and epoch_index < warmup_epochs:
        return base_lr * float(epoch_index + 1) / float(warmup_epochs)
    cosine_epochs = max(1, epochs - warmup_epochs)
    cosine_index = min(max(epoch_index - warmup_epochs, 0), cosine_epochs)
    cosine_scale = 0.5 * (1.0 + math.cos(math.pi * float(cosine_index) / float(cosine_epochs)))
    return min_lr + (base_lr - min_lr) * cosine_scale


def _set_optimizer_lr(optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def _train_classifier_epoch(
    model,
    train_loader,
    optimizer,
    *,
    device: str,
    class_weights,
    sample_weights: dict[str, float],
    label_smoothing: float,
    loss_name: str,
    focal_gamma: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_probability: float,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in train_loader:
        images = batch["image"].to(device=device, dtype=torch.float32)
        labels = batch["label"].to(device=device)
        batch_weights = _batch_sample_weights(
            [str(value) for value in batch["sample_id"]],
            sample_weights,
            device=device,
        )
        (
            images,
            labels_a,
            labels_b,
            mix_lambda,
            weights_a,
            weights_b,
        ) = _maybe_apply_mix_augmentation(
            images,
            labels,
            batch_weights,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mix_probability=mix_probability,
        )
        optimizer.zero_grad()
        logits = model(images)
        if labels_b is None:
            loss = _classification_loss(
                logits,
                labels_a,
                class_weights=class_weights,
                sample_weights=weights_a,
                label_smoothing=label_smoothing,
                loss_name=loss_name,
                focal_gamma=focal_gamma,
            )
        else:
            loss_a = _classification_loss(
                logits,
                labels_a,
                class_weights=class_weights,
                sample_weights=weights_a,
                label_smoothing=label_smoothing,
                loss_name=loss_name,
                focal_gamma=focal_gamma,
            )
            loss_b = _classification_loss(
                logits,
                labels_b,
                class_weights=class_weights,
                sample_weights=weights_b,
                label_smoothing=label_smoothing,
                loss_name=loss_name,
                focal_gamma=focal_gamma,
            )
            loss = float(mix_lambda) * loss_a + (1.0 - float(mix_lambda)) * loss_b
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def _score_checkpoint_candidate(
    *,
    epoch_metrics: dict[str, Any],
    y_true: list[int],
    malignant_probabilities: list[float],
    checkpoint_strategy: str,
    selection_threshold: float,
    sensitivity_weight: float,
    min_specificity: float,
) -> tuple[dict[str, Any], float]:
    # Selection strategies reuse the validation probabilities already computed
    # for epoch metrics, so one epoch does one validation forward pass.
    if checkpoint_strategy in {"selected_threshold", "sensitivity", "high_sensitivity"}:
        threshold_metrics = classification_metrics(
            y_true,
            malignant_probabilities,
            threshold=selection_threshold,
        )
        score_metrics = threshold_metrics | {"auc": epoch_metrics.get("auc")}
    elif checkpoint_strategy == "youden":
        score_metrics = best_threshold_by_youden(y_true, malignant_probabilities)
    else:
        score_metrics = epoch_metrics

    if checkpoint_strategy in {"best_auc", "auc"}:
        score = float(score_metrics.get("auc") or 0.0)
    else:
        score = _constrained_score(
            score_metrics,
            sensitivity_weight=sensitivity_weight,
            min_specificity=min_specificity,
        )
    return score_metrics, score


def _snapshot_state_dict(model) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    atomic_torch_save(payload, destination)


def _resolve_sample_weight_path(paths, training_cfg: dict[str, Any]) -> Path | None:
    sample_weight_path = training_cfg.get("sample_weight_path")
    if sample_weight_path is None:
        return None
    resolved = Path(sample_weight_path)
    if not resolved.is_absolute():
        resolved = (paths.project_root / resolved).resolve()
    return resolved


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


def _build_classifier_model_and_transforms(
    *,
    config: dict[str, Any],
    data_cfg: dict[str, Any],
    device: str,
):
    preprocess_cfg = data_cfg.get("preprocess", {})
    augmentation_cfg = data_cfg.get("augmentation", {})
    model_cfg = config.get("model", {})
    model = create_classifier(
        model_name=model_cfg.get("name", "resnet18"),
        pretrained=bool(model_cfg.get("pretrained", True)),
        in_chans=int(model_cfg.get("in_chans", 3)),
        num_classes=int(model_cfg.get("num_classes", 2)),
        **_extra_model_kwargs(model_cfg),
    ).to(device)
    model_data_settings = _pretrained_data_settings(model, preprocess_cfg)
    preprocess_settings = _resolve_preprocess_settings(
        data_cfg,
        preprocess_cfg,
        model_data_settings,
    )
    image_size = int(preprocess_settings["image_size"])
    train_transform = build_classifier_transform(
        image_size=image_size,
        apply_clahe_enabled=bool(preprocess_cfg.get("clahe", False)),
        horizontal_flip=bool(augmentation_cfg.get("horizontal_flip", False)),
        flip_probability=float(augmentation_cfg.get("flip_probability", 0.5)),
        rotation_degrees=float(augmentation_cfg.get("rotation_degrees", 0.0)),
        brightness=float(augmentation_cfg.get("brightness", 0.0)),
        contrast=float(augmentation_cfg.get("contrast", 0.0)),
        scale_min=float(augmentation_cfg.get("scale_min", 1.0)),
        scale_max=float(augmentation_cfg.get("scale_max", 1.0)),
        mean=preprocess_settings["mean"],
        std=preprocess_settings["std"],
        interpolation=preprocess_settings["interpolation"],
        crop_pct=preprocess_settings["crop_pct"],
    )
    eval_transform = build_classifier_transform(
        image_size=image_size,
        apply_clahe_enabled=bool(preprocess_cfg.get("clahe", False)),
        mean=preprocess_settings["mean"],
        std=preprocess_settings["std"],
        interpolation=preprocess_settings["interpolation"],
        crop_pct=preprocess_settings["crop_pct"],
    )
    return model, image_size, train_transform, eval_transform, preprocess_settings


def _build_classifier_loaders(
    *,
    train_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
    image_size: int,
    train_transform,
    eval_transform,
    data_cfg: dict[str, Any],
    device: str,
):
    train_dataset = BUSBRAClassificationDataset(
        train_manifest, image_size=image_size, transform=train_transform
    )
    val_dataset = BUSBRAClassificationDataset(
        val_manifest, image_size=image_size, transform=eval_transform
    )
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = device.startswith("cuda")
    # Persistent workers only make sense when DataLoader starts child workers.
    loader_kwargs = {
        "batch_size": int(data_cfg.get("batch_size", 8)),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = torch_utils_data.DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = torch_utils_data.DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def _build_class_weights(train_manifest: pd.DataFrame, training_cfg: dict[str, Any], *, device: str):
    class_weight_cfg = training_cfg.get("class_weights")
    class_weights = None
    if class_weight_cfg == "balanced":
        class_weights = _class_weights(train_manifest, positive_weight=None)
    elif isinstance(class_weight_cfg, dict):
        class_weights = torch.tensor(
            [
                float(class_weight_cfg.get("benign", 1.0)),
                float(class_weight_cfg.get("malignant", 1.0)),
            ],
            dtype=torch.float32,
        )
    positive_weight = training_cfg.get("positive_class_weight")
    if positive_weight is not None:
        class_weights = _class_weights(train_manifest, positive_weight=float(positive_weight))
    if class_weights is not None:
        class_weights = class_weights.to(device=device)
    return class_weights


def _write_classifier_training_outputs(
    *,
    config: dict[str, Any],
    paths,
    output_cfg: dict[str, Any],
    fold: int,
    device: str,
    model,
    metrics: dict[str, Any],
    best_epoch: int,
    best_metrics: dict[str, Any] | None,
    checkpoint_strategy: str,
    preprocess_settings: dict[str, Any],
    scheduler_cfg: dict[str, Any],
    class_weights,
    label_smoothing: float,
    loss_name: str,
    focal_gamma: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_probability: float,
    sample_weight_path: Path | None,
    sample_weights: dict[str, float],
    min_specificity: float,
    epoch_reports: list[dict[str, Any]],
    train_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
) -> dict[str, Any]:
    checkpoints_dir = ensure_dir(paths.checkpoints_root)
    reports_dir = ensure_dir(paths.reports_root)
    checkpoint_path = checkpoints_dir / output_cfg.get("checkpoint_name", "classifier_fold{fold}.pt").format(fold=fold)
    report_path = reports_dir / output_cfg.get("report_name", "train_cls_fold{fold}.json").format(fold=fold)

    _atomic_torch_save(
        {
            "state_dict": model.state_dict(),
            "model_config": config.get("model", {}),
            "fold": fold,
            "metrics": metrics,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
        },
        checkpoint_path,
    )
    report = {
        "fold": fold,
        "device": device,
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "checkpoint_strategy": checkpoint_strategy,
        "preprocess_settings": preprocess_settings,
        "scheduler": scheduler_cfg,
        "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
        "label_smoothing": label_smoothing,
        "loss": loss_name,
        "focal_gamma": focal_gamma if loss_name == "focal" else None,
        "mixup_alpha": mixup_alpha,
        "cutmix_alpha": cutmix_alpha,
        "mix_probability": mix_probability,
        "sample_weight_path": str(sample_weight_path) if sample_weight_path is not None else None,
        "sample_weight_count": len(sample_weights),
        "min_specificity": min_specificity,
        "epoch_reports": epoch_reports,
        "train_size": int(len(train_manifest)),
        "val_size": int(len(val_manifest)),
    }
    write_json_report(report_path, report)
    return report


@dataclasses.dataclass(slots=True)
class _TrainingLoopConfig:
    """All settings the training loop needs to pick checkpoints."""

    epochs: int
    scheduler_cfg: dict[str, Any]
    checkpoint_strategy: str
    selection_threshold: float
    sensitivity_weight: float
    min_specificity: float
    class_weights: Any
    label_smoothing: float
    loss_name: str
    focal_gamma: float
    mixup_alpha: float
    cutmix_alpha: float
    mix_probability: float
    sample_weights: dict[str, float]


@dataclasses.dataclass(slots=True)
class _PreparedTrainingRun:
    """Objects and settings prepared before classifier epochs start."""

    model: Any
    train_loader: Any
    val_loader: Any
    train_manifest: pd.DataFrame
    val_manifest: pd.DataFrame
    preprocess_settings: dict[str, Any]
    class_weights: Any
    sample_weight_path: Path | None
    sample_weights: dict[str, float]
    loop_config: _TrainingLoopConfig
    base_learning_rate: float
    optimizer: Any


def _prepare_classifier_training_run(
    *,
    config: dict[str, Any],
    paths,
    fold: int,
    seed: int,
    device: str,
    epochs_override: int | None,
) -> _PreparedTrainingRun:
    """Prepare data splits, loaders, model, optimizer, and loop settings."""
    manifest = load_busbra_manifest(paths.busbra_root)
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    train_manifest, val_manifest = _prepare_fold_manifests(
        manifest=manifest,
        paths=paths,
        training_cfg=training_cfg,
        fold=fold,
        seed=seed,
    )
    model, image_size, train_transform, eval_transform, preprocess_settings = (
        _build_classifier_model_and_transforms(
            config=config,
            data_cfg=data_cfg,
            device=device,
        )
    )
    train_loader, val_loader = _build_classifier_loaders(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        image_size=image_size,
        train_transform=train_transform,
        eval_transform=eval_transform,
        data_cfg=data_cfg,
        device=device,
    )

    base_learning_rate = float(training_cfg.get("learning_rate", 3e-4))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_learning_rate,
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    class_weights = _build_class_weights(train_manifest, training_cfg, device=device)
    sample_weight_path = _resolve_sample_weight_path(paths, training_cfg)
    sample_weights = _load_sample_weights(
        sample_weight_path,
        weight_column=str(training_cfg.get("sample_weight_column", "sample_weight")),
    )
    loop_config = _TrainingLoopConfig(
        epochs=int(epochs_override or training_cfg.get("epochs", 5)),
        scheduler_cfg=training_cfg.get("scheduler", {}) or {},
        checkpoint_strategy=str(training_cfg.get("checkpoint_strategy", "last")).lower(),
        selection_threshold=float(training_cfg.get("selection_threshold", 0.5)),
        sensitivity_weight=float(training_cfg.get("sensitivity_weight", 0.0)),
        min_specificity=float(training_cfg.get("min_specificity", 0.0)),
        class_weights=class_weights,
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0)),
        loss_name=str(training_cfg.get("loss", "cross_entropy")).lower(),
        focal_gamma=float(training_cfg.get("focal_gamma", 2.0)),
        mixup_alpha=float(training_cfg.get("mixup_alpha", 0.0)),
        cutmix_alpha=float(training_cfg.get("cutmix_alpha", 0.0)),
        mix_probability=float(training_cfg.get("mix_probability", 0.0)),
        sample_weights=sample_weights,
    )
    return _PreparedTrainingRun(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        preprocess_settings=preprocess_settings,
        class_weights=class_weights,
        sample_weight_path=sample_weight_path,
        sample_weights=sample_weights,
        loop_config=loop_config,
        base_learning_rate=base_learning_rate,
        optimizer=optimizer,
    )


def _run_training_loop(
    *,
    model,
    train_loader,
    val_loader,
    optimizer,
    device: str,
    fold: int,
    loop_config: _TrainingLoopConfig,
    base_learning_rate: float,
    logger,
) -> tuple[dict[str, Any] | None, int, dict[str, Any] | None, list[dict[str, Any]]]:
    """Return (best_state_dict, best_epoch, best_metrics, epoch_reports)."""
    cfg = loop_config
    best_state_dict = None
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    best_score = float("-inf")
    epoch_reports: list[dict[str, Any]] = []

    for epoch in range(cfg.epochs):
        current_lr = _lr_for_epoch(
            epoch,
            base_lr=base_learning_rate,
            epochs=cfg.epochs,
            scheduler_cfg=cfg.scheduler_cfg,
        )
        _set_optimizer_lr(optimizer, current_lr)
        mean_loss = _train_classifier_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            class_weights=cfg.class_weights,
            sample_weights=cfg.sample_weights,
            label_smoothing=cfg.label_smoothing,
            loss_name=cfg.loss_name,
            focal_gamma=cfg.focal_gamma,
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            mix_probability=cfg.mix_probability,
        )
        val_y_true, val_malignant_probabilities = _collect_validation_probabilities(
            model, val_loader, device,
        )
        # Keep epoch metrics and checkpoint selection on the same validation pass.
        epoch_metrics = classification_metrics(val_y_true, val_malignant_probabilities)
        score_metrics, score = _score_checkpoint_candidate(
            epoch_metrics=epoch_metrics,
            y_true=val_y_true,
            malignant_probabilities=val_malignant_probabilities,
            checkpoint_strategy=cfg.checkpoint_strategy,
            selection_threshold=cfg.selection_threshold,
            sensitivity_weight=cfg.sensitivity_weight,
            min_specificity=cfg.min_specificity,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_metrics = score_metrics
            best_state_dict = _snapshot_state_dict(model)
        epoch_reports.append({
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "loss": mean_loss,
            "metrics": epoch_metrics,
            "score_metrics": score_metrics,
            "selection_score": score,
        })
        logger.info(
            "fold=%s epoch=%s loss=%.4f auc=%.4f sens=%.4f score=%.4f",
            fold,
            epoch + 1,
            mean_loss,
            float(epoch_metrics.get("auc") or 0.0),
            float(score_metrics.get("sensitivity", 0.0)),
            score,
        )
    return best_state_dict, best_epoch, best_metrics, epoch_reports


def run_classifier_training(
    config_path: str | Path,
    *,
    fold: int = 1,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    """Train one BUSBRA classification fold and write checkpoint plus JSON report."""
    require_dependency("torch", torch)
    require_dependency("torch.optim", optim)
    require_dependency("torch.utils.data", torch_utils_data)

    config, paths = load_project_config(config_path)
    logger = get_logger("train_cls")
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = select_device(str(config.get("device", "auto")))
    output_cfg = config.get("output", {})
    prepared = _prepare_classifier_training_run(
        config=config,
        paths=paths,
        fold=fold,
        seed=seed,
        device=device,
        epochs_override=epochs_override,
    )
    best_state_dict, best_epoch, best_metrics, epoch_reports = _run_training_loop(
        model=prepared.model,
        train_loader=prepared.train_loader,
        val_loader=prepared.val_loader,
        optimizer=prepared.optimizer,
        device=device,
        fold=fold,
        loop_config=prepared.loop_config,
        base_learning_rate=prepared.base_learning_rate,
        logger=logger,
    )

    if prepared.loop_config.checkpoint_strategy != "last" and best_state_dict is not None:
        prepared.model.load_state_dict(best_state_dict)
    metrics = _evaluate_model(prepared.model, prepared.val_loader, device)
    return _write_classifier_training_outputs(
        config=config,
        paths=paths,
        output_cfg=output_cfg,
        fold=fold,
        device=device,
        model=prepared.model,
        metrics=metrics,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        checkpoint_strategy=prepared.loop_config.checkpoint_strategy,
        preprocess_settings=prepared.preprocess_settings,
        scheduler_cfg=prepared.loop_config.scheduler_cfg,
        class_weights=prepared.class_weights,
        label_smoothing=prepared.loop_config.label_smoothing,
        loss_name=prepared.loop_config.loss_name,
        focal_gamma=prepared.loop_config.focal_gamma,
        mixup_alpha=prepared.loop_config.mixup_alpha,
        cutmix_alpha=prepared.loop_config.cutmix_alpha,
        mix_probability=prepared.loop_config.mix_probability,
        sample_weight_path=prepared.sample_weight_path,
        sample_weights=prepared.sample_weights,
        min_specificity=prepared.loop_config.min_specificity,
        epoch_reports=epoch_reports,
        train_manifest=prepared.train_manifest,
        val_manifest=prepared.val_manifest,
    )
