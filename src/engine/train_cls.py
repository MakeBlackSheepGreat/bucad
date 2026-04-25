from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.datasets.busbra import BUSBRAClassificationDataset, generate_busbra_split_assignments, load_busbra_manifest
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


def _evaluate_model(model, loader, device: str) -> dict[str, Any]:
    require_dependency("torch", torch)
    model.eval()
    y_true: list[int] = []
    malignant_probabilities: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            malignant_probabilities.extend(probs[:, 1].cpu().tolist())
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


def run_classifier_training(
    config_path: str | Path,
    *,
    fold: int = 1,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    require_dependency("torch", torch)
    require_dependency("torch.optim", optim)
    require_dependency("torch.utils.data", torch_utils_data)

    config, paths = load_project_config(config_path)
    logger = get_logger("train_cls")
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = select_device(str(config.get("device", "auto")))

    manifest = load_busbra_manifest(paths.busbra_root)
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})

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

    train_dataset = BUSBRAClassificationDataset(
        train_manifest, image_size=image_size, transform=train_transform
    )
    val_dataset = BUSBRAClassificationDataset(
        val_manifest, image_size=image_size, transform=eval_transform
    )
    train_loader = torch_utils_data.DataLoader(
        train_dataset,
        batch_size=int(data_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    val_loader = torch_utils_data.DataLoader(
        val_dataset,
        batch_size=int(data_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )

    base_learning_rate = float(training_cfg.get("learning_rate", 3e-4))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_learning_rate,
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
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
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0)),
    )

    epochs = int(epochs_override or training_cfg.get("epochs", 5))
    scheduler_cfg = training_cfg.get("scheduler", {}) or {}
    checkpoint_strategy = str(training_cfg.get("checkpoint_strategy", "last")).lower()
    selection_threshold = float(training_cfg.get("selection_threshold", 0.5))
    sensitivity_weight = float(training_cfg.get("sensitivity_weight", 0.0))
    min_specificity = float(training_cfg.get("min_specificity", 0.0))
    best_state_dict = None
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    best_score = float("-inf")
    epoch_reports: list[dict[str, Any]] = []
    for epoch in range(epochs):
        current_lr = _lr_for_epoch(
            epoch,
            base_lr=base_learning_rate,
            epochs=epochs,
            scheduler_cfg=scheduler_cfg,
        )
        _set_optimizer_lr(optimizer, current_lr)
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses))
        epoch_metrics = _evaluate_model(model, val_loader, device)
        threshold_metrics = classification_metrics(
            [0, 1],
            [0.0, 1.0],
            threshold=selection_threshold,
        )
        if checkpoint_strategy in {"selected_threshold", "sensitivity", "high_sensitivity"}:
            y_true: list[int] = []
            malignant_probabilities: list[float] = []
            model.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    images = val_batch["image"].to(device=device, dtype=torch.float32)
                    labels = val_batch["label"].to(device=device)
                    probs = torch.softmax(model(images), dim=1)
                    y_true.extend(labels.cpu().tolist())
                    malignant_probabilities.extend(probs[:, 1].cpu().tolist())
            threshold_metrics = classification_metrics(
                y_true,
                malignant_probabilities,
                threshold=selection_threshold,
            )
            score_metrics = threshold_metrics | {"auc": epoch_metrics.get("auc")}
        elif checkpoint_strategy == "youden":
            y_true = []
            malignant_probabilities = []
            model.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    images = val_batch["image"].to(device=device, dtype=torch.float32)
                    labels = val_batch["label"].to(device=device)
                    probs = torch.softmax(model(images), dim=1)
                    y_true.extend(labels.cpu().tolist())
                    malignant_probabilities.extend(probs[:, 1].cpu().tolist())
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
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_metrics = score_metrics
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        epoch_reports.append(
            {
                "epoch": epoch + 1,
                "learning_rate": current_lr,
                "loss": mean_loss,
                "metrics": epoch_metrics,
                "score_metrics": score_metrics,
                "selection_score": score,
            }
        )
        logger.info(
            "fold=%s epoch=%s loss=%.4f auc=%.4f sens=%.4f score=%.4f",
            fold,
            epoch + 1,
            mean_loss,
            float(epoch_metrics.get("auc") or 0.0),
            float(score_metrics.get("sensitivity", 0.0)),
            score,
        )

    if checkpoint_strategy != "last" and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    metrics = _evaluate_model(model, val_loader, device)
    checkpoints_dir = ensure_dir(paths.checkpoints_root)
    reports_dir = ensure_dir(paths.reports_root)
    checkpoint_path = checkpoints_dir / output_cfg.get("checkpoint_name", "classifier_fold{fold}.pt").format(fold=fold)
    report_path = reports_dir / output_cfg.get("report_name", "train_cls_fold{fold}.json").format(fold=fold)

    torch.save(
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
        "label_smoothing": float(training_cfg.get("label_smoothing", 0.0)),
        "min_specificity": min_specificity,
        "epoch_reports": epoch_reports,
        "train_size": int(len(train_manifest)),
        "val_size": int(len(val_manifest)),
    }
    write_json_report(report_path, report)
    return report
