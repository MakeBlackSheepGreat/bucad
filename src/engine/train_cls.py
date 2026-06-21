"""Classifier training loop and report writer for BUSBRA folds."""

from __future__ import annotations

import copy
import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.datasets.busbra import (
    BUSBRAClassificationDataset,
    BUSBRAClassificationDualViewDataset,
    generate_busbra_split_assignments,
    load_busbra_manifest,
)
from src.engine.checkpoints import atomic_torch_save
from src.models.classifier import create_classifier
from src.models.segmenter import load_segmenter
from src.preprocess.io import read_image, save_image
from src.preprocess.io import cv2
from src.preprocess.transforms import build_classifier_transform, prepare_classifier_input
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
    """Load existing fold assignments or create a case-level BUSBRA split file."""
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
    """Split a manifest into train/validation rows for one fold."""
    fold_assignments = assignments[assignments["fold_id"] == fold]
    train_ids = set(fold_assignments.loc[fold_assignments["stage"] == "train", "sample_id"])
    val_ids = set(fold_assignments.loc[fold_assignments["stage"] == "val", "sample_id"])
    train_manifest = manifest[manifest["sample_id"].isin(train_ids)].reset_index(drop=True)
    val_manifest = manifest[manifest["sample_id"].isin(val_ids)].reset_index(drop=True)
    return train_manifest, val_manifest


def _collect_validation_probabilities(model, loader, device: str) -> tuple[list[int], list[float]]:
    """Collect labels and malignant probabilities from one validation pass."""
    require_dependency("torch", torch)
    y_true: list[int] = []
    malignant_probabilities: list[float] = []
    model.eval()
    inference_context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with inference_context():
        for batch in loader:
            images, labels, _batch_weights = _prepare_classifier_batch(
                batch,
                {},
                device=device,
            )
            logits = _model_logits(model, images)
            probs = torch.softmax(logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            malignant_probabilities.extend(probs[:, 1].cpu().tolist())
    return y_true, malignant_probabilities


def _evaluate_model(model, loader, device: str) -> dict[str, Any]:
    """Evaluate classification metrics over a validation loader."""
    y_true, malignant_probabilities = _collect_validation_probabilities(model, loader, device)
    return classification_metrics(y_true, malignant_probabilities)


def _weighted_score(metrics: dict[str, Any], *, sensitivity_weight: float) -> float:
    """Score metrics with optional extra emphasis on sensitivity."""
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
    """Score metrics while penalizing specificity below a configured floor."""
    score = _weighted_score(metrics, sensitivity_weight=sensitivity_weight)
    specificity = float(metrics.get("specificity", 0.0))
    if min_specificity > 0 and specificity < min_specificity:
        score -= (min_specificity - specificity) * 2.0
    return score


def _class_weights(train_manifest: pd.DataFrame, positive_weight: float | None):
    """Build class weights from the training fold label distribution."""
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
    """Load optional per-sample weights from a CSV file."""
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


def _count_sample_weight_hits(train_manifest: pd.DataFrame, sample_weights: dict[str, float]) -> int:
    """Count training samples that receive a non-default sample weight."""
    if not sample_weights:
        return 0
    count = 0
    for sample_id in train_manifest["sample_id"].astype(str):
        if float(sample_weights.get(sample_id, 1.0)) != 1.0:
            count += 1
    return count


def _batch_sample_weights(
    sample_ids: list[str],
    sample_weights: dict[str, float],
    *,
    device: str,
):
    """Return a tensor of sample weights aligned with the current batch."""
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
    class_priors,
    sample_weights,
    label_smoothing: float,
    loss_name: str,
    focal_gamma: float,
    balanced_softmax_tau: float,
):
    """Compute weighted cross-entropy or focal loss with optional per-sample weights."""
    adjusted_logits = logits
    if loss_name == "balanced_softmax":
        if class_priors is None:
            raise ValueError("balanced_softmax requires class_priors.")
        adjusted_logits = logits + float(balanced_softmax_tau) * class_priors.clamp_min(1e-12).log().to(
            device=logits.device,
            dtype=logits.dtype,
        )
    if loss_name == "focal":
        per_sample_loss = torch.nn.functional.cross_entropy(
            adjusted_logits,
            labels,
            weight=class_weights,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        probabilities = torch.softmax(adjusted_logits, dim=1)
        pt = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
        per_sample_loss = ((1.0 - pt) ** float(focal_gamma)) * per_sample_loss
    else:
        per_sample_loss = torch.nn.functional.cross_entropy(
            adjusted_logits,
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


def _pairwise_auc_regularizer(
    logits,
    labels,
    *,
    sample_weights,
    margin: float,
):
    """Compute a weighted pairwise logistic surrogate for AUC ranking."""
    score_margin = logits[:, 1] - logits[:, 0]
    positive_mask = labels == 1
    negative_mask = labels == 0
    positive_scores = score_margin[positive_mask]
    negative_scores = score_margin[negative_mask]
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return logits.new_zeros(())
    pairwise_margin = positive_scores.unsqueeze(1) - negative_scores.unsqueeze(0)
    per_pair_loss = torch.nn.functional.softplus(float(margin) - pairwise_margin)
    if sample_weights is None:
        return per_pair_loss.mean()
    positive_weights = sample_weights[positive_mask]
    negative_weights = sample_weights[negative_mask]
    pairwise_weights = positive_weights.unsqueeze(1) * negative_weights.unsqueeze(0)
    denominator = pairwise_weights.sum().clamp_min(1e-6)
    return (per_pair_loss * pairwise_weights).sum() / denominator


def _rand_bbox(width: int, height: int, lam: float) -> tuple[int, int, int, int]:
    """Sample a CutMix patch box from the Beta-mixed area ratio."""
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
    """Apply MixUp or CutMix to one batch and keep paired labels/weights."""
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
    """Read timm pretrained data settings when config enables them."""
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
    """Return model config keys that should be passed through to timm."""
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
    """Merge config and timm pretrained data settings into concrete preprocessing values."""
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
    """Compute the learning rate for the current epoch scheduler step."""
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
    """Apply one learning rate value to every optimizer parameter group."""
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def _move_dual_view_batch(batch: dict[str, Any], *, device: str) -> dict[str, Any]:
    """Move dual-view image and descriptor tensors to the training device."""
    moved = {
        "image_full": batch["image_full"].to(device=device, dtype=torch.float32, non_blocking=True),
        "image_roi": batch["image_roi"].to(device=device, dtype=torch.float32, non_blocking=True),
        "roi_descriptor": batch["roi_descriptor"].to(device=device, dtype=torch.float32, non_blocking=True),
    }
    if "roi_valid" in batch and hasattr(batch["roi_valid"], "to"):
        moved["roi_valid"] = batch["roi_valid"].to(device=device, dtype=torch.float32, non_blocking=True)
    return moved


def _prepare_classifier_batch(batch: dict[str, Any], sample_weights: dict[str, float], *, device: str):
    """Move one training batch to device and attach optional sample weights."""
    if "image_full" in batch:
        images = _move_dual_view_batch(batch, device=device)
    else:
        images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
    labels = batch["label"].to(device=device, non_blocking=True)
    batch_weights = _batch_sample_weights(
        [str(value) for value in batch["sample_id"]],
        sample_weights,
        device=device,
    )
    return images, labels, batch_weights


def _model_logits(model, images, labels=None):
    """Call model forward, passing labels only for heads that support label-aware logits."""
    if isinstance(images, dict):
        if labels is None:
            return model(**images)
        try:
            return model(**images, labels=labels)
        except TypeError:
            return model(**images)
    if labels is None:
        return model(images)
    try:
        return model(images, labels=labels)
    except TypeError:
        return model(images)


def _model_logits_and_embedding(model, images, labels=None):
    """Return logits and optional training embedding when the model exposes it."""
    forward_with_embedding = getattr(model, "forward_with_embedding", None)
    if callable(forward_with_embedding):
        try:
            if isinstance(images, dict):
                return forward_with_embedding(**images, labels=labels)
            return forward_with_embedding(images, labels=labels)
        except TypeError:
            if isinstance(images, dict):
                return forward_with_embedding(**images)
            return forward_with_embedding(images)
    return _model_logits(model, images, labels=labels), None


def _supervised_contrastive_loss(embeddings, labels, *, temperature: float) -> Any:
    """Compute a simple supervised contrastive loss over one batch embedding tensor."""
    require_dependency("torch", torch)
    if embeddings is None or labels is None:
        return None
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return embeddings.new_zeros(()) if embeddings is not None else None
    normalized = torch.nn.functional.normalize(embeddings, dim=1)
    logits = torch.matmul(normalized, normalized.T) / float(max(temperature, 1e-6))
    mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    exp_logits = torch.exp(logits) * (~mask).to(dtype=logits.dtype)
    positive_mask = labels.view(-1, 1).eq(labels.view(1, -1)) & (~mask)
    if not positive_mask.any():
        return embeddings.new_zeros(())
    denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_prob = logits - torch.log(denom)
    log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0
    if not valid.any():
        return embeddings.new_zeros(())
    mean_log_prob_pos = (log_prob * positive_mask).sum(dim=1) / positive_counts.clamp_min(1)
    loss = -mean_log_prob_pos[valid].mean()
    return loss


def _moe_load_balance_loss(expert_weights, *, num_experts: int | None = None):
    """Penalize collapsed MoE routing by matching the batch mean to uniform use."""
    if expert_weights is None:
        return None
    if expert_weights.ndim != 2 or expert_weights.shape[0] <= 0:
        return expert_weights.new_zeros(())
    expert_count = int(num_experts or expert_weights.shape[1])
    target = expert_weights.new_full((expert_count,), 1.0 / float(expert_count))
    mean_weights = expert_weights.mean(dim=0)
    return torch.nn.functional.mse_loss(mean_weights, target)


def _mixed_classification_loss(
    logits,
    labels_a,
    labels_b,
    mix_lambda: float,
    weights_a,
    weights_b,
    *,
    class_weights,
    class_priors,
    label_smoothing: float,
    loss_name: str,
    focal_gamma: float,
    balanced_softmax_tau: float,
    pairwise_auc_weight: float,
    pairwise_auc_margin: float,
):
    """Blend normal and mixed-label losses for MixUp/CutMix batches."""
    if labels_b is None:
        classification_loss = _classification_loss(
            logits,
            labels_a,
            class_weights=class_weights,
            class_priors=class_priors,
            sample_weights=weights_a,
            label_smoothing=label_smoothing,
            loss_name=loss_name,
            focal_gamma=focal_gamma,
            balanced_softmax_tau=balanced_softmax_tau,
        )
        if pairwise_auc_weight <= 0.0:
            return classification_loss
        pairwise_loss = _pairwise_auc_regularizer(
            logits,
            labels_a,
            sample_weights=weights_a,
            margin=pairwise_auc_margin,
        )
        return classification_loss + float(pairwise_auc_weight) * pairwise_loss

    loss_a = _classification_loss(
        logits,
        labels_a,
        class_weights=class_weights,
        class_priors=class_priors,
        sample_weights=weights_a,
        label_smoothing=label_smoothing,
        loss_name=loss_name,
        focal_gamma=focal_gamma,
        balanced_softmax_tau=balanced_softmax_tau,
    )
    loss_b = _classification_loss(
        logits,
        labels_b,
        class_weights=class_weights,
        class_priors=class_priors,
        sample_weights=weights_b,
        label_smoothing=label_smoothing,
        loss_name=loss_name,
        focal_gamma=focal_gamma,
        balanced_softmax_tau=balanced_softmax_tau,
    )
    return float(mix_lambda) * loss_a + (1.0 - float(mix_lambda)) * loss_b


def _train_classifier_epoch(
    model,
    train_loader,
    optimizer,
    model_ema,
    *,
    device: str,
    class_weights,
    class_priors,
    sample_weights: dict[str, float],
    label_smoothing: float,
    loss_name: str,
    focal_gamma: float,
    balanced_softmax_tau: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_probability: float,
    pairwise_auc_weight: float,
    pairwise_auc_margin: float,
    supcon_weight: float,
    supcon_temperature: float,
    moe_load_balance_weight: float,
    use_sam: bool,
) -> float:
    """Run one classifier epoch with optional sample weights and mix augmentations."""
    model.train()
    losses: list[float] = []
    for batch in train_loader:
        images, labels, batch_weights = _prepare_classifier_batch(
            batch,
            sample_weights,
            device=device,
        )
        if isinstance(images, dict):
            labels_a = labels
            labels_b = None
            mix_lambda = 1.0
            weights_a = batch_weights
            weights_b = None
        else:
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
        logits, embeddings = _model_logits_and_embedding(model, images, labels_a)
        loss = _mixed_classification_loss(
            logits,
            labels_a,
            labels_b,
            mix_lambda,
            weights_a,
            weights_b,
            class_weights=class_weights,
            class_priors=class_priors,
            label_smoothing=label_smoothing,
            loss_name=loss_name,
            focal_gamma=focal_gamma,
            balanced_softmax_tau=balanced_softmax_tau,
            pairwise_auc_weight=pairwise_auc_weight,
            pairwise_auc_margin=pairwise_auc_margin,
        )
        if (
            float(supcon_weight) > 0.0
            and labels_b is None
            and embeddings is not None
        ):
            supcon_loss = _supervised_contrastive_loss(
                embeddings,
                labels_a,
                temperature=float(supcon_temperature),
            )
            if supcon_loss is not None:
                loss = loss + float(supcon_weight) * supcon_loss
        if float(moe_load_balance_weight) > 0.0:
            expert_weights = getattr(model, "last_expert_weights", None)
            load_balance_loss = _moe_load_balance_loss(
                expert_weights,
                num_experts=getattr(model, "num_experts", None),
            )
            if load_balance_loss is not None:
                loss = loss + float(moe_load_balance_weight) * load_balance_loss
        loss.backward()
        if bool(use_sam):
            optimizer.first_step()
            optimizer.zero_grad()
            logits_second, embeddings_second = _model_logits_and_embedding(model, images, labels_a)
            second_loss = _mixed_classification_loss(
                logits_second,
                labels_a,
                labels_b,
                mix_lambda,
                weights_a,
                weights_b,
                class_weights=class_weights,
                class_priors=class_priors,
                label_smoothing=label_smoothing,
                loss_name=loss_name,
                focal_gamma=focal_gamma,
                balanced_softmax_tau=balanced_softmax_tau,
                pairwise_auc_weight=pairwise_auc_weight,
                pairwise_auc_margin=pairwise_auc_margin,
            )
            if (
                float(supcon_weight) > 0.0
                and labels_b is None
                and embeddings_second is not None
            ):
                supcon_loss_second = _supervised_contrastive_loss(
                    embeddings_second,
                    labels_a,
                    temperature=float(supcon_temperature),
                )
                if supcon_loss_second is not None:
                    second_loss = second_loss + float(supcon_weight) * supcon_loss_second
            if float(moe_load_balance_weight) > 0.0:
                expert_weights_second = getattr(model, "last_expert_weights", None)
                load_balance_loss_second = _moe_load_balance_loss(
                    expert_weights_second,
                    num_experts=getattr(model, "num_experts", None),
                )
                if load_balance_loss_second is not None:
                    second_loss = second_loss + float(moe_load_balance_weight) * load_balance_loss_second
            second_loss.backward()
            optimizer.second_step()
        else:
            optimizer.step()
        if model_ema is not None:
            model_ema.update(model)
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
    """Score one epoch according to the configured checkpoint selection strategy."""
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
    """Clone model weights to CPU for best-checkpoint restoration."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


class _ModelEma:
    """Maintain an exponential moving average of model parameters."""

    def __init__(self, model, *, decay: float) -> None:
        self.decay = float(decay)
        self.state_dict = _snapshot_state_dict(model)

    def update(self, model) -> None:
        """Update EMA weights from the latest model state."""
        current_state = model.state_dict()
        decay = float(self.decay)
        one_minus_decay = 1.0 - decay
        for key, ema_value in self.state_dict.items():
            model_value = current_state[key].detach().cpu()
            if not torch.is_floating_point(model_value):
                self.state_dict[key] = model_value.clone()
                continue
            self.state_dict[key] = ema_value.mul(decay).add(model_value, alpha=one_minus_decay)


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    """Compatibility wrapper around the shared atomic checkpoint writer."""
    atomic_torch_save(payload, destination)


def _resolve_sample_weight_path(paths, training_cfg: dict[str, Any]) -> Path | None:
    """Resolve the optional sample-weight CSV path relative to project root."""
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
    """Prepare non-empty train/validation manifests for one classifier fold."""
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


def _resolve_project_path(value: str | Path, paths) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (paths.project_root / path).resolve()


def _segmenter_oof_cache_dir(paths, roi_cfg: dict[str, Any]) -> Path:
    """Return the cache directory for training-time predicted ROI masks."""
    cache_dir = roi_cfg.get("mask_cache_dir", "artifacts/cache/segmenter_oof_masks")
    return _resolve_project_path(cache_dir, paths)


def _segmenter_checkpoint_for_fold(paths, roi_cfg: dict[str, Any], fold_id: int) -> Path:
    """Resolve one segmenter OOF checkpoint path for a validation fold id."""
    pattern = str(roi_cfg.get("segmenter_checkpoint_pattern", "artifacts/checkpoints/segmenter_5fold_fold{fold}.pt"))
    return _resolve_project_path(pattern.format(fold=int(fold_id)), paths)


def _predict_segmenter_mask(model, image: np.ndarray, *, image_size: int, device: str) -> np.ndarray:
    """Predict a probability mask with one loaded segmenter and resize it to image shape."""
    input_tensor = prepare_classifier_input(image, int(image_size), apply_clahe_enabled=False)
    batch = input_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
    inference_context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with inference_context():
        logits = model(batch)
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    if cv2 is not None:
        return cv2.resize(
            probability.astype(np.float32),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    y_index = np.linspace(0, probability.shape[0] - 1, image.shape[0]).round().astype(int)
    x_index = np.linspace(0, probability.shape[1] - 1, image.shape[1]).round().astype(int)
    return probability[np.ix_(y_index, x_index)].astype(np.float32)


def _ensure_segmenter_oof_mask_cache(
    *,
    manifest: pd.DataFrame,
    assignments: pd.DataFrame,
    paths,
    roi_cfg: dict[str, Any],
    device: str,
) -> dict[str, Path]:
    """Generate or reuse BUSBRA OOF predicted masks for the requested samples."""
    require_dependency("torch", torch)
    cache_dir = _segmenter_oof_cache_dir(paths, roi_cfg)
    ensure_dir(cache_dir)
    assignment_by_sample = {}
    for row in assignments.itertuples(index=False):
        if str(getattr(row, "stage", "")) != "val":
            continue
        fold_value = getattr(row, "fold_id", None)
        if fold_value is None:
            fold_value = getattr(row, "fold")
        assignment_by_sample[str(row.sample_id)] = int(fold_value)
    mask_paths: dict[str, Path] = {}
    loaded_segmenters: dict[int, Any] = {}
    segmenter_image_size = int(roi_cfg.get("segmenter_image_size", 256))
    for row in manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        fold_id = assignment_by_sample.get(sample_id)
        if fold_id is None:
            raise ValueError(f"No OOF segmenter fold assignment found for sample {sample_id}.")
        mask_path = cache_dir / f"fold{fold_id}_{sample_id}.png"
        mask_paths[sample_id] = mask_path
        if mask_path.exists():
            continue
        checkpoint = _segmenter_checkpoint_for_fold(paths, roi_cfg, fold_id)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Segmenter OOF checkpoint not found: {checkpoint}")
        if fold_id not in loaded_segmenters:
            model = load_segmenter(
                {"architecture": "unet", "encoder_name": "resnet18", "encoder_weights": None, "in_channels": 3, "classes": 1},
                checkpoint,
                map_location=device,
            ).to(device)
            model.eval()
            loaded_segmenters[fold_id] = model
        image = read_image(row.image_path, grayscale=True)
        probability = _predict_segmenter_mask(
            loaded_segmenters[fold_id],
            image,
            image_size=segmenter_image_size,
            device=device,
        )
        save_image(mask_path, (np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8))
    return mask_paths


def _apply_roi_mask_source(
    *,
    train_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
    assignments: pd.DataFrame,
    paths,
    data_cfg: dict[str, Any],
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach configured ROI mask source columns to dual-view manifests."""
    roi_cfg = data_cfg.get("roi", {}) or {}
    mask_source = str(roi_cfg.get("mask_source", "gt_mask")).lower()
    if mask_source in {"gt", "gt_mask", "ground_truth"}:
        return train_manifest, val_manifest
    if mask_source != "segmenter_oof":
        raise ValueError(f"Unsupported ROI mask_source: {mask_source}")
    combined = pd.concat([train_manifest, val_manifest], ignore_index=True)
    mask_paths = _ensure_segmenter_oof_mask_cache(
        manifest=combined,
        assignments=assignments,
        paths=paths,
        roi_cfg=roi_cfg,
        device=device,
    )
    train_manifest = train_manifest.copy()
    val_manifest = val_manifest.copy()
    train_manifest["roi_mask_path"] = train_manifest["sample_id"].astype(str).map(lambda sample_id: str(mask_paths[sample_id]))
    val_manifest["roi_mask_path"] = val_manifest["sample_id"].astype(str).map(lambda sample_id: str(mask_paths[sample_id]))
    return train_manifest, val_manifest


def _build_classifier_model_and_transforms(
    *,
    config: dict[str, Any],
    data_cfg: dict[str, Any],
    device: str,
):
    """Construct the classifier plus train/eval transforms from one config."""
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
    """Create train/validation DataLoaders with Windows-safe worker defaults."""
    input_mode = str(data_cfg.get("input_mode", "single_image")).lower()
    if input_mode == "dual_view_roi":
        roi_cfg = data_cfg.get("roi", {}) or {}
        roi_kwargs = {
            "mask_threshold": float(roi_cfg.get("mask_threshold", 0.4)),
            "margin_ratio": float(roi_cfg.get("margin_ratio", 0.35)),
            "min_area_ratio": float(roi_cfg.get("min_area_ratio", 0.08)),
            "max_area_ratio": float(roi_cfg.get("max_area_ratio", 0.75)),
            "largest_component": bool(roi_cfg.get("largest_component", True)),
        }
        train_dataset = BUSBRAClassificationDualViewDataset(
            train_manifest,
            image_size=image_size,
            transform=train_transform,
            roi_transform=train_transform,
            **roi_kwargs,
        )
        val_dataset = BUSBRAClassificationDualViewDataset(
            val_manifest,
            image_size=image_size,
            transform=eval_transform,
            roi_transform=eval_transform,
            **roi_kwargs,
        )
    else:
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
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
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
    """Resolve configured class weights and move them to the training device."""
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


def _build_class_priors(train_manifest: pd.DataFrame, training_cfg: dict[str, Any], *, device: str):
    """Build empirical class priors for loss functions that correct class-frequency bias."""
    if str(training_cfg.get("loss", "cross_entropy")).lower() != "balanced_softmax":
        return None
    labels = train_manifest["pathology_label"].astype(str).str.lower()
    benign_count = float((labels == "benign").sum())
    malignant_count = float((labels == "malignant").sum())
    counts = torch.tensor([benign_count, malignant_count], dtype=torch.float32)
    priors = counts / counts.sum().clamp_min(1e-12)
    return priors.to(device=device)


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
    balanced_softmax_tau: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_probability: float,
    pairwise_auc_weight: float,
    pairwise_auc_margin: float,
    supcon_weight: float,
    supcon_temperature: float,
    moe_load_balance_weight: float,
    use_sam: bool,
    sam_rho: float,
    sam_adaptive: bool,
    use_ema: bool,
    ema_decay: float,
    sample_weight_path: Path | None,
    sample_weights: dict[str, float],
    sample_weight_hit_count: int,
    min_specificity: float,
    epoch_reports: list[dict[str, Any]],
    train_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
) -> dict[str, Any]:
    """Persist the selected classifier checkpoint and its training report."""
    checkpoints_dir = ensure_dir(paths.checkpoints_root)
    reports_dir = ensure_dir(paths.reports_root)
    checkpoint_path = checkpoints_dir / output_cfg.get("checkpoint_name", "classifier_fold{fold}.pt").format(fold=fold)
    report_path = reports_dir / output_cfg.get("report_name", "train_cls_fold{fold}.json").format(fold=fold)

    _atomic_torch_save(
        _classifier_checkpoint_payload(
            config=config,
            model=model,
            fold=fold,
            metrics=metrics,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
        ),
        checkpoint_path,
    )
    report = _classifier_training_report(
        fold=fold,
        device=device,
        checkpoint_path=checkpoint_path,
        model_config=copy.deepcopy(config.get("model", {})),
        data_config=copy.deepcopy(config.get("data", {})),
        experiment=copy.deepcopy(config.get("experiment", {})),
        metrics=metrics,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        checkpoint_strategy=checkpoint_strategy,
        preprocess_settings=preprocess_settings,
        scheduler_cfg=scheduler_cfg,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        loss_name=loss_name,
        focal_gamma=focal_gamma,
        balanced_softmax_tau=balanced_softmax_tau,
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        mix_probability=mix_probability,
        pairwise_auc_weight=pairwise_auc_weight,
        pairwise_auc_margin=pairwise_auc_margin,
        supcon_weight=supcon_weight,
        supcon_temperature=supcon_temperature,
        moe_load_balance_weight=moe_load_balance_weight,
        use_sam=use_sam,
        sam_rho=sam_rho,
        sam_adaptive=sam_adaptive,
        use_ema=use_ema,
        ema_decay=ema_decay,
        sample_weight_path=sample_weight_path,
        sample_weights=sample_weights,
        sample_weight_hit_count=sample_weight_hit_count,
        min_specificity=min_specificity,
        epoch_reports=epoch_reports,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
    )
    write_json_report(report_path, report)
    return report


def _classifier_checkpoint_payload(
    *,
    config: dict[str, Any],
    model,
    fold: int,
    metrics: dict[str, Any],
    best_epoch: int,
    best_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep the checkpoint schema in one place for training and tests."""
    return {
        "state_dict": model.state_dict(),
        "model_config": config.get("model", {}),
        "fold": fold,
        "metrics": metrics,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
    }


def _classifier_training_report(
    *,
    fold: int,
    device: str,
    checkpoint_path: Path,
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    experiment: dict[str, Any],
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
    balanced_softmax_tau: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_probability: float,
    pairwise_auc_weight: float,
    pairwise_auc_margin: float,
    supcon_weight: float,
    supcon_temperature: float,
    moe_load_balance_weight: float,
    use_sam: bool,
    sam_rho: float,
    sam_adaptive: bool,
    use_ema: bool,
    ema_decay: float,
    sample_weight_path: Path | None,
    sample_weights: dict[str, float],
    sample_weight_hit_count: int,
    min_specificity: float,
    epoch_reports: list[dict[str, Any]],
    train_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
) -> dict[str, Any]:
    """Build the JSON report without performing filesystem writes."""
    return {
        "fold": fold,
        "device": device,
        "checkpoint_path": str(checkpoint_path),
        "model_config": model_config,
        "data_config": data_config,
        "experiment": experiment,
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
        "balanced_softmax_tau": balanced_softmax_tau if loss_name == "balanced_softmax" else None,
        "mixup_alpha": mixup_alpha,
        "cutmix_alpha": cutmix_alpha,
        "mix_probability": mix_probability,
        "pairwise_auc_weight": pairwise_auc_weight,
        "pairwise_auc_margin": pairwise_auc_margin,
        "supcon_weight": supcon_weight,
        "supcon_temperature": supcon_temperature if supcon_weight > 0.0 else None,
        "moe_load_balance_weight": moe_load_balance_weight,
        "optimizer": "sam" if use_sam else "adamw",
        "sam_rho": sam_rho if use_sam else None,
        "sam_adaptive": sam_adaptive if use_sam else None,
        "use_ema": use_ema,
        "ema_decay": ema_decay if use_ema else None,
        "sample_weight_path": str(sample_weight_path) if sample_weight_path is not None else None,
        "sample_weight_count": len(sample_weights),
        "sample_weight_hit_count": int(sample_weight_hit_count),
        "min_specificity": min_specificity,
        "epoch_reports": epoch_reports,
        "train_size": int(len(train_manifest)),
        "val_size": int(len(val_manifest)),
    }


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
    class_priors: Any
    label_smoothing: float
    loss_name: str
    focal_gamma: float
    balanced_softmax_tau: float
    mixup_alpha: float
    cutmix_alpha: float
    mix_probability: float
    pairwise_auc_weight: float
    pairwise_auc_margin: float
    supcon_weight: float
    supcon_temperature: float
    moe_load_balance_weight: float
    use_sam: bool
    sam_rho: float
    sam_adaptive: bool
    use_ema: bool
    ema_decay: float
    sample_weights: dict[str, float]


@dataclasses.dataclass(slots=True)
class _TrainingLoopResult:
    """Best checkpoint candidate and per-epoch metrics from a training loop."""

    best_state_dict: dict[str, Any] | None
    best_epoch: int
    best_metrics: dict[str, Any] | None
    epoch_reports: list[dict[str, Any]]
    best_source: str


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
    sample_weight_hit_count: int
    loop_config: _TrainingLoopConfig
    base_learning_rate: float
    optimizer: Any
    model_ema: Any


def _build_classifier_optimizer(model, training_cfg: dict[str, Any], *, learning_rate: float):
    """Build the AdamW optimizer used by classifier training."""
    optimizer_name = str(training_cfg.get("optimizer", "adamw")).lower()
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    if optimizer_name == "sam":
        return _SAMOptimizer(
            model.parameters(),
            optim.AdamW,
            rho=float(training_cfg.get("sam_rho", 0.05)),
            adaptive=bool(training_cfg.get("sam_adaptive", False)),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    return optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def _build_training_loop_config(
    *,
    training_cfg: dict[str, Any],
    epochs_override: int | None,
    class_weights,
    class_priors,
    sample_weights: dict[str, float],
) -> _TrainingLoopConfig:
    """Normalize training config values needed inside the epoch loop."""
    return _TrainingLoopConfig(
        epochs=int(epochs_override or training_cfg.get("epochs", 5)),
        scheduler_cfg=training_cfg.get("scheduler", {}) or {},
        checkpoint_strategy=str(training_cfg.get("checkpoint_strategy", "last")).lower(),
        selection_threshold=float(training_cfg.get("selection_threshold", 0.5)),
        sensitivity_weight=float(training_cfg.get("sensitivity_weight", 0.0)),
        min_specificity=float(training_cfg.get("min_specificity", 0.0)),
        class_weights=class_weights,
        class_priors=class_priors,
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0)),
        loss_name=str(training_cfg.get("loss", "cross_entropy")).lower(),
        focal_gamma=float(training_cfg.get("focal_gamma", 2.0)),
        balanced_softmax_tau=float(training_cfg.get("balanced_softmax_tau", 1.0)),
        mixup_alpha=float(training_cfg.get("mixup_alpha", 0.0)),
        cutmix_alpha=float(training_cfg.get("cutmix_alpha", 0.0)),
        mix_probability=float(training_cfg.get("mix_probability", 0.0)),
        pairwise_auc_weight=float(training_cfg.get("pairwise_auc_weight", 0.0)),
        pairwise_auc_margin=float(training_cfg.get("pairwise_auc_margin", 0.0)),
        supcon_weight=float(training_cfg.get("supcon_weight", 0.0)),
        supcon_temperature=float(training_cfg.get("supcon_temperature", 0.07)),
        moe_load_balance_weight=float(training_cfg.get("moe_load_balance_weight", 0.0)),
        use_sam=str(training_cfg.get("optimizer", "adamw")).lower() == "sam",
        sam_rho=float(training_cfg.get("sam_rho", 0.05)),
        sam_adaptive=bool(training_cfg.get("sam_adaptive", False)),
        use_ema=bool(training_cfg.get("use_ema", False)),
        ema_decay=float(training_cfg.get("ema_decay", 0.9998)),
        sample_weights=sample_weights,
    )


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
    roi_cfg = data_cfg.get("roi", {}) or {}
    if str(roi_cfg.get("mask_source", "gt_mask")).lower() == "segmenter_oof":
        split_path = Path(training_cfg.get("split_path", paths.reports_root / "busbra_5fold_splits.csv"))
        if not split_path.is_absolute():
            split_path = (paths.project_root / split_path).resolve()
        assignments = _load_or_create_splits(
            manifest,
            split_path,
            fold_count=int(training_cfg.get("fold_count", 5)),
            seed=seed,
        )
        train_manifest, val_manifest = _apply_roi_mask_source(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            assignments=assignments,
            paths=paths,
            data_cfg=data_cfg,
            device=device,
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
    optimizer = _build_classifier_optimizer(
        model,
        training_cfg,
        learning_rate=base_learning_rate,
    )
    class_weights = _build_class_weights(train_manifest, training_cfg, device=device)
    class_priors = _build_class_priors(train_manifest, training_cfg, device=device)
    sample_weight_path = _resolve_sample_weight_path(paths, training_cfg)
    sample_weights = _load_sample_weights(
        sample_weight_path,
        weight_column=str(training_cfg.get("sample_weight_column", "sample_weight")),
    )
    sample_weight_hit_count = _count_sample_weight_hits(train_manifest, sample_weights)
    if sample_weight_path is not None and sample_weights and sample_weight_hit_count <= 0:
        raise ValueError(
            f"Sample weight file {sample_weight_path} does not match any training samples for fold {fold}."
        )
    loop_config = _build_training_loop_config(
        training_cfg=training_cfg,
        epochs_override=epochs_override,
        class_weights=class_weights,
        class_priors=class_priors,
        sample_weights=sample_weights,
    )
    model_ema = _ModelEma(model, decay=loop_config.ema_decay) if loop_config.use_ema else None
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
        sample_weight_hit_count=sample_weight_hit_count,
        loop_config=loop_config,
        base_learning_rate=base_learning_rate,
        optimizer=optimizer,
        model_ema=model_ema,
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
    model_ema,
    logger,
) -> _TrainingLoopResult:
    """Train epochs and return the selected checkpoint candidate plus reports."""
    cfg = loop_config
    best_state_dict = None
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    best_score = float("-inf")
    epoch_reports: list[dict[str, Any]] = []
    best_source = "model"

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
            model_ema,
            device=device,
            class_weights=cfg.class_weights,
            class_priors=cfg.class_priors,
            sample_weights=cfg.sample_weights,
            label_smoothing=cfg.label_smoothing,
            loss_name=cfg.loss_name,
            focal_gamma=cfg.focal_gamma,
            balanced_softmax_tau=cfg.balanced_softmax_tau,
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            mix_probability=cfg.mix_probability,
            pairwise_auc_weight=cfg.pairwise_auc_weight,
            pairwise_auc_margin=cfg.pairwise_auc_margin,
            supcon_weight=cfg.supcon_weight,
            supcon_temperature=cfg.supcon_temperature,
            moe_load_balance_weight=cfg.moe_load_balance_weight,
            use_sam=cfg.use_sam,
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
            best_source = "model"
        ema_epoch_metrics = None
        ema_score_metrics = None
        ema_score = None
        if model_ema is not None:
            original_state = _snapshot_state_dict(model)
            model.load_state_dict(model_ema.state_dict)
            ema_y_true, ema_malignant_probabilities = _collect_validation_probabilities(
                model, val_loader, device,
            )
            ema_epoch_metrics = classification_metrics(ema_y_true, ema_malignant_probabilities)
            ema_score_metrics, ema_score = _score_checkpoint_candidate(
                epoch_metrics=ema_epoch_metrics,
                y_true=ema_y_true,
                malignant_probabilities=ema_malignant_probabilities,
                checkpoint_strategy=cfg.checkpoint_strategy,
                selection_threshold=cfg.selection_threshold,
                sensitivity_weight=cfg.sensitivity_weight,
                min_specificity=cfg.min_specificity,
            )
            model.load_state_dict(original_state)
            if float(ema_score) > best_score:
                best_score = float(ema_score)
                best_epoch = epoch + 1
                best_metrics = ema_score_metrics
                best_state_dict = {
                    key: value.clone()
                    for key, value in model_ema.state_dict.items()
                }
                best_source = "ema"
        epoch_reports.append({
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "loss": mean_loss,
            "metrics": epoch_metrics,
            "score_metrics": score_metrics,
            "selection_score": score,
            "ema_metrics": ema_epoch_metrics,
            "ema_score_metrics": ema_score_metrics,
            "ema_selection_score": ema_score,
        })
        logger.info(
            "fold=%s epoch=%s loss=%.4f auc=%.4f sens=%.4f score=%.4f%s",
            fold,
            epoch + 1,
            mean_loss,
            float(epoch_metrics.get("auc") or 0.0),
            float(score_metrics.get("sensitivity", 0.0)),
            score,
            (
                f" ema_auc={float((ema_epoch_metrics or {}).get('auc') or 0.0):.4f}"
                f" ema_score={float(ema_score or 0.0):.4f}"
                if model_ema is not None
                else ""
            ),
        )
    return _TrainingLoopResult(
        best_state_dict=best_state_dict,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        epoch_reports=epoch_reports,
        best_source=best_source,
    )


def _final_classifier_metrics(
    *,
    model,
    val_loader,
    device: str,
    checkpoint_strategy: str,
    epoch_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return final validation metrics, reusing the last epoch pass when possible."""
    if checkpoint_strategy == "last" and epoch_reports:
        return dict(epoch_reports[-1]["metrics"])
    return _evaluate_model(model, val_loader, device)


def _finish_classifier_training_run(
    *,
    config: dict[str, Any],
    paths,
    output_cfg: dict[str, Any],
    fold: int,
    device: str,
    prepared: _PreparedTrainingRun,
    loop_result: _TrainingLoopResult,
) -> dict[str, Any]:
    """Restore the selected checkpoint when needed and persist training outputs."""
    if prepared.loop_config.checkpoint_strategy != "last" and loop_result.best_state_dict is not None:
        prepared.model.load_state_dict(loop_result.best_state_dict)
    metrics = _final_classifier_metrics(
        model=prepared.model,
        val_loader=prepared.val_loader,
        device=device,
        checkpoint_strategy=prepared.loop_config.checkpoint_strategy,
        epoch_reports=loop_result.epoch_reports,
    )
    return _write_classifier_training_outputs(
        config=config,
        paths=paths,
        output_cfg=output_cfg,
        fold=fold,
        device=device,
        model=prepared.model,
        metrics=metrics,
        best_epoch=loop_result.best_epoch,
        best_metrics=loop_result.best_metrics,
        checkpoint_strategy=prepared.loop_config.checkpoint_strategy,
        preprocess_settings=prepared.preprocess_settings,
        scheduler_cfg=prepared.loop_config.scheduler_cfg,
        class_weights=prepared.class_weights,
        label_smoothing=prepared.loop_config.label_smoothing,
        loss_name=prepared.loop_config.loss_name,
        focal_gamma=prepared.loop_config.focal_gamma,
        balanced_softmax_tau=prepared.loop_config.balanced_softmax_tau,
        mixup_alpha=prepared.loop_config.mixup_alpha,
        cutmix_alpha=prepared.loop_config.cutmix_alpha,
        mix_probability=prepared.loop_config.mix_probability,
        pairwise_auc_weight=prepared.loop_config.pairwise_auc_weight,
        pairwise_auc_margin=prepared.loop_config.pairwise_auc_margin,
        supcon_weight=prepared.loop_config.supcon_weight,
        supcon_temperature=prepared.loop_config.supcon_temperature,
        moe_load_balance_weight=prepared.loop_config.moe_load_balance_weight,
        use_sam=prepared.loop_config.use_sam,
        sam_rho=prepared.loop_config.sam_rho,
        sam_adaptive=prepared.loop_config.sam_adaptive,
        use_ema=prepared.loop_config.use_ema,
        ema_decay=prepared.loop_config.ema_decay,
        sample_weight_path=prepared.sample_weight_path,
        sample_weights=prepared.sample_weights,
        sample_weight_hit_count=prepared.sample_weight_hit_count,
        min_specificity=prepared.loop_config.min_specificity,
        epoch_reports=loop_result.epoch_reports,
        train_manifest=prepared.train_manifest,
        val_manifest=prepared.val_manifest,
    )


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
    loop_result = _run_training_loop(
        model=prepared.model,
        train_loader=prepared.train_loader,
        val_loader=prepared.val_loader,
        optimizer=prepared.optimizer,
        device=device,
        fold=fold,
        loop_config=prepared.loop_config,
        base_learning_rate=prepared.base_learning_rate,
        model_ema=prepared.model_ema,
        logger=logger,
    )
    return _finish_classifier_training_run(
        config=config,
        paths=paths,
        output_cfg=output_cfg,
        fold=fold,
        device=device,
        prepared=prepared,
        loop_result=loop_result,
    )
class _SAMOptimizer:
    """Minimal SAM wrapper around a base optimizer for classification experiments."""

    def __init__(self, params, base_optimizer_cls, *, rho: float, adaptive: bool = False, **kwargs) -> None:
        self.param_groups = []
        self.base_optimizer = base_optimizer_cls(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.rho = float(rho)
        self.adaptive = bool(adaptive)
        self._cached_perturbations: list[tuple[Any, Any]] = []

    def zero_grad(self) -> None:
        """Delegate gradient clearing to the wrapped optimizer."""
        self.base_optimizer.zero_grad()

    def first_step(self) -> None:
        """Perturb parameters toward the local sharpness ascent direction."""
        require_dependency("torch", torch)
        grad_norm = self._grad_norm()
        scale = self.rho / (float(grad_norm.item()) + 1e-12)
        self._cached_perturbations = []
        with torch.no_grad():
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    if self.adaptive:
                        e_w = (param.pow(2) * param.grad) * scale
                    else:
                        e_w = param.grad * scale
                    param.add_(e_w)
                    self._cached_perturbations.append((param, e_w))

    def second_step(self) -> None:
        """Restore parameters and apply the wrapped optimizer update."""
        with torch.no_grad():
            for param, e_w in self._cached_perturbations:
                param.sub_(e_w)
        self.base_optimizer.step()
        self._cached_perturbations = []

    def _grad_norm(self):
        """Return the global L2 norm over all parameter gradients."""
        require_dependency("torch", torch)
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if self.adaptive:
                    grad = grad * param.abs()
                norms.append(torch.norm(grad, p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)
