"""Segmentation loss components for mask, boundary, affinity, and prototype terms."""

from __future__ import annotations

from typing import Any

from src.utils.runtime import optional_import, require_dependency


torch = optional_import("torch")
F = optional_import("torch.nn.functional")


def _resize_like(values, target):
    """Resize feature maps to match the target spatial dimensions."""
    require_dependency("torch.nn.functional", F)
    if values.shape[-2:] == target.shape[-2:]:
        return values
    return F.interpolate(values, size=target.shape[-2:], mode="bilinear", align_corners=False)


def _binary_target(mask):
    """Convert a probability mask to a binary target tensor."""
    return (mask > 0.5).to(dtype=mask.dtype)


def dice_loss_from_logits(logits, targets, *, eps: float = 1e-6):
    """Compute a soft Dice loss from raw logits and binary targets."""
    require_dependency("torch", torch)
    probabilities = torch.sigmoid(_resize_like(logits, targets))
    targets = targets.to(dtype=probabilities.dtype)
    dims = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def boundary_target_from_mask(mask, *, kernel_size: int = 3):
    """Derive a single-pixel boundary target from a mask using dilation minus erosion."""
    require_dependency("torch", torch)
    require_dependency("torch.nn.functional", F)
    target = _binary_target(mask)
    padding = int(kernel_size) // 2
    dilation = F.max_pool2d(target, kernel_size=kernel_size, stride=1, padding=padding)
    erosion = -F.max_pool2d(-target, kernel_size=kernel_size, stride=1, padding=padding)
    return (dilation - erosion).clamp(0.0, 1.0)


def boundary_bce_loss(boundary_logits, targets):
    """Compute binary cross-entropy between predicted and target boundaries."""
    require_dependency("torch.nn.functional", F)
    resized_logits = _resize_like(boundary_logits, targets)
    boundary_targets = boundary_target_from_mask(targets)
    return F.binary_cross_entropy_with_logits(resized_logits, boundary_targets)


def pixel_affinity_loss(mask_logits, targets, *, shifts: tuple[tuple[int, int], ...] | None = None):
    """Penalize neighboring pixel prediction inconsistency for mask refinement."""
    require_dependency("torch", torch)
    require_dependency("torch.nn.functional", F)
    if shifts is None:
        shifts = ((1, 0), (0, 1), (1, 1), (1, -1))
    probabilities = torch.sigmoid(_resize_like(mask_logits, targets))
    targets = _binary_target(targets).to(dtype=probabilities.dtype)
    losses = []
    for dy, dx in shifts:
        pred_a = probabilities[:, :, max(dy, 0): probabilities.shape[-2] + min(dy, 0), max(dx, 0): probabilities.shape[-1] + min(dx, 0)]
        pred_b = probabilities[:, :, max(-dy, 0): probabilities.shape[-2] - max(dy, 0), max(-dx, 0): probabilities.shape[-1] - max(dx, 0)]
        tgt_a = targets[:, :, max(dy, 0): targets.shape[-2] + min(dy, 0), max(dx, 0): targets.shape[-1] + min(dx, 0)]
        tgt_b = targets[:, :, max(-dy, 0): targets.shape[-2] - max(dy, 0), max(-dx, 0): targets.shape[-1] - max(dx, 0)]
        if pred_a.numel() == 0:
            continue
        pred_affinity = 1.0 - torch.abs(pred_a - pred_b)
        target_affinity = 1.0 - torch.abs(tgt_a - tgt_b)
        losses.append(F.mse_loss(pred_affinity, target_affinity))
    if not losses:
        return probabilities.sum() * 0.0
    return torch.stack(losses).mean()


def _weighted_prototype(features, weights):
    """Compute a spatially weighted feature prototype."""
    require_dependency("torch", torch)
    weights = weights.to(dtype=features.dtype)
    numerator = (features * weights).sum(dim=(2, 3))
    denominator = weights.sum(dim=(2, 3)).clamp_min(1e-6)
    return numerator / denominator


def prototype_contrast_loss(features, targets, *, positive_region: str = "foreground"):
    """Contrast foreground or edge prototypes against their complementary region."""
    require_dependency("torch", torch)
    require_dependency("torch.nn.functional", F)
    resized_targets = F.interpolate(
        _binary_target(targets).to(dtype=features.dtype),
        size=features.shape[-2:],
        mode="nearest",
    )
    if positive_region == "edge":
        positive = F.interpolate(
            boundary_target_from_mask(targets).to(dtype=features.dtype),
            size=features.shape[-2:],
            mode="nearest",
        )
    else:
        positive = resized_targets
    negative = 1.0 - positive
    has_positive = positive.sum(dim=(1, 2, 3)) > 0
    has_negative = negative.sum(dim=(1, 2, 3)) > 0
    valid = has_positive & has_negative
    if not bool(valid.any()):
        return features.sum() * 0.0
    selected_features = F.normalize(features[valid], dim=1)
    selected_positive = positive[valid]
    selected_negative = negative[valid]
    positive_proto = F.normalize(_weighted_prototype(selected_features, selected_positive), dim=1)
    negative_proto = F.normalize(_weighted_prototype(selected_features, selected_negative), dim=1)
    sim_positive = (selected_features * positive_proto[:, :, None, None]).sum(dim=1, keepdim=True)
    sim_negative = (selected_features * negative_proto[:, :, None, None]).sum(dim=1, keepdim=True)
    logits = sim_positive - sim_negative
    return F.binary_cross_entropy_with_logits(logits, selected_positive)


def _extract_mask_logits(outputs: Any):
    """Unwrap mask logits from dict or raw model outputs."""
    if isinstance(outputs, dict):
        return outputs.get("mask")
    return outputs


def segmentation_loss(outputs: Any, targets, config: dict[str, Any] | None = None):
    """Combine configured mask, boundary, affinity, and prototype loss terms."""
    require_dependency("torch", torch)
    require_dependency("torch.nn.functional", F)
    cfg = config or {}
    mask_logits = _extract_mask_logits(outputs)
    if mask_logits is None:
        raise ValueError("Segmentation model output does not contain mask logits.")
    mask_logits = _resize_like(mask_logits, targets)
    loss_name = str(cfg.get("name", "bce")).lower()
    bce_weight = float(cfg.get("bce_weight", 1.0 if "bce" in loss_name else 0.0))
    dice_weight = float(cfg.get("dice_weight", 1.0 if "dice" in loss_name else 0.0))
    boundary_weight = float(cfg.get("boundary_weight", 0.0))
    pal_weight = float(cfg.get("pal_weight", 0.0))
    foreground_prototype_weight = float(cfg.get("foreground_prototype_weight", 0.0))
    edge_prototype_weight = float(cfg.get("edge_prototype_weight", 0.0))
    if "boundary" in loss_name and boundary_weight == 0.0:
        boundary_weight = 1.0
    if "pal" in loss_name and pal_weight == 0.0:
        pal_weight = 0.2
    if "prototype" in loss_name and foreground_prototype_weight == 0.0:
        foreground_prototype_weight = 0.1

    components = {}
    total = mask_logits.sum() * 0.0
    if bce_weight:
        components["bce"] = F.binary_cross_entropy_with_logits(mask_logits, targets)
        total = total + bce_weight * components["bce"]
    if dice_weight:
        components["dice"] = dice_loss_from_logits(mask_logits, targets)
        total = total + dice_weight * components["dice"]
    if boundary_weight:
        boundary_logits = outputs.get("boundary") if isinstance(outputs, dict) else mask_logits
        components["boundary"] = boundary_bce_loss(boundary_logits, targets)
        total = total + boundary_weight * components["boundary"]
    if pal_weight:
        components["pal"] = pixel_affinity_loss(mask_logits, targets)
        total = total + pal_weight * components["pal"]
    features = outputs.get("prototype_features") if isinstance(outputs, dict) else None
    if features is not None and foreground_prototype_weight:
        components["foreground_prototype"] = prototype_contrast_loss(features, targets, positive_region="foreground")
        total = total + foreground_prototype_weight * components["foreground_prototype"]
    if features is not None and edge_prototype_weight:
        components["edge_prototype"] = prototype_contrast_loss(features, targets, positive_region="edge")
        total = total + edge_prototype_weight * components["edge_prototype"]
    if not components:
        components["bce"] = F.binary_cross_entropy_with_logits(mask_logits, targets)
        total = components["bce"]
    return total, components
