"""Image overlay rendering utilities for segmentation masks and heatmaps."""

from __future__ import annotations

import numpy as np

from src.preprocess.io import cv2, ensure_three_channels


def _resize_like(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == target_shape:
        return image
    if cv2 is not None:
        return cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    return image


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    array = image.astype(np.float32)
    if array.max() <= 1.0:
        array *= 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def render_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (255, 64, 64),
    alpha: float = 0.35,
) -> np.ndarray:
    base = normalize_uint8(ensure_three_channels(image))
    resized_mask = _resize_like(mask.astype(np.float32), base.shape[:2])
    binary_mask = resized_mask > 0.5
    overlay = base.copy()
    overlay[binary_mask] = (
        (1.0 - alpha) * overlay[binary_mask] + alpha * np.asarray(color, dtype=np.float32)
    ).astype(np.uint8)
    return overlay


def render_heatmap_overlay(image: np.ndarray, heatmap: np.ndarray, *, alpha: float = 0.4) -> np.ndarray:
    base = normalize_uint8(ensure_three_channels(image))
    heat = _resize_like(heatmap.astype(np.float32), base.shape[:2])
    heat = heat - heat.min()
    denom = float(heat.max()) if float(heat.max()) > 0 else 1.0
    heat = heat / denom

    if cv2 is not None:
        color_map = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    else:
        color_map = np.stack(
            [(heat * 255), np.zeros_like(heat), ((1.0 - heat) * 255)],
            axis=-1,
        ).astype(np.uint8)
    blended = ((1.0 - alpha) * base + alpha * color_map).astype(np.uint8)
    return blended
