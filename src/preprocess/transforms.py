from __future__ import annotations

from typing import Any

import numpy as np

from src.preprocess.io import cv2, ensure_three_channels
from src.utils.runtime import optional_import


torch = optional_import("torch")


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    if cv2 is not None:
        interpolation = cv2.INTER_AREA if image.ndim == 2 else cv2.INTER_LINEAR
        return cv2.resize(image, (size, size), interpolation=interpolation)

    pil_module = optional_import("PIL.Image")
    if pil_module is None:
        raise RuntimeError("Resize requires cv2 or Pillow.")
    pil_image = pil_module.fromarray(image)
    return np.asarray(pil_image.resize((size, size)))


def normalize_image(image: np.ndarray) -> np.ndarray:
    array = image.astype(np.float32)
    if array.max() > 1.0:
        array /= 255.0
    return array


def apply_clahe(image: np.ndarray, *, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Apply local contrast enhancement for ultrasound-style grayscale images."""
    if cv2 is None:
        return image
    gray = image[..., 0] if image.ndim == 3 else image
    uint8_image = gray.astype(np.uint8) if gray.dtype != np.uint8 else gray
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    enhanced = clahe.apply(uint8_image)
    if image.ndim == 3:
        return ensure_three_channels(enhanced)
    return enhanced


def to_chw(image: np.ndarray) -> np.ndarray:
    image = ensure_three_channels(image)
    return np.transpose(image, (2, 0, 1)).astype(np.float32)


def to_tensor_if_available(array: np.ndarray) -> Any:
    if torch is None:
        return array
    return torch.from_numpy(array)


def prepare_classifier_input(
    image: np.ndarray,
    image_size: int,
    *,
    apply_clahe_enabled: bool = False,
) -> Any:
    processed = apply_clahe(image) if apply_clahe_enabled else image
    resized = resize_image(ensure_three_channels(processed), image_size)
    normalized = normalize_image(resized)
    chw = to_chw(normalized)
    return to_tensor_if_available(chw)


def build_classifier_transform(
    *,
    image_size: int,
    apply_clahe_enabled: bool = False,
    horizontal_flip: bool = False,
    flip_probability: float = 0.5,
):
    def transform(image: np.ndarray) -> Any:
        processed = image
        if horizontal_flip and np.random.random() < float(flip_probability):
            processed = np.fliplr(processed).copy()
        return prepare_classifier_input(
            processed,
            image_size,
            apply_clahe_enabled=apply_clahe_enabled,
        )

    return transform


def prepare_mask_target(mask: np.ndarray, image_size: int) -> Any:
    resized = resize_image(mask.astype(np.uint8), image_size)
    channel = resized.astype(np.float32)[None, ...]
    return to_tensor_if_available(channel)
