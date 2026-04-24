from __future__ import annotations

from typing import Any

import numpy as np

from src.preprocess.io import cv2, ensure_three_channels
from src.utils.runtime import optional_import


torch = optional_import("torch")


def _random_scale_crop(image: np.ndarray, *, scale_min: float, scale_max: float) -> np.ndarray:
    if cv2 is None or scale_min <= 0 or scale_max <= 0:
        return image
    scale = float(np.random.uniform(scale_min, scale_max))
    if abs(scale - 1.0) < 1e-3:
        return image

    height, width = image.shape[:2]
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    if scale >= 1.0:
        start_x = max(0, (new_width - width) // 2)
        start_y = max(0, (new_height - height) // 2)
        return resized[start_y:start_y + height, start_x:start_x + width]

    pad_left = (width - new_width) // 2
    pad_right = width - new_width - pad_left
    pad_top = (height - new_height) // 2
    pad_bottom = height - new_height - pad_top
    return np.pad(
        resized,
        ((pad_top, pad_bottom), (pad_left, pad_right)) + (() if image.ndim == 2 else ((0, 0),)),
        mode="edge",
    )


def _random_rotate(image: np.ndarray, *, max_degrees: float) -> np.ndarray:
    if cv2 is None or max_degrees <= 0:
        return image
    angle = float(np.random.uniform(-max_degrees, max_degrees))
    if abs(angle) < 1e-3:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _random_brightness_contrast(
    image: np.ndarray,
    *,
    brightness: float,
    contrast: float,
) -> np.ndarray:
    if brightness <= 0 and contrast <= 0:
        return image
    array = image.astype(np.float32)
    contrast_factor = 1.0 + float(np.random.uniform(-contrast, contrast))
    brightness_delta = float(np.random.uniform(-brightness, brightness)) * 255.0
    adjusted = array * contrast_factor + brightness_delta
    return np.clip(adjusted, 0, 255).astype(image.dtype)


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
    rotation_degrees: float = 0.0,
    brightness: float = 0.0,
    contrast: float = 0.0,
    scale_min: float = 1.0,
    scale_max: float = 1.0,
):
    def transform(image: np.ndarray) -> Any:
        processed = image
        if scale_min != 1.0 or scale_max != 1.0:
            processed = _random_scale_crop(
                processed,
                scale_min=float(scale_min),
                scale_max=float(scale_max),
            )
        if rotation_degrees > 0:
            processed = _random_rotate(processed, max_degrees=float(rotation_degrees))
        if horizontal_flip and np.random.random() < float(flip_probability):
            processed = np.fliplr(processed).copy()
        if brightness > 0 or contrast > 0:
            processed = _random_brightness_contrast(
                processed,
                brightness=float(brightness),
                contrast=float(contrast),
            )
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
