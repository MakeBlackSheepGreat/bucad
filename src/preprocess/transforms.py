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


def to_chw(image: np.ndarray) -> np.ndarray:
    image = ensure_three_channels(image)
    return np.transpose(image, (2, 0, 1)).astype(np.float32)


def to_tensor_if_available(array: np.ndarray) -> Any:
    if torch is None:
        return array
    return torch.from_numpy(array)


def prepare_classifier_input(image: np.ndarray, image_size: int) -> Any:
    resized = resize_image(ensure_three_channels(image), image_size)
    normalized = normalize_image(resized)
    chw = to_chw(normalized)
    return to_tensor_if_available(chw)


def prepare_mask_target(mask: np.ndarray, image_size: int) -> Any:
    resized = resize_image(mask.astype(np.uint8), image_size)
    channel = resized.astype(np.float32)[None, ...]
    return to_tensor_if_available(channel)
