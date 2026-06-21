"""Classifier and segmenter preprocessing transforms for numpy and torch inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.preprocess.io import cv2, ensure_three_channels
from src.utils.runtime import optional_import


torch = optional_import("torch")


def _random_scale_crop(image: np.ndarray, *, scale_min: float, scale_max: float) -> np.ndarray:
    """Randomly zoom or pad an image while preserving the original output size."""
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
    """Randomly rotate an image within +/- max_degrees, preserving size."""
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
    """Randomly adjust brightness and contrast while preserving dtype."""
    if brightness <= 0 and contrast <= 0:
        return image
    array = image.astype(np.float32)
    contrast_factor = 1.0 + float(np.random.uniform(-contrast, contrast))
    brightness_delta = float(np.random.uniform(-brightness, brightness)) * 255.0
    adjusted = array * contrast_factor + brightness_delta
    return np.clip(adjusted, 0, 255).astype(image.dtype)


def _cv2_interpolation(name: str):
    """Map a readable interpolation name to an OpenCV interpolation code."""
    if cv2 is None:
        return None
    normalized = str(name).lower()
    mapping = {
        "area": cv2.INTER_AREA,
        "linear": cv2.INTER_LINEAR,
        "bilinear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "bicubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    return mapping.get(normalized, cv2.INTER_AREA)


def resize_image(image: np.ndarray, size: int, *, interpolation: str = "area") -> np.ndarray:
    """Resize an image or mask to a square size."""
    if cv2 is not None:
        interpolation_code = _cv2_interpolation(interpolation)
        return cv2.resize(image, (size, size), interpolation=interpolation_code)

    pil_module = optional_import("PIL.Image")
    if pil_module is None:
        raise RuntimeError("Resize requires cv2 or Pillow.")
    pil_image = pil_module.fromarray(image)
    resample = getattr(pil_module, "BICUBIC", 3) if str(interpolation).lower() == "bicubic" else getattr(pil_module, "BILINEAR", 2)
    return np.asarray(pil_image.resize((size, size), resample=resample))


def _resize_shorter_side(image: np.ndarray, size: int, *, interpolation: str) -> np.ndarray:
    """Resize an image so its shorter side equals the requested size."""
    if cv2 is None:
        pil_module = optional_import("PIL.Image")
        if pil_module is None:
            raise RuntimeError("Resize requires cv2 or Pillow.")
        height, width = image.shape[:2]
        scale = float(size) / float(min(height, width))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        pil_image = pil_module.fromarray(image)
        resample = getattr(pil_module, "BICUBIC", 3) if str(interpolation).lower() == "bicubic" else getattr(pil_module, "BILINEAR", 2)
        return np.asarray(pil_image.resize((new_width, new_height), resample=resample))

    height, width = image.shape[:2]
    scale = float(size) / float(min(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=_cv2_interpolation(interpolation),
    )


def center_crop(image: np.ndarray, size: int) -> np.ndarray:
    """Crop the center square region, resizing first when the image is too small."""
    height, width = image.shape[:2]
    if height < size or width < size:
        return resize_image(image, size)
    top = max(0, (height - size) // 2)
    left = max(0, (width - size) // 2)
    return image[top:top + size, left:left + size]


def resize_with_optional_crop(
    image: np.ndarray,
    size: int,
    *,
    interpolation: str = "area",
    crop_pct: float = 1.0,
) -> np.ndarray:
    """Resize directly or use timm-style resize-then-center-crop preprocessing."""
    crop_pct = float(crop_pct or 1.0)
    if crop_pct >= 0.999:
        return resize_image(image, size, interpolation=interpolation)
    resize_size = max(size, int(round(size / crop_pct)))
    resized = _resize_shorter_side(image, resize_size, interpolation=interpolation)
    return center_crop(resized, size)


def _channel_values(values: Sequence[float] | None) -> np.ndarray | None:
    """Normalize one or three channel constants into broadcastable RGB shape."""
    if values is None:
        return None
    array = np.asarray(list(values), dtype=np.float32)
    if array.size == 1:
        array = np.repeat(array, 3)
    if array.size != 3:
        raise ValueError("Image normalization values must contain 1 or 3 numbers.")
    return array.reshape(1, 1, 3)


def normalize_image(
    image: np.ndarray,
    *,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> np.ndarray:
    """Scale image pixels to float range and apply optional mean/std normalization."""
    array = image.astype(np.float32)
    if array.max() > 1.0:
        array /= 255.0
    mean_array = _channel_values(mean)
    std_array = _channel_values(std)
    if mean_array is not None:
        array = array - mean_array
    if std_array is not None:
        array = array / np.maximum(std_array, 1e-6)
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
    """Convert an image from HWC to CHW float32 layout."""
    image = ensure_three_channels(image)
    return np.transpose(image, (2, 0, 1)).astype(np.float32)


def to_tensor_if_available(array: np.ndarray) -> Any:
    """Convert a NumPy array to torch.Tensor when PyTorch is installed."""
    if torch is None:
        return array
    return torch.from_numpy(array)


def prepare_classifier_input(
    image: np.ndarray,
    image_size: int,
    *,
    apply_clahe_enabled: bool = False,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    interpolation: str = "area",
    crop_pct: float = 1.0,
) -> Any:
    """Apply classifier preprocessing and return a CHW tensor/array."""
    processed = apply_clahe(image) if apply_clahe_enabled else image
    resized = resize_with_optional_crop(
        ensure_three_channels(processed),
        image_size,
        interpolation=interpolation,
        crop_pct=crop_pct,
    )
    normalized = normalize_image(resized, mean=mean, std=std)
    chw = to_chw(normalized)
    return to_tensor_if_available(chw)


@dataclass(frozen=True)
class ClassifierTransform:
    """Pickle-safe stochastic classifier transform for DataLoader workers."""

    image_size: int
    apply_clahe_enabled: bool = False
    horizontal_flip: bool = False
    flip_probability: float = 0.5
    rotation_degrees: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    scale_min: float = 1.0
    scale_max: float = 1.0
    mean: Sequence[float] | None = None
    std: Sequence[float] | None = None
    interpolation: str = "area"
    crop_pct: float = 1.0

    def __call__(self, image: np.ndarray) -> Any:
        """Apply stochastic image augmentation then standard classifier preprocessing."""
        processed = image
        if self.scale_min != 1.0 or self.scale_max != 1.0:
            processed = _random_scale_crop(
                processed,
                scale_min=float(self.scale_min),
                scale_max=float(self.scale_max),
            )
        if self.rotation_degrees > 0:
            processed = _random_rotate(processed, max_degrees=float(self.rotation_degrees))
        if self.horizontal_flip and np.random.random() < float(self.flip_probability):
            processed = np.fliplr(processed).copy()
        if self.brightness > 0 or self.contrast > 0:
            processed = _random_brightness_contrast(
                processed,
                brightness=float(self.brightness),
                contrast=float(self.contrast),
            )
        return prepare_classifier_input(
            processed,
            self.image_size,
            apply_clahe_enabled=self.apply_clahe_enabled,
            mean=self.mean,
            std=self.std,
            interpolation=self.interpolation,
            crop_pct=self.crop_pct,
        )


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
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    interpolation: str = "area",
    crop_pct: float = 1.0,
):
    """Create the stochastic training/evaluation transform used by classifiers."""
    return ClassifierTransform(
        image_size=image_size,
        apply_clahe_enabled=apply_clahe_enabled,
        horizontal_flip=horizontal_flip,
        flip_probability=flip_probability,
        rotation_degrees=rotation_degrees,
        brightness=brightness,
        contrast=contrast,
        scale_min=scale_min,
        scale_max=scale_max,
        mean=mean,
        std=std,
        interpolation=interpolation,
        crop_pct=crop_pct,
    )


def prepare_mask_target(mask: np.ndarray, image_size: int) -> Any:
    """Resize a binary mask target and return it with a channel dimension."""
    resized = resize_image(mask.astype(np.uint8), image_size)
    channel = resized.astype(np.float32)[None, ...]
    return to_tensor_if_available(channel)
