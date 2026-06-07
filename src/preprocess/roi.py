"""ROI bounding-box extraction and mask crop helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np

from src.preprocess.io import cv2
from src.preprocess.transforms import resize_image


def _resize_mask_to_image(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Resize a mask to match an image before bbox extraction."""
    if mask.shape[:2] == image.shape[:2]:
        return mask
    height, width = image.shape[:2]
    if cv2 is not None:
        return cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return resize_image(mask.astype(np.float32), max(height, width))[:height, :width]


def _largest_connected_component(binary: np.ndarray) -> np.ndarray:
    """Return only the largest foreground component, with a NumPy fallback."""
    mask = binary.astype(np.uint8)
    if cv2 is None:
        height, width = mask.shape[:2]
        labels = np.zeros((height, width), dtype=np.int32)
        label_areas: list[int] = [0]
        current_label = 0
        for start_y, start_x in zip(*np.where(mask > 0), strict=False):
            if labels[start_y, start_x] != 0:
                continue
            current_label += 1
            area = 0
            stack = [(int(start_y), int(start_x))]
            labels[start_y, start_x] = current_label
            while stack:
                y, x = stack.pop()
                area += 1
                for ny in range(max(0, y - 1), min(height, y + 2)):
                    for nx in range(max(0, x - 1), min(width, x + 2)):
                        if labels[ny, nx] == 0 and mask[ny, nx] > 0:
                            labels[ny, nx] = current_label
                            stack.append((ny, nx))
            label_areas.append(area)
        if current_label == 0:
            return binary
        largest_index = int(np.argmax(label_areas[1:])) + 1
        return labels == largest_index
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if component_count <= 1:
        return binary
    largest_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_index


def mask_bbox(
    mask: np.ndarray,
    *,
    threshold: float = 0.5,
    min_area_ratio: float = 0.001,
    largest_component: bool = False,
) -> tuple[int, int, int, int] | None:
    """Return the foreground bbox for a mask, or None when the mask is too small."""
    binary = np.asarray(mask) > float(threshold)
    if binary.ndim == 3:
        binary = binary[..., 0]
    if largest_component:
        binary = _largest_connected_component(binary)
    height, width = binary.shape[:2]
    ys, xs = np.where(binary)
    if len(xs) == 0 or len(ys) == 0:
        return None
    area_ratio = float(len(xs)) / float(max(1, height * width))
    if area_ratio < float(min_area_ratio):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_bbox(
    bbox: tuple[int, int, int, int],
    *,
    image_shape: tuple[int, int] | tuple[int, int, int],
    margin_ratio: float = 0.25,
    square: bool = True,
) -> tuple[int, int, int, int]:
    """Expand a bbox with optional square normalization while staying in image bounds."""
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    if square:
        side = max(box_width, box_height)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        half = side / 2.0
        x1, x2 = int(round(cx - half)), int(round(cx + half))
        y1, y2 = int(round(cy - half)), int(round(cy + half))
        box_width = x2 - x1
        box_height = y2 - y1
    margin_x = int(round(box_width * float(margin_ratio)))
    margin_y = int(round(box_height * float(margin_ratio)))
    return (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(width, x2 + margin_x),
        min(height, y2 + margin_y),
    )


def crop_to_mask_bbox(
    image: np.ndarray,
    mask: np.ndarray | None,
    *,
    threshold: float = 0.5,
    margin_ratio: float = 0.25,
    min_area_ratio: float = 0.001,
    largest_component: bool = False,
    square: bool = True,
    fallback: Literal["full"] = "full",
) -> np.ndarray:
    """Crop an image around a mask bbox, falling back to the full image."""
    if mask is None:
        return image.copy()
    resized_mask = _resize_mask_to_image(mask, image)
    bbox = mask_bbox(
        resized_mask,
        threshold=threshold,
        min_area_ratio=min_area_ratio,
        largest_component=largest_component,
    )
    if bbox is None:
        return image.copy()
    x1, y1, x2, y2 = expand_bbox(
        bbox,
        image_shape=image.shape,
        margin_ratio=margin_ratio,
        square=square,
    )
    if x2 <= x1 or y2 <= y1:
        return image.copy()
    return image[y1:y2, x1:x2].copy()
