from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.preprocess.io import cv2
from src.preprocess.roi import expand_bbox, mask_bbox


BASE_ROUTER_FEATURES = (
    "full_probability",
    "roi_probability",
    "stacked_probability",
    "full_logit",
    "roi_logit",
    "stacked_logit",
    "full_roi_delta",
    "abs_full_roi_delta",
    "full_uncertainty",
    "roi_uncertainty",
    "stacked_uncertainty",
)

ROI_DESCRIPTOR_FEATURES = (
    "roi_valid",
    "roi_area_ratio",
    "mask_area_ratio",
    "lesion_bbox_area_ratio",
    "lesion_aspect_ratio",
    "mask_extent",
    "mask_compactness",
    "boundary_complexity",
    "component_count",
    "mask_mean_probability",
    "edge_contrast",
    "image_mean",
    "image_std",
    "image_sharpness",
)

DEFAULT_ROUTER_FEATURES = BASE_ROUTER_FEATURES + ROI_DESCRIPTOR_FEATURES


def logit(value: float) -> float:
    clipped = float(np.clip(value, 1e-6, 1.0 - 1e-6))
    return float(math.log(clipped / (1.0 - clipped)))


def _as_unit_float_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 3:
        array = array[..., 0]
    if array.size == 0:
        return array
    if float(np.nanmax(array)) > 1.5:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _resize_mask_to_image(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    mask_array = np.asarray(mask, dtype=np.float32)
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]
    if mask_array.shape[:2] == image.shape[:2]:
        return mask_array
    height, width = image.shape[:2]
    if cv2 is not None:
        return cv2.resize(mask_array, (width, height), interpolation=cv2.INTER_LINEAR)
    y_index = np.linspace(0, mask_array.shape[0] - 1, height).round().astype(int)
    x_index = np.linspace(0, mask_array.shape[1] - 1, width).round().astype(int)
    return mask_array[np.ix_(y_index, x_index)]


def _connected_component_summary(binary: np.ndarray) -> tuple[int, np.ndarray]:
    binary_bool = np.asarray(binary, dtype=bool)
    if not np.any(binary_bool):
        return 0, binary_bool
    if cv2 is None:
        return 1, binary_bool
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_bool.astype(np.uint8),
        8,
    )
    foreground_count = max(0, int(component_count) - 1)
    if foreground_count == 0:
        return 0, binary_bool
    largest_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return foreground_count, labels == largest_index


def _boundary_stats(binary: np.ndarray, image: np.ndarray) -> tuple[float, float, float]:
    binary_uint8 = np.asarray(binary, dtype=np.uint8)
    area = float(binary_uint8.sum())
    if area <= 0:
        return 0.0, 0.0, 0.0

    if cv2 is not None:
        contours, _ = cv2.findContours(
            binary_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded = cv2.erode(binary_uint8, kernel, iterations=1).astype(bool)
        dilated = cv2.dilate(binary_uint8, kernel, iterations=1).astype(bool)
    else:
        shifted = [
            np.roll(binary_uint8, shift=1, axis=0),
            np.roll(binary_uint8, shift=-1, axis=0),
            np.roll(binary_uint8, shift=1, axis=1),
            np.roll(binary_uint8, shift=-1, axis=1),
        ]
        boundary = np.logical_or.reduce([binary_uint8 != item for item in shifted])
        perimeter = float(boundary.sum())
        eroded = binary_uint8.astype(bool) & ~boundary
        dilated = np.logical_or.reduce([binary_uint8.astype(bool), *[item.astype(bool) for item in shifted]])

    if perimeter <= 0.0:
        compactness = 1.0
        complexity = 1.0
    else:
        compactness = float(np.clip(4.0 * math.pi * area / (perimeter * perimeter), 0.0, 1.0))
        complexity = float(min(20.0, 1.0 / max(compactness, 1e-6)))

    inside_edge = binary_uint8.astype(bool) & ~eroded
    outside_edge = dilated & ~binary_uint8.astype(bool)
    if np.any(inside_edge) and np.any(outside_edge):
        edge_contrast = float(abs(image[inside_edge].mean() - image[outside_edge].mean()))
    else:
        edge_contrast = 0.0
    return compactness, complexity, edge_contrast


def _image_sharpness(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    if cv2 is not None:
        laplacian = cv2.Laplacian(image.astype(np.float32), cv2.CV_32F)
        return float(np.var(laplacian))
    gy, gx = np.gradient(image.astype(np.float64))
    return float(np.var(gx) + np.var(gy))


def extract_roi_descriptors(
    image: np.ndarray,
    mask: np.ndarray | None,
    *,
    threshold: float = 0.5,
    margin_ratio: float = 0.35,
    min_area_ratio: float = 0.001,
    largest_component: bool = False,
) -> dict[str, float]:
    unit_image = _as_unit_float_image(image)
    descriptors: dict[str, float] = {
        "roi_valid": 0.0,
        "roi_area_ratio": 1.0,
        "mask_area_ratio": 0.0,
        "lesion_bbox_area_ratio": 1.0,
        "lesion_aspect_ratio": 1.0,
        "mask_extent": 0.0,
        "mask_compactness": 0.0,
        "boundary_complexity": 0.0,
        "component_count": 0.0,
        "mask_mean_probability": 0.0,
        "edge_contrast": 0.0,
        "image_mean": float(unit_image.mean()) if unit_image.size else 0.0,
        "image_std": float(unit_image.std()) if unit_image.size else 0.0,
        "image_sharpness": _image_sharpness(unit_image),
    }
    if mask is None or unit_image.size == 0:
        return descriptors

    resized_mask = np.clip(_resize_mask_to_image(mask, unit_image), 0.0, 1.0)
    binary_initial = resized_mask > float(threshold)
    component_count, largest_binary = _connected_component_summary(binary_initial)
    descriptors["component_count"] = float(component_count)
    binary = largest_binary if largest_component else binary_initial
    if not np.any(binary):
        return descriptors

    bbox = mask_bbox(
        binary.astype(np.uint8),
        threshold=0.5,
        min_area_ratio=float(min_area_ratio),
        largest_component=False,
    )
    if bbox is None:
        return descriptors

    x1, y1, x2, y2 = bbox
    height, width = binary.shape[:2]
    total_area = float(max(1, height * width))
    lesion_width = max(1, x2 - x1)
    lesion_height = max(1, y2 - y1)
    lesion_bbox_area = float(lesion_width * lesion_height)
    expanded = expand_bbox(
        bbox,
        image_shape=binary.shape,
        margin_ratio=float(margin_ratio),
        square=True,
    )
    ex1, ey1, ex2, ey2 = expanded
    roi_area = float(max(1, ex2 - ex1) * max(1, ey2 - ey1))
    mask_area = float(binary.sum())
    compactness, complexity, edge_contrast = _boundary_stats(binary, unit_image)

    descriptors.update(
        {
            "roi_valid": 1.0,
            "roi_area_ratio": float(roi_area / total_area),
            "mask_area_ratio": float(mask_area / total_area),
            "lesion_bbox_area_ratio": float(lesion_bbox_area / total_area),
            "lesion_aspect_ratio": float(lesion_width / lesion_height),
            "mask_extent": float(mask_area / max(1.0, lesion_bbox_area)),
            "mask_compactness": compactness,
            "boundary_complexity": complexity,
            "mask_mean_probability": float(resized_mask[binary].mean()),
            "edge_contrast": edge_contrast,
        }
    )
    return descriptors


def build_router_feature_map(
    *,
    full_probability: float,
    roi_probability: float,
    stacked_probability: float,
    descriptors: dict[str, float] | None = None,
) -> dict[str, float]:
    full = float(full_probability)
    roi = float(roi_probability)
    stacked = float(stacked_probability)
    values: dict[str, float] = {
        "full_probability": full,
        "roi_probability": roi,
        "stacked_probability": stacked,
        "full_logit": logit(full),
        "roi_logit": logit(roi),
        "stacked_logit": logit(stacked),
        "full_roi_delta": float(roi - full),
        "abs_full_roi_delta": float(abs(roi - full)),
        "full_uncertainty": float(1.0 - 2.0 * abs(full - 0.5)),
        "roi_uncertainty": float(1.0 - 2.0 * abs(roi - 0.5)),
        "stacked_uncertainty": float(1.0 - 2.0 * abs(stacked - 0.5)),
    }
    if descriptors:
        values.update({str(key): float(value) for key, value in descriptors.items()})
    return values


def router_feature_vector(
    feature_map: dict[str, float],
    feature_names: list[str] | tuple[str, ...],
) -> np.ndarray:
    missing = [name for name in feature_names if name not in feature_map]
    if missing:
        raise KeyError(f"Router feature(s) missing: {', '.join(missing)}")
    return np.asarray([feature_map[name] for name in feature_names], dtype=np.float64)
