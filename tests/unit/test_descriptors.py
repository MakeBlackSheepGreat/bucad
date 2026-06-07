"""Unit tests for descriptors."""

from __future__ import annotations

import numpy as np

from src.engine.descriptors import (
    build_router_feature_map,
    extract_roi_descriptors,
    router_feature_vector,
)


def test_extract_roi_descriptors_reports_shape_and_quality_features() -> None:
    """Verify extract roi descriptors reports shape and quality features."""
    image = np.zeros((20, 30), dtype=np.uint8)
    image[6:14, 10:20] = 180
    mask = np.zeros((20, 30), dtype=np.float32)
    mask[6:14, 10:20] = 0.9

    descriptors = extract_roi_descriptors(
        image,
        mask,
        threshold=0.5,
        margin_ratio=0.0,
    )

    assert descriptors["roi_valid"] == 1.0
    assert 0.12 < descriptors["mask_area_ratio"] < 0.14
    assert 0.16 < descriptors["roi_area_ratio"] < 0.17
    assert descriptors["mask_extent"] == 1.0
    assert descriptors["mask_mean_probability"] > 0.8
    assert descriptors["image_std"] > 0.0
    assert descriptors["edge_contrast"] > 0.0


def test_router_feature_vector_combines_probabilities_and_descriptors() -> None:
    """Verify router feature vector combines probabilities and descriptors."""
    feature_map = build_router_feature_map(
        full_probability=0.2,
        roi_probability=0.8,
        stacked_probability=0.7,
        descriptors={"roi_area_ratio": 0.3, "image_std": 0.2},
    )

    vector = router_feature_vector(
        feature_map,
        ["full_probability", "roi_probability", "full_roi_delta", "roi_area_ratio", "image_std"],
    )

    assert np.allclose(vector, [0.2, 0.8, 0.6, 0.3, 0.2])
