from __future__ import annotations

import numpy as np

from src.preprocess.roi import crop_to_mask_bbox, expand_bbox, mask_bbox


def test_mask_bbox_returns_foreground_bounds() -> None:
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:5, 3:8] = 1

    assert mask_bbox(mask) == (3, 2, 8, 5)


def test_expand_bbox_clamps_to_image() -> None:
    assert expand_bbox((1, 1, 4, 3), image_shape=(5, 5), margin_ratio=1.0) == (0, 0, 5, 5)


def test_crop_to_mask_bbox_falls_back_to_full_image_for_empty_mask() -> None:
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)
    mask = np.zeros((5, 5), dtype=np.uint8)

    cropped = crop_to_mask_bbox(image, mask)

    assert np.array_equal(cropped, image)


def test_crop_to_mask_bbox_returns_context_crop() -> None:
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 4:6] = 1

    cropped = crop_to_mask_bbox(image, mask, margin_ratio=0.5, square=True)

    assert cropped.shape == (4, 4)
    assert cropped[0, 0] == image[3, 3]


def test_crop_to_mask_bbox_can_use_largest_component() -> None:
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[6:9, 6:9] = 1

    cropped = crop_to_mask_bbox(
        image,
        mask,
        margin_ratio=0.0,
        square=False,
        largest_component=True,
    )

    assert cropped.shape == (3, 3)
    assert cropped[0, 0] == image[6, 6]
