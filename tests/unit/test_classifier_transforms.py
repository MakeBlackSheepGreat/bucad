from __future__ import annotations

import numpy as np

from src.preprocess.transforms import build_classifier_transform
from src.utils.config import load_yaml


def test_augmented_classifier_transform_preserves_tensor_shape() -> None:
    np.random.seed(42)
    image = np.random.randint(0, 255, size=(96, 80), dtype=np.uint8)
    transform = build_classifier_transform(
        image_size=256,
        apply_clahe_enabled=True,
        horizontal_flip=True,
        flip_probability=1.0,
        rotation_degrees=8,
        brightness=0.08,
        contrast=0.12,
        scale_min=0.92,
        scale_max=1.08,
    )

    tensor = transform(image)

    assert tuple(tensor.shape) == (3, 256, 256)
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_optimized_classifier_config_uses_higher_resolution_and_augments() -> None:
    config = load_yaml("configs/classifier/efficientnetv2_s_256_aug.yml")

    assert config["model"]["name"] == "tf_efficientnetv2_s"
    assert config["data"]["image_size"] == 256
    assert config["data"]["augmentation"]["rotation_degrees"] > 0
    assert config["data"]["augmentation"]["brightness"] > 0
    assert config["data"]["augmentation"]["contrast"] > 0
    assert config["output"]["checkpoint_name"].startswith("efficientnetv2_s_256_aug")


def test_resolution_only_classifier_config_keeps_baseline_augmentation() -> None:
    config = load_yaml("configs/classifier/efficientnetv2_s_256.yml")

    assert config["model"]["name"] == "tf_efficientnetv2_s"
    assert config["data"]["image_size"] == 256
    assert config["data"]["augmentation"]["horizontal_flip"] is True
    assert "rotation_degrees" not in config["data"]["augmentation"]
    assert config["output"]["checkpoint_name"].startswith("efficientnetv2_s_256")
