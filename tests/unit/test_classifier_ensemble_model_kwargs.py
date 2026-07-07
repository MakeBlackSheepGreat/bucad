"""Unit tests for classifier ensemble model kwargs propagation."""

from __future__ import annotations

import numpy as np

from src.engine.classifier_ensemble import ClassifierEnsemble
from src.engine.runtime_config import RuntimeConfig
from src.utils.runtime import optional_import


torch = optional_import("torch")


def test_classifier_ensemble_includes_top_level_model_kwargs_in_fallback_members() -> None:
    """Verify fallback member configs preserve top-level model kwargs."""
    ensemble = ClassifierEnsemble(
        RuntimeConfig.from_mapping(
            {
                "classifier_model": "convnext_tiny",
                "classifier_checkpoints": ["./artifacts/checkpoints/demo.pt"],
                "classifier_model_kwargs": {"proj_dim": 192},
            }
        )
    )

    members = ensemble.resolved_classifier_member_configs()

    assert members == [
        {
            "model": "convnext_tiny",
            "checkpoint": "./artifacts/checkpoints/demo.pt",
            "weight": 1.0,
            "model_kwargs": {"proj_dim": 192},
        }
    ]


def test_classifier_ensemble_member_model_kwargs_override_top_level_defaults() -> None:
    """Verify member-level model kwargs override top-level model kwargs."""
    ensemble = ClassifierEnsemble(
        RuntimeConfig.from_mapping(
            {
                "classifier_model": "convnext_tiny",
                "classifier_model_kwargs": {"proj_dim": 192, "attn_heads": 4},
                "classifier_members": [
                    {
                        "model": "convnext_tiny",
                        "checkpoint": "./artifacts/checkpoints/demo.pt",
                        "model_kwargs": {"proj_dim": 256},
                    }
                ],
            }
        )
    )

    model_config = ensemble._model_config_for_member(ensemble.resolved_classifier_member_configs()[0])

    assert model_config["model_kwargs"] == {"proj_dim": 256, "attn_heads": 4}


def test_classifier_ensemble_predicts_dualview_member_with_mask(monkeypatch) -> None:
    """Verify dual-view members receive full image, ROI image, and descriptors."""
    if torch is None:
        return
    ensemble = ClassifierEnsemble(
        RuntimeConfig.from_mapping(
            {
                "classifier_model": "roi_dualview_convnext_tiny",
                "classifier_checkpoints": ["./artifacts/checkpoints/demo.pt"],
                "classifier_input_mode": "dual_view_roi",
                "classifier_image_size": 8,
                "classifier_roi": {
                    "mask_threshold": 0.4,
                    "margin_ratio": 0.1,
                    "min_area_ratio": 0.01,
                    "max_area_ratio": 0.75,
                    "largest_component": True,
                },
            }
        )
    )

    class _Model(torch.nn.Module):
        """Small dual-view classifier used by ensemble tests."""

        input_mode = "dual_view_roi"

        def forward(self, image_full, image_roi, roi_descriptor):
            """Return fixed logits after validating input tensors."""
            assert tuple(image_full.shape) == (1, 3, 8, 8)
            assert tuple(image_roi.shape) == (1, 3, 8, 8)
            assert tuple(roi_descriptor.shape) == (1, 14)
            return torch.tensor([[0.0, 2.0]], dtype=torch.float32, device=image_full.device)

    monkeypatch.setattr(
        ensemble,
        "_ensure_classifier_members_loaded",
        lambda _configs: None,
    )
    ensemble._classifier_members = [
        {
            "model": "roi_dualview_convnext_tiny",
            "checkpoint": "./artifacts/checkpoints/demo.pt",
            "weight": 1.0,
            "model_instance": _Model(),
        }
    ]
    ensemble._classifier_models = [ensemble._classifier_members[0]["model_instance"]]
    image = np.zeros((16, 16), dtype=np.uint8)
    image[5:11, 5:11] = 200
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[5:11, 5:11] = 1

    benign, malignant = ensemble.predict_on_image(image, mask=mask)

    assert malignant > benign


def test_dualview_tta_variants_are_batched_for_one_forward() -> None:
    """Verify dual-view TTA uses one batched forward while preserving averaged probabilities."""
    if torch is None:
        return
    ensemble = ClassifierEnsemble(
        RuntimeConfig.from_mapping(
            {
                "classifier_input_mode": "dual_view_roi",
                "classifier_image_size": 8,
                "classifier_tta_variants": ["original", "hflip"],
                "classifier_roi": {
                    "mask_threshold": 0.4,
                    "margin_ratio": 0.1,
                    "min_area_ratio": 0.01,
                    "max_area_ratio": 0.75,
                    "largest_component": True,
                },
            }
        )
    )

    class _Model(torch.nn.Module):
        """Small dual-view model that records batched TTA calls."""

        def __init__(self) -> None:
            """Initialize call counter."""
            super().__init__()
            self.calls = 0

        def forward(self, image_full, image_roi, roi_descriptor):
            """Return logits from deterministic full/ROI/descriptor scores."""
            self.calls += 1
            assert tuple(image_full.shape) == (2, 3, 8, 8)
            assert tuple(image_roi.shape) == (2, 3, 8, 8)
            assert tuple(roi_descriptor.shape) == (2, 14)
            score = image_full.mean((1, 2, 3)) + 0.5 * image_roi.mean((1, 2, 3))
            score = score + roi_descriptor[:, 0] * 0.01
            return torch.stack([-score, score], dim=1)

    model = _Model()
    member = {"model": "dummy_dualview", "model_instance": model, "input_mode": "dual_view_roi"}
    image = np.zeros((16, 16), dtype=np.uint8)
    image[:, 8:] = 128
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 6:14] = 1

    probabilities = ensemble._predict_dual_view_member_probabilities(
        image,
        mask,
        member,
        device="cpu",
    )

    assert model.calls == 1
    assert probabilities.shape == (2,)
    assert np.isclose(float(probabilities.sum()), 1.0, atol=1e-6)

