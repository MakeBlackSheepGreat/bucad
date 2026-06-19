"""Unit tests for classifier ensemble model kwargs propagation."""

from __future__ import annotations

from src.engine.classifier_ensemble import ClassifierEnsemble
from src.engine.runtime_config import RuntimeConfig


def test_classifier_ensemble_includes_top_level_model_kwargs_in_fallback_members() -> None:
    """Verify fallback member configs preserve top-level model kwargs."""
    ensemble = ClassifierEnsemble(
        RuntimeConfig.from_mapping(
            {
                "classifier_model": "sonoglore_convnext_tiny",
                "classifier_checkpoints": ["./artifacts/checkpoints/demo.pt"],
                "classifier_model_kwargs": {"proj_dim": 192},
            }
        )
    )

    members = ensemble.resolved_classifier_member_configs()

    assert members == [
        {
            "model": "sonoglore_convnext_tiny",
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
                "classifier_model": "sonoglore_convnext_tiny",
                "classifier_model_kwargs": {"proj_dim": 192, "attn_heads": 4},
                "classifier_members": [
                    {
                        "model": "sonoglore_convnext_tiny",
                        "checkpoint": "./artifacts/checkpoints/demo.pt",
                        "model_kwargs": {"proj_dim": 256},
                    }
                ],
            }
        )
    )

    model_config = ensemble._model_config_for_member(ensemble.resolved_classifier_member_configs()[0])

    assert model_config["model_kwargs"] == {"proj_dim": 256, "attn_heads": 4}
