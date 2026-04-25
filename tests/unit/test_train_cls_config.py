from __future__ import annotations

from src.engine.train_cls import _extra_model_kwargs


def test_extra_model_kwargs_excludes_standard_classifier_fields() -> None:
    assert _extra_model_kwargs(
        {
            "name": "convnext_small",
            "pretrained": True,
            "in_chans": 3,
            "num_classes": 2,
            "drop_path_rate": 0.2,
        }
    ) == {"drop_path_rate": 0.2}
