"""Canonical DARA paper names and AERIS/QDHE compatibility metadata."""

from __future__ import annotations

from typing import Any


DARA_MODEL_NAME = "DARA"
DARA_T_NAME = "DARA-ConvNeXt-Tiny"
AERIS_MODEL_NAME = "AERIS-Net"  # Historical compatibility name.
AERIS_T_NAME = "AERIS-T"  # Historical compatibility name.

AERIS_VARIANTS: dict[str, dict[str, Any]] = {
    "AERIS-Base": {
        "role": "inter-stage selection without routed experts",
        "classifier_alias": "aeris_base_t",
        "segmenter_alias": "aeris_base",
    },
    "AERIS-Cls": {
        "role": "scene classifier with shared and reliability-gated routed experts",
        "classifier_alias": "aeris_cls_t",
        "segmenter_alias": None,
    },
    "AERIS-Seg": {
        "role": "boundary-aware semantic segmenter with spatial experts",
        "classifier_alias": None,
        "segmenter_alias": "aeris_seg_t",
    },
    "AERIS-Full": {
        "role": "inter-stage selection, delta-history monitor, and reliability-gated experts",
        "classifier_alias": "aeris_t",
        "segmenter_alias": "aeris_t",
    },
}

HISTORICAL_MODEL_ALIASES = {
    "aeris": "DARA-Full historical compatibility name",
    "aeris_t": "DARA-Full historical compatibility name",
    "aeris_full": "DARA-Full historical compatibility name",
    "aeris_full_t": "DARA-Full historical compatibility name",
    "aeris_cls": "DARA-Cls historical compatibility name",
    "aeris_cls_t": "DARA-Cls historical compatibility name",
    "aeris_seg": "DARA-Seg historical compatibility name",
    "aeris_seg_t": "DARA-Seg historical compatibility name",
    "qdhe": "DARA-Full delta-history enhancement branch",
    "qdhe_moe": "DARA-Full delta-history segmentation branch",
    "qdhe_moe_tiny": "DARA-Full delta-history classification branch",
}


def aeris_identity(model_config: dict[str, Any]) -> dict[str, str | None]:
    """Return canonical DARA identity while preserving every configured alias."""
    configured = str(
        model_config.get("name", model_config.get("architecture", "unknown"))
    ).lower()
    legacy = configured if configured in HISTORICAL_MODEL_ALIASES else None
    if configured in {"aeris_base", "aeris_base_t", "dara_base", "dara_base_t"}:
        variant = "DARA-Base"
    elif configured in {"aeris_cls", "aeris_cls_t", "dara_cls", "dara_cls_t"}:
        variant = "DARA-Cls"
    elif configured in {"aeris_seg", "aeris_seg_t", "dara_seg", "dara_seg_t"}:
        variant = "DARA-Seg"
    elif configured.startswith("qdhe"):
        variant = "DARA-Full (QDHE enhancement)"
    elif configured.startswith(("aeris", "dara")):
        variant = "DARA-Full"
    else:
        variant = configured
    is_dara_family = (
        configured.startswith("aeris")
        or configured.startswith("dara")
        or configured.startswith("qdhe")
    )
    return {
        "network": DARA_MODEL_NAME if is_dara_family else configured,
        "model": DARA_T_NAME if is_dara_family else configured,
        "variant": variant,
        "configured_alias": configured,
        "legacy_alias": legacy,
    }
