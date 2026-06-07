from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.engine.descriptors import (
    build_router_feature_map,
    extract_roi_descriptors,
    router_feature_vector,
)
from src.engine.errors import ClassificationUnavailableError
from src.preprocess.roi import crop_to_mask_bbox, expand_bbox, mask_bbox


class RoiEnhancer:
    @staticmethod
    def stacker_features(
        full_probability: float, roi_probability: float, feature_mode: str
    ) -> np.ndarray:
        values = np.asarray([full_probability, roi_probability], dtype=np.float64)
        if feature_mode == "logit":
            clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
            values = np.log(clipped / (1.0 - clipped))
        return values

    def apply_roi_stacker(
        self,
        *,
        full_probability: float,
        roi_probability: float,
        stacker: dict[str, Any],
    ) -> float:
        feature_mode = str(stacker.get("feature_mode", "probability"))
        features = self.stacker_features(full_probability, roi_probability, feature_mode)
        mean = np.asarray(stacker.get("scaler_mean", [0.0, 0.0]), dtype=np.float64)
        scale = np.asarray(stacker.get("scaler_scale", [1.0, 1.0]), dtype=np.float64)
        coef = np.asarray(stacker.get("coef", [1.0, 0.0]), dtype=np.float64)
        if mean.shape != (2,) or scale.shape != (2,) or coef.shape != (2,):
            raise ClassificationUnavailableError("ROI stacker parameters must contain two feature values.")
        scaled = (features - mean) / np.maximum(scale, 1e-6)
        logit = float(np.dot(coef, scaled) + float(stacker.get("intercept", 0.0)))
        probability = 1.0 / (1.0 + float(np.exp(-logit)))
        return float(np.clip(probability, 0.0, 1.0))

    def apply_descriptor_router(
        self,
        *,
        full_probability: float,
        roi_probability: float,
        stacked_probability: float,
        descriptors: dict[str, float],
        router: dict[str, Any],
    ) -> float:
        if not bool(router.get("enabled", False)):
            return float(stacked_probability)
        feature_names = [str(name) for name in router.get("feature_names", [])]
        if not feature_names:
            raise ClassificationUnavailableError("Descriptor router requires feature_names.")
        feature_map = build_router_feature_map(
            full_probability=full_probability,
            roi_probability=roi_probability,
            stacked_probability=stacked_probability,
            descriptors=descriptors,
        )
        try:
            features = router_feature_vector(feature_map, feature_names)
        except KeyError as exc:
            raise ClassificationUnavailableError(str(exc)) from exc
        mean = np.asarray(router.get("scaler_mean", [0.0] * len(feature_names)), dtype=np.float64)
        scale = np.asarray(router.get("scaler_scale", [1.0] * len(feature_names)), dtype=np.float64)
        coef = np.asarray(router.get("coef", []), dtype=np.float64)
        if mean.shape != features.shape or scale.shape != features.shape or coef.shape != features.shape:
            raise ClassificationUnavailableError(
                "Descriptor router parameters must match feature_names length."
            )
        scaled = (features - mean) / np.maximum(scale, 1e-6)
        logit_value = float(np.dot(coef, scaled) + float(router.get("intercept", 0.0)))
        routed_probability = 1.0 / (1.0 + float(np.exp(-logit_value)))
        blend_weight = float(router.get("blend_weight", 1.0))
        if blend_weight < 1.0:
            blend_weight = float(np.clip(blend_weight, 0.0, 1.0))
            routed_probability = (
                blend_weight * routed_probability
                + (1.0 - blend_weight) * float(stacked_probability)
            )
        return float(np.clip(routed_probability, 0.0, 1.0))

    @staticmethod
    def roi_area_ratio(mask: np.ndarray | None, config: dict[str, Any]) -> tuple[float, bool]:
        if mask is None:
            return 1.0, True
        threshold = float(config.get("mask_threshold", 0.5))
        largest_component = bool(config.get("largest_component", False))
        min_mask_area_ratio = float(config.get("min_mask_area_ratio", 0.001))
        bbox = mask_bbox(
            mask,
            threshold=threshold,
            min_area_ratio=min_mask_area_ratio,
            largest_component=largest_component,
        )
        if bbox is None:
            return 1.0, True
        x1, y1, x2, y2 = expand_bbox(
            bbox,
            image_shape=mask.shape,
            margin_ratio=float(config.get("margin_ratio", 0.35)),
            square=True,
        )
        area = max(1, (x2 - x1) * (y2 - y1))
        total = max(1, int(mask.shape[0]) * int(mask.shape[1]))
        return float(area / total), False

    @staticmethod
    def roi_area_gate_config(config: dict[str, Any]) -> dict[str, Any] | None:
        gate = config.get("quality_gate")
        if not isinstance(gate, dict):
            gate = {}
        min_area = gate.get("min_area_ratio", config.get("min_roi_area_ratio"))
        max_area = gate.get("max_area_ratio", config.get("max_roi_area_ratio"))
        if min_area is None and max_area is None:
            return None
        return {
            "enabled": bool(gate.get("enabled", True)),
            "min_area_ratio": None if min_area is None else float(min_area),
            "max_area_ratio": None if max_area is None else float(max_area),
            "fallback_to_full": bool(gate.get("fallback_to_full", True)),
        }

    def should_fallback_by_area(
        self,
        mask: np.ndarray | None,
        config: dict[str, Any],
    ) -> bool:
        gate = self.roi_area_gate_config(config)
        if gate is None or not gate["enabled"] or not gate["fallback_to_full"]:
            return False
        area_ratio, used_fallback = self.roi_area_ratio(mask, config)
        if used_fallback:
            return True
        min_area = gate["min_area_ratio"]
        max_area = gate["max_area_ratio"]
        if min_area is not None and area_ratio < min_area:
            return True
        return bool(max_area is not None and area_ratio > max_area)

    def predict(
        self,
        image: np.ndarray,
        *,
        full_benign_probability: float,
        full_malignant_probability: float,
        config: dict[str, Any],
        classifier_predictor: Callable[..., tuple[float, float]],
        segmenter_predictor: Callable[[np.ndarray], np.ndarray],
    ) -> tuple[float, float]:
        mask = segmenter_predictor(image)
        if self.should_fallback_by_area(mask, config):
            return full_benign_probability, full_malignant_probability
        roi_image = crop_to_mask_bbox(
            image,
            mask,
            threshold=float(config.get("mask_threshold", 0.5)),
            margin_ratio=float(config.get("margin_ratio", 0.35)),
            largest_component=bool(config.get("largest_component", False)),
        )
        raw_weight_overrides = config.get("classifier_weight_overrides", {})
        member_weight_overrides = (
            {str(key): float(value) for key, value in raw_weight_overrides.items()}
            if isinstance(raw_weight_overrides, dict) and raw_weight_overrides
            else None
        )
        _, roi_malignant_probability = classifier_predictor(
            roi_image,
            member_weight_overrides=member_weight_overrides,
        )
        malignant_probability = self.apply_roi_stacker(
            full_probability=full_malignant_probability,
            roi_probability=roi_malignant_probability,
            stacker=config["stacker"],
        )
        blend_weight = float(config.get("roi_stack_blend_weight", 1.0))
        if blend_weight < 1.0:
            blend_weight = float(np.clip(blend_weight, 0.0, 1.0))
            malignant_probability = (
                blend_weight * malignant_probability
                + (1.0 - blend_weight) * full_malignant_probability
            )
        descriptor_router = config.get("descriptor_router")
        if isinstance(descriptor_router, dict) and bool(descriptor_router.get("enabled", False)):
            descriptors = extract_roi_descriptors(
                image,
                mask,
                threshold=float(config.get("mask_threshold", 0.5)),
                margin_ratio=float(config.get("margin_ratio", 0.35)),
                min_area_ratio=float(config.get("min_mask_area_ratio", 0.001)),
                largest_component=bool(config.get("largest_component", False)),
            )
            malignant_probability = self.apply_descriptor_router(
                full_probability=full_malignant_probability,
                roi_probability=roi_malignant_probability,
                stacked_probability=malignant_probability,
                descriptors=descriptors,
                router=descriptor_router,
            )
        return 1.0 - malignant_probability, malignant_probability
