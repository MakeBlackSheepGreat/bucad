from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.datasets.busi import load_busi_manifest
from src.engine.descriptors import (
    build_router_feature_map,
    extract_roi_descriptors,
    router_feature_vector,
)
from src.engine.errors import (
    BucadError,
    ClassificationUnavailableError,
    InvalidInputError,
    OptionalOutputUnavailableError,
    QualityBlockedError,
    UnexpectedRuntimeError,
)
from src.models.classifier import classifier_probabilities, load_classifier
from src.models.segmenter import load_segmenter
from src.explain.gradcam import generate_gradcam_map
from src.explain.overlay import render_heatmap_overlay, render_mask_overlay
from src.preprocess.io import cv2, ensure_three_channels, read_image, validate_image_array
from src.preprocess.roi import crop_to_mask_bbox, expand_bbox, mask_bbox
from src.preprocess.transforms import prepare_classifier_input
from src.utils.config import load_project_config
from src.utils.metrics import best_threshold_by_youden, classification_metrics, threshold_sweep
from src.utils.paths import resolve_path
from src.utils.reporting import write_json_report, write_markdown_report
from src.utils.results import InferenceResponse, build_diagnostic_result
from src.utils.runtime import optional_import


torch = optional_import("torch")


def _resolve_runtime_checkpoint_paths(runtime_config: dict[str, Any], *, project_root: Path) -> dict[str, Any]:
    resolved = dict(runtime_config)

    def resolve_checkpoint(value: Any) -> Any:
        if not value:
            return value
        return str(resolve_path(str(value), base_dir=project_root))

    if "classifier_checkpoint" in resolved:
        resolved["classifier_checkpoint"] = resolve_checkpoint(resolved.get("classifier_checkpoint"))
    if isinstance(resolved.get("classifier_checkpoints"), list):
        resolved["classifier_checkpoints"] = [
            resolve_checkpoint(checkpoint)
            for checkpoint in resolved["classifier_checkpoints"]
            if checkpoint
        ]
    if isinstance(resolved.get("classifier_members"), list):
        members = []
        for member in resolved["classifier_members"]:
            if not isinstance(member, dict):
                continue
            member_config = dict(member)
            if "checkpoint" in member_config:
                member_config["checkpoint"] = resolve_checkpoint(member_config.get("checkpoint"))
            members.append(member_config)
        resolved["classifier_members"] = members
    if "segmenter_checkpoint" in resolved:
        resolved["segmenter_checkpoint"] = resolve_checkpoint(resolved.get("segmenter_checkpoint"))
    if isinstance(resolved.get("segmenter_checkpoints"), list):
        resolved["segmenter_checkpoints"] = [
            resolve_checkpoint(checkpoint)
            for checkpoint in resolved["segmenter_checkpoints"]
            if checkpoint
        ]
    return resolved


def assess_image_quality(image: np.ndarray) -> str:
    if image.ndim < 2 or min(image.shape[:2]) < 32:
        return "invalid"
    if float(np.std(image)) < 3.0:
        return "low_quality"
    return "valid"


class BreastUltrasoundInferenceService:
    def __init__(
        self,
        runtime_config: dict[str, Any],
        *,
        paths=None,
        classifier_predictor: Callable[[np.ndarray], tuple[float, float]] | None = None,
        segmenter_predictor: Callable[[np.ndarray], np.ndarray] | None = None,
        explanation_generator: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.runtime_config = runtime_config
        self.paths = paths
        self.classifier_predictor = classifier_predictor
        self.segmenter_predictor = segmenter_predictor
        self.explanation_generator = explanation_generator
        self._classifier_model = None
        self._classifier_models = None
        self._classifier_members = None
        self._segmenter_model = None
        self._segmenter_models = None

    @classmethod
    def from_config(cls, config_path: str | Path) -> "BreastUltrasoundInferenceService":
        config, paths = load_project_config(config_path)
        runtime_config = dict(config.get("runtime", {}))
        runtime_config = _resolve_runtime_checkpoint_paths(runtime_config, project_root=paths.project_root)
        runtime_config.setdefault("device", config.get("device", "cpu"))
        return cls(runtime_config, paths=paths)

    def diagnose(
        self,
        image_input: str | Path | np.ndarray,
        *,
        input_filename: str | None = None,
        decision_threshold: float | None = None,
        need_segmentation: bool = True,
        need_explanation: bool = True,
    ) -> InferenceResponse:
        filename = input_filename or (
            Path(image_input).name if isinstance(image_input, (str, Path)) else "uploaded.png"
        )
        try:
            image = read_image(image_input, grayscale=True)
            validate_image_array(image)
            quality = assess_image_quality(image)
            if quality == "invalid":
                raise InvalidInputError("Uploaded image is invalid or too small.")
            if quality == "low_quality":
                raise QualityBlockedError("Image quality is too poor for reliable analysis.")

            benign_probability, malignant_probability = self._predict_classification(image)
            threshold = float(
                decision_threshold
                if decision_threshold is not None
                else self.runtime_config.get("default_threshold", 0.5)
            )
            result = build_diagnostic_result(
                benign_probability,
                malignant_probability,
                threshold=threshold,
                borderline_margin=float(self.runtime_config.get("borderline_margin", 0.08)),
                model_version=self._model_identifier(),
            )
            response = InferenceResponse(
                status="completed",
                input_filename=filename,
                result=result,
                original_image_view=ensure_three_channels(image),
                metadata={
                    "model_identifier": self._model_identifier(),
                    "decision_threshold": threshold,
                    "primary_model": self.runtime_config.get(
                        "primary_classifier_model",
                        "unknown",
                    ),
                    "ensemble_display_name": self.runtime_config.get("ensemble_display_name"),
                },
            )
            self._attach_optional_visuals(
                response,
                image,
                need_segmentation=need_segmentation,
                need_explanation=need_explanation,
            )
            return response
        except InvalidInputError as exc:
            return InferenceResponse(status="invalid_input", input_filename=filename, result=None, warnings=[str(exc)])
        except QualityBlockedError as exc:
            return InferenceResponse(status="quality_blocked", input_filename=filename, result=None, warnings=[str(exc)])
        except ClassificationUnavailableError as exc:
            return InferenceResponse(status="classification_unavailable", input_filename=filename, result=None, warnings=[str(exc)])
        except BucadError as exc:
            return InferenceResponse(status="failed", input_filename=filename, result=None, warnings=[str(exc)])
        except Exception as exc:  # pragma: no cover - defensive branch
            wrapped = UnexpectedRuntimeError(str(exc))
            return InferenceResponse(status="unexpected_runtime_error", input_filename=filename, result=None, warnings=[str(wrapped)])

    def _resolved_classifier_checkpoint(self) -> str | None:
        checkpoints = self._resolved_classifier_checkpoints()
        return checkpoints[0] if checkpoints else None

    def _resolved_classifier_member_configs(self) -> list[dict[str, Any]]:
        members = self.runtime_config.get("classifier_members")
        if isinstance(members, list) and members:
            resolved = []
            for member in members:
                if not isinstance(member, dict) or not member.get("checkpoint"):
                    continue
                resolved_member = dict(member)
                resolved_member["model"] = str(
                    member.get(
                        "model",
                        self.runtime_config.get("classifier_model", "resnet18"),
                    )
                )
                resolved_member["checkpoint"] = str(member["checkpoint"])
                resolved_member["weight"] = float(member.get("weight", 1.0))
                resolved.append(resolved_member)
            return resolved
        return [
            {
                "model": str(self.runtime_config.get("classifier_model", "resnet18")),
                "checkpoint": checkpoint,
                "weight": 1.0,
            }
            for checkpoint in self._resolved_classifier_checkpoints()
        ]

    def _resolved_classifier_checkpoints(self) -> list[str]:
        checkpoint_list = self.runtime_config.get("classifier_checkpoints")
        if isinstance(checkpoint_list, list) and checkpoint_list:
            return [str(checkpoint) for checkpoint in checkpoint_list if checkpoint]
        checkpoint = self.runtime_config.get("classifier_checkpoint")
        if checkpoint:
            return [str(checkpoint)]
        if self.paths is not None and self.paths.default_classifier_ckpt.exists():
            return [str(self.paths.default_classifier_ckpt)]
        return []

    def _resolved_segmenter_checkpoints(self) -> list[str]:
        checkpoint_list = self.runtime_config.get("segmenter_checkpoints")
        if isinstance(checkpoint_list, list) and checkpoint_list:
            return [str(checkpoint) for checkpoint in checkpoint_list if checkpoint]
        checkpoint = self.runtime_config.get("segmenter_checkpoint")
        if checkpoint:
            return [str(checkpoint)]
        if self.paths is not None and self.paths.default_segmenter_ckpt.exists():
            return [str(self.paths.default_segmenter_ckpt)]
        return []

    def _resolved_segmenter_checkpoint(self) -> str | None:
        checkpoints = self._resolved_segmenter_checkpoints()
        return checkpoints[0] if checkpoints else None

    def _model_identifier(self) -> str:
        members = self._resolved_classifier_member_configs()
        if len(members) > 1:
            model_names = sorted({member["model"] for member in members})
            return "mixed-ensemble:" + "+".join(model_names)
        if members:
            return Path(members[0]["checkpoint"]).name
        return str(self.runtime_config.get("classifier_model", "unknown"))

    def _member_config_value(
        self,
        member: dict[str, Any] | None,
        key: str,
        runtime_key: str,
        default: Any,
    ) -> Any:
        if member is not None:
            if key in member:
                return member[key]
            if runtime_key in member:
                return member[runtime_key]
        return self.runtime_config.get(runtime_key, default)

    def _classifier_preprocess_kwargs(
        self,
        variant: dict[str, Any] | None = None,
        member: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        variant = variant or {}
        crop_pct = self._member_config_value(member, "crop_pct", "classifier_crop_pct", 1.0)
        return {
            "apply_clahe_enabled": bool(
                self._member_config_value(
                    member,
                    "apply_clahe",
                    "classifier_apply_clahe",
                    False,
                )
            ),
            "mean": self._member_config_value(member, "mean", "classifier_mean", None),
            "std": self._member_config_value(member, "std", "classifier_std", None),
            "interpolation": str(
                self._member_config_value(member, "interpolation", "classifier_interpolation", "area")
            ),
            "crop_pct": float(variant.get("crop_pct", crop_pct)),
        }

    def _classifier_device(self) -> str:
        requested = str(
            self.runtime_config.get(
                "classifier_device",
                self.runtime_config.get("device", "cpu"),
            )
        ).lower()
        if requested == "auto":
            return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        return requested

    def _classifier_tta_variants(self, member: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        configured = self._member_config_value(
            member,
            "tta_variants",
            "classifier_tta_variants",
            None,
        )
        if isinstance(configured, list) and configured:
            variants = []
            for item in configured:
                if isinstance(item, str):
                    variants.append({"name": item})
                elif isinstance(item, dict):
                    variants.append(dict(item))
            if variants:
                return variants
        variants = [{"name": "identity"}]
        if bool(
            self._member_config_value(
                member,
                "tta_horizontal_flip",
                "classifier_tta_horizontal_flip",
                False,
            )
        ):
            variants.append({"name": "hflip"})
        return variants

    def _apply_classifier_tta_variant(self, image: np.ndarray, variant: dict[str, Any]) -> np.ndarray:
        name = str(variant.get("name", "identity")).lower()
        if name in {"identity", "none", "original"}:
            return image
        if name in {"hflip", "horizontal_flip"}:
            return np.fliplr(image).copy()
        if name in {"rotate", "rotation"}:
            degrees = float(variant.get("degrees", 0.0))
        elif name.startswith("rotate_"):
            suffix = name.replace("rotate_", "", 1).replace("p", "+").replace("m", "-")
            degrees = float(suffix)
        else:
            return image
        if abs(degrees) < 1e-6 or cv2 is None:
            return image
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _classifier_input_tensors(
        self,
        image: np.ndarray,
        member: dict[str, Any] | None = None,
    ) -> list[Any]:
        tensors = []
        image_size = int(
            self._member_config_value(member, "image_size", "classifier_image_size", 224)
        )
        for variant in self._classifier_tta_variants(member):
            variant_image = self._apply_classifier_tta_variant(image, variant)
            tensor = prepare_classifier_input(
                variant_image,
                image_size,
                **self._classifier_preprocess_kwargs(variant, member),
            )
            tensors.append(tensor)
        return tensors

    def _predict_classifier_ensemble_on_image(
        self,
        image: np.ndarray,
        *,
        member_weight_overrides: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        member_configs = self._resolved_classifier_member_configs()
        if not member_configs:
            raise ClassificationUnavailableError("No classifier checkpoint is configured.")
        if torch is None:
            raise ClassificationUnavailableError("Torch is not available in the current environment.")

        if self._classifier_members is None:
            self._classifier_members = []
            self._classifier_models = []
            for member in member_configs:
                model_config = {
                    "name": member["model"],
                    "pretrained": bool(
                        self._member_config_value(
                            member,
                            "pretrained",
                            "classifier_pretrained",
                            False,
                        )
                    ),
                    "in_chans": 3,
                    "num_classes": 2,
                }
                model = load_classifier(
                    model_config,
                    checkpoint_path=member["checkpoint"],
                    map_location="cpu",
                )
                model.eval()
                self._classifier_models.append(model)
                self._classifier_members.append({**member, "model_instance": model})
            self._classifier_model = self._classifier_members[0]["model_instance"]

        device = self._classifier_device()
        ensemble_probs = []
        weights = []
        for member in self._classifier_members:
            weight = float(member.get("weight", 1.0))
            if member_weight_overrides:
                weight = float(member_weight_overrides.get(str(member.get("model")), weight))
            if weight <= 0.0:
                continue
            input_tensors = self._classifier_input_tensors(image, member)
            if not input_tensors or not hasattr(input_tensors[0], "unsqueeze"):
                raise ClassificationUnavailableError("Torch tensor conversion failed for classifier input.")
            batch = torch.stack(input_tensors)
            model = member["model_instance"]
            tta_probs = classifier_probabilities(model, batch, device=device)
            ensemble_probs.append(np.mean(tta_probs, axis=0) * weight)
            weights.append(weight)
        if not weights:
            raise ClassificationUnavailableError("No classifier member has a positive weight.")
        probs = np.sum(np.asarray(ensemble_probs, dtype=np.float32), axis=0) / float(sum(weights))
        return float(probs[0]), float(probs[1])

    def _roi_enhancement_config(self) -> dict[str, Any] | None:
        config = self.runtime_config.get("roi_enhancement")
        if not isinstance(config, dict) or not bool(config.get("enabled", False)):
            return None
        stacker = config.get("stacker")
        if not isinstance(stacker, dict):
            return None
        return config

    @staticmethod
    def _stacker_features(full_probability: float, roi_probability: float, feature_mode: str) -> np.ndarray:
        values = np.asarray([full_probability, roi_probability], dtype=np.float64)
        if feature_mode == "logit":
            clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
            values = np.log(clipped / (1.0 - clipped))
        return values

    def _apply_roi_stacker(
        self,
        *,
        full_probability: float,
        roi_probability: float,
        stacker: dict[str, Any],
    ) -> float:
        feature_mode = str(stacker.get("feature_mode", "probability"))
        features = self._stacker_features(full_probability, roi_probability, feature_mode)
        mean = np.asarray(stacker.get("scaler_mean", [0.0, 0.0]), dtype=np.float64)
        scale = np.asarray(stacker.get("scaler_scale", [1.0, 1.0]), dtype=np.float64)
        coef = np.asarray(stacker.get("coef", [1.0, 0.0]), dtype=np.float64)
        if mean.shape != (2,) or scale.shape != (2,) or coef.shape != (2,):
            raise ClassificationUnavailableError("ROI stacker parameters must contain two feature values.")
        scaled = (features - mean) / np.maximum(scale, 1e-6)
        logit = float(np.dot(coef, scaled) + float(stacker.get("intercept", 0.0)))
        probability = 1.0 / (1.0 + float(np.exp(-logit)))
        return float(np.clip(probability, 0.0, 1.0))

    def _apply_descriptor_router(
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
    def _roi_area_ratio(mask: np.ndarray | None, config: dict[str, Any]) -> tuple[float, bool]:
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
    def _roi_area_gate_config(config: dict[str, Any]) -> dict[str, Any] | None:
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

    def _should_fallback_roi_by_area(
        self,
        mask: np.ndarray | None,
        config: dict[str, Any],
    ) -> bool:
        gate = self._roi_area_gate_config(config)
        if gate is None or not gate["enabled"] or not gate["fallback_to_full"]:
            return False
        area_ratio, used_fallback = self._roi_area_ratio(mask, config)
        if used_fallback:
            return True
        min_area = gate["min_area_ratio"]
        max_area = gate["max_area_ratio"]
        if min_area is not None and area_ratio < min_area:
            return True
        return bool(max_area is not None and area_ratio > max_area)

    def _predict_roi_enhanced_classification(
        self,
        image: np.ndarray,
        *,
        full_benign_probability: float,
        full_malignant_probability: float,
        config: dict[str, Any],
    ) -> tuple[float, float]:
        mask = (
            self.segmenter_predictor(image)
            if self.segmenter_predictor is not None
            else self._predict_segmentation(image)
        )
        if self._should_fallback_roi_by_area(mask, config):
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
        _, roi_malignant_probability = self._predict_classifier_ensemble_on_image(
            roi_image,
            member_weight_overrides=member_weight_overrides,
        )
        malignant_probability = self._apply_roi_stacker(
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
            malignant_probability = self._apply_descriptor_router(
                full_probability=full_malignant_probability,
                roi_probability=roi_malignant_probability,
                stacked_probability=malignant_probability,
                descriptors=descriptors,
                router=descriptor_router,
            )
        return 1.0 - malignant_probability, malignant_probability

    def _predict_classification(self, image: np.ndarray) -> tuple[float, float]:
        if self.classifier_predictor is not None:
            benign, malignant = self.classifier_predictor(image)
            return float(benign), float(malignant)

        full_benign, full_malignant = self._predict_classifier_ensemble_on_image(image)
        roi_config = self._roi_enhancement_config()
        if roi_config is None:
            return full_benign, full_malignant
        try:
            return self._predict_roi_enhanced_classification(
                image,
                full_benign_probability=full_benign,
                full_malignant_probability=full_malignant,
                config=roi_config,
            )
        except BucadError:
            if bool(roi_config.get("fallback_to_full", True)):
                return full_benign, full_malignant
            raise

    def _attach_optional_visuals(
        self,
        response: InferenceResponse,
        image: np.ndarray,
        *,
        need_segmentation: bool,
        need_explanation: bool,
    ) -> None:
        partial = False
        if need_segmentation:
            try:
                mask = (
                    self.segmenter_predictor(image)
                    if self.segmenter_predictor is not None
                    else self._predict_segmentation(image)
                )
                response.lesion_overlay_view = render_mask_overlay(image, mask)
            except OptionalOutputUnavailableError as exc:
                partial = True
                response.lesion_visualization_missing_reason = str(exc)
                response.warnings.append(str(exc))
        if need_explanation:
            try:
                heatmap = (
                    self.explanation_generator(image)
                    if self.explanation_generator is not None
                    else self._predict_explanation(image)
                )
                response.explanation_view = render_heatmap_overlay(image, heatmap)
            except OptionalOutputUnavailableError as exc:
                partial = True
                response.explanation_missing_reason = str(exc)
                response.warnings.append(str(exc))
        if partial:
            response.status = "partial"

    def _predict_segmentation(self, image: np.ndarray) -> np.ndarray:
        checkpoints = self._resolved_segmenter_checkpoints()
        if not checkpoints or not self.runtime_config.get("segmentation_enabled", True):
            raise OptionalOutputUnavailableError("Segmentation weights are not available.")
        if torch is None:
            raise OptionalOutputUnavailableError("Torch is not available for segmentation.")
        if self._segmenter_models is None:
            model_config = {
                "architecture": "unet",
                "encoder_name": "resnet18",
                "encoder_weights": None,
                "in_channels": 3,
                "classes": 1,
            }
            self._segmenter_models = []
            for checkpoint in checkpoints:
                model = load_segmenter(
                    model_config,
                    checkpoint_path=checkpoint,
                    map_location="cpu",
                )
                model.eval()
                self._segmenter_models.append(model)
            self._segmenter_model = self._segmenter_models[0] if self._segmenter_models else None
        input_tensor = prepare_classifier_input(
            image, int(self.runtime_config.get("segmenter_image_size", 256))
        )
        if not hasattr(input_tensor, "unsqueeze"):
            raise OptionalOutputUnavailableError("Torch tensor conversion failed for segmentation.")
        masks = []
        with torch.no_grad():
            batch = input_tensor.unsqueeze(0).to(dtype=torch.float32)
            for model in self._segmenter_models:
                logits = model(batch)
                masks.append(torch.sigmoid(logits)[0, 0].cpu().numpy())
        if not masks:
            raise OptionalOutputUnavailableError("Segmentation weights are not available.")
        return np.mean(np.asarray(masks, dtype=np.float32), axis=0)

    def _predict_explanation(self, image: np.ndarray) -> np.ndarray:
        if not self.runtime_config.get("gradcam_enabled", True):
            raise OptionalOutputUnavailableError("Grad-CAM is disabled in runtime config.")
        if self._classifier_model is None:
            raise OptionalOutputUnavailableError("Classifier model is not loaded for explanation.")
        input_tensor = prepare_classifier_input(
            image,
            int(self.runtime_config.get("classifier_image_size", 224)),
            **self._classifier_preprocess_kwargs(),
        )
        if not hasattr(input_tensor, "unsqueeze"):
            raise OptionalOutputUnavailableError("Torch tensor conversion failed for explanation.")
        device = self._classifier_device()
        return generate_gradcam_map(
            self._classifier_model.to(device),
            input_tensor.unsqueeze(0).to(device=device, dtype=torch.float32),
        )


def evaluate_busi_dataset(
    config_path: str | Path,
    *,
    classifier_predictor: Callable[[np.ndarray], tuple[float, float]] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    config, paths = load_project_config(config_path)
    runtime_config = dict(config.get("runtime", {}))
    runtime_config.setdefault("device", config.get("device", "cpu"))
    service = BreastUltrasoundInferenceService(
        runtime_config,
        paths=paths,
        classifier_predictor=classifier_predictor,
    )
    manifest = load_busi_manifest(paths.busi_root, include_normal=False)
    malignant_probabilities: list[float] = []
    y_true: list[int] = []
    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        response = service.diagnose(
            row.image_path,
            input_filename=Path(row.image_path).name,
            need_segmentation=False,
            need_explanation=False,
        )
        if response.result is None:
            continue
        malignant_probabilities.append(response.result.malignant_probability)
        y_true.append(1 if row.pathology_label == "malignant" else 0)
        rows.append(
            {
                "sample_id": row.sample_id,
                "pathology_label": row.pathology_label,
                "malignant_probability": response.result.malignant_probability,
                "final_label": response.result.final_label,
            }
        )

    default_threshold = float(service.runtime_config.get("default_threshold", 0.5))
    metrics = classification_metrics(
        y_true,
        malignant_probabilities,
        threshold=default_threshold,
    )
    threshold_rows = threshold_sweep(y_true, malignant_probabilities)
    best_threshold = best_threshold_by_youden(y_true, malignant_probabilities)
    report = {
        "config_path": str(config_path),
        "model_identifier": service._model_identifier(),
        "runtime_summary": {
            "classifier_image_size": int(service.runtime_config.get("classifier_image_size", 224)),
            "classifier_crop_pct": float(service.runtime_config.get("classifier_crop_pct", 1.0)),
            "classifier_tta_variants": service._classifier_tta_variants(),
            "classifier_member_count": len(service._resolved_classifier_member_configs()),
        },
        "sample_count": len(rows),
        "metrics": metrics,
        "threshold_analysis": {
            "best_by_youden": best_threshold,
            "rows": threshold_rows,
        },
        "rows": rows,
    }
    destination = Path(output_path) if output_path is not None else paths.reports_root / "busi_eval.json"
    write_json_report(destination, report)
    threshold_markdown = _threshold_analysis_markdown(metrics, best_threshold, threshold_rows)
    write_markdown_report(destination.parent / "threshold_analysis.md", threshold_markdown)
    if output_path is not None:
        write_markdown_report(
            destination.with_name(f"threshold_analysis_{destination.stem}.md"),
            threshold_markdown,
        )
    return report


def _threshold_analysis_markdown(
    default_metrics: dict[str, Any],
    best_threshold: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "# Threshold Analysis",
        "",
        "This report is generated from BUSI external evaluation outputs.",
        "",
        "## Default Threshold",
        "",
        f"- Threshold: `{default_metrics.get('threshold', 0.5):.2f}`",
        f"- AUC: `{default_metrics.get('auc')}`",
        f"- Accuracy: `{default_metrics.get('accuracy', 0.0):.4f}`",
        f"- Recall/Sensitivity: `{default_metrics.get('sensitivity', 0.0):.4f}`",
        f"- Precision: `{default_metrics.get('precision', 0.0):.4f}`",
        f"- Specificity: `{default_metrics.get('specificity', 0.0):.4f}`",
        f"- F1-Score: `{default_metrics.get('f1_score', 0.0):.4f}`",
        "",
        "## Best Threshold By Youden J",
        "",
    ]
    if best_threshold:
        lines.extend(
            [
                f"- Threshold: `{best_threshold.get('threshold', 0.5):.2f}`",
                f"- Youden J: `{best_threshold.get('youden_j', 0.0):.4f}`",
                f"- Accuracy: `{best_threshold.get('accuracy', 0.0):.4f}`",
                f"- Recall/Sensitivity: `{best_threshold.get('sensitivity', 0.0):.4f}`",
                f"- Precision: `{best_threshold.get('precision', 0.0):.4f}`",
                f"- Specificity: `{best_threshold.get('specificity', 0.0):.4f}`",
                f"- F1-Score: `{best_threshold.get('f1_score', 0.0):.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Sweep",
            "",
            "| Threshold | Recall/Sensitivity | Precision | Specificity | Accuracy | F1-Score | Youden J |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['threshold']:.2f} | {row['sensitivity']:.4f} | "
            f"{row.get('precision', 0.0):.4f} | {row['specificity']:.4f} | "
            f"{row['accuracy']:.4f} | {row.get('f1_score', 0.0):.4f} | {row['youden_j']:.4f} |"
        )
    return lines
