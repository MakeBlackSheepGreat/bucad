"""Classifier ensemble loading, preprocessing, TTA, and probability averaging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.engine.descriptors import ROI_DESCRIPTOR_FEATURES, extract_roi_descriptors
from src.engine.devices import resolve_torch_device
from src.engine.errors import ClassificationUnavailableError
from src.engine.runtime_config import RuntimeConfig
from src.models.classifier import classifier_probabilities, load_classifier
from src.preprocess.roi import crop_to_mask_bbox
from src.preprocess.io import cv2
from src.preprocess.transforms import prepare_classifier_input
from src.utils.runtime import optional_import


torch = optional_import("torch")


class ClassifierEnsemble:
    """Load configured classifier members, apply TTA, and average probabilities."""

    def __init__(self, runtime_config: RuntimeConfig, *, paths=None) -> None:
        """Initialize lazy model caches from parsed runtime config and project paths."""
        self.runtime_config = runtime_config
        self.paths = paths
        self._classifier_model = None
        self._classifier_models = None
        self._classifier_members = None
        self._model_device = None

    @property
    def primary_model(self):
        """Return the first loaded classifier model, or None when unloaded."""
        return self._classifier_model

    def resolved_classifier_checkpoint(self) -> str | None:
        """Return the first resolved checkpoint path, if any."""
        checkpoints = self.resolved_classifier_checkpoints()
        return checkpoints[0] if checkpoints else None

    def resolved_classifier_member_configs(self) -> list[dict[str, Any]]:
        """Return the configured ensemble member dicts, falling back to single-checkpoint mode."""
        members = self.runtime_config.classifier_member_dicts()
        if not members:
            members = [
                {
                    "model": str(self.runtime_config.get("classifier_model", "resnet18")),
                    "checkpoint": checkpoint,
                    "weight": 1.0,
                }
                for checkpoint in self.resolved_classifier_checkpoints()
            ]
        top_level_model_kwargs = self.runtime_config.get("classifier_model_kwargs", None)
        resolved_members = []
        for member in members:
            resolved_member = dict(member)
            model_kwargs: dict[str, Any] = {}
            if isinstance(top_level_model_kwargs, dict):
                model_kwargs.update(top_level_model_kwargs)
            member_model_kwargs = resolved_member.get("model_kwargs")
            if isinstance(member_model_kwargs, dict):
                model_kwargs.update(member_model_kwargs)
            if model_kwargs:
                resolved_member["model_kwargs"] = model_kwargs
            resolved_members.append(resolved_member)
        return resolved_members

    def resolved_classifier_checkpoints(self) -> list[str]:
        """Return the resolved checkpoint path list from config or project defaults."""
        if self.runtime_config.classifier_checkpoints:
            return list(self.runtime_config.classifier_checkpoints)
        if self.paths is not None and self.paths.default_classifier_ckpt.exists():
            return [str(self.paths.default_classifier_ckpt)]
        return []

    def model_identifier(self) -> str:
        """Return a short human-readable name for the current classifier ensemble."""
        members = self.resolved_classifier_member_configs()
        if len(members) > 1:
            model_names = sorted({member["model"] for member in members})
            return "mixed-ensemble:" + "+".join(model_names)
        if members:
            return Path(members[0]["checkpoint"]).name
        return str(self.runtime_config.get("classifier_model", "unknown"))

    def member_config_value(
        self,
        member: dict[str, Any] | None,
        key: str,
        runtime_key: str,
        default: Any,
    ) -> Any:
        """Look up a member-level override, then fall back to the runtime config."""
        return self.runtime_config.member_config_value(member, key, runtime_key, default)

    def preprocess_kwargs(
        self,
        variant: dict[str, Any] | None = None,
        member: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return classifier preprocessing kwargs for the given variant and member."""
        variant = variant or {}
        crop_pct = self.member_config_value(member, "crop_pct", "classifier_crop_pct", 1.0)
        return {
            "apply_clahe_enabled": bool(
                self.member_config_value(
                    member,
                    "apply_clahe",
                    "classifier_apply_clahe",
                    False,
                )
            ),
            "mean": self.member_config_value(member, "mean", "classifier_mean", None),
            "std": self.member_config_value(member, "std", "classifier_std", None),
            "interpolation": str(
                self.member_config_value(member, "interpolation", "classifier_interpolation", "area")
            ),
            "crop_pct": float(variant.get("crop_pct", crop_pct)),
        }

    def device(self) -> str:
        """Resolve the classifier device string, preferring CUDA when available."""
        requested = self.runtime_config.get(
            "classifier_device",
            self.runtime_config.get("device", "cpu"),
        )
        return resolve_torch_device(requested, torch)

    def _ensure_models_on_device(self, device: str) -> None:
        """Move cached models to *device* only when it differs from the current cache."""
        if self._classifier_models is None or self._model_device == device:
            return
        # Keep model migration explicit so repeated inference calls do not hide
        # device movement inside the low-level probability helper.
        for model in self._classifier_models:
            model.to(device)
        self._model_device = device

    def _model_config_for_member(self, member: dict[str, Any]) -> dict[str, Any]:
        """Build the model constructor dict for a single ensemble member."""
        model_config = {
            "name": member["model"],
            "pretrained": bool(
                self.member_config_value(
                    member,
                    "pretrained",
                    "classifier_pretrained",
                    False,
                )
            ),
            "in_chans": 3,
            "num_classes": 2,
        }
        top_level_model_kwargs = self.runtime_config.get("classifier_model_kwargs", None)
        model_kwargs: dict[str, Any] = {}
        if isinstance(top_level_model_kwargs, dict):
            model_kwargs.update(top_level_model_kwargs)
        member_model_kwargs = member.get("model_kwargs")
        if isinstance(member_model_kwargs, dict):
            model_kwargs.update(member_model_kwargs)
        if model_kwargs:
            model_config["model_kwargs"] = model_kwargs
        return model_config

    def _ensure_classifier_members_loaded(self, member_configs: list[dict[str, Any]]) -> None:
        """Lazy-load all member checkpoints once, caching models and metadata."""
        if self._classifier_members is not None:
            return
        # Classifier checkpoints are loaded lazily because Gradio can build the
        # app before a user submits an image.
        self._classifier_members = []
        self._classifier_models = []
        for member in member_configs:
            model = load_classifier(
                self._model_config_for_member(member),
                checkpoint_path=member["checkpoint"],
                map_location="cpu",
            )
            model.eval()
            self._classifier_models.append(model)
            self._classifier_members.append({**member, "model_instance": model})
        self._classifier_model = self._classifier_members[0]["model_instance"]

    @staticmethod
    def _member_weight(
        member: dict[str, Any],
        member_weight_overrides: dict[str, float] | None,
    ) -> float:
        """Return the ensemble weight for one member, honoring optional overrides."""
        weight = float(member.get("weight", 1.0))
        if member_weight_overrides:
            return float(member_weight_overrides.get(str(member.get("model")), weight))
        return weight

    def _predict_member_probabilities(
        self,
        image: np.ndarray,
        member: dict[str, Any],
        *,
        device: str,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run TTA variants for one member and return averaged class probabilities."""
        if self._member_input_mode(member) == "dual_view_roi":
            return self._predict_dual_view_member_probabilities(
                image,
                mask,
                member,
                device=device,
            )
        input_tensors = self.input_tensors(image, member)
        if not input_tensors or not hasattr(input_tensors[0], "unsqueeze"):
            raise ClassificationUnavailableError("Torch tensor conversion failed for classifier input.")
        batch = torch.stack(input_tensors)
        tta_probs = classifier_probabilities(
            member["model_instance"],
            batch,
            device=device,
            move_model=False,
        )
        return np.mean(tta_probs, axis=0)

    def _member_input_mode(self, member: dict[str, Any]) -> str:
        """Return the configured or model-declared classifier input mode."""
        configured = self.member_config_value(member, "input_mode", "classifier_input_mode", None)
        if configured is not None:
            return str(configured).lower()
        model = member.get("model_instance")
        return str(getattr(model, "input_mode", "single_image")).lower()

    def requires_roi_mask(self) -> bool:
        """Return True when any configured classifier member consumes ROI inputs."""
        for member in self.resolved_classifier_member_configs():
            configured = self.member_config_value(member, "input_mode", "classifier_input_mode", None)
            if configured is not None and str(configured).lower() == "dual_view_roi":
                return True
            if str(member.get("model", "")).lower() in {"roi_dualview_convnext_tiny", "dualview_convnext_tiny"}:
                return True
        if self._classifier_members is not None:
            for member in self._classifier_members:
                if self._member_input_mode(member) == "dual_view_roi":
                    return True
        return False

    def _roi_config_for_member(self, member: dict[str, Any]) -> dict[str, Any]:
        """Resolve dual-view ROI preprocessing settings for one member."""
        config = self.member_config_value(member, "roi", "classifier_roi", {}) or {}
        return dict(config) if isinstance(config, dict) else {}

    def _dual_view_roi_image_and_descriptors(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        member: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build ROI image and descriptor vector for a dual-view classifier member."""
        roi_cfg = self._roi_config_for_member(member)
        mask_threshold = float(roi_cfg.get("mask_threshold", 0.4))
        margin_ratio = float(roi_cfg.get("margin_ratio", 0.35))
        min_area_ratio = float(roi_cfg.get("min_area_ratio", 0.08))
        max_area_ratio = float(roi_cfg.get("max_area_ratio", 0.75))
        largest_component = bool(roi_cfg.get("largest_component", True))
        descriptors = extract_roi_descriptors(
            image,
            mask,
            threshold=mask_threshold,
            margin_ratio=margin_ratio,
            min_area_ratio=min_area_ratio,
            largest_component=largest_component,
        )
        roi_valid = bool(descriptors.get("roi_valid", 0.0) >= 0.5)
        roi_area = float(descriptors.get("roi_area_ratio", 1.0))
        if roi_valid and min_area_ratio <= roi_area <= max_area_ratio:
            roi_image = crop_to_mask_bbox(
                image,
                mask,
                threshold=mask_threshold,
                margin_ratio=margin_ratio,
                min_area_ratio=min_area_ratio,
                largest_component=largest_component,
            )
        else:
            roi_image = image.copy()
            descriptors = dict(descriptors)
            descriptors["roi_valid"] = 0.0
            descriptors["roi_area_ratio"] = 1.0
        descriptor_vector = np.asarray(
            [float(descriptors.get(name, 0.0)) for name in ROI_DESCRIPTOR_FEATURES],
            dtype=np.float32,
        )
        return roi_image, descriptor_vector

    def _predict_dual_view_member_probabilities(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        member: dict[str, Any],
        *,
        device: str,
    ) -> np.ndarray:
        """Run TTA variants for one dual-view member and return averaged probabilities."""
        roi_image, descriptor_vector = self._dual_view_roi_image_and_descriptors(image, mask, member)
        image_size = int(
            self.member_config_value(member, "image_size", "classifier_image_size", 224)
        )
        full_tensors = []
        roi_tensors = []
        for variant in self.tta_variants(member):
            full_variant = self.apply_tta_variant(image, variant)
            roi_variant = self.apply_tta_variant(roi_image, variant)
            preprocess_kwargs = self.preprocess_kwargs(variant, member)
            full_tensor = prepare_classifier_input(
                full_variant,
                image_size,
                **preprocess_kwargs,
            )
            roi_tensor = prepare_classifier_input(
                roi_variant,
                image_size,
                **preprocess_kwargs,
            )
            full_tensors.append(full_tensor)
            roi_tensors.append(roi_tensor)
        if not full_tensors or not hasattr(full_tensors[0], "unsqueeze"):
            raise ClassificationUnavailableError("Torch tensor conversion failed for dual-view classifier input.")
        descriptor_tensor = torch.from_numpy(descriptor_vector).repeat(len(full_tensors), 1)
        batch = {
            "image_full": torch.stack(full_tensors).to(device=device, dtype=torch.float32),
            "image_roi": torch.stack(roi_tensors).to(device=device, dtype=torch.float32),
            "roi_descriptor": descriptor_tensor.to(device=device, dtype=torch.float32),
        }
        inference_context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with inference_context():
            logits = member["model_instance"](**batch)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        return np.mean(np.asarray(probabilities, dtype=np.float32), axis=0)

    @staticmethod
    def _weighted_average_probabilities(
        weighted_probabilities: list[np.ndarray],
        weights: list[float],
    ) -> np.ndarray:
        """Average weighted probability arrays and normalize by the total weight."""
        if not weights:
            raise ClassificationUnavailableError("No classifier member has a positive weight.")
        return np.sum(np.asarray(weighted_probabilities, dtype=np.float32), axis=0) / float(sum(weights))

    def tta_variants(self, member: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Resolve member-specific TTA variants with legacy horizontal-flip fallback."""
        configured = self.member_config_value(
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
            self.member_config_value(
                member,
                "tta_horizontal_flip",
                "classifier_tta_horizontal_flip",
                False,
            )
        ):
            variants.append({"name": "hflip"})
        return variants

    def apply_tta_variant(self, image: np.ndarray, variant: dict[str, Any]) -> np.ndarray:
        """Apply a single TTA variant such as horizontal flip or rotation."""
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

    def input_tensors(
        self,
        image: np.ndarray,
        member: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Build one input tensor per TTA variant for the given member."""
        tensors = []
        image_size = int(
            self.member_config_value(member, "image_size", "classifier_image_size", 224)
        )
        for variant in self.tta_variants(member):
            variant_image = self.apply_tta_variant(image, variant)
            tensor = prepare_classifier_input(
                variant_image,
                image_size,
                **self.preprocess_kwargs(variant, member),
            )
            tensors.append(tensor)
        return tensors

    def predict_on_image(
        self,
        image: np.ndarray,
        *,
        member_weight_overrides: dict[str, float] | None = None,
        mask: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Return ensemble benign/malignant probabilities for one image."""
        member_configs = self.resolved_classifier_member_configs()
        if not member_configs:
            raise ClassificationUnavailableError("No classifier checkpoint is configured.")
        if torch is None:
            raise ClassificationUnavailableError("Torch is not available in the current environment.")

        self._ensure_classifier_members_loaded(member_configs)
        device = self.device()
        self._ensure_models_on_device(device)
        weighted_probabilities = []
        weights = []
        for member in self._classifier_members:
            weight = self._member_weight(member, member_weight_overrides)
            if weight <= 0.0:
                continue
            member_probabilities = self._predict_member_probabilities(
                image,
                member,
                device=device,
                mask=mask,
            )
            weighted_probabilities.append(member_probabilities * weight)
            weights.append(weight)
        probs = self._weighted_average_probabilities(weighted_probabilities, weights)
        return float(probs[0]), float(probs[1])
