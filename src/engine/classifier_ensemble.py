"""Classifier ensemble loading, preprocessing, TTA, and probability averaging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.engine.devices import resolve_torch_device
from src.engine.errors import ClassificationUnavailableError
from src.engine.runtime_config import RuntimeConfig
from src.models.classifier import classifier_probabilities, load_classifier
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
        if members:
            return members
        return [
            {
                "model": str(self.runtime_config.get("classifier_model", "resnet18")),
                "checkpoint": checkpoint,
                "weight": 1.0,
            }
            for checkpoint in self.resolved_classifier_checkpoints()
        ]

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
        return {
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
    ) -> np.ndarray:
        """Run TTA variants for one member and return averaged class probabilities."""
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
            )
            weighted_probabilities.append(member_probabilities * weight)
            weights.append(weight)
        probs = self._weighted_average_probabilities(weighted_probabilities, weights)
        return float(probs[0]), float(probs[1])
