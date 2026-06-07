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
    def __init__(self, runtime_config: RuntimeConfig, *, paths=None) -> None:
        self.runtime_config = runtime_config
        self.paths = paths
        self._classifier_model = None
        self._classifier_models = None
        self._classifier_members = None
        self._model_device = None

    @property
    def primary_model(self):
        return self._classifier_model

    def resolved_classifier_checkpoint(self) -> str | None:
        checkpoints = self.resolved_classifier_checkpoints()
        return checkpoints[0] if checkpoints else None

    def resolved_classifier_member_configs(self) -> list[dict[str, Any]]:
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
        if self.runtime_config.classifier_checkpoints:
            return list(self.runtime_config.classifier_checkpoints)
        if self.paths is not None and self.paths.default_classifier_ckpt.exists():
            return [str(self.paths.default_classifier_ckpt)]
        return []

    def model_identifier(self) -> str:
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
        return self.runtime_config.member_config_value(member, key, runtime_key, default)

    def preprocess_kwargs(
        self,
        variant: dict[str, Any] | None = None,
        member: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        requested = self.runtime_config.get(
            "classifier_device",
            self.runtime_config.get("device", "cpu"),
        )
        return resolve_torch_device(requested, torch)

    def _ensure_models_on_device(self, device: str) -> None:
        if self._classifier_models is None or self._model_device == device:
            return
        # Keep model migration explicit so repeated inference calls do not hide
        # device movement inside the low-level probability helper.
        for model in self._classifier_models:
            model.to(device)
        self._model_device = device

    def tta_variants(self, member: dict[str, Any] | None = None) -> list[dict[str, Any]]:
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
        member_configs = self.resolved_classifier_member_configs()
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
                model = load_classifier(
                    model_config,
                    checkpoint_path=member["checkpoint"],
                    map_location="cpu",
                )
                model.eval()
                self._classifier_models.append(model)
                self._classifier_members.append({**member, "model_instance": model})
            self._classifier_model = self._classifier_members[0]["model_instance"]

        device = self.device()
        self._ensure_models_on_device(device)
        ensemble_probs = []
        weights = []
        for member in self._classifier_members:
            weight = float(member.get("weight", 1.0))
            if member_weight_overrides:
                weight = float(member_weight_overrides.get(str(member.get("model")), weight))
            if weight <= 0.0:
                continue
            input_tensors = self.input_tensors(image, member)
            if not input_tensors or not hasattr(input_tensors[0], "unsqueeze"):
                raise ClassificationUnavailableError("Torch tensor conversion failed for classifier input.")
            batch = torch.stack(input_tensors)
            model = member["model_instance"]
            tta_probs = classifier_probabilities(
                model,
                batch,
                device=device,
                move_model=False,
            )
            ensemble_probs.append(np.mean(tta_probs, axis=0) * weight)
            weights.append(weight)
        if not weights:
            raise ClassificationUnavailableError("No classifier member has a positive weight.")
        probs = np.sum(np.asarray(ensemble_probs, dtype=np.float32), axis=0) / float(sum(weights))
        return float(probs[0]), float(probs[1])
