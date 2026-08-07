"""Optional segmentation overlays and Grad-CAM evidence generation for inference."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.engine.classifier_ensemble import ClassifierEnsemble
from src.engine.devices import resolve_torch_device
from src.engine.errors import OptionalOutputUnavailableError
from src.engine.runtime_config import RuntimeConfig
from src.explain.gradcam import generate_gradcam_map
from src.explain.overlay import render_heatmap_overlay, render_mask_overlay
from src.models.segmenter import load_segmenter
from src.preprocess.transforms import prepare_classifier_input
from src.utils.results import InferenceResponse
from src.utils.runtime import optional_import


torch = optional_import("torch")


class VisualEvidenceService:
    """Attach optional mask overlays and Grad-CAM maps to diagnosis responses."""

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        classifier_ensemble: ClassifierEnsemble,
        *,
        paths=None,
    ) -> None:
        """Initialize lazy segmenter caches and the classifier dependency."""
        self.runtime_config = runtime_config
        self.classifier_ensemble = classifier_ensemble
        self.paths = paths
        self._segmenter_model = None
        self._segmenter_models = None
        self._segmenter_device = None

    @property
    def segmenter_model(self):
        """Return the primary loaded segmenter model, if any."""
        return self._segmenter_model

    @property
    def segmenter_models(self):
        """Return all loaded segmenter models, if they have been loaded."""
        return self._segmenter_models

    def resolved_segmenter_checkpoints(self) -> list[str]:
        """Return configured segmenter checkpoints, falling back to the project default."""
        if self.runtime_config.segmenter_checkpoints:
            return list(self.runtime_config.segmenter_checkpoints)
        if self.paths is not None and self.paths.default_segmenter_ckpt.exists():
            return [str(self.paths.default_segmenter_ckpt)]
        return []

    def resolved_segmenter_checkpoint(self) -> str | None:
        """Return the first resolved segmenter checkpoint, if any."""
        checkpoints = self.resolved_segmenter_checkpoints()
        return checkpoints[0] if checkpoints else None

    def segmenter_device(self) -> str:
        """Resolve the segmenter device from runtime config with CUDA fallback."""
        requested = self.runtime_config.get(
            "segmenter_device",
            self.runtime_config.get("device", "cpu"),
        )
        return resolve_torch_device(requested, torch)

    def _ensure_segmenters_on_device(self, device: str) -> None:
        """Move cached segmenter models to the selected device only when needed."""
        if self._segmenter_models is None or self._segmenter_device == device:
            return
        # Segmenter visuals are optional but expensive; cache the chosen device
        # so repeated UI requests do not migrate models on every image.
        for model in self._segmenter_models:
            model.to(device)
        self._segmenter_device = device

    @staticmethod
    def _default_segmenter_model_config() -> dict[str, Any]:
        """Return the default segmenter architecture used for runtime checkpoints."""
        return {
            "architecture": "unet",
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "in_channels": 3,
            "classes": 1,
        }

    def _ensure_segmenters_loaded(self, checkpoints: list[str]) -> None:
        """Lazy-load all configured segmenter checkpoints on CPU."""
        if self._segmenter_models is not None:
            return
        self._segmenter_models = []
        for checkpoint in checkpoints:
            model = load_segmenter(
                self._default_segmenter_model_config(),
                checkpoint_path=checkpoint,
                map_location="cpu",
            )
            model.eval()
            self._segmenter_models.append(model)
        self._segmenter_model = self._segmenter_models[0] if self._segmenter_models else None

    def _segmenter_input_batch(self, image: np.ndarray, *, device: str):
        """Convert a single image into a one-item segmenter input batch."""
        input_tensor = prepare_classifier_input(
            image, int(self.runtime_config.get("segmenter_image_size", 256))
        )
        if not hasattr(input_tensor, "unsqueeze"):
            raise OptionalOutputUnavailableError("Torch tensor conversion failed for segmentation.")
        return input_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)

    @staticmethod
    def _average_masks(masks: list[np.ndarray]) -> np.ndarray:
        """Average a list of predicted probability masks."""
        if not masks:
            raise OptionalOutputUnavailableError("Segmentation weights are not available.")
        return np.mean(np.asarray(masks, dtype=np.float32), axis=0)

    def attach_optional_visuals(
        self,
        response: InferenceResponse,
        image: np.ndarray,
        *,
        need_segmentation: bool,
        need_explanation: bool,
        segmenter_predictor: Callable[[np.ndarray], np.ndarray] | None = None,
        explanation_generator: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Populate visual fields and downgrade the response to partial on optional failures."""
        partial = False
        if need_segmentation:
            try:
                mask = (
                    segmenter_predictor(image)
                    if segmenter_predictor is not None
                    else self.predict_segmentation(image)
                )
                response.lesion_overlay_view = render_mask_overlay(image, mask)
            except OptionalOutputUnavailableError as exc:
                partial = True
                response.lesion_visualization_missing_reason = str(exc)
                response.warnings.append(str(exc))
        if need_explanation:
            try:
                heatmap = (
                    explanation_generator(image)
                    if explanation_generator is not None
                    else self.predict_explanation(image)
                )
                response.explanation_view = render_heatmap_overlay(image, heatmap)
            except OptionalOutputUnavailableError as exc:
                partial = True
                response.explanation_missing_reason = str(exc)
                response.warnings.append(str(exc))
        if partial:
            response.status = "partial"

    def predict_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Run the configured segmenter ensemble and average probability masks."""
        checkpoints = self.resolved_segmenter_checkpoints()
        if not checkpoints or not self.runtime_config.get("segmentation_enabled", True):
            raise OptionalOutputUnavailableError("Segmentation weights are not available.")
        if torch is None:
            raise OptionalOutputUnavailableError("Torch is not available for segmentation.")
        self._ensure_segmenters_loaded(checkpoints)
        device = self.segmenter_device()
        self._ensure_segmenters_on_device(device)
        batch = self._segmenter_input_batch(image, device=device)
        masks = []
        inference_context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with inference_context():
            for model in self._segmenter_models:
                outputs = model(batch)
                logits = outputs["mask"] if isinstance(outputs, dict) else outputs
                masks.append(torch.sigmoid(logits)[0, 0].cpu().numpy())
        return self._average_masks(masks)

    def predict_explanation(self, image: np.ndarray) -> np.ndarray:
        """Generate Grad-CAM from the already-loaded primary classifier."""
        if not self.runtime_config.get("gradcam_enabled", True):
            raise OptionalOutputUnavailableError("Grad-CAM is disabled in runtime config.")
        classifier_model = self.classifier_ensemble.primary_model
        if classifier_model is None:
            raise OptionalOutputUnavailableError("Classifier model is not loaded for explanation.")
        input_tensor = prepare_classifier_input(
            image,
            int(self.runtime_config.get("classifier_image_size", 224)),
            **self.classifier_ensemble.preprocess_kwargs(),
        )
        if not hasattr(input_tensor, "unsqueeze"):
            raise OptionalOutputUnavailableError("Torch tensor conversion failed for explanation.")
        device = self.classifier_ensemble.device()
        return generate_gradcam_map(
            classifier_model.to(device),
            input_tensor.unsqueeze(0).to(device=device, dtype=torch.float32),
        )
