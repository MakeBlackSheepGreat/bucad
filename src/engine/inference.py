from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.datasets.busi import load_busi_manifest
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
from src.preprocess.io import ensure_three_channels, read_image, validate_image_array
from src.preprocess.transforms import prepare_classifier_input
from src.utils.config import load_project_config
from src.utils.metrics import classification_metrics
from src.utils.reporting import write_json_report
from src.utils.results import InferenceResponse, build_diagnostic_result
from src.utils.runtime import optional_import


torch = optional_import("torch")


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
        self._segmenter_model = None

    @classmethod
    def from_config(cls, config_path: str | Path) -> "BreastUltrasoundInferenceService":
        config, paths = load_project_config(config_path)
        return cls(config.get("runtime", {}), paths=paths)

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
                metadata={"model_identifier": self._model_identifier()},
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
        checkpoint = self.runtime_config.get("classifier_checkpoint")
        if checkpoint:
            return str(checkpoint)
        if self.paths is not None and self.paths.default_classifier_ckpt.exists():
            return str(self.paths.default_classifier_ckpt)
        return None

    def _resolved_segmenter_checkpoint(self) -> str | None:
        checkpoint = self.runtime_config.get("segmenter_checkpoint")
        if checkpoint:
            return str(checkpoint)
        if self.paths is not None and self.paths.default_segmenter_ckpt.exists():
            return str(self.paths.default_segmenter_ckpt)
        return None

    def _model_identifier(self) -> str:
        checkpoint = self._resolved_classifier_checkpoint()
        if checkpoint:
            return Path(checkpoint).name
        return str(self.runtime_config.get("classifier_model", "unknown"))

    def _predict_classification(self, image: np.ndarray) -> tuple[float, float]:
        if self.classifier_predictor is not None:
            benign, malignant = self.classifier_predictor(image)
            return float(benign), float(malignant)

        checkpoint = self._resolved_classifier_checkpoint()
        if not checkpoint:
            raise ClassificationUnavailableError("No classifier checkpoint is configured.")
        if torch is None:
            raise ClassificationUnavailableError("Torch is not available in the current environment.")

        if self._classifier_model is None:
            model_config = {
                "name": self.runtime_config.get("classifier_model", "resnet18"),
                "pretrained": bool(self.runtime_config.get("classifier_pretrained", False)),
                "in_chans": 3,
                "num_classes": 2,
            }
            self._classifier_model = load_classifier(
                model_config,
                checkpoint_path=checkpoint,
                map_location="cpu",
            )
            self._classifier_model.eval()

        input_tensor = prepare_classifier_input(
            image, int(self.runtime_config.get("classifier_image_size", 224))
        )
        if not hasattr(input_tensor, "unsqueeze"):
            raise ClassificationUnavailableError("Torch tensor conversion failed for classifier input.")
        probs = classifier_probabilities(self._classifier_model, input_tensor, device="cpu")[0]
        return float(probs[0]), float(probs[1])

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
        checkpoint = self._resolved_segmenter_checkpoint()
        if not checkpoint or not self.runtime_config.get("segmentation_enabled", True):
            raise OptionalOutputUnavailableError("Segmentation weights are not available.")
        if torch is None:
            raise OptionalOutputUnavailableError("Torch is not available for segmentation.")
        if self._segmenter_model is None:
            model_config = {
                "architecture": "unet",
                "encoder_name": "resnet18",
                "encoder_weights": None,
                "in_channels": 3,
                "classes": 1,
            }
            self._segmenter_model = load_segmenter(
                model_config,
                checkpoint_path=checkpoint,
                map_location="cpu",
            )
            self._segmenter_model.eval()
        input_tensor = prepare_classifier_input(
            image, int(self.runtime_config.get("segmenter_image_size", 256))
        )
        if not hasattr(input_tensor, "unsqueeze"):
            raise OptionalOutputUnavailableError("Torch tensor conversion failed for segmentation.")
        with torch.no_grad():
            logits = self._segmenter_model(input_tensor.unsqueeze(0).to(dtype=torch.float32))
            mask = torch.sigmoid(logits)[0, 0].cpu().numpy()
        return mask

    def _predict_explanation(self, image: np.ndarray) -> np.ndarray:
        if not self.runtime_config.get("gradcam_enabled", True):
            raise OptionalOutputUnavailableError("Grad-CAM is disabled in runtime config.")
        if self._classifier_model is None:
            raise OptionalOutputUnavailableError("Classifier model is not loaded for explanation.")
        input_tensor = prepare_classifier_input(
            image, int(self.runtime_config.get("classifier_image_size", 224))
        )
        if not hasattr(input_tensor, "unsqueeze"):
            raise OptionalOutputUnavailableError("Torch tensor conversion failed for explanation.")
        return generate_gradcam_map(self._classifier_model, input_tensor.unsqueeze(0).to(dtype=torch.float32))


def evaluate_busi_dataset(
    config_path: str | Path,
    *,
    classifier_predictor: Callable[[np.ndarray], tuple[float, float]] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    config, paths = load_project_config(config_path)
    service = BreastUltrasoundInferenceService(
        config.get("runtime", {}),
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

    metrics = classification_metrics(y_true, malignant_probabilities)
    report = {
        "sample_count": len(rows),
        "metrics": metrics,
        "rows": rows,
    }
    destination = Path(output_path) if output_path is not None else paths.reports_root / "busi_eval.json"
    write_json_report(destination, report)
    return report
