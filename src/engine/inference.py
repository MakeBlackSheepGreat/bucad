"""Public inference service facade for single-image diagnosis and BUSI evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.datasets.busi import load_busi_manifest
from src.datasets.external_bus import DATASET_REGISTRY, load_external_manifest
from src.engine.classifier_ensemble import ClassifierEnsemble
from src.engine.devices import resolve_torch_device
from src.engine.errors import (
    BucadError,
    ClassificationUnavailableError,
    InvalidInputError,
    QualityBlockedError,
    UnexpectedRuntimeError,
)
from src.engine.roi_enhancer import RoiEnhancer
from src.engine.runtime_config import (
    RuntimeConfig,
    resolve_runtime_checkpoint_paths,
)
from src.engine.visual_evidence import VisualEvidenceService
from src.models.segmenter import load_segmenter
from src.preprocess.io import ensure_three_channels, read_image, read_mask, validate_image_array
from src.preprocess.transforms import prepare_classifier_input
from src.utils.config import load_project_config
from src.utils.metrics import (
    best_threshold_by_youden,
    classification_metrics,
    dice_score,
    iou_score,
    threshold_sweep,
)
from src.utils.reporting import write_json_report, write_markdown_report
from src.utils.results import InferenceResponse, build_diagnostic_result
from src.utils.runtime import optional_import


torch = optional_import("torch")


def _resolve_runtime_checkpoint_paths(
    runtime_config: dict[str, Any], *, project_root: Path
) -> dict[str, Any]:
    """Resolve runtime checkpoint paths relative to the project root."""
    return resolve_runtime_checkpoint_paths(runtime_config, project_root=project_root)


def _inference_service_from_project_config(
    config_path: str | Path,
    *,
    classifier_predictor: Callable[[np.ndarray], tuple[float, float]] | None = None,
) -> tuple["BreastUltrasoundInferenceService", Any]:
    """Create an inference service and resolved project paths from a config file."""
    config, paths = load_project_config(config_path)
    runtime_config = dict(config.get("runtime", {}))
    runtime_config = _resolve_runtime_checkpoint_paths(runtime_config, project_root=paths.project_root)
    runtime_config.setdefault("device", config.get("device", "cpu"))
    service = BreastUltrasoundInferenceService(
        runtime_config,
        paths=paths,
        classifier_predictor=classifier_predictor,
    )
    return service, paths


def assess_image_quality(image: np.ndarray) -> str:
    """Return a coarse quality gate result before expensive model inference."""
    if image.ndim < 2 or min(image.shape[:2]) < 32:
        return "invalid"
    if float(np.std(image)) < 3.0:
        return "low_quality"
    return "valid"


class BreastUltrasoundInferenceService:
    """High-level diagnosis service kept stable for CLI, tests, and Gradio."""

    def __init__(
        self,
        runtime_config: dict[str, Any],
        *,
        paths=None,
        classifier_predictor: Callable[[np.ndarray], tuple[float, float]] | None = None,
        segmenter_predictor: Callable[[np.ndarray], np.ndarray] | None = None,
        explanation_generator: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Initialize inference collaborators and optional injected predictors."""
        self.runtime = RuntimeConfig.from_mapping(runtime_config)
        self.runtime_config = self.runtime.raw
        self.paths = paths
        self.classifier_predictor = classifier_predictor
        self.segmenter_predictor = segmenter_predictor
        self.explanation_generator = explanation_generator
        self.classifier_ensemble = ClassifierEnsemble(self.runtime, paths=paths)
        self.roi_enhancer = RoiEnhancer()
        self._last_roi_fallback_reason: str | None = None
        self.visual_evidence = VisualEvidenceService(
            self.runtime,
            self.classifier_ensemble,
            paths=paths,
        )
        self._lesionext_model = None
        self._lesionext_device = None
        self._last_lesionext_mask: np.ndarray | None = None
        self._last_lesionext_router_weight: float | None = None

    def _uses_lesionext(self) -> bool:
        """Return whether the runtime selects the unified LesioNeXt path."""
        return str(self.runtime_config.get("model_family", "")).lower() in {"lesionext", "lesionext_bus"}

    def _lesionext_checkpoint(self) -> str | None:
        """Return the unified-model checkpoint configured for LesioNeXt inference."""
        checkpoint = self.runtime_config.get("lesionext_checkpoint")
        return str(checkpoint) if checkpoint else None

    def _ensure_lesionext_loaded(self):
        """Load the unified LesioNeXt checkpoint once and move it on demand."""
        if torch is None:
            raise ClassificationUnavailableError("Torch is not available for LesioNeXt inference.")
        checkpoint = self._lesionext_checkpoint()
        if checkpoint is None or not Path(checkpoint).exists():
            raise ClassificationUnavailableError("LesioNeXt checkpoint is not available.")
        if self._lesionext_model is None:
            model_config = dict(self.runtime_config.get("lesionext_model", {}))
            model_config.setdefault("architecture", "lesionext")
            model_config.setdefault("in_channels", 3)
            model_config.setdefault("classes", 1)
            self._lesionext_model = load_segmenter(model_config, checkpoint_path=checkpoint, map_location="cpu")
            self._lesionext_model.eval()
        requested = self.runtime_config.get("lesionext_device", self.runtime_config.get("device", "cpu"))
        device = resolve_torch_device(str(requested), torch)
        if self._lesionext_device != device:
            self._lesionext_model.to(device)
            self._lesionext_device = device
        return self._lesionext_model, device

    def _predict_lesionext(self, image: np.ndarray) -> tuple[float, float]:
        """Run one unified LesioNeXt pass and cache mask plus reliability metadata."""
        model, device = self._ensure_lesionext_loaded()
        image_size = int(self.runtime_config.get("lesionext_image_size", 256))
        tensor = prepare_classifier_input(image, image_size)
        if not hasattr(tensor, "unsqueeze"):
            raise ClassificationUnavailableError("Torch tensor conversion failed for LesioNeXt inference.")
        inference_context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with inference_context():
            outputs = model.forward_with_aux(tensor.unsqueeze(0).to(device=device, dtype=torch.float32))
            probabilities = torch.softmax(outputs["class_logits"], dim=1)[0].cpu().numpy()
            self._last_lesionext_mask = torch.sigmoid(outputs["mask"])[0, 0].cpu().numpy()
            self._last_lesionext_router_weight = float(outputs["roi_weight"][0, 0].detach().cpu().item())
        return float(probabilities[0]), float(probabilities[1])

    @property
    def _classifier_model(self):
        """Proxy to the primary classifier model on the ensemble."""
        return self.classifier_ensemble.primary_model

    @property
    def _classifier_models(self):
        """Proxy to all loaded classifier models on the ensemble."""
        return self.classifier_ensemble._classifier_models

    @property
    def _classifier_members(self):
        """Proxy to classifier member dicts on the ensemble."""
        return self.classifier_ensemble._classifier_members

    @property
    def _segmenter_model(self):
        """Proxy to the primary segmenter model on the visual evidence service."""
        return self.visual_evidence.segmenter_model

    @property
    def _segmenter_models(self):
        """Proxy to all loaded segmenter models on the visual evidence service."""
        return self.visual_evidence.segmenter_models

    @classmethod
    def from_config(cls, config_path: str | Path) -> "BreastUltrasoundInferenceService":
        """Build the service from a project YAML config and resolve checkpoint paths."""
        service, _paths = _inference_service_from_project_config(config_path)
        return service

    def diagnose(
        self,
        image_input: str | Path | np.ndarray,
        *,
        input_filename: str | None = None,
        decision_threshold: float | None = None,
        need_segmentation: bool = True,
        need_explanation: bool = True,
    ) -> InferenceResponse:
        """Run single-image diagnosis with optional segmentation and Grad-CAM evidence."""
        filename = input_filename or (
            Path(image_input).name if isinstance(image_input, (str, Path)) else "uploaded.png"
        )
        try:
            image = self._validated_input_image(image_input)
            benign_probability, malignant_probability = self._predict_classification(image)
            response = self._build_completed_response(
                filename=filename,
                image=image,
                benign_probability=benign_probability,
                malignant_probability=malignant_probability,
                decision_threshold=decision_threshold,
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

    def _validated_input_image(self, image_input: str | Path | np.ndarray) -> np.ndarray:
        """Read, validate, and quality-check an uploaded image before inference."""
        image = read_image(image_input, grayscale=True)
        validate_image_array(image)
        quality = assess_image_quality(image)
        if quality == "invalid":
            raise InvalidInputError("Uploaded image is invalid or too small.")
        if quality == "low_quality":
            raise QualityBlockedError("Image quality is too poor for reliable analysis.")
        return image

    def _decision_threshold(self, override: float | None) -> float:
        """Return the override threshold when provided, else the configured default."""
        return float(
            override
            if override is not None
            else self.runtime_config.get("default_threshold", 0.5)
        )

    def _response_metadata(self, decision_threshold: float) -> dict[str, Any]:
        """Build the metadata dict attached to every completed response."""
        metadata = {
            "model_identifier": self._model_identifier(),
            "decision_threshold": decision_threshold,
            "primary_model": self.runtime_config.get(
                "primary_classifier_model",
                "unknown",
            ),
            "ensemble_display_name": self.runtime_config.get("ensemble_display_name"),
            "roi_fallback_reason": self._last_roi_fallback_reason,
        }
        if self._uses_lesionext():
            metadata["roi_reliability"] = self._last_lesionext_router_weight
        return metadata

    def _build_completed_response(
        self,
        *,
        filename: str,
        image: np.ndarray,
        benign_probability: float,
        malignant_probability: float,
        decision_threshold: float | None,
    ) -> InferenceResponse:
        """Wrap classification probabilities into a completed InferenceResponse."""
        threshold = self._decision_threshold(decision_threshold)
        result = build_diagnostic_result(
            benign_probability,
            malignant_probability,
            threshold=threshold,
            borderline_margin=float(self.runtime_config.get("borderline_margin", 0.08)),
            model_version=self._model_identifier(),
        )
        return InferenceResponse(
            status="completed",
            input_filename=filename,
            result=result,
            original_image_view=ensure_three_channels(image),
            metadata=self._response_metadata(threshold),
        )

    def _resolved_classifier_checkpoint(self) -> str | None:
        """Return the first resolved classifier checkpoint."""
        return self.classifier_ensemble.resolved_classifier_checkpoint()

    def _resolved_classifier_member_configs(self) -> list[dict[str, Any]]:
        """Return classifier member configs for compatibility callers."""
        return self.classifier_ensemble.resolved_classifier_member_configs()

    def _resolved_classifier_checkpoints(self) -> list[str]:
        """Return all resolved classifier checkpoint paths."""
        return self.classifier_ensemble.resolved_classifier_checkpoints()

    def _resolved_segmenter_checkpoints(self) -> list[str]:
        """Return all resolved segmenter checkpoint paths."""
        return self.visual_evidence.resolved_segmenter_checkpoints()

    def _resolved_segmenter_checkpoint(self) -> str | None:
        """Return the first resolved segmenter checkpoint."""
        return self.visual_evidence.resolved_segmenter_checkpoint()

    def _model_identifier(self) -> str:
        """Return the active classifier ensemble identifier."""
        if self._uses_lesionext():
            return "LesioNeXt-BUS"
        return self.classifier_ensemble.model_identifier()

    def _member_config_value(
        self,
        member: dict[str, Any] | None,
        key: str,
        runtime_key: str,
        default: Any,
    ) -> Any:
        """Look up a member-level runtime override through the ensemble."""
        return self.classifier_ensemble.member_config_value(member, key, runtime_key, default)

    def _classifier_preprocess_kwargs(
        self,
        variant: dict[str, Any] | None = None,
        member: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return classifier preprocessing kwargs for compatibility callers."""
        return self.classifier_ensemble.preprocess_kwargs(variant, member)

    def _classifier_device(self) -> str:
        """Return the resolved classifier device."""
        return self.classifier_ensemble.device()

    def _classifier_tta_variants(self, member: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return classifier TTA variants for a member."""
        return self.classifier_ensemble.tta_variants(member)

    def _apply_classifier_tta_variant(self, image: np.ndarray, variant: dict[str, Any]) -> np.ndarray:
        """Apply one classifier TTA variant to an image."""
        return self.classifier_ensemble.apply_tta_variant(image, variant)

    def _classifier_input_tensors(
        self,
        image: np.ndarray,
        member: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Build classifier input tensors for compatibility callers."""
        return self.classifier_ensemble.input_tensors(image, member)

    def _predict_classifier_ensemble_on_image(
        self,
        image: np.ndarray,
        *,
        member_weight_overrides: dict[str, float] | None = None,
        mask: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Predict benign/malignant probabilities with the classifier ensemble."""
        return self.classifier_ensemble.predict_on_image(
            image,
            member_weight_overrides=member_weight_overrides,
            mask=mask,
        )

    def _roi_enhancement_config(self) -> dict[str, Any] | None:
        """Return the enabled ROI enhancement config, if present."""
        return self.runtime.roi_enhancement_config()

    @staticmethod
    def _stacker_features(full_probability: float, roi_probability: float, feature_mode: str) -> np.ndarray:
        """Build ROI stacker features from full-image and ROI probabilities."""
        return RoiEnhancer.stacker_features(full_probability, roi_probability, feature_mode)

    def _apply_roi_stacker(
        self,
        *,
        full_probability: float,
        roi_probability: float,
        stacker: dict[str, Any],
    ) -> float:
        """Apply the ROI stacker through the RoiEnhancer service."""
        return self.roi_enhancer.apply_roi_stacker(
            full_probability=full_probability,
            roi_probability=roi_probability,
            stacker=stacker,
        )

    def _apply_descriptor_router(
        self,
        *,
        full_probability: float,
        roi_probability: float,
        stacked_probability: float,
        descriptors: dict[str, float],
        router: dict[str, Any],
    ) -> float:
        """Apply descriptor-based routing through the RoiEnhancer service."""
        return self.roi_enhancer.apply_descriptor_router(
            full_probability=full_probability,
            roi_probability=roi_probability,
            stacked_probability=stacked_probability,
            descriptors=descriptors,
            router=router,
        )

    @staticmethod
    def _roi_area_ratio(mask: np.ndarray | None, config: dict[str, Any]) -> tuple[float, bool]:
        """Return the ROI crop area ratio for a segmentation mask."""
        return RoiEnhancer.roi_area_ratio(mask, config)

    @staticmethod
    def _roi_area_gate_config(config: dict[str, Any]) -> dict[str, Any] | None:
        """Parse ROI area gate settings from an ROI config dict."""
        return RoiEnhancer.roi_area_gate_config(config)

    def _should_fallback_roi_by_area(
        self,
        mask: np.ndarray | None,
        config: dict[str, Any],
    ) -> bool:
        """Return True when ROI mask area should fall back to full-image prediction."""
        return self.roi_enhancer.should_fallback_by_area(mask, config)

    def _predict_roi_enhanced_classification(
        self,
        image: np.ndarray,
        *,
        full_benign_probability: float,
        full_malignant_probability: float,
        config: dict[str, Any],
    ) -> tuple[float, float]:
        """Apply ROI reclassification while preserving the reason for full-image fallback."""
        benign_probability, malignant_probability, fallback_reason = self.roi_enhancer.predict(
            image,
            full_benign_probability=full_benign_probability,
            full_malignant_probability=full_malignant_probability,
            config=config,
            classifier_predictor=self._predict_classifier_ensemble_on_image,
            segmenter_predictor=(
                self.segmenter_predictor
                if self.segmenter_predictor is not None
                else self._predict_segmentation
            ),
        )
        self._last_roi_fallback_reason = fallback_reason
        return benign_probability, malignant_probability

    def _predict_classification(self, image: np.ndarray) -> tuple[float, float]:
        """Run full-image classification and optional ROI-enhanced refinement."""
        self._last_roi_fallback_reason = None
        self._last_lesionext_mask = None
        self._last_lesionext_router_weight = None
        if self._uses_lesionext():
            return self._predict_lesionext(image)
        if self.classifier_predictor is not None:
            benign, malignant = self.classifier_predictor(image)
            return float(benign), float(malignant)

        classifier_mask = None
        if self.classifier_ensemble.requires_roi_mask():
            classifier_mask = (
                self.segmenter_predictor(image)
                if self.segmenter_predictor is not None
                else self._predict_segmentation(image)
            )
        try:
            if classifier_mask is None:
                full_benign, full_malignant = self._predict_classifier_ensemble_on_image(image)
            else:
                full_benign, full_malignant = self._predict_classifier_ensemble_on_image(
                    image,
                    mask=classifier_mask,
                )
        except TypeError:
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
                self._last_roi_fallback_reason = "roi_enhancement_failed"
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
        """Attach optional lesion and explanation views to a completed response."""
        lesionext_segmenter = None
        if self._uses_lesionext() and self._last_lesionext_mask is not None:
            lesionext_segmenter = lambda _image: self._last_lesionext_mask
        self.visual_evidence.attach_optional_visuals(
            response,
            image,
            need_segmentation=need_segmentation,
            need_explanation=need_explanation,
            segmenter_predictor=lesionext_segmenter or self.segmenter_predictor,
            explanation_generator=self.explanation_generator,
        )

    def _predict_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Predict a lesion mask through the visual evidence service."""
        if self._uses_lesionext():
            if self._last_lesionext_mask is None:
                self._predict_lesionext(image)
            return self._last_lesionext_mask
        return self.visual_evidence.predict_segmentation(image)

    def _predict_explanation(self, image: np.ndarray) -> np.ndarray:
        """Generate a Grad-CAM explanation through the visual evidence service."""
        return self.visual_evidence.predict_explanation(image)


def evaluate_busi_dataset(
    config_path: str | Path,
    *,
    classifier_predictor: Callable[[np.ndarray], tuple[float, float]] | None = None,
    output_path: str | Path | None = None,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Evaluate the configured classifier path on BUSI benign/malignant samples."""
    service, paths = _inference_service_from_project_config(
        config_path,
        classifier_predictor=classifier_predictor,
    )
    manifest = load_busi_manifest(paths.busi_root, include_normal=False)
    y_true, malignant_probabilities, rows = _collect_busi_predictions(service, manifest)

    report = _build_busi_report(
        config_path=config_path,
        service=service,
        y_true=y_true,
        malignant_probabilities=malignant_probabilities,
        rows=rows,
        auc_bootstrap_ci=_bootstrap_auc_ci(
            y_true,
            malignant_probabilities,
            replicates=bootstrap_replicates,
        ),
    )
    destination = Path(output_path) if output_path is not None else paths.reports_root / "busi_eval.json"
    _write_busi_report_outputs(report, destination, write_named_threshold_report=output_path is not None)
    return report


def _bootstrap_auc_ci(
    y_true: list[int],
    probabilities: list[float],
    *,
    replicates: int = 2000,
    seed: int = 20260806,
) -> dict[str, Any] | None:
    """Estimate a percentile confidence interval without touching model selection."""
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(probabilities, dtype=np.float64)
    if len(labels) < 2 or len(np.unique(labels)) < 2 or replicates <= 0:
        return None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(int(replicates)):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        values.append(float(roc_auc_score(sampled_labels, scores[indices])))
    if not values:
        return None
    low, high = np.percentile(np.asarray(values), [2.5, 97.5])
    return {"confidence": 0.95, "lower": float(low), "upper": float(high), "replicates": len(values)}


def _resize_mask_for_eval(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a ground-truth mask to the model output shape with nearest-neighbor semantics."""
    if mask.shape[:2] == shape:
        return mask
    cv2_module = optional_import("cv2")
    if cv2_module is not None:
        return cv2_module.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2_module.INTER_NEAREST)
    y = np.linspace(0, mask.shape[0] - 1, shape[0]).astype(int)
    x = np.linspace(0, mask.shape[1] - 1, shape[1]).astype(int)
    return mask[np.ix_(y, x)]


def evaluate_external_bus_dataset(
    dataset_id: str,
    root: str | Path,
    *,
    config_path: str | Path,
    manifest_path: str | Path | None = None,
    output_path: str | Path | None = None,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Evaluate a frozen config on an independent dataset with no external tuning."""
    if dataset_id not in DATASET_REGISTRY:
        raise KeyError(f"Unknown external dataset '{dataset_id}'.")
    service, paths = _inference_service_from_project_config(config_path)
    manifest = load_external_manifest(dataset_id, root, manifest_path=manifest_path)
    y_true: list[int] = []
    probabilities: list[float] = []
    rows: list[dict[str, Any]] = []
    dice_values: list[float] = []
    iou_values: list[float] = []
    skipped: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        response = service.diagnose(
            row.image_path,
            input_filename=Path(row.image_path).name,
            need_segmentation=False,
            need_explanation=False,
        )
        if response.result is None:
            skipped.append({"sample_id": row.sample_id, "reason": response.status})
            continue
        malignant_probability = float(response.result.malignant_probability)
        label = 1 if row.pathology_label == "malignant" else 0
        y_true.append(label)
        probabilities.append(malignant_probability)
        prediction_row = {
            "sample_id": row.sample_id,
            "case_id": row.case_id,
            "dataset_name": dataset_id,
            "image_path": row.image_path,
            "mask_path": row.mask_path,
            "pathology_label": row.pathology_label,
            "malignant_probability": malignant_probability,
            "final_label": response.result.final_label,
            "status": response.status,
        }
        if row.mask_path:
            try:
                predicted_mask = service._predict_segmentation(read_image(row.image_path, grayscale=True))
                true_mask = read_mask(row.mask_path)
                if true_mask is None:
                    raise ValueError("Ground-truth mask is unavailable.")
                true_mask = _resize_mask_for_eval(true_mask, predicted_mask.shape[:2])
                prediction_row["dice"] = dice_score(predicted_mask, true_mask)
                prediction_row["iou"] = iou_score(predicted_mask, true_mask)
                dice_values.append(prediction_row["dice"])
                iou_values.append(prediction_row["iou"])
            except Exception as exc:  # segmentation is an optional external metric
                prediction_row["segmentation_error"] = str(exc)
        rows.append(prediction_row)
    threshold = float(service.runtime_config.get("default_threshold", 0.5))
    metrics = classification_metrics(y_true, probabilities, threshold=threshold)
    segmentation = None
    if dice_values:
        segmentation = {
            "sample_count": len(dice_values),
            "dice_mean": float(np.mean(dice_values)),
            "iou_mean": float(np.mean(iou_values)),
        }
    report = {
        "dataset_id": dataset_id,
        "dataset_card": {"dataset_id": dataset_id, **DATASET_REGISTRY[dataset_id]},
        "config_path": str(config_path),
        "model_identifier": service._model_identifier(),
        "sample_count": len(rows),
        "case_count": int(manifest["case_id"].nunique()),
        "label_counts": {
            str(label): int(count)
            for label, count in manifest["pathology_label"].value_counts().to_dict().items()
        },
        "skipped_count": len(skipped),
        "skipped": skipped,
        "threshold_source": "frozen_runtime_default; no external threshold tuning",
        "metrics": metrics,
        "auc_bootstrap_ci": _bootstrap_auc_ci(y_true, probabilities, replicates=bootstrap_replicates),
        "segmentation": segmentation,
        "rows": rows,
    }
    destination = Path(output_path) if output_path is not None else paths.reports_root / f"external_{dataset_id}.json"
    write_json_report(destination, report)
    prediction_path = destination.with_name(f"{destination.stem}_predictions.csv")
    pd.DataFrame.from_records(rows).to_csv(prediction_path, index=False, encoding="utf-8-sig")
    return report


def _collect_busi_predictions(
    service: BreastUltrasoundInferenceService,
    manifest,
) -> tuple[list[int], list[float], list[dict[str, Any]]]:
    """Run BUSI samples through diagnosis and keep only completed predictions."""
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
    return y_true, malignant_probabilities, rows


def _build_busi_report(
    *,
    config_path: str | Path,
    service: BreastUltrasoundInferenceService,
    y_true: list[int],
    malignant_probabilities: list[float],
    rows: list[dict[str, Any]],
    auc_bootstrap_ci: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the BUSI report payload without touching the filesystem."""
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
        "auc_bootstrap_ci": auc_bootstrap_ci,
        "threshold_analysis": {
            "best_by_youden": best_threshold,
            "rows": threshold_rows,
        },
        "rows": rows,
    }
    return report


def _write_busi_report_outputs(
    report: dict[str, Any],
    destination: Path,
    *,
    write_named_threshold_report: bool,
) -> None:
    """Write JSON plus the standard and optional named threshold reports."""
    write_json_report(destination, report)
    metrics = report["metrics"]
    threshold_analysis = report["threshold_analysis"]
    threshold_markdown = _threshold_analysis_markdown(
        metrics,
        threshold_analysis.get("best_by_youden", {}),
        threshold_analysis.get("rows", []),
    )
    write_markdown_report(destination.parent / "threshold_analysis.md", threshold_markdown)
    if write_named_threshold_report:
        write_markdown_report(
            destination.with_name(f"threshold_analysis_{destination.stem}.md"),
            threshold_markdown,
        )


def _threshold_analysis_markdown(
    default_metrics: dict[str, Any],
    best_threshold: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[str]:
    """Render BUSI threshold sweep metrics as a small Markdown report."""
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
