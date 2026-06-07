"""Public inference service facade for single-image diagnosis and BUSI evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.datasets.busi import load_busi_manifest
from src.engine.classifier_ensemble import ClassifierEnsemble
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
from src.preprocess.io import ensure_three_channels, read_image, validate_image_array
from src.utils.config import load_project_config
from src.utils.metrics import best_threshold_by_youden, classification_metrics, threshold_sweep
from src.utils.reporting import write_json_report, write_markdown_report
from src.utils.results import InferenceResponse, build_diagnostic_result


def _resolve_runtime_checkpoint_paths(
    runtime_config: dict[str, Any], *, project_root: Path
) -> dict[str, Any]:
    return resolve_runtime_checkpoint_paths(runtime_config, project_root=project_root)


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

    @property
    def _classifier_model(self):
        return self.classifier_ensemble.primary_model

    @property
    def _classifier_models(self):
        return self.classifier_ensemble._classifier_models

    @property
    def _classifier_members(self):
        return self.classifier_ensemble._classifier_members

    @property
    def _segmenter_model(self):
        return self.visual_evidence.segmenter_model

    @property
    def _segmenter_models(self):
        return self.visual_evidence.segmenter_models

    @classmethod
    def from_config(cls, config_path: str | Path) -> "BreastUltrasoundInferenceService":
        """Build the service from a project YAML config and resolve checkpoint paths."""
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
        """Run single-image diagnosis with optional segmentation and Grad-CAM evidence."""
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
                    "roi_fallback_reason": self._last_roi_fallback_reason,
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
        return self.classifier_ensemble.resolved_classifier_checkpoint()

    def _resolved_classifier_member_configs(self) -> list[dict[str, Any]]:
        return self.classifier_ensemble.resolved_classifier_member_configs()

    def _resolved_classifier_checkpoints(self) -> list[str]:
        return self.classifier_ensemble.resolved_classifier_checkpoints()

    def _resolved_segmenter_checkpoints(self) -> list[str]:
        return self.visual_evidence.resolved_segmenter_checkpoints()

    def _resolved_segmenter_checkpoint(self) -> str | None:
        return self.visual_evidence.resolved_segmenter_checkpoint()

    def _model_identifier(self) -> str:
        return self.classifier_ensemble.model_identifier()

    def _member_config_value(
        self,
        member: dict[str, Any] | None,
        key: str,
        runtime_key: str,
        default: Any,
    ) -> Any:
        return self.classifier_ensemble.member_config_value(member, key, runtime_key, default)

    def _classifier_preprocess_kwargs(
        self,
        variant: dict[str, Any] | None = None,
        member: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.classifier_ensemble.preprocess_kwargs(variant, member)

    def _classifier_device(self) -> str:
        return self.classifier_ensemble.device()

    def _classifier_tta_variants(self, member: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self.classifier_ensemble.tta_variants(member)

    def _apply_classifier_tta_variant(self, image: np.ndarray, variant: dict[str, Any]) -> np.ndarray:
        return self.classifier_ensemble.apply_tta_variant(image, variant)

    def _classifier_input_tensors(
        self,
        image: np.ndarray,
        member: dict[str, Any] | None = None,
    ) -> list[Any]:
        return self.classifier_ensemble.input_tensors(image, member)

    def _predict_classifier_ensemble_on_image(
        self,
        image: np.ndarray,
        *,
        member_weight_overrides: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        return self.classifier_ensemble.predict_on_image(
            image,
            member_weight_overrides=member_weight_overrides,
        )

    def _roi_enhancement_config(self) -> dict[str, Any] | None:
        return self.runtime.roi_enhancement_config()

    @staticmethod
    def _stacker_features(full_probability: float, roi_probability: float, feature_mode: str) -> np.ndarray:
        return RoiEnhancer.stacker_features(full_probability, roi_probability, feature_mode)

    def _apply_roi_stacker(
        self,
        *,
        full_probability: float,
        roi_probability: float,
        stacker: dict[str, Any],
    ) -> float:
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
        return self.roi_enhancer.apply_descriptor_router(
            full_probability=full_probability,
            roi_probability=roi_probability,
            stacked_probability=stacked_probability,
            descriptors=descriptors,
            router=router,
        )

    @staticmethod
    def _roi_area_ratio(mask: np.ndarray | None, config: dict[str, Any]) -> tuple[float, bool]:
        return RoiEnhancer.roi_area_ratio(mask, config)

    @staticmethod
    def _roi_area_gate_config(config: dict[str, Any]) -> dict[str, Any] | None:
        return RoiEnhancer.roi_area_gate_config(config)

    def _should_fallback_roi_by_area(
        self,
        mask: np.ndarray | None,
        config: dict[str, Any],
    ) -> bool:
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
        self._last_roi_fallback_reason = None
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
        self.visual_evidence.attach_optional_visuals(
            response,
            image,
            need_segmentation=need_segmentation,
            need_explanation=need_explanation,
            segmenter_predictor=self.segmenter_predictor,
            explanation_generator=self.explanation_generator,
        )

    def _predict_segmentation(self, image: np.ndarray) -> np.ndarray:
        return self.visual_evidence.predict_segmentation(image)

    def _predict_explanation(self, image: np.ndarray) -> np.ndarray:
        return self.visual_evidence.predict_explanation(image)


def evaluate_busi_dataset(
    config_path: str | Path,
    *,
    classifier_predictor: Callable[[np.ndarray], tuple[float, float]] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate the configured classifier path on BUSI benign/malignant samples."""
    config, paths = load_project_config(config_path)
    runtime_config = dict(config.get("runtime", {}))
    runtime_config = _resolve_runtime_checkpoint_paths(runtime_config, project_root=paths.project_root)
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
