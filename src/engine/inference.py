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
from src.preprocess.io import cv2, ensure_three_channels, read_image, validate_image_array
from src.preprocess.transforms import prepare_classifier_input
from src.utils.config import load_project_config
from src.utils.metrics import best_threshold_by_youden, classification_metrics, threshold_sweep
from src.utils.reporting import write_json_report, write_markdown_report
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
        self._classifier_models = None
        self._classifier_members = None
        self._segmenter_model = None

    @classmethod
    def from_config(cls, config_path: str | Path) -> "BreastUltrasoundInferenceService":
        config, paths = load_project_config(config_path)
        runtime_config = dict(config.get("runtime", {}))
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

    def _resolved_segmenter_checkpoint(self) -> str | None:
        checkpoint = self.runtime_config.get("segmenter_checkpoint")
        if checkpoint:
            return str(checkpoint)
        if self.paths is not None and self.paths.default_segmenter_ckpt.exists():
            return str(self.paths.default_segmenter_ckpt)
        return None

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

    def _predict_classification(self, image: np.ndarray) -> tuple[float, float]:
        if self.classifier_predictor is not None:
            benign, malignant = self.classifier_predictor(image)
            return float(benign), float(malignant)

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
            input_tensors = self._classifier_input_tensors(image, member)
            if not input_tensors or not hasattr(input_tensors[0], "unsqueeze"):
                raise ClassificationUnavailableError("Torch tensor conversion failed for classifier input.")
            batch = torch.stack(input_tensors)
            model = member["model_instance"]
            weight = float(member.get("weight", 1.0))
            tta_probs = classifier_probabilities(model, batch, device=device)
            ensemble_probs.append(np.mean(tta_probs, axis=0) * weight)
            weights.append(weight)
        probs = np.sum(np.asarray(ensemble_probs, dtype=np.float32), axis=0) / float(sum(weights))
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
        f"- Sensitivity: `{default_metrics.get('sensitivity', 0.0):.4f}`",
        f"- Specificity: `{default_metrics.get('specificity', 0.0):.4f}`",
        "",
        "## Best Threshold By Youden J",
        "",
    ]
    if best_threshold:
        lines.extend(
            [
                f"- Threshold: `{best_threshold.get('threshold', 0.5):.2f}`",
                f"- Youden J: `{best_threshold.get('youden_j', 0.0):.4f}`",
                f"- Sensitivity: `{best_threshold.get('sensitivity', 0.0):.4f}`",
                f"- Specificity: `{best_threshold.get('specificity', 0.0):.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Sweep",
            "",
            "| Threshold | Sensitivity | Specificity | Accuracy | Youden J |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['threshold']:.2f} | {row['sensitivity']:.4f} | "
            f"{row['specificity']:.4f} | {row['accuracy']:.4f} | {row['youden_j']:.4f} |"
        )
    return lines
