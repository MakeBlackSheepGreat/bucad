"""Diagnosis result dataclasses and clinical recommendation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.utils.metrics import normalize_binary_probs
from src.utils.runtime import timestamp_now


AUXILIARY_USE_DISCLAIMER = (
    "This system is for auxiliary review only and does not replace clinician judgement."
)


@dataclass(slots=True)
class DiagnosticResult:
    benign_probability: float
    malignant_probability: float
    final_label: str
    confidence_band: str
    recommendation_text: str
    analysis_timestamp: str
    model_version: str
    auxiliary_use_disclaimer: str = AUXILIARY_USE_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VisualEvidence:
    overlay_type: str
    generation_status: str
    artifact_path: str | None = None
    reason_if_missing: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InferenceResponse:
    status: str
    input_filename: str
    result: DiagnosticResult | None
    warnings: list[str] = field(default_factory=list)
    original_image_view: Any | None = None
    lesion_overlay_view: Any | None = None
    explanation_view: Any | None = None
    lesion_visualization_missing_reason: str | None = None
    explanation_missing_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "input_filename": self.input_filename,
            "warnings": self.warnings,
            "lesion_visualization_missing_reason": self.lesion_visualization_missing_reason,
            "explanation_missing_reason": self.explanation_missing_reason,
            "metadata": self.metadata,
        }
        if self.result is not None:
            payload.update(self.result.to_dict())
        return payload


def confidence_band_from_probability(
    malignant_probability: float, *, threshold: float = 0.5, borderline_margin: float = 0.08
) -> str:
    distance = abs(malignant_probability - threshold)
    max_prob = max(malignant_probability, 1.0 - malignant_probability)
    if distance <= borderline_margin:
        return "borderline"
    if max_prob >= 0.8:
        return "high"
    return "low"


def recommendation_for_band(confidence_band: str, final_label: str) -> str:
    if confidence_band == "borderline":
        return "Result is borderline. Manual review is strongly recommended."
    if confidence_band == "low":
        return f"Model leans {final_label}, but confidence is limited. Review the image carefully."
    if final_label == "malignant":
        return "High-risk appearance detected. Prompt manual review is recommended."
    return "Findings lean benign, but clinician review remains required."


def build_diagnostic_result(
    benign_probability: float,
    malignant_probability: float,
    *,
    threshold: float = 0.5,
    borderline_margin: float = 0.08,
    model_version: str = "unknown",
) -> DiagnosticResult:
    """Normalize probabilities and assemble the user-facing diagnostic result."""
    benign_probability, malignant_probability = normalize_binary_probs(
        benign_probability, malignant_probability
    )
    final_label = "malignant" if malignant_probability >= threshold else "benign"
    band = confidence_band_from_probability(
        malignant_probability,
        threshold=threshold,
        borderline_margin=borderline_margin,
    )
    return DiagnosticResult(
        benign_probability=benign_probability,
        malignant_probability=malignant_probability,
        final_label=final_label,
        confidence_band=band,
        recommendation_text=recommendation_for_band(band, final_label),
        analysis_timestamp=timestamp_now(),
        model_version=model_version,
    )
