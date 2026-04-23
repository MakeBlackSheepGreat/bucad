from __future__ import annotations

from src.utils.results import InferenceResponse


def diagnosis_markdown(response: InferenceResponse) -> str:
    if response.result is None:
        return "## Diagnosis\nNo diagnosis result is available."
    result = response.result
    return "\n".join(
        [
            "## Diagnosis",
            f"- Final label: **{result.final_label}**",
            f"- Malignant probability: **{result.malignant_probability:.3f}**",
            f"- Benign probability: **{result.benign_probability:.3f}**",
            f"- Confidence band: **{result.confidence_band}**",
            f"- Recommendation: {result.recommendation_text}",
            f"- Disclaimer: {result.auxiliary_use_disclaimer}",
        ]
    )
