from __future__ import annotations

from app.components.localization import (
    localize_confidence_band,
    localize_disclaimer,
    localize_final_label,
    localize_recommendation,
)
from src.utils.results import InferenceResponse


def diagnosis_markdown(response: InferenceResponse) -> str:
    if response.result is None:
        return "## 诊断结果\n暂无可展示的诊断结果。"
    result = response.result
    return "\n".join(
        [
            "## 诊断结果",
            f"- 最终判断：**{localize_final_label(result.final_label)}**",
            f"- 恶性概率：**{result.malignant_probability:.3f}**",
            f"- 良性概率：**{result.benign_probability:.3f}**",
            f"- 置信等级：**{localize_confidence_band(result.confidence_band)}**",
            f"- 诊断建议：{localize_recommendation(result.final_label, result.confidence_band)}",
            f"- 免责声明：{localize_disclaimer(result.auxiliary_use_disclaimer)}",
        ]
    )
