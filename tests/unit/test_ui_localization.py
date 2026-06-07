"""Unit tests for ui localization."""

from __future__ import annotations

from app.components.result_panels import diagnosis_markdown
from app.components.status_panels import status_markdown, warnings_markdown
from src.utils.results import InferenceResponse, build_diagnostic_result


def test_diagnosis_markdown_is_localized_to_chinese() -> None:
    """Verify diagnosis markdown is localized to chinese."""
    response = InferenceResponse(
        status="completed",
        input_filename="sample.png",
        result=build_diagnostic_result(0.2, 0.8, threshold=0.5),
    )

    markdown = diagnosis_markdown(response)

    assert "诊断结果" in markdown
    assert "最终判定" in markdown
    assert "恶性" in markdown
    assert "恶性概率" in markdown
    assert "80.0%" in markdown
    assert "本系统仅用于辅助分析和原型演示，不能替代医生诊断。" in markdown


def test_status_and_warning_panels_are_localized_to_chinese() -> None:
    """Verify status and warning panels are localized to chinese."""
    response = InferenceResponse(
        status="quality_blocked",
        input_filename="sample.png",
        result=None,
        warnings=["Image quality is too poor for reliable analysis."],
    )

    status = status_markdown(response)
    warnings = warnings_markdown(response)

    assert "运行状态" in status
    assert "图像质量不足" in status
    assert "提示信息" in warnings
    assert "图像质量过低，当前无法给出可靠分析结果。" in warnings
