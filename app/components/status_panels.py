from __future__ import annotations

from app.components.localization import localize_status, localize_warning
from src.utils.results import InferenceResponse


def status_markdown(response: InferenceResponse) -> str:
    return f"## 运行状态\n当前状态：**{localize_status(response.status)}**"


def warnings_markdown(response: InferenceResponse) -> str:
    if not response.warnings:
        return "## 提示信息\n当前没有额外提示。"
    items = "\n".join(f"- {localize_warning(message)}" for message in response.warnings)
    return f"## 提示信息\n{items}"
