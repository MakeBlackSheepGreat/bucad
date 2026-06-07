"""HTML rendering helpers for BUCAD runtime status and warnings."""

from __future__ import annotations

from html import escape

from app.components.localization import localize_status, localize_warning
from src.utils.results import InferenceResponse


def status_markdown(response: InferenceResponse) -> str:
    status = escape(localize_status(response.status))
    status_class = "ok" if response.status == "completed" else "warn"
    return f"""
<div class="status-card">
  <div class="section-title">运行状态</div>
  <div class="status-line {status_class}">
    <span class="status-dot"></span>
    <span>{status}</span>
  </div>
</div>
"""


def warnings_markdown(response: InferenceResponse) -> str:
    if not response.warnings:
        return """
<div class="status-card">
  <div class="section-title">提示信息</div>
  <div class="hint-text">暂无异常提示。请结合原图、叠加图和热力图综合查看。</div>
</div>
"""

    items = "\n".join(
        f"<li>{escape(localize_warning(warning))}</li>" for warning in response.warnings
    )
    return f"""
<div class="status-card">
  <div class="section-title">提示信息</div>
  <ul class="warning-list">{items}</ul>
</div>
"""
