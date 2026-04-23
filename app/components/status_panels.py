from __future__ import annotations

from src.utils.results import InferenceResponse


def status_markdown(response: InferenceResponse) -> str:
    return f"## Status\nCurrent status: **{response.status}**"


def warnings_markdown(response: InferenceResponse) -> str:
    if not response.warnings:
        return "## Warnings\nNo additional warnings."
    items = "\n".join(f"- {message}" for message in response.warnings)
    return f"## Warnings\n{items}"
