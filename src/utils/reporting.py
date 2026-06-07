"""Small report writers for JSON and Markdown artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def write_json_report(path: str | Path, payload: dict[str, Any]) -> Path:
    """Serialize *payload* as pretty-printed JSON and return the written path."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return report_path


def write_markdown_report(path: str | Path, lines: Iterable[str]) -> Path:
    """Join *lines* with newlines, write as UTF-8, and return the written path."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
