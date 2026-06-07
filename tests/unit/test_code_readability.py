"""Unit tests for lightweight source readability guardrails."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path


def test_tracked_python_files_have_module_docstrings() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    missing: list[str] = []
    for relative_path in result.stdout.splitlines():
        source_path = root / relative_path
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        if ast.get_docstring(tree) is None:
            missing.append(relative_path)

    assert missing == []
