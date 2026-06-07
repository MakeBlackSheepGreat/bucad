"""Unit tests for lightweight source readability guardrails."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path


def _tracked_python_files(root: Path) -> list[str]:
    """Return tracked Python paths using Git so generated caches stay out of scope."""
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.splitlines()


def test_tracked_python_files_have_module_docstrings() -> None:
    """Verify every tracked Python file has a module-level docstring."""
    root = Path(__file__).resolve().parents[2]

    missing: list[str] = []
    for relative_path in _tracked_python_files(root):
        source_path = root / relative_path
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        if ast.get_docstring(tree) is None:
            missing.append(relative_path)

    assert missing == []


def test_tracked_code_objects_have_docstrings() -> None:
    """Verify every tracked function and class has a docstring."""
    root = Path(__file__).resolve().parents[2]

    missing: list[str] = []
    for relative_path in _tracked_python_files(root):
        source_path = root / relative_path
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if ast.get_docstring(node) is None:
                missing.append(f"{relative_path}:{node.lineno} {node.name}")

    assert missing == []
