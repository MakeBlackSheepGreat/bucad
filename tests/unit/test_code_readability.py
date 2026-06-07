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


def test_long_code_objects_have_docstrings() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "ls-files", "src/**/*.py", "app/**/*.py", "scripts/**/*.py"],
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
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            length = int(node.end_lineno or node.lineno) - int(node.lineno) + 1
            if length >= 30 and ast.get_docstring(node) is None:
                missing.append(f"{relative_path}:{node.lineno} {node.name}")

    assert missing == []
