"""Unit tests for lightweight source readability guardrails."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path


CORE_CODE_PREFIXES = (
    "src/",
    "app/",
)
MAINTAINED_SCRIPT_PATHS = {
    "scripts/batch_infer.py",
    "scripts/eval_busi.py",
    "scripts/export_demo_assets.py",
    "scripts/export_report_documents.py",
    "scripts/export_visual_evidence.py",
    "scripts/make_split.py",
    "scripts/run_comparison.py",
    "scripts/train_cls.py",
    "scripts/train_seg.py",
}


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


def _is_readability_guarded(relative_path: str) -> bool:
    """Keep strict object docstring checks on core code and maintained CLI scripts."""
    return relative_path.startswith(CORE_CODE_PREFIXES) or relative_path in MAINTAINED_SCRIPT_PATHS


def test_tracked_python_files_have_module_docstrings() -> None:
    root = Path(__file__).resolve().parents[2]

    missing: list[str] = []
    for relative_path in _tracked_python_files(root):
        source_path = root / relative_path
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        if ast.get_docstring(tree) is None:
            missing.append(relative_path)

    assert missing == []


def test_long_code_objects_have_docstrings() -> None:
    root = Path(__file__).resolve().parents[2]

    missing: list[str] = []
    for relative_path in _tracked_python_files(root):
        if not _is_readability_guarded(relative_path):
            continue
        source_path = root / relative_path
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            length = int(node.end_lineno or node.lineno) - int(node.lineno) + 1
            if length >= 25 and ast.get_docstring(node) is None:
                missing.append(f"{relative_path}:{node.lineno} {node.name}")

    assert missing == []
