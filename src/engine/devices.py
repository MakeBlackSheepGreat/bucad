"""Shared torch device resolution helpers for training and inference."""

from __future__ import annotations

from typing import Any


def resolve_torch_device(requested: str | None, torch_module: Any) -> str:
    value = str(requested or "cpu").lower()
    cuda_available = bool(
        torch_module is not None
        and hasattr(torch_module, "cuda")
        and torch_module.cuda.is_available()
    )
    if value == "auto":
        return "cuda" if cuda_available else "cpu"
    if value.startswith("cuda") and not cuda_available:
        return "cpu"
    return value
