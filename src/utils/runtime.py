"""Runtime dependency, seeding, device, and filesystem helpers."""

from __future__ import annotations

import importlib
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def optional_import(module_name: str) -> Any | None:
    """Import a module by name, returning None when it is not installed."""
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def require_dependency(module_name: str, module: Any | None) -> Any:
    """Raise a clear error when an optional dependency is missing."""
    if module is None:
        raise RuntimeError(
            f"Optional dependency '{module_name}' is required for this action. "
            f"Install project requirements first."
        )
    return module


def ensure_dir(path: str | Path) -> Path:
    """Create the directory (including parents) and return its resolved path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def timestamp_now() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and (when available) PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch = optional_import("torch")
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - requires CUDA
            torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> str:
    """Resolve a human-readable device string such as ``'auto'`` or ``'cuda'``."""
    from src.engine.devices import resolve_torch_device

    torch = optional_import("torch")
    return resolve_torch_device(requested, torch)


def is_torch_available() -> bool:
    """Return True when PyTorch can be imported in the current environment."""
    return optional_import("torch") is not None
