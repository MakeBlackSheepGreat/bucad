from __future__ import annotations

import importlib
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def optional_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def require_dependency(module_name: str, module: Any | None) -> Any:
    if module is None:
        raise RuntimeError(
            f"Optional dependency '{module_name}' is required for this action. "
            f"Install project requirements first."
        )
    return module


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = optional_import("torch")
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - requires CUDA
            torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> str:
    torch = optional_import("torch")
    requested = requested.lower()
    if requested != "auto":
        return requested
    if torch is not None and torch.cuda.is_available():  # pragma: no cover - env dependent
        return "cuda"
    return "cpu"


def is_torch_available() -> bool:
    return optional_import("torch") is not None
