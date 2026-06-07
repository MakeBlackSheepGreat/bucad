"""Checkpoint write helpers shared by classifier and segmenter training."""

from __future__ import annotations

import gc
import os
import time
import uuid
from pathlib import Path
from typing import Any

from src.utils.runtime import optional_import, require_dependency


torch = optional_import("torch")


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    """Write a torch checkpoint through a sibling temp file, then replace it."""
    require_dependency("torch", torch)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary_path)
    gc.collect()

    # Windows can briefly hold a checkpoint handle after torch's zip writer
    # closes. A short retry keeps replacement atomic while avoiding half files.
    last_error: PermissionError | None = None
    for attempt in range(5):
        try:
            temporary_path.replace(destination)
            return
        except PermissionError as exc:  # pragma: no cover - timing dependent
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
            gc.collect()
    try:
        temporary_path.unlink(missing_ok=True)
    finally:
        if last_error is not None:
            raise last_error
