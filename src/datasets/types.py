"""Shared dataset record types for breast ultrasound samples."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class CaseSample:
    """Normalized metadata for one breast ultrasound image sample."""

    sample_id: str
    case_id: str
    dataset_name: str
    image_path: str
    mask_path: str | None
    pathology_label: str
    view_side: str | None = None
    quality_flag: str = "valid"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the sample metadata to a plain dict."""
        return asdict(self)
