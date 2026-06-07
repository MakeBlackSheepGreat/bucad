"""Report required and optional Python dependencies for the BUCAD environment."""

from __future__ import annotations

import importlib
import json
import platform
import sys
from typing import Dict


REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "yaml",
    "sklearn",
]

OPTIONAL_MODULES = [
    "torch",
    "torchvision",
    "cv2",
    "timm",
    "segmentation_models_pytorch",
    "gradio",
]


def check_module(name: str) -> Dict[str, str | bool]:
    """Import one dependency and return availability plus version details."""
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "detail": str(exc)}

    version = getattr(module, "__version__", "unknown")
    return {"available": True, "detail": str(version)}


def main() -> int:
    """Print the environment report and fail only when required modules are missing."""
    report: Dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "required": {name: check_module(name) for name in REQUIRED_MODULES},
        "optional": {name: check_module(name) for name in OPTIONAL_MODULES},
    }

    torch_report = report["optional"].get("torch", {})
    if isinstance(torch_report, dict) and torch_report.get("available"):
        try:
            torch = importlib.import_module("torch")
            report["cuda_available"] = bool(torch.cuda.is_available())
            report["cuda_device_count"] = int(torch.cuda.device_count())
        except Exception as exc:  # pragma: no cover - environment dependent
            report["cuda_available"] = False
            report["cuda_error"] = str(exc)
    else:
        report["cuda_available"] = False

    print(json.dumps(report, indent=2, ensure_ascii=False))
    required_ok = all(
        item["available"] for item in report["required"].values()  # type: ignore[index]
    )
    return 0 if required_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
