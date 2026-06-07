"""Grad-CAM generation wrapper used by the BUCAD visual evidence path."""

from __future__ import annotations

import numpy as np

from src.engine.errors import OptionalOutputUnavailableError
from src.models.classifier import resolve_gradcam_target_layer
from src.utils.runtime import optional_import


torch = optional_import("torch")
pytorch_grad_cam = optional_import("pytorch_grad_cam")


def generate_gradcam_map(model, input_tensor, *, target_layer=None) -> np.ndarray:
    """Generate a normalized Grad-CAM map for one classifier input tensor."""
    if torch is None or pytorch_grad_cam is None:
        raise OptionalOutputUnavailableError(
            "Grad-CAM dependency is not installed in the current environment."
        )

    target_layer = target_layer or resolve_gradcam_target_layer(model)
    if target_layer is None:
        raise OptionalOutputUnavailableError("Could not resolve a Grad-CAM target layer.")

    grad_cam = pytorch_grad_cam.GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = grad_cam(input_tensor=input_tensor)
    if grayscale_cam is None or len(grayscale_cam) == 0:
        raise OptionalOutputUnavailableError("Grad-CAM returned no explanation map.")
    cam = grayscale_cam[0]
    cam = cam - cam.min()
    denom = cam.max() if float(cam.max()) > 0 else 1.0
    return (cam / denom).astype(np.float32)
