from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.engine.errors import InvalidInputError
from src.utils.runtime import optional_import, require_dependency


cv2 = optional_import("cv2")
PIL_Image = None
pil_module = optional_import("PIL.Image")
if pil_module is not None:
    PIL_Image = pil_module

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def validate_suffix(path: str | Path) -> None:
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise InvalidInputError(f"Unsupported image file type: {suffix}")


def _read_with_cv2(path: Path, grayscale: bool) -> np.ndarray:
    require_dependency("cv2", cv2)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise InvalidInputError(f"Unable to read image: {path}")
    if not grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _read_with_pil(path: Path, grayscale: bool) -> np.ndarray:
    if PIL_Image is None:
        raise InvalidInputError(
            "No image backend is installed. Install opencv-python or Pillow."
        )
    with PIL_Image.open(path) as image:
        if grayscale:
            image = image.convert("L")
        else:
            image = image.convert("RGB")
        return np.asarray(image)


def read_image(image: str | Path | np.ndarray, *, grayscale: bool = True) -> np.ndarray:
    if isinstance(image, np.ndarray):
        array = image.copy()
        if grayscale and array.ndim == 3:
            array = array[..., 0]
        return array

    path = Path(image)
    validate_suffix(path)
    if not path.exists():
        raise InvalidInputError(f"Image does not exist: {path}")

    if cv2 is not None:
        return _read_with_cv2(path, grayscale)
    return _read_with_pil(path, grayscale)


def save_image(path: str | Path, image: np.ndarray) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is not None:
        array = image
        if image.ndim == 3:
            array = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), array)
        return output_path
    if PIL_Image is not None:
        PIL_Image.fromarray(image).save(output_path)
        return output_path
    raise InvalidInputError("No image backend is installed for saving.")


def read_mask(mask_path: str | Path | None) -> np.ndarray | None:
    if mask_path is None:
        return None
    mask = read_image(mask_path, grayscale=True)
    return (mask > 0).astype(np.uint8)


def ensure_three_channels(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    return image


def validate_image_array(image: np.ndarray) -> None:
    if image.size == 0:
        raise InvalidInputError("Image is empty.")
    if image.ndim not in (2, 3):
        raise InvalidInputError(f"Unexpected image shape: {image.shape}")
