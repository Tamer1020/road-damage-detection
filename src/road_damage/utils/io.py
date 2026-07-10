"""Image IO helpers: disk read/write and in-memory encode/decode."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Could not write image: {path}")


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes (e.g. an upload) into a BGR ndarray."""
    array = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid or unsupported image data.")
    return image


def encode_image(image: np.ndarray, ext: str = ".jpg") -> bytes:
    ok, buffer = cv2.imencode(ext, image)
    if not ok:
        raise ValueError("Could not encode image.")
    return buffer.tobytes()


def to_base64(image: np.ndarray, ext: str = ".jpg") -> str:
    return base64.b64encode(encode_image(image, ext)).decode("utf-8")
