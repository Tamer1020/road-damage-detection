"""Draw detection boxes and labels onto images with OpenCV."""

from __future__ import annotations

import cv2
import numpy as np

from .detection import Detection

# BGR palette cycled by class id.
_COLORS = [
    (56, 56, 255),
    (255, 168, 56),
    (56, 255, 120),
    (56, 180, 255),
    (180, 56, 255),
]


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return a copy of `image` with boxes/labels drawn for each detection."""
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        color = _COLORS[det.class_id % len(_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{det.label} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(
            out, label, (x1 + 1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out
