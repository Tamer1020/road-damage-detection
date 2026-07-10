"""Thin wrapper around the Ultralytics YOLO model for inference.

Isolating the framework here means the rest of the app depends on plain
`Detection` objects, not on Ultralytics internals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from ultralytics import YOLO

from .detection import Detection


class RoadDamageModel:
    def __init__(self, weights: str | Path, device: Union[int, str] = "cpu") -> None:
        self.model = YOLO(str(weights))
        self.device = device
        # Ultralytics exposes the class-id -> name mapping learned at train time.
        self.names: dict[int, str] = self.model.names

    def predict_image(
        self,
        source: str | Path | np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ) -> list[Detection]:
        """Run detection on a single image and return structured detections."""
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )
        result = results[0]  # single image -> single result
        detections: list[Detection] = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.append(
                Detection(
                    class_id=cls_id,
                    label=self.names.get(cls_id, str(cls_id)),
                    confidence=float(box.conf[0]),
                    bbox=tuple(float(v) for v in box.xyxy[0].tolist()),
                )
            )
        return detections
