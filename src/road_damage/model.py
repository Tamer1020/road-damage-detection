"""Thin wrapper around the Ultralytics YOLO model for inference.

Isolating the framework here means the rest of the app depends on plain
`Detection` objects, not on Ultralytics internals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .detection import Detection


class RoadDamageModel:
    def __init__(self, weights: str | Path, device: int | str = "cpu") -> None:
        if not Path(weights).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {weights}. Download the release first.")
        self.model = YOLO(str(weights), task="detect")
        self.device = device
        # Read names after prediction: accessing YOLO.names earlier can initialize
        # an ONNX backend before the requested device has been applied.
        self.names: dict[int, str] = {}

    def predict_image(
        self,
        source: str | Path | np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        rect: bool = True,
    ) -> list[Detection]:
        """Run detection on a single image and return structured detections."""
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            rect=rect,
            device=self.device,
            verbose=False,
        )
        result = results[0]  # single image -> single result
        self.names = result.names
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
