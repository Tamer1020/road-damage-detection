"""Lightweight detection type shared across the codebase.

Kept free of heavy imports (no torch/ultralytics) so it can be used by the
visualizer and unit tests without loading the model stack.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 2) for v in self.bbox],
        }
