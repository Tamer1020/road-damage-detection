"""Typed configuration loaded from YAML.

Using dataclasses gives us autocompletion, defaults, and a single source of
truth for the config schema. `Config.load` reads the YAML and fills any
missing keys from the dataclass defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    name: str = "road-damage-detection"
    seed: int = 42


@dataclass
class DataConfig:
    root: str = "data/rdd2022"
    yaml: str = "data/rdd2022/data.yaml"
    classes: list[str] = field(default_factory=lambda: ["D00", "D10", "D20", "D40"])


@dataclass
class ModelConfig:
    weights: str = "yolov8n.pt"
    checkpoint: str = "models/best.pt"


@dataclass
class TrainConfig:
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: int | str = 0
    project: str = "runs"
    name: str = "rdd_exp"
    patience: int = 20
    lr0: float = 0.01


@dataclass
class InferenceConfig:
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    device: int | str = "cpu"


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 10
    max_image_pixels: int = 20_000_000

    def __post_init__(self):
        if self.max_upload_mb <= 0 or self.max_image_pixels <= 0:
            raise ValueError("API upload and pixel limits must be positive")


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    api: ApiConfig = field(default_factory=ApiConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> Config:
        raw = raw or {}
        return cls(
            project=ProjectConfig(**raw.get("project", {})),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            inference=InferenceConfig(**raw.get("inference", {})),
            api=ApiConfig(**raw.get("api", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)
