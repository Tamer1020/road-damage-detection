"""Evaluation entry point: reports mAP metrics on a chosen split."""

from __future__ import annotations

import argparse

from ultralytics import YOLO

from .config import Config
from .dataset import build_data_yaml
from .utils.logging import get_logger

logger = get_logger(__name__)


def evaluate(config: Config, split: str = "val"):
    data_yaml = build_data_yaml(config)
    model = YOLO(config.model.checkpoint)
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=config.inference.imgsz,
        device=config.inference.device,
    )
    logger.info("mAP50-95: %.4f | mAP50: %.4f", metrics.box.map, metrics.box.map50)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the road damage detector.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()
    evaluate(Config.load(args.config), split=args.split)


if __name__ == "__main__":
    main()
