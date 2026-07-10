"""Training entry point.

Fine-tunes a pretrained YOLO model on the road-damage dataset using the
hyperparameters from config.yaml.
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO

from .config import Config
from .dataset import build_data_yaml
from .utils.logging import get_logger

logger = get_logger(__name__)


def train(config: Config):
    data_yaml = build_data_yaml(config)
    logger.info("Dataset descriptor: %s", data_yaml)

    model = YOLO(config.model.weights)
    results = model.train(
        data=str(data_yaml),
        epochs=config.train.epochs,
        imgsz=config.train.imgsz,
        batch=config.train.batch,
        device=config.train.device,
        project=config.train.project,
        name=config.train.name,
        patience=config.train.patience,
        lr0=config.train.lr0,
        seed=config.project.seed,
    )
    logger.info("Training finished. Best weights: %s", model.trainer.best)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the road damage detector.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(Config.load(args.config))


if __name__ == "__main__":
    main()
