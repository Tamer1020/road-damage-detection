"""Single-image inference CLI.

Loads the trained checkpoint, runs detection, prints JSON, and optionally
writes an annotated image to the outputs directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Config
from .model import RoadDamageModel
from .utils.io import read_image, write_image
from .utils.logging import get_logger
from .visualize import draw_detections

logger = get_logger(__name__)


def run(image_path: str, config: Config, save_annotated: bool = False,
        output_dir: str = "outputs") -> list[dict]:
    model = RoadDamageModel(config.model.checkpoint, device=config.inference.device)
    detections = model.predict_image(
        image_path,
        conf=config.inference.conf,
        iou=config.inference.iou,
        imgsz=config.inference.imgsz,
    )
    logger.info("Found %d damage instance(s) in %s", len(detections), image_path)

    if save_annotated:
        image = read_image(image_path)
        annotated = draw_detections(image, detections)
        out_path = Path(output_dir) / f"annotated_{Path(image_path).name}"
        write_image(out_path, annotated)
        logger.info("Annotated image saved to %s", out_path)

    return [d.to_dict() for d in detections]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on one image.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument("--save", action="store_true", help="Save annotated output.")
    args = parser.parse_args()

    config = Config.load(args.config)
    results = run(args.image, config, save_annotated=args.save)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
