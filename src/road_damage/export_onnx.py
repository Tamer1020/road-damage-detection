"""Export the trained YOLO checkpoint to ONNX.

ONNX is a portable, framework-agnostic model format. Exporting lets the same
trained weights run under ONNX Runtime, TensorRT, OpenVINO, etc. — the usual
path for edge / embedded deployment where a full PyTorch install is undesirable.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from .config import Config
from .utils.logging import get_logger

logger = get_logger(__name__)


def export_onnx(
    config: Config,
    output: str | None = None,
    imgsz: int | None = None,
    opset: int | None = None,
    dynamic: bool = False,
) -> Path:
    """Export config.model.checkpoint to ONNX and return the output path."""
    checkpoint = Path(config.model.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Train first with `rdd-train`, "
            f"or point model.checkpoint in the config at a valid .pt file."
        )

    imgsz = imgsz or config.inference.imgsz
    logger.info("Loading checkpoint: %s", checkpoint)
    model = YOLO(str(checkpoint))

    # Only pass opset when the user set it; otherwise let Ultralytics pick a
    # version compatible with the installed torch/onnx (avoids forced failures).
    export_kwargs: dict = {"format": "onnx", "imgsz": imgsz, "dynamic": dynamic}
    if opset is not None:
        export_kwargs["opset"] = opset

    logger.info(
        "Exporting to ONNX (imgsz=%s, opset=%s, dynamic=%s)",
        imgsz, opset if opset is not None else "auto", dynamic,
    )
    exported = Path(model.export(**export_kwargs))

    # Ultralytics writes the .onnx next to the checkpoint; move it if the user
    # asked for a specific output path.
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != exported.resolve():
            shutil.move(str(exported), str(target))
        exported = target

    logger.info("ONNX model saved to: %s", exported)
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the trained model to ONNX.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--output", default=None,
        help="Output .onnx path (default: next to the checkpoint).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=None,
        help="Export image size (default: inference.imgsz from config).",
    )
    parser.add_argument(
        "--opset", type=int, default=None,
        help="ONNX opset version (default: let Ultralytics choose).",
    )
    parser.add_argument(
        "--dynamic", action="store_true",
        help="Export with dynamic input axes (variable batch/size).",
    )
    args = parser.parse_args()

    config = Config.load(args.config)
    export_onnx(
        config,
        output=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
    )


if __name__ == "__main__":
    main()