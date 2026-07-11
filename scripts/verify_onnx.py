"""Verify that the exported ONNX model loads and runs with ONNX Runtime.

This script checks two things:
1. ONNX Runtime can create a valid inference session.
2. The ONNX model gives similar detections to the original PyTorch checkpoint
   on the same image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

from road_damage.config import Config
from road_damage.model import RoadDamageModel
from road_damage.utils.io import read_image


def inspect_onnx(onnx_path: Path) -> None:
    size_mb = onnx_path.stat().st_size / 1024 / 1024
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    print("\n--- ONNX Runtime session ---")
    print(f"File      : {onnx_path}")
    print(f"Size      : {size_mb:.2f} MB")
    print(f"Providers : {session.get_providers()}")

    for model_input in session.get_inputs():
        print(
            f"Input     : {model_input.name} | "
            f"shape={model_input.shape} | dtype={model_input.type}"
        )

    for model_output in session.get_outputs():
        print(
            f"Output    : {model_output.name} | "
            f"shape={model_output.shape} | dtype={model_output.type}"
        )

    print("ONNX Runtime session created successfully.")


def compare_predictions(
    pt_path: str,
    onnx_path: str,
    image_path: str,
    config: Config,
) -> None:
    image = read_image(image_path)

    prediction_kwargs = {
        "conf": config.inference.conf,
        "iou": config.inference.iou,
        "imgsz": config.inference.imgsz,
    }

    pytorch_model = RoadDamageModel(pt_path, device="cpu")
    onnx_model = RoadDamageModel(onnx_path, device="cpu")

    pytorch_detections = pytorch_model.predict_image(image, **prediction_kwargs)
    onnx_detections = onnx_model.predict_image(image, **prediction_kwargs)

    print("\n--- PyTorch vs ONNX Runtime ---")
    print(f"Image              : {image_path}")
    print(f"PyTorch detections : {len(pytorch_detections)}")
    print(f"ONNX detections    : {len(onnx_detections)}")

    if len(pytorch_detections) != len(onnx_detections):
        print("WARNING: Detection counts are different.")
        print("This does not always mean the export failed, but it should be inspected.")
        return

    if not pytorch_detections:
        print("No detections found on this image.")
        print("Use another image with visible damage for a stronger verification.")
        return

    pytorch_sorted = sorted(
        pytorch_detections,
        key=lambda detection: -detection.confidence,
    )
    onnx_sorted = sorted(
        onnx_detections,
        key=lambda detection: -detection.confidence,
    )

    labels_match = all(
        pytorch_detection.class_id == onnx_detection.class_id
        for pytorch_detection, onnx_detection in zip(pytorch_sorted, onnx_sorted, strict=True)
    )

    max_confidence_delta = max(
        abs(pytorch_detection.confidence - onnx_detection.confidence)
        for pytorch_detection, onnx_detection in zip(pytorch_sorted, onnx_sorted, strict=True)

    )

    max_box_delta = max(
        float(
            np.max(
                np.abs(
                    np.array(pytorch_detection.bbox)
                    - np.array(onnx_detection.bbox)
                )
            )
        )
        for pytorch_detection, onnx_detection in zip(pytorch_sorted, onnx_sorted, strict=True)

    )

    print(f"Class labels match     : {labels_match}")
    print(f"Max confidence delta   : {max_confidence_delta:.6f}")
    print(f"Max box coordinate diff: {max_box_delta:.4f} px")
    print("ONNX verification completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify exported ONNX model.")
    parser.add_argument("--config", default="configs/config_train.yaml")
    parser.add_argument("--pt", required=True, help="Path to PyTorch .pt checkpoint.")
    parser.add_argument("--onnx", required=True, help="Path to exported ONNX model.")
    parser.add_argument("--image", required=True, help="Image used for comparison.")
    args = parser.parse_args()

    onnx_path = Path(args.onnx)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    config = Config.load(args.config)

    inspect_onnx(onnx_path)
    compare_predictions(
        pt_path=args.pt,
        onnx_path=args.onnx,
        image_path=args.image,
        config=config,
    )


if __name__ == "__main__":
    main()