"""Latency / FPS benchmark for the road damage detector.

Measures pure inference time (images are decoded once, up front, so disk IO is
not counted). Reports latency percentiles and average throughput — the numbers
you cite when arguing a model fits an edge device's real-time budget.

The --weights and --device flags override the config so the same image set can
be benchmarked across backends (PyTorch .pt vs ONNX .onnx) and devices (GPU vs
CPU) without editing config.yaml between runs.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from .config import Config
from .model import RoadDamageModel
from .utils.io import read_image
from .utils.logging import get_logger

logger = get_logger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_images(image: str | None, folder: str | None) -> list[Path]:
    paths: list[Path] = []
    if image:
        paths.append(Path(image))
    if folder:
        folder_path = Path(folder)
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder_path}")
        paths.extend(
            sorted(p for p in folder_path.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        )

    if not paths:
        raise ValueError("No images found. Provide --image or a non-empty --folder.")

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Image(s) not found: {missing}")
    return paths


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile without numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _backend_of(weights: str) -> str:
    return "ONNX Runtime" if str(weights).endswith(".onnx") else "PyTorch"


def _summarize(
    latencies_ms: list[float],
    config: Config,
    weights: str,
    device: str,
    n_images: int,
    warmup: int,
    runs: int,
) -> dict:
    mean_ms = statistics.mean(latencies_ms)
    return {
        "backend": _backend_of(weights),
        "weights": str(weights),
        "device": str(device),
        "imgsz": config.inference.imgsz,
        "conf": config.inference.conf,
        "iou": config.inference.iou,
        "num_images": n_images,
        "warmup": warmup,
        "runs": runs,
        "machine": f"{platform.system()} {platform.machine()}",
        "latency_ms": {
            "mean": round(mean_ms, 3),
            "median": round(statistics.median(latencies_ms), 3),
            "min": round(min(latencies_ms), 3),
            "max": round(max(latencies_ms), 3),
            "p95": round(_percentile(latencies_ms, 95), 3),
        },
        "fps_mean": round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0,
    }


def _print_summary(s: dict) -> None:
    lat = s["latency_ms"]
    print("\n=== Benchmark summary ===")
    print(f"Backend       : {s['backend']}")
    print(f"Weights       : {s['weights']}")
    print(f"Device        : {s['device']}")
    print(f"Image size    : {s['imgsz']}")
    print(f"Images        : {s['num_images']}")
    print(f"Warmup / Runs : {s['warmup']} / {s['runs']}")
    print(
        f"Latency (ms)  : mean {lat['mean']} | median {lat['median']} | "
        f"min {lat['min']} | max {lat['max']} | p95 {lat['p95']}"
    )
    print(f"Throughput    : {s['fps_mean']} FPS")
    print("=========================\n")


def benchmark(
    config: Config,
    image: str | None = None,
    folder: str | None = None,
    warmup: int = 5,
    runs: int = 50,
    save_json: bool = False,
    tag: str | None = None,
    weights: str | None = None,
    device: str | None = None,
    output_dir: str = "outputs",
) -> dict:
    # CLI flags win over config; config supplies the fallback.
    weights = weights or config.model.checkpoint
    device = device if device is not None else str(config.inference.device)

    paths = _collect_images(image, folder)
    images = [read_image(p) for p in paths]  # decode once; keep IO out of timing
    logger.info("Loaded %d image(s) for benchmarking", len(images))
    logger.info("Backend=%s | weights=%s | device=%s", _backend_of(weights), weights, device)

    model = RoadDamageModel(weights, device=device)

    def infer(img):
        return model.predict_image(
            img,
            conf=config.inference.conf,
            iou=config.inference.iou,
            imgsz=config.inference.imgsz,
        )

    # Warmup runs are not timed — they absorb lazy CUDA/kernel/graph init.
    for i in range(warmup):
        infer(images[i % len(images)])

    latencies_ms: list[float] = []
    for i in range(runs):
        img = images[i % len(images)]
        start = time.perf_counter()
        infer(img)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    summary = _summarize(
        latencies_ms, config, weights, device, len(images), warmup, runs
    )
    _print_summary(summary)

    if save_json:
        name = f"benchmark_{tag}.json" if tag else "benchmark.json"
        out_path = Path(output_dir) / name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Benchmark results saved to %s", out_path)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference latency / FPS.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--image", default=None, help="Path to a single image.")
    parser.add_argument("--folder", default=None, help="Path to a folder of images.")
    parser.add_argument(
        "--weights", default=None,
        help="Override model weights (.pt or .onnx). Default: model.checkpoint.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Override device, e.g. 'cpu' or '0'. Default: inference.device.",
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations (not timed)."
    )
    parser.add_argument("--runs", type=int, default=50, help="Timed iterations.")
    parser.add_argument(
        "--save-json", action="store_true",
        help="Save results to outputs/benchmark[_TAG].json.",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Suffix for the JSON filename, e.g. 'pt_gpu' or 'onnx_cpu'.",
    )
    args = parser.parse_args()

    # Fail clearly (exit code 2 + usage) if the user gave no input source.
    if not args.image and not args.folder:
        parser.error("Provide --image or --folder.")

    config = Config.load(args.config)
    benchmark(
        config,
        image=args.image,
        folder=args.folder,
        warmup=args.warmup,
        runs=args.runs,
        save_json=args.save_json,
        tag=args.tag,
        weights=args.weights,
        device=args.device,
    )


if __name__ == "__main__":
    main()