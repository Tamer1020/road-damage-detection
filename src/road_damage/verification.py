"""Strict, order-independent comparison of post-NMS detection sets.

Matching uses class, IoU, confidence and coordinate tolerances, then finds a
one-to-one assignment. Empty/empty images alone cannot verify an export.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config
from .detection import Detection


@dataclass(frozen=True)
class Tolerances:
    min_iou: float = 0.99
    max_confidence_delta: float = 1e-4
    max_box_delta_px: float = 0.1

    def __post_init__(self):
        if not math.isfinite(self.min_iou) or not 0 < self.min_iou <= 1:
            raise ValueError("min_iou must be in (0, 1]")
        for value in (self.max_confidence_delta, self.max_box_delta_px):
            if not math.isfinite(value) or value < 0:
                raise ValueError("delta tolerances must be finite and non-negative")


def box_iou(a, b) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * (
        max(0, min(a[3], b[3]) - max(a[1], b[1])))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _valid(d: Detection) -> bool:
    return (len(d.bbox) == 4 and all(math.isfinite(v) for v in (*d.bbox, d.confidence))
            and 0 <= d.confidence <= 1 and d.bbox[2] > d.bbox[0]
            and d.bbox[3] > d.bbox[1])


def compare_detections(reference: list[Detection], candidate: list[Detection],
                       tolerances: Tolerances | None = None) -> dict:
    tolerances = tolerances or Tolerances()
    report = {"reference_count": len(reference), "candidate_count": len(candidate),
              "passed": False, "matches": []}
    if not all(_valid(d) for d in reference + candidate):
        return {**report, "reason": "invalid_or_nonfinite_detection"}
    if len(reference) != len(candidate):
        return {**report, "reason": "detection_count_mismatch"}
    if not reference:
        return {**report, "passed": True, "reason": "both_empty"}

    edges = {}
    differences = {}
    for i, a in enumerate(reference):
        edges[i] = []
        for j, b in enumerate(candidate):
            if a.class_id != b.class_id or a.label != b.label:
                continue
            overlap = box_iou(a.bbox, b.bbox)
            confidence = abs(a.confidence - b.confidence)
            box = max(abs(x - y) for x, y in zip(a.bbox, b.bbox, strict=True))
            if (overlap >= tolerances.min_iou and confidence <= tolerances.max_confidence_delta
                    and box <= tolerances.max_box_delta_px):
                edges[i].append(j)
                differences[i, j] = {"iou": overlap, "confidence_delta": confidence,
                                     "box_delta_px": box}
        edges[i].sort(key=lambda j: (-differences[i, j]["iou"], j))

    # Augmenting paths avoid greedy matching failures when several boxes overlap.
    owner: dict[int, int] = {}

    def assign(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or assign(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(len(reference)):
        if not assign(i, set()):
            return {**report, "reason": "no_class_and_tolerance_preserving_assignment"}
    matches = [{"reference_index": i, "candidate_index": j, **differences[i, j]}
               for j, i in sorted(owner.items(), key=lambda item: item[1])]
    return {**report, "passed": True, "reason": "matched", "matches": matches}


def summarize_reports(reports: list[dict], min_nonempty: int = 1) -> dict:
    if min_nonempty < 1:
        raise ValueError("min_nonempty must be at least 1")
    nonempty = sum(r["reference_count"] > 0 for r in reports)
    failed = sum(not r["passed"] for r in reports)
    enough = nonempty >= min_nonempty
    return {"passed": bool(reports) and failed == 0 and enough,
            "images": len(reports), "failed_images": failed,
            "nonempty_reference_images": nonempty,
            "required_nonempty_images": min_nonempty,
            "status": "failed" if failed else "passed" if enough else "inconclusive"}


def verify(config: Config, pt: Path, onnx: Path, images: list[Path],
           tolerances: Tolerances, min_nonempty: int = 1) -> dict:
    import onnxruntime as ort

    from .model import RoadDamageModel
    from .utils.io import read_image

    for path in (pt, onnx):
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")
    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    inputs = [{"name": x.name, "shape": x.shape, "dtype": x.type}
              for x in session.get_inputs()]
    providers = session.get_providers()
    del session
    models = [RoadDamageModel(path, device="cpu") for path in (pt, onnx)]
    kwargs = {"conf": config.inference.conf, "iou": config.inference.iou,
              "imgsz": config.inference.imgsz, "rect": False}
    reports = []
    for path in images:
        image = read_image(path)
        reference, candidate = [model.predict_image(image, **kwargs) for model in models]
        report = compare_detections(reference, candidate, tolerances)
        reports.append({"image": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **report})
    packages = {}
    for name in ("ultralytics", "torch", "torchvision", "onnxruntime", "numpy"):
        packages[name] = importlib.metadata.version(name)
    return {
        "schema_version": 1, "scope": "post-NMS detection parity; not an accuracy evaluation",
        "summary": summarize_reports(reports, min_nonempty),
        "tolerances": asdict(tolerances), "inference": {**kwargs, "device": "cpu"},
        "models": {name: {"file": path.name,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in (("pytorch", pt), ("onnx", onnx))},
        "onnx": {"inputs": inputs, "providers": providers},
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "packages": packages},
        "images": reports,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify PyTorch/ONNX detection parity.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--pt", required=True)
    parser.add_argument("--onnx", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--folder")
    parser.add_argument("--min-iou", type=float, default=0.99)
    parser.add_argument("--max-confidence-delta", type=float, default=1e-4)
    parser.add_argument("--max-box-delta-px", type=float, default=0.1)
    parser.add_argument("--min-nonempty", type=int, default=1)
    parser.add_argument("--output", default="outputs/onnx_verification.json")
    args = parser.parse_args(argv)
    try:
        limits = Tolerances(args.min_iou, args.max_confidence_delta, args.max_box_delta_px)
        if args.min_nonempty < 1:
            raise ValueError("--min-nonempty must be at least 1")
        if args.folder and not Path(args.folder).is_dir():
            raise FileNotFoundError(f"Image folder not found: {args.folder}")
        images = [Path(args.image)] if args.image else sorted(
            p for p in Path(args.folder).iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not images:
            raise ValueError("No images found")
        report = verify(Config.load(args.config), Path(args.pt), Path(args.onnx), images,
                        limits, args.min_nonempty)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(report["summary"], indent=2))
        print(f"Report: {output}")
        return 0 if report["summary"]["passed"] else 1
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Verification could not run: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
