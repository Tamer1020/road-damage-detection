"""Validate VOC annotations and publish a reproducible, disjoint YOLO split.

Use a new or empty destination. Existing datasets are never merged or deleted.
The original sorted-file / seeded-shuffle split is preserved on valid inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ["D00", "D10", "D20", "D40"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}


def _find_dir(src: Path, candidates: list[str]) -> Path:
    for name in candidates:
        path = src / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"None of {candidates} found under {src}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert_annotation(xml_path: Path) -> list[str]:
    """Convert target classes, rejecting invalid sizes and target boxes.

    Unknown classes are ignored, as in the original four-class baseline.
    No coordinate clipping or automatic annotation repair is performed.
    """
    try:
        root = ET.parse(xml_path).getroot()
        width = int(root.findtext("size/width", "0"))
        height = int(root.findtext("size/height", "0"))
        if width <= 0 or height <= 0:
            raise ValueError("image width and height must be positive")
        lines = []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name not in CLASS_TO_ID:
                continue
            coords = [float(obj.findtext(f"bndbox/{key}", "nan"))
                      for key in ("xmin", "ymin", "xmax", "ymax")]
            xmin, ymin, xmax, ymax = coords
            if not all(math.isfinite(v) for v in coords):
                raise ValueError("box coordinates must be finite numbers")
            if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
                raise ValueError(f"invalid {name} box {coords} for {width}x{height}")
            values = ((xmin + xmax) / (2 * width), (ymin + ymax) / (2 * height),
                      (xmax - xmin) / width, (ymax - ymin) / height)
            lines.append(f"{CLASS_TO_ID[name]} " + " ".join(f"{v:.6f}" for v in values))
        return lines
    except (ET.ParseError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid annotation {xml_path.name}: {exc}") from exc


def prepare(src: Path, dst: Path, val_ratio: float, seed: int) -> dict:
    """Prepare once, record input/output hashes, and return the split manifest."""
    src, dst = Path(src).resolve(), Path(dst).resolve()
    if not math.isfinite(val_ratio) or not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1 (exclusive)")
    if dst.is_relative_to(src) or src.is_relative_to(dst):
        raise ValueError("Source and destination must not contain each other")
    if dst.exists() and (not dst.is_dir() or any(dst.iterdir())):
        raise FileExistsError(f"Destination is not empty: {dst}. Choose a new --dst.")
    ann_dir = _find_dir(src, ["annotations/xmls", "Annotations", "annotations"])
    img_dir = _find_dir(src, ["images", "JPEGImages"])
    xml_files = sorted(ann_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML annotations found in {ann_dir}")
    random.Random(seed).shuffle(xml_files)
    n_val = int(len(xml_files) * val_ratio)
    if not 0 < n_val < len(xml_files):
        raise ValueError("This val_ratio must produce at least one image in each split")
    splits = {"val": xml_files[:n_val], "train": xml_files[n_val:]}
    records: dict[str, list[tuple[Path, str, dict]]] = {"train": [], "val": []}
    seen: dict[str, tuple[str, str]] = {}
    for split, files in splits.items():
        for xml_path in files:
            image_path = img_dir / f"{xml_path.stem}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing image for {xml_path.name}: {image_path}")
            lines = convert_annotation(xml_path)
            label = "\n".join(lines)
            digest = _sha256(image_path)
            if digest in seen and seen[digest][0] != split:
                raise ValueError(f"Identical image content crosses splits: "
                                 f"{seen[digest][1]} and {image_path.name}")
            seen[digest] = (split, image_path.name)
            record = {
                "id": xml_path.stem, "image": f"images/{split}/{image_path.name}",
                "image_sha256": digest, "annotation_sha256": _sha256(xml_path),
                "label_sha256": hashlib.sha256(label.encode("utf-8")).hexdigest(),
                "objects": len(lines),
            }
            records[split].append((image_path, label, record))
    manifest = {
        "schema_version": 1, "seed": seed, "val_ratio": val_ratio, "classes": CLASSES,
        "split_method": "sorted XML names; Python random.Random(seed).shuffle",
        "duplicate_check": "exact image SHA-256 across splits; not near-duplicate detection",
        "counts": {split: len(items) for split, items in records.items()},
        "splits": {split: [item[2] for item in sorted(items, key=lambda x: x[2]["id"])]
                   for split, items in records.items()},
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dst.name}-prepare-", dir=dst.parent))
    try:
        for split, items in records.items():
            (staging / "images" / split).mkdir(parents=True)
            (staging / "labels" / split).mkdir(parents=True)
            for image_path, label, record in items:
                shutil.copy2(image_path, staging / record["image"])
                target = staging / "labels" / split / f"{record['id']}.txt"
                target.write_text(label, encoding="utf-8")
        (staging / "split_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        descriptor = {"path": str(dst), "train": "images/train", "val": "images/val",
                      "names": dict(enumerate(CLASSES))}
        # JSON is valid YAML and avoids an extra preparation dependency.
        (staging / "data.yaml").write_text(json.dumps(descriptor, indent=2) + "\n",
                                           encoding="utf-8")
        if dst.exists():
            dst.rmdir()  # Only an empty directory can be removed here.
        staging.rename(dst)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    for split, count in manifest["counts"].items():
        print(f"{split}: {count} images")
    print(f"Manifest: {dst / 'split_manifest.json'}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate VOC labels and create a new YOLO split.")
    parser.add_argument("--src", required=True, help="RDD country training folder.")
    parser.add_argument("--dst", default="data/rdd2022", help="New or empty output folder.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        prepare(Path(args.src), Path(args.dst), args.val_ratio, args.seed)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
