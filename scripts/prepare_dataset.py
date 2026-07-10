"""Convert RDD (PASCAL VOC XML) annotations to YOLO format and split.

Input layout (typical RDD country folder):
    <src>/
    ├── annotations/xmls/*.xml   (or Annotations/*.xml)
    └── images/*.jpg             (or JPEGImages/*.jpg)

Output layout:
    <dst>/
    ├── images/{train,val}/*.jpg
    └── labels/{train,val}/*.txt

Usage:
    python scripts/prepare_dataset.py --src data/raw/Japan --dst data/rdd2022 \
        --val-ratio 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# Class order MUST match configs/config.yaml -> data.classes
CLASSES = ["D00", "D10", "D20", "D40"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}


def _find_dir(src: Path, candidates: list[str]) -> Path:
    for name in candidates:
        path = src / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"None of {candidates} found under {src}")


def convert_annotation(xml_path: Path) -> list[str]:
    """Return YOLO label lines for one VOC XML file."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    lines: list[str] = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS_TO_ID:
            continue  # ignore classes we don't model
        class_id = CLASS_TO_ID[name]
        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        # YOLO wants normalized center-x, center-y, width, height.
        xc = ((xmin + xmax) / 2) / width
        yc = ((ymin + ymax) / 2) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def prepare(src: Path, dst: Path, val_ratio: float, seed: int) -> None:
    ann_dir = _find_dir(src, ["annotations/xmls", "Annotations", "annotations"])
    img_dir = _find_dir(src, ["images", "JPEGImages"])

    xml_files = sorted(ann_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML annotations found in {ann_dir}")

    random.seed(seed)
    random.shuffle(xml_files)
    n_val = int(len(xml_files) * val_ratio)
    splits = {"val": xml_files[:n_val], "train": xml_files[n_val:]}

    for split, files in splits.items():
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

        kept = 0
        for xml_path in files:
            image_path = img_dir / f"{xml_path.stem}.jpg"
            if not image_path.exists():
                continue
            lines = convert_annotation(xml_path)
            # Copy image and write label (empty label = background image).
            shutil.copy2(image_path, dst / "images" / split / image_path.name)
            label_path = dst / "labels" / split / f"{xml_path.stem}.txt"
            label_path.write_text("\n".join(lines), encoding="utf-8")
            kept += 1
        print(f"{split}: {kept} images")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RDD VOC XML to YOLO format.")
    parser.add_argument("--src", required=True, help="RDD country folder.")
    parser.add_argument("--dst", default="data/rdd2022", help="Output dataset root.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(Path(args.src), Path(args.dst), args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
