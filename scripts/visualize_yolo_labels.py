from pathlib import Path

from road_damage.detection import Detection
from road_damage.utils.io import read_image, write_image
from road_damage.visualize import draw_detections

CLASSES = ["D00", "D10", "D20", "D40"]


def yolo_to_pixel_box(line: str, image_width: int, image_height: int):
    parts = line.strip().split()

    class_id = int(parts[0])
    x_center = float(parts[1]) * image_width
    y_center = float(parts[2]) * image_height
    box_width = float(parts[3]) * image_width
    box_height = float(parts[4]) * image_height

    x1 = x_center - box_width / 2
    y1 = y_center - box_height / 2
    x2 = x_center + box_width / 2
    y2 = y_center + box_height / 2

    return class_id, (x1, y1, x2, y2)


def main():
    split = "val"
    max_images = 8

    images_dir = Path("data/rdd2022/images") / split
    labels_dir = Path("data/rdd2022/labels") / split
    output_dir = Path("assets/ground_truth_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    for image_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        lines = label_path.read_text(encoding="utf-8").strip().splitlines()

        if not lines:
            continue

        image = read_image(image_path)
        image_height, image_width = image.shape[:2]

        detections = []

        for line in lines:
            class_id, bbox = yolo_to_pixel_box(line, image_width, image_height)

            detections.append(
                Detection(
                    class_id=class_id,
                    label=f"GT-{CLASSES[class_id]}",
                    confidence=1.0,
                    bbox=bbox,
                )
            )

        annotated = draw_detections(image, detections)

        output_path = output_dir / f"gt_{image_path.name}"
        write_image(output_path, annotated)

        print(f"Saved: {output_path}")
        saved += 1

        if saved >= max_images:
            break

    print(f"Done. Saved {saved} ground-truth sample images.")


if __name__ == "__main__":
    main()