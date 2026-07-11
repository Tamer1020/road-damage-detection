from pathlib import Path

from road_damage.config import Config
from road_damage.model import RoadDamageModel
from road_damage.utils.io import read_image, write_image
from road_damage.visualize import draw_detections


def label_file_has_objects(label_path: Path) -> bool:
    if not label_path.exists():
        return False

    text = label_path.read_text(encoding="utf-8").strip()
    return bool(text)


def main():
    config = Config.load("configs/config_train.yaml")

    images_dir = Path("data/rdd2022/images/val")
    labels_dir = Path("data/rdd2022/labels/val")

    output_dir = Path("assets/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = RoadDamageModel(config.model.checkpoint, device=config.inference.device)

    saved = 0
    max_samples = 8

    summary_lines = [
        "# Prediction Samples",
        "",
        "Predictions generated on validation images from the RDD2022 Czech subset.",
        "",
        "| Image | Number of detections |",
        "|---|---:|",
    ]

    for image_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / f"{image_path.stem}.txt"

        # Prefer validation images that actually contain annotated damage.
        if not label_file_has_objects(label_path):
            continue

        detections = model.predict_image(
            image_path,
            conf=config.inference.conf,
            iou=config.inference.iou,
            imgsz=config.inference.imgsz,
        )

        # Save only images where the trained model predicts at least one box.
        if not detections:
            continue

        image = read_image(image_path)
        annotated = draw_detections(image, detections)

        output_path = output_dir / f"pred_{image_path.name}"
        write_image(output_path, annotated)

        print(f"Saved: {output_path} | detections: {len(detections)}")

        summary_lines.append(f"| `{output_path.name}` | {len(detections)} |")

        saved += 1
        if saved >= max_samples:
            break

    summary_path = output_dir / "prediction_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    if saved == 0:
        print("No prediction samples were saved. Try lowering confidence threshold.")
    else:
        print(f"Done. Saved {saved} prediction sample images.")
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()