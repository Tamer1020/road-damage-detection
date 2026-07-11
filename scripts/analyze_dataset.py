from collections import Counter
from pathlib import Path

CLASSES = ["D00", "D10", "D20", "D40"]


def count_images(images_dir: Path) -> int:
    return len(list(images_dir.glob("*.jpg")))


def analyze_labels(labels_dir: Path) -> tuple[int, int, Counter]:
    label_files = sorted(labels_dir.glob("*.txt"))

    empty_files = 0
    class_counter = Counter()

    for label_path in label_files:
        text = label_path.read_text(encoding="utf-8").strip()

        if not text:
            empty_files += 1
            continue

        for line in text.splitlines():
            parts = line.split()
            class_id = int(parts[0])
            class_name = CLASSES[class_id]
            class_counter[class_name] += 1

    return len(label_files), empty_files, class_counter


def main():
    dataset_root = Path("data/rdd2022")

    train_images_dir = dataset_root / "images" / "train"
    val_images_dir = dataset_root / "images" / "val"

    train_labels_dir = dataset_root / "labels" / "train"
    val_labels_dir = dataset_root / "labels" / "val"

    train_images = count_images(train_images_dir)
    val_images = count_images(val_images_dir)

    train_label_files, train_empty_labels, train_class_counts = analyze_labels(train_labels_dir)
    val_label_files, val_empty_labels, val_class_counts = analyze_labels(val_labels_dir)

    total_class_counts = train_class_counts + val_class_counts

    print("\n=== Dataset Summary ===")
    print(f"Train images      : {train_images}")
    print(f"Validation images : {val_images}")
    print(f"Train label files : {train_label_files}")
    print(f"Val label files   : {val_label_files}")
    print(f"Empty train labels: {train_empty_labels}")
    print(f"Empty val labels  : {val_empty_labels}")

    print("\n=== Object Count per Class ===")
    for class_name in CLASSES:
        print(f"{class_name}: {total_class_counts[class_name]}")

    output_dir = Path("assets/dataset_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "dataset_summary.md"

    lines = [
        "# Dataset Summary",
        "",
        "RDD2022 Czech subset after Pascal VOC XML to YOLO TXT conversion.",
        "",
        "## Image Counts",
        "",
        "| Split | Images | Label files | Empty label files |",
        "|---|---:|---:|---:|",
        f"| Train | {train_images} | {train_label_files} | {train_empty_labels} |",
        f"| Validation | {val_images} | {val_label_files} | {val_empty_labels} |",
        "",
        "## Object Count per Class",
        "",
        "| Class | Meaning | Count |",
        "|---|---|---:|",
        f"| D00 | Longitudinal crack | {total_class_counts['D00']} |",
        f"| D10 | Transverse crack | {total_class_counts['D10']} |",
        f"| D20 | Alligator crack | {total_class_counts['D20']} |",
        f"| D40 | Pothole | {total_class_counts['D40']} |",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()