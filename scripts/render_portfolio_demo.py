"""Render a 60-second evidence replay; requires Pillow, ffmpeg and DejaVu fonts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

W, H, FPS = 1280, 720, 25
BG, PANEL, LINE = "#0b1422", "#14243a", "#31435b"
WHITE, MUTED, GREEN, AMBER = "#edf3fc", "#aabbd0", "#6ce4c7", "#ffc982"
CHAPTERS = [
    (0, 6, "Road damage detection: released YOLOv8n weights and real road images."),
    (6, 18, "A real HTTP request returns detections, coordinates and confidence scores."),
    (18, 30, "An annotated crack is missed. No detections does not mean no damage."),
    (30, 42, "The published D20 reference is not recovered by the D00 prediction."),
    (42, 48, "PyTorch and ONNX agree within explicit tolerances on scoped fixtures."),
    (48, 54, "Historical CPU throughput and validation accuracy measure different things."),
    (54, 60, "Inspect the evidence, reproduce the run and explore the API on GitHub."),
]


class Renderer:
    def __init__(self, capture: Path, fonts: Path):
        self.capture = capture
        self.fonts = fonts
        self.manifest = json.loads((capture / "manifest.json").read_text())
        self.responses = {
            name: json.loads((capture / f"{name}.json").read_text())
            for name in ("crack", "missed_damage", "class_confusion")
        }
        for case in self.manifest["cases"]:
            for path, expected in (
                (Path(case["input"]), case["input_sha256"]),
                (capture / case["annotation"], case["annotation_sha256"]),
            ):
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Capture evidence changed: {path}")
        if (
            self.responses["crack"]["count"] < 1
            or self.responses["missed_damage"]["count"] != 0
            or not any(d["label"] == "D00" for d in self.responses["class_confusion"]["detections"])
        ):
            raise ValueError(
                "The captured behavior changed; revise the storyboard before rendering"
            )
        self.parity = [
            json.loads(Path(f"assets/evaluation/onnx_parity_{name}.json").read_text())
            for name in ("benchmark_images", "demo_images")
        ]
        for report in self.parity:
            if not report["summary"]["passed"]:
                raise ValueError("Cannot present an unsuccessful parity check as passing")
            if report["models"]["pytorch"]["sha256"] != self.manifest["model"]["sha256"]:
                raise ValueError("Capture and verification weights differ")
        self.benchmark = json.loads(Path("assets/evaluation/benchmark_onnx_cpu.json").read_text())

    def font(self, size, bold=False, mono=False):
        name = (
            "DejaVuSansMono.ttf" if mono else ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        )
        return ImageFont.truetype(str(self.fonts / name), size)

    def text(self, draw, xy, value, size=24, color=WHITE, bold=False, mono=False):
        draw.text(xy, str(value), font=self.font(size, bold, mono), fill=color)

    def lines(self, draw, xy, text, width, size=22, color=MUTED, bold=False):
        x, y = xy
        line = ""
        for word in text.split():
            candidate = f"{line} {word}".strip()
            if self.font(size, bold).getlength(candidate) > width and line:
                self.text(draw, (x, y), line, size, color, bold)
                y += size + 10
                line = word
            else:
                line = candidate
        if line:
            self.text(draw, (x, y), line, size, color, bold)
        return y + size + 10

    def base(self, title, subtitle):
        image = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(image)
        self.text(draw, (40, 24), "ROAD DAMAGE DETECTION  /  ENGINEERING DEMO", 15, GREEN, True)
        self.text(draw, (40, 54), title, 35, WHITE, True)
        self.text(draw, (40, 103), subtitle, 19, MUTED)
        draw.line((40, 668, 1240, 668), fill=LINE, width=1)
        self.text(draw, (40, 684), "Tamer1020 / road-damage-detection", 15, MUTED)
        self.text(draw, (735, 684), "Recorded evidence replay · chapter timing edited", 15, MUTED)
        return image, draw

    def pane(self, image, draw, box, path, label, color=GREEN):
        x, y, right, bottom = box
        draw.rounded_rectangle(box, radius=12, fill=PANEL, outline=LINE, width=1)
        self.text(draw, (x + 18, y + 12), label, 19, color, True)
        with Image.open(path) as source:
            picture = ImageOps.contain(source.convert("RGB"), (right - x - 24, bottom - y - 56))
        image.paste(picture, (x + (right - x - picture.width) // 2, y + 48))

    def comparison(self, title, subtitle, left, right, left_label, right_label, note):
        image, draw = self.base(title, subtitle)
        self.pane(image, draw, (40, 145, 612, 610), left, left_label)
        self.pane(image, draw, (668, 145, 1240, 610), right, right_label, AMBER)
        self.text(draw, (40, 631), note, 20, WHITE)
        return image

    def scene(self, index):
        cap = self.capture
        if index == 0:
            return self.comparison(
                "From a road image to an inspectable prediction",
                "YOLOv8n · OpenCV · FastAPI · ONNX Runtime",
                "assets/demo/inputs/Czech_000047.jpg",
                cap / "crack.jpg",
                "INPUT IMAGE",
                "CAPTURED API OUTPUT",
                "Released weights. Real HTTP requests. Successes and failures shown.",
            )
        if index == 1:
            image, draw = self.base(
                "A prediction you can inspect",
                "POST /predict?annotate=true  ·  HTTP 200  ·  CPU inference",
            )
            self.pane(image, draw, (40, 145, 572, 630), cap / "crack.jpg", "RETURNED ANNOTATION")
            draw.rounded_rectangle((600, 145, 1240, 630), radius=12, fill=PANEL, outline=LINE)
            response = self.responses["crack"]
            detection = response["detections"][0]
            lines = [
                "{",
                f'  "filename": "{response["filename"]}",',
                f'  "count": {response["count"]},',
                '  "detections": [{',
                f'    "label": "{detection["label"]}",',
                f'    "confidence": {detection["confidence"]},',
                '    "bbox": ' + json.dumps(detection["bbox"]),
                "  }],",
                f'  "inference_ms": {response["inference_ms"]}',
                "}",
            ]
            for i, line in enumerate(lines):
                self.text(draw, (620, 164 + i * 33), line, 19, WHITE, mono=True)
            self.lines(
                draw,
                (620, 523),
                "Confidence is a detection score, not model accuracy.",
                590,
                23,
                AMBER,
                True,
            )
            self.text(draw, (620, 598), "Individual call timing; not a benchmark.", 19, MUTED)
            return image
        if index == 2:
            return self.comparison(
                "No detections does not mean no damage",
                "Czech_000006 · original benchmark input · confidence threshold 0.25",
                "assets/ground_truth_samples/gt_Czech_000006.jpg",
                cap / "missed_damage.jpg",
                "GROUND TRUTH: D00",
                "API: 0 DETECTIONS",
                "The marked crack is missed. This is a model limitation to investigate.",
            )
        if index == 3:
            return self.comparison(
                "A box can still miss the annotated damage",
                "Czech_000113 · original published validation artifacts",
                "assets/ground_truth_samples/gt_Czech_000113.jpg",
                "assets/predictions/pred_Czech_000113.jpg",
                "GROUND TRUTH: D20 REGION",
                "PREDICTION: D00 REGION",
                "Different class and extent. Review failure cases before tuning the model.",
            )
        if index == 4:
            image, draw = self.base(
                "Verify the export, image by image",
                "Same released weights · fixed 640×640 preprocessing · both backends on CPU",
            )
            values = [
                (40, "10", "benchmark images", "1 non-empty reference image"),
                (452, "4", "demo fixtures", "3 non-empty reference images"),
                (864, "PASS", "explicit tolerances", "Class-aware one-to-one matching"),
            ]
            for x, value, label, detail in values:
                draw.rounded_rectangle((x, 178, x + 376, 460), radius=14, fill=PANEL, outline=LINE)
                self.text(draw, (x + 24, 212), value, 60, GREEN, True)
                self.text(draw, (x + 24, 310), label, 25, WHITE, True)
                self.lines(draw, (x + 24, 360), detail, 328, 20)
            self.text(
                draw,
                (40, 496),
                "IoU ≥ 0.99    |    confidence delta ≤ 0.0001    |    box delta ≤ 0.1 px",
                24,
            )
            self.lines(
                draw,
                (40, 551),
                "One scene appears in both sets. This checks backend agreement, "
                "not detection accuracy. Image and model hashes are included in the reports.",
                1170,
            )
            return image
        if index == 5:
            image, draw = self.base(
                "Measure speed. State the limits.",
                "Historical measurements from the original baseline; not this capture run",
            )
            draw.rounded_rectangle((40, 164, 624, 616), radius=14, fill=PANEL, outline=LINE)
            draw.rounded_rectangle((656, 164, 1240, 616), radius=14, fill=PANEL, outline=LINE)
            self.text(draw, (64, 197), "ONNX RUNTIME / CPU", 20, GREEN, True)
            self.text(draw, (64, 244), f"{self.benchmark['fps_mean']} images/s", 44, WHITE, True)
            self.lines(
                draw,
                (64, 327),
                "Intel i7-13650HX · 640 input · 10 images · 100 timed calls",
                520,
                23,
            )
            self.lines(
                draw, (64, 432), "21.466 ms mean / 22.870 ms p95. Image decoding excluded.", 510, 23
            )
            self.text(draw, (64, 561), "Laptop CPU, not embedded hardware.", 21, AMBER)
            self.text(draw, (680, 197), "CZECH VALIDATION SPLIT", 20, GREEN, True)
            self.text(draw, (680, 244), "24.5% mAP@50", 44, WHITE, True)
            self.lines(
                draw,
                (680, 327),
                "565 images · 34.0% overall recall · 16.2% pothole recall",
                520,
                23,
            )
            self.lines(
                draw, (680, 432), "A measured baseline with substantial missed damage.", 510, 23
            )
            self.text(draw, (680, 561), "Independent field validation is needed.", 21, AMBER)
            return image
        image, draw = self.base(
            "Inspect. Reproduce. Improve.",
            "Computer vision engineering with visible evidence and honest boundaries",
        )
        self.text(draw, (60, 222), "Try the local /demo interface", 42, WHITE, True)
        self.lines(
            draw,
            (60, 305),
            "Download verified weights. Run the CLI and API. Inspect raw reports "
            "and failure examples. Reproduce the export checks.",
            1080,
            29,
        )
        self.text(draw, (60, 493), "github.com/Tamer1020/road-damage-detection", 31, GREEN, True)
        self.text(draw, (60, 569), "Python · PyTorch · OpenCV · FastAPI · ONNX Runtime", 24, MUTED)
        return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", default="assets/demo/capture")
    parser.add_argument("--output", default="assets/demo/road-damage-demo.mp4")
    parser.add_argument("--font-dir", default="/usr/share/fonts/truetype/dejavu")
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = Renderer(Path(args.capture), Path(args.font_dir))
    frames = [renderer.scene(i) for i in range(len(CHAPTERS))]
    preview = output.parent / "preview"
    preview.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(preview / f"scene-{i}.jpg", quality=90)
    frames[0].save(output.parent / "poster.jpg", quality=92)
    srt = []
    for i, (start, end, text) in enumerate(CHAPTERS, 1):
        srt.append(f"{i}\n00:00:{start:02},000 --> 00:{end // 60:02}:{end % 60:02},000\n{text}\n")
    output.with_suffix(".srt").write_text("\n".join(srt), encoding="utf-8")
    if args.preview_only:
        print(preview)
        return
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index, (start, end, _) in enumerate(CHAPTERS):
            for frame_number in range(start * FPS, end * FPS):
                frame = frames[index].copy()
                draw = ImageDraw.Draw(frame)
                draw.rectangle(
                    (40, 661, 40 + int(1200 * frame_number / (60 * FPS)), 664), fill=GREEN
                )
                process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"Rendered {output}: 60 seconds, {W}x{H}, {FPS} fps")


if __name__ == "__main__":
    main()
