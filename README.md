# Road Damage Detection

> **Status:** 🚧 Actively developed. The full pipeline — data prep → training → evaluation → inference → FastAPI API → ONNX export → benchmarking — is implemented, tested, and runs end to end. Trained weights and measured results are the next milestone; no metrics are reported here until they are actually measured.

## Project Summary

**Road Damage Detection** is an end-to-end computer-vision pipeline that detects and classifies road-surface damage — longitudinal, transverse, and alligator cracks, and potholes — from street-level images using **YOLOv8**.

It covers the complete workflow: **RDD2022** dataset preparation (PASCAL VOC → YOLO conversion), configuration-driven **training** and **evaluation**, and **inference** via both a CLI and a **FastAPI** service. For deployment it adds **ONNX export** and an **FPS / latency benchmark**, making it a practical starting point for edge and real-time use.

## Project Overview

Road infrastructure inspection is usually expensive, slow, and manual. This project explores how deep learning can be used to automatically detect visible road damage from camera images.

The project focuses on four common road damage classes from the RDD2022 dataset:

| ID | Code | Meaning |
|---:|:-----|:--------|
| 0 | D00 | Longitudinal crack |
| 1 | D10 | Transverse crack |
| 2 | D20 | Alligator crack |
| 3 | D40 | Pothole |

## Features

- YOLO-based road damage detection
- RDD2022 dataset support
- Pascal VOC XML to YOLO TXT annotation conversion
- Automatic train / validation split
- Config-driven training and inference
- CLI tools for training, evaluation, and inference
- FastAPI inference service
- ONNX export for deployment
- FPS / latency benchmark tool
- Unit tests with pytest
- Docker support
- GitHub Actions CI workflow

## Repository Structure

```text
road-damage-detection/
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   └── config.yaml
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── notebooks/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── scripts/
│   ├── download_data.py
│   └── prepare_dataset.py
├── src/
│   └── road_damage/
│       ├── __init__.py
│       ├── api.py
│       ├── benchmark.py
│       ├── config.py
│       ├── dataset.py
│       ├── detection.py
│       ├── evaluate.py
│       ├── export_onnx.py
│       ├── inference.py
│       ├── model.py
│       ├── train.py
│       ├── visualize.py
│       └── utils/
│           ├── __init__.py
│           ├── io.py
│           └── logging.py
├── tests/
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Dataset

This project expects the RDD2022 dataset.

The raw dataset annotations are provided in Pascal VOC XML format. The script `scripts/prepare_dataset.py` converts these XML annotations into YOLO TXT format and creates a train / validation split.

Source repository:

```text
https://github.com/sekilab/RoadDamageDetector
```

Expected raw dataset example:

```text
data/raw/Japan/
├── Annotations/
│   ├── image_001.xml
│   └── ...
└── JPEGImages/
    ├── image_001.jpg
    └── ...
```

Expected prepared dataset layout:

```text
data/rdd2022/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

Large datasets, trained weights, and generated outputs are ignored by Git and should not be uploaded to the repository.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux / macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For development and testing:

```bash
pip install -r requirements-dev.txt
```

## Configuration

The project is controlled through:

```text
configs/config.yaml
```

Important configuration sections:

- `data`: dataset paths and class names
- `model`: pretrained weights and trained checkpoint path
- `train`: training hyperparameters
- `inference`: confidence threshold, IoU threshold, image size, and device
- `api`: FastAPI host, port, and upload size limit

Example:

```yaml
model:
  weights: yolov8n.pt
  checkpoint: models/best.pt

inference:
  conf: 0.25
  iou: 0.45
  imgsz: 640
  device: cpu
```

## Data Preparation

Convert RDD2022 Pascal VOC XML annotations to YOLO format:

```bash
python scripts/prepare_dataset.py ^
    --src data/raw/Japan ^
    --dst data/rdd2022 ^
    --val-ratio 0.2 ^
    --seed 42
```

Linux / macOS version:

```bash
python scripts/prepare_dataset.py \
    --src data/raw/Japan \
    --dst data/rdd2022 \
    --val-ratio 0.2 \
    --seed 42
```

The script creates:

```text
data/rdd2022/images/train
data/rdd2022/images/val
data/rdd2022/labels/train
data/rdd2022/labels/val
```

The YOLO dataset descriptor `data.yaml` is generated automatically during training and evaluation.

## Training

Train the model:

```bash
rdd-train --config configs/config.yaml
```

Alternative:

```bash
python -m road_damage.train --config configs/config.yaml
```

Training outputs are saved under:

```text
runs/
```

After training, copy the best weights to:

```text
models/best.pt
```

The inference tools use the checkpoint path defined in:

```yaml
model:
  checkpoint: models/best.pt
```

## Evaluation

Evaluate the trained model on the validation split:

```bash
rdd-eval --config configs/config.yaml --split val
```

The evaluation reports YOLO metrics such as:

- mAP@50
- mAP@50-95

## Inference CLI

Run inference on one image:

```bash
rdd-infer --config configs/config.yaml --image path/to/street.jpg --save
```

The command prints detections as JSON and optionally saves an annotated image to:

```text
outputs/
```

Example output format:

```json
[
  {
    "class_id": 0,
    "label": "D00",
    "confidence": 0.87,
    "bbox": [120.5, 85.0, 310.2, 160.8]
  }
]
```

## Inference API

Start the FastAPI server:

```bash
rdd-serve
```

Alternative:

```bash
uvicorn road_damage.api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

Prediction endpoint:

```text
POST /predict
```

Example request:

```bash
curl -s -F "file=@street.jpg" "http://127.0.0.1:8000/predict?annotate=false"
```

Use `annotate=true` to return a base64 encoded annotated image.

## Export to ONNX

After training, export the trained YOLO checkpoint to ONNX:

```bash
rdd-export-onnx --config configs/config.yaml
```

Custom output path:

```bash
rdd-export-onnx --output models/road_damage.onnx --imgsz 640
```

Export with a fixed ONNX opset and dynamic input axes:

```bash
rdd-export-onnx --opset 12 --dynamic
```

ONNX is useful because it allows the trained model to run outside the PyTorch training environment, for example with ONNX Runtime, TensorRT, or OpenVINO.

This is important for Edge AI because deployment devices often need smaller and faster runtimes than a full PyTorch installation.

## FPS / Latency Benchmark

Measure inference speed on one image:

```bash
rdd-benchmark --image samples/street.jpg
```

Measure inference speed on a folder of images:

```bash
rdd-benchmark --folder samples/ --warmup 10 --runs 200 --save-json
```

The benchmark reports:

- mean latency
- median latency
- minimum latency
- maximum latency
- p95 latency
- average FPS

Example output shape:

```text
=== Benchmark summary ===
Device        : cpu
Image size    : 640
Images        : 1
Warmup / Runs : 5 / 50
Latency (ms)  : mean <ms> | median <ms> | min <ms> | max <ms> | p95 <ms>
Throughput    : <fps> FPS
=========================
```

No fake benchmark numbers are included in this repository. The benchmark values depend on the trained model, input resolution, and hardware.

## Why Edge AI Matters

For road damage detection, accuracy is not the only important factor. A model may also need to run in real time on limited hardware, such as:

- vehicle-mounted cameras
- industrial edge devices
- roadside inspection systems
- embedded GPU platforms
- CPU-only gateways

That is why this project includes ONNX export and FPS benchmarking. These tools make it possible to compare model speed, latency, and deployment readiness.

## Docker

Build the Docker image:

```bash
docker build -t road-damage-detection .
```

Run the API container:

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models" road-damage-detection
```

The Docker image runs the API as a non-root user and includes a health check.

## Testing

Run the test suite:

```bash
pytest -q
```

Current test coverage includes:

- config loading
- dataset YAML generation
- image encoding / decoding
- visualization utilities

Expected result:

```text
9 passed
```

## About This Project

This is an actively developed portfolio project. The codebase is complete, unit-tested, and CI-checked; the next milestone is training on RDD2022 and publishing real evaluation metrics, benchmark numbers, and prediction samples. This README reports only measured results — never placeholder or estimated numbers.

## Current Status

| Component | Status |
|---|---|
| Dataset pipeline (VOC → YOLO conversion + train/val split) | ✅ Implemented |
| Config-driven training (`rdd-train`) | ✅ Implemented |
| Evaluation (`rdd-eval`, mAP@50 / mAP@50-95) | ✅ Implemented |
| Inference — CLI (`rdd-infer`) | ✅ Implemented |
| Inference — FastAPI API (`/health`, `/predict`) | ✅ Implemented |
| ONNX export (`rdd-export-onnx`) | ✅ Implemented |
| FPS / latency benchmark (`rdd-benchmark`) | ✅ Implemented |
| Unit tests + CI | ✅ 9 tests passing (pytest + GitHub Actions) |
| Docker image (non-root, healthcheck) | ✅ Implemented |
| **Trained model weights** | ⏳ In progress |
| **Measured results (mAP, FPS, sample predictions)** | ⏳ In progress |

## Dataset Label Verification

Before training, I verified the converted YOLO labels by drawing the ground-truth bounding boxes on real validation images from the RDD2022 Czech subset.

This step confirms that the Pascal VOC XML to YOLO TXT conversion works correctly and that the bounding boxes align with visible road cracks and potholes.

Example ground-truth visualizations:

![Ground truth sample 1](assets/ground_truth_samples/gt_Czech_000006.jpg)

More samples are available in:

```text
assets/ground_truth_samples/

## Project Results

This section should be updated after training the model.

Planned results to include:

- example prediction images
- validation metrics
- inference speed
- ONNX export result
- benchmark JSON output

No fake results are reported before actual training and testing.

## What I Learned

This project demonstrates practical experience with:

- object detection using YOLO
- road damage detection datasets
- annotation format conversion
- Pascal VOC XML parsing
- YOLO TXT label generation
- model training and evaluation
- inference visualization with OpenCV
- API deployment with FastAPI
- ONNX model export
- FPS and latency benchmarking
- Python package structure
- unit testing
- Docker-based deployment

## Limitations

Current limitations:

- The model must be trained before real inference can be performed.
- The repository does not include the full RDD2022 dataset because it is too large.
- Benchmark results depend strongly on the hardware.
- The first version focuses on bounding-box detection only.
- No INT8 quantization is implemented yet.
- No TensorRT deployment is included yet.

## Next Steps

Possible future improvements:

- Train on multiple RDD2022 country subsets
- Compare YOLOv8n, YOLOv8s, and YOLOv8m
- Add TensorRT export
- Add ONNX Runtime inference
- Add INT8 quantization
- Add confusion matrix visualization
- Add example prediction images to the README
- Add a small demo video
- Add experiment tracking
- Deploy the API on a cloud or edge device

## Roadmap / Next Steps

- [ ] Train YOLOv8n on an RDD2022 country subset; commit the training config and logs
- [ ] Add measured evaluation metrics (mAP@50, mAP@50-95) from `rdd-eval`
- [ ] Add annotated sample prediction images for cracks and potholes
- [ ] Add training curves and confusion matrix (`results.png`, `confusion_matrix.png`)
- [ ] Add measured latency / FPS from `rdd-benchmark`, with the test device stated
- [ ] Verify the exported ONNX model runs under ONNX Runtime and report its file size
- [ ] Add a short demo GIF or video of the FastAPI `/predict` endpoint
- [ ] Document the dataset subset, image counts, and class distribution

## License

MIT — see `LICENSE`.