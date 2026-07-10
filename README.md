# Road Damage Detection

Detect and classify road surface damage (cracks and potholes) from street-level
imagery using a YOLO object detector. The project ships a training pipeline, an
evaluation step, a CLI for single-image inference, and a FastAPI service for
online prediction.

## Damage classes

Following the RDD2022 convention (4-class subset):

| ID | Code | Meaning |
|---:|:-----|:--------|
| 0 | D00 | Longitudinal crack |
| 1 | D10 | Transverse crack |
| 2 | D20 | Alligator crack |
| 3 | D40 | Pothole |

## Dataset

This project expects the **RDD2022** dataset (Road Damage Detector, Arya et al.).
The raw annotations are PASCAL VOC XML; `scripts/prepare_dataset.py` converts them
to the YOLO text format and creates the `train/val` split.

- Source repo: https://github.com/sekilab/RoadDamageDetector
- Download the archive, unpack it under `data/`, then run the prepare script.

Expected on-disk layout after preparation:

```text
data/rdd2022/
├── images/
│ ├── train/
│ └── val/
├── labels/
│ ├── train/
│ └── val/
└── data.yaml # generated automatically by the pipeline
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                   # installs the road_damage package + CLIs
```

> GPU users: install the CUDA build of PyTorch first (see pytorch.org), then the
> rest of the requirements.

## Data preparation

```bash
# 1. Point --src at the unpacked RDD country folder (contains Annotations/ + JPEGImages/)
python scripts/prepare_dataset.py \
    --src data/raw/Japan \
    --dst data/rdd2022 \
    --val-ratio 0.2 \
    --seed 42
```

## Training

```bash
rdd-train --config configs/config.yaml
# or: python -m road_damage.train --config configs/config.yaml
```

Weights land under `runs/<name>/weights/best.pt`. Copy that to the path in
`model.checkpoint` (default `models/best.pt`) for inference.

## Evaluation

```bash
rdd-eval --config configs/config.yaml --split val
```

Reports mAP@50 and mAP@50-95.

## Inference (CLI)

```bash
rdd-infer --config configs/config.yaml --image path/to/street.jpg --save
```

Prints detections as JSON and writes an annotated image to `outputs/`.

## Inference (API)

```bash
rdd-serve
# or: uvicorn road_damage.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET  /health` → liveness probe
- `POST /predict` → multipart image upload, returns JSON detections; pass
  `?annotate=true` to also get a base64 annotated image.

```bash
curl -s -F "file=@street.jpg" "http://localhost:8000/predict?annotate=false" | jq
```

## Docker

```bash
docker build -t road-damage-detection .
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models" road-damage-detection
```

The image runs the API as a non-root user with a container HEALTHCHECK.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Configuration

All runtime behaviour is driven by `configs/config.yaml` and loaded into typed
dataclasses in `road_damage/config.py`. Override any key by editing the YAML;
no code changes required.

## License

MIT — see `LICENSE`.
