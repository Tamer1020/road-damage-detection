# Reproduction guide

Run commands from the repository root. The root README gives the dataset-free CPU quickstart. `constraints-cpu.txt` records the direct runtime versions used for the new Linux/Python 3.12 checks; it is not a complete transitive lockfile or the original Windows training environment.

## Installation and GPU use

For the verified CPU route, install PyTorch 2.11.0 / torchvision 0.26.0 from the CPU wheel index, then `pip install -e . -c constraints-cpu.txt`. Download weights with `python scripts/download_models.py`.

For GPU training, install a CUDA-enabled PyTorch/torchvision combination supported by your driver using [PyTorch's installation guide](https://pytorch.org/get-started/locally/), then install the project without the CPU-specific constraints. Set `train.device` and `inference.device` appropriately. Check the environment before training:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The original baseline ran on an RTX 5070 Laptop GPU. A different GPU/runtime is a new experiment and should be documented as such.

## Dataset preparation

Obtain the Czech training subset from the [RDD2022 authors](https://github.com/sekilab/RoadDamageDetector) or [official dataset record](https://doi.org/10.6084/m9.figshare.21431547.v1). Dataset archives are external and may change hosting. Do not mix the unlabeled challenge test images into a labeled validation split.

Expected source: `data/raw/Czech/annotations/xmls/*.xml` and `data/raw/Czech/images/*.jpg`.

```bash
rdd-prepare --src data/raw/Czech --dst data/rdd2022 --val-ratio 0.2 --seed 42
```

The command creates `images/{train,val}`, `labels/{train,val}`, `data.yaml` and `split_manifest.json`. It validates target-class box geometry, rejects missing images and exact content duplicates across splits, and only publishes after preparation succeeds. Unknown classes are ignored, matching the original four-class baseline.

**Use a new or empty destination.** Reusing a populated destination fails without deleting or modifying it. To try another split, choose a new `--dst` and update both `data.root` and `data.yaml` in a copied config. This prevents old split files from surviving a new seed or validation ratio.

The manifest records class order, seed, split membership, original image/XML hashes and output label hashes. It does not detect neighboring frames or other near-duplicates. For a robust generalization experiment, add route/group separation and an independent test set.

The old script entry point remains available:

```bash
python scripts/prepare_dataset.py --src data/raw/Czech --dst data/rdd2022-new --seed 42
```

## Training and evaluation

```bash
rdd-train --config configs/config_smoke.yaml
rdd-train --config configs/config_train.yaml
```

Training prints the best checkpoint path. Copy that checkpoint to the `model.checkpoint` path in your experiment config, or update the config to point to it. The downloader deliberately refuses to overwrite a different locally trained checkpoint.

```bash
rdd-eval --config configs/config_train.yaml --split val
python scripts/analyze_dataset.py --help
python scripts/visualize_yolo_labels.py --help
```

See each script before use: the historical sample-generation scripts use their documented repository paths. The published baseline used 100 epochs, 640 input, batch 8 and seed 42. Keep raw logs, config, model hashes, split manifest and versions for every new run. The historical training curves and metrics in `assets/training` and `assets/evaluation` are preserved.

## Inference and API

```bash
rdd-infer --config configs/config.yaml --image assets/demo/inputs/Czech_000047.jpg --save
rdd-serve
```

- `GET /health`: process liveness; does not load weights.
- `GET /ready`: load and warm up the configured model once; return 503 if initialization/inference fails.
- `POST /predict`: JPEG/PNG upload, detection JSON, optional `annotate=true` for a base64 JPEG.
- `GET /demo`: local image-upload interface. `GET /docs`: OpenAPI UI.

Default limits: 10 MB compressed upload and 20 million decoded pixels. Invalid image data returns 400, excessive size/dimensions 413 and unsupported declared content type 415. Read limits apply to the uploaded file after multipart parsing; this is not a reverse-proxy request-body limit.

The shared predictor is serialized with a lock. Blocking inference runs in the framework's worker thread pool. This keeps model access consistent but is not a high-concurrency service design; load testing, worker sizing, authentication and deployment infrastructure are separate work.

To use another configuration, set `RDD_CONFIG` before starting:

```bash
# Linux
RDD_CONFIG=configs/config.yaml rdd-serve
```

```powershell
# PowerShell
$env:RDD_CONFIG = 'configs/config.yaml'
rdd-serve
```

## Export verification

```bash
rdd-export-onnx --config configs/config.yaml --output models/new_export.onnx --imgsz 640
rdd-verify-onnx --config configs/config.yaml --pt models/best.pt --onnx models/new_export.onnx --folder assets/demo/inputs --output outputs/new_export_parity.json
```

Do not replace the released ONNX file while trying a new export. The verifier uses the same fixed-size preprocessing on both CPU backends. It writes a JSON report even when predictions mismatch. Exit 0 means all images pass and at least one PyTorch image is non-empty; 1 means mismatch/inconclusive; 2 means input/execution error. `--min-nonempty` can require more positive cases. Class, confidence and box tolerances are explicit CLI options; record them instead of loosening them silently to obtain a pass.

For a stronger evaluation, use the full validation split rather than the small bundled fixtures. Post-NMS parity is different from raw-tensor equivalence and from mAP.

## Benchmarking

Use one decoded image set and identical settings for each backend:

```bash
rdd-benchmark --config configs/config.yaml --folder assets/bench_images --weights models/best.pt --device cpu --warmup 10 --runs 100 --save-json --tag pt_cpu
rdd-benchmark --config configs/config.yaml --folder assets/bench_images --weights models/road_damage_640.onnx --device cpu --warmup 10 --runs 100 --save-json --tag onnx_cpu
rdd-benchmark --config configs/config.yaml --folder assets/bench_images --weights models/best.pt --device 0 --warmup 10 --runs 100 --save-json --tag pt_gpu
```

Only run the GPU command on a compatible CUDA environment. Record CPU/GPU, runtime versions, power mode, thread settings and timing scope. The benchmark times the full model wrapper on predecoded images, excluding HTTP, disk/image decoding and rendering. New timings belong to a new report; the existing Windows results are historical evidence.

## Docker

```bash
docker build -t road-damage-detection .
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models:ro" road-damage-detection
```

This is the Linux shell example; use an absolute host path for the model mount on other shells. Download the release before running. The image runs as a non-root user and uses model readiness as its health check. No embedded-hardware or production-load claim is implied by supplying a Dockerfile.

## Tests and recorded demo

```bash
pip install -r requirements-test.txt
pip install --no-deps -e .
pytest -q
ruff check src tests scripts
```

Unit tests need no weights. The full CPU integration job downloads weights with verified hashes, runs parity and makes real local HTTP requests. To reproduce the demo capture/render, follow [DEMO.md](DEMO.md).
