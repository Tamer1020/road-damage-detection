# Demo: inspect predictions and their limits

[Watch the 60-second video](../assets/demo/road-damage-demo.mp4) · [Poster](../assets/demo/poster.jpg)

## What is recorded

`scripts/capture_demo.py` starts a local uvicorn/FastAPI process and makes real HTTP requests. It uses the unchanged `v0.1.0` PyTorch weights, CPU configuration, confidence 0.25 and NMS IoU 0.45. Each request saves its JSON response and the returned annotated JPEG. The base64 image field is decoded into that JPEG; other response fields are unchanged.

The video is a replay/composition of those captured responses, not a browser screen recording. Chapter durations are chosen for readability and do not represent inference time or throughput. No boxes or confidence scores are invented.

| Segment | Evidence |
|---|---|
| Input → crack detection | `Czech_000047`, 640×640 demo copy; real HTTP 200 response |
| Structured API output | Detection JSON from the same request |
| Missed damage | Original `Czech_000006` benchmark image returns no detections; historical ground truth marks D00 |
| Class/region mismatch | Original published `Czech_000113` ground-truth/prediction pair; current demo-copy response also reports D00 |
| Export agreement and speed | Scoped parity reports and historical CPU benchmark, clearly identified |
| Limits | Czech baseline accuracy and future independent evaluation |

The new 640×640 inputs came from a public dataset mirror. They are demo fixtures, not replacements for the historical validation images. [Attribution and source hashes](ATTRIBUTION.md).

## Inspect the evidence

- [Capture manifest](../assets/demo/capture/manifest.json): input/config/model hashes, environment and capture method.
- [Crack response](../assets/demo/capture/crack.json), [missed-damage response](../assets/demo/capture/missed_damage.json), [class-confusion response](../assets/demo/capture/class_confusion.json).
- [Invalid upload response](../assets/demo/capture/invalid_upload.json): HTTP 400 after model loading.
- [Benchmark-image parity](../assets/evaluation/onnx_parity_benchmark_images.json), [demo-image parity](../assets/evaluation/onnx_parity_demo_images.json).

The shown `inference_ms` values are individual local model-call timings, not a new benchmark. The performance card uses the original Windows CPU report, with its device and excluded decoding scope.

## Reproduce

Install the tested CPU environment and release weights as described in the root README. Capture to a new folder when experimenting:

```bash
python scripts/capture_demo.py --config configs/config.yaml --output outputs/demo_capture
python scripts/render_portfolio_demo.py --capture outputs/demo_capture --output outputs/road-damage-demo.mp4
```

Rendering additionally requires **ffmpeg** on PATH and DejaVu Sans fonts. On Linux the default font directory is `/usr/share/fonts/truetype/dejavu`; use `--font-dir` elsewhere. The renderer uses Pillow to compose exact image/JSON evidence and ffmpeg for H.264 encoding. It writes an English SRT and a poster alongside the MP4.

To explore interactively, start `rdd-serve` and visit `http://127.0.0.1:8000/demo`. The upload interface shows the current model response; it does not claim a damage-free road when the output is empty.
