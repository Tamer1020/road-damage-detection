# Model card and evidence

## Model and intended use

YOLOv8n fine-tuned for four RDD classes: D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole. Intended for learning, reproducible experimentation and assisted inspection. A missing detection is not a road-safety assessment.

Release: [v0.1.0](https://github.com/Tamer1020/road-damage-detection/releases/tag/v0.1.0). The published model weights and historical experiment files are unchanged by the portfolio update.

| Artifact | SHA-256 |
|---|---|
| `best.pt` | `12f65922fe28fbc963ba3bafb95390d850e96ca8fd99058f40f4ce00e54feddb` |
| `road_damage_640.onnx` | `d898d2d4860ab44949276d50f01610f350244d378042659072b4110cd9d677da` |

## Historical training and validation

- RDD2022 Czech subset: 2,264 training images and 565 validation images.
- Seed 42, random image split at 20% validation; no independent route/country test is reported.
- 100 epochs, image size 640, batch 8, NVIDIA GeForce RTX 5070 Laptop GPU.
- Background-only label files: 1,403 train and 354 validation. Empty target-class labels do not guarantee the absence of every possible defect.
- Raw evaluation contains 321 target instances in validation, including only 28 pothole instances. Small per-class samples limit interpretation.

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| All | 0.315 | 0.340 | 0.245 | 0.0954 |
| D00 | 0.408 | 0.456 | 0.417 | 0.171 |
| D10 | 0.415 | 0.296 | 0.248 | 0.0759 |
| D20 | 0.208 | 0.448 | 0.238 | 0.104 |
| D40 | 0.231 | 0.162 | 0.0752 | 0.0303 |

Source: [raw evaluation log](../assets/evaluation/eval_output.txt). These are the recorded validator metrics; they should not be described as API precision/recall at the demo's confidence threshold. [Training arguments](../assets/training/args.yaml), [epoch log](../assets/training/results.csv), [curves](../assets/training/results.png), [confusion matrix](../assets/training/confusion_matrix.png), [dataset summary](../assets/dataset_analysis/dataset_summary.md).

Class imbalance is a plausible contributor to weak rare-class results, not a demonstrated causal explanation. Changes to sampling or augmentation need controlled experiments.

## Historical latency benchmark

| Backend | Device | Mean ms | p95 ms | Images/s |
|---|---|---:|---:|---:|
| PyTorch | RTX 5070 Laptop GPU | 7.632 | 8.938 | 131.02 |
| PyTorch | i7-13650HX CPU | 59.187 | 69.414 | 16.90 |
| ONNX Runtime | i7-13650HX CPU | 21.466 | 22.870 | 46.59 |

Windows AMD64, reported RAM 31.7 GB, 640 input, confidence 0.25, NMS IoU 0.45. Ten decoded images, ten warmups, 100 timed calls. `59.187 / 21.466 ≈ 2.76` is the ratio of mean CPU call latency in this experiment.

The timed wrapper includes preprocessing, model execution, postprocessing and conversion to `Detection` objects. Disk/image decoding is outside timing. These are sequential image-call throughput estimates, not camera-to-result, HTTP service, embedded hardware or power measurements. The original JSON does not capture every runtime/thread setting, so exact speed reproduction is not guaranteed.

Raw reports: [PyTorch CPU](../assets/evaluation/benchmark_pt_cpu.json), [PyTorch GPU](../assets/evaluation/benchmark_pt_gpu.json), [ONNX CPU](../assets/evaluation/benchmark_onnx_cpu.json). The portfolio update does not replace these measurements with timings from its own environment.

## Current export verification

The original [one-image verification log](../assets/evaluation/onnx_verification.txt) is preserved. The new verifier checks post-NMS detections using same-class, one-to-one matching rather than pairing detections by confidence rank.

| Fixture set | Images | Non-empty reference images | Outcome |
|---|---:|---:|---|
| Original benchmark images | 10 | 1 | All pass configured tolerances |
| Additional 640×640 demo fixtures | 4 | 3 | All pass configured tolerances |

One scene ID (`Czech_000006`) appears in both sets at different resolutions/encodings. These are 14 input fixtures across 13 scene IDs, not an independent held-out evaluation.

Conditions: both models on CPU, `rect=False`, 640×640 input, confidence 0.25, NMS IoU 0.45. Match requires class and label equality, IoU ≥ 0.99, confidence difference ≤ 0.0001 and maximum coordinate difference ≤ 0.1 pixel. Empty/empty images alone cannot produce a passing run.

The [benchmark-fixture report](../assets/evaluation/onnx_parity_benchmark_images.json) and [demo-fixture report](../assets/evaluation/onnx_parity_demo_images.json) include model/image hashes, versions and every image result. Export parity establishes agreement between these backends on these inputs; it does not establish detection correctness or accuracy on the full validation split.

## Dataset integrity change

The old preparation script could leave files in both splits when rerun into the same destination with a different seed. This was reproduced using controlled fixtures; it is not evidence that the published training run was contaminated.

Preparation now refuses non-empty output folders, validates annotations before publishing, checks exact image hashes across splits and saves a split manifest. The original seeded splitting algorithm is retained on valid inputs. Existing historical metrics are preserved; the full historical dataset was not rebuilt or retrained as part of this update. Adjacent/near-duplicate frames and geographic generalization still require separate assessment.
