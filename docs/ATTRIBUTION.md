# Data and third-party attribution

The project uses the RDD road-damage imagery described by Arya et al., **RDD2022: A multi-national image dataset for automatic Road Damage Detection**. Primary sources: [authors' repository](https://github.com/sekilab/RoadDamageDetector), [paper](https://arxiv.org/abs/2209.08538), [Figshare dataset](https://doi.org/10.6084/m9.figshare.21431547.v1).

The Figshare article metadata identifies its dataset license as **CC BY 4.0** ([metadata endpoint](https://api.figshare.com/v2/articles/21431547), [license](https://creativecommons.org/licenses/by/4.0/)). Retain dataset attribution when reusing the image examples; the project code's MIT license does not replace third-party terms.

## Assets in this repository

- `assets/bench_images`: original repository benchmark inputs. These historical fixtures are unchanged.
- `assets/ground_truth_samples`, `assets/predictions`: historical visualization artifacts from the Czech baseline. Added annotation overlays are visible in the images.
- `assets/demo/inputs`: four 640×640 demo copies of Czech scenes from the public [Mustela4/Road-Damage-Detection](https://github.com/Mustela4/Road-Damage-Detection) dataset mirror at commit `c3af2ac1ffc275547563830e02790fdf32b169b7`. Retrieved LFS objects were checked against their SHA-256 pointers. These copies differ in resolution/encoding from the repository's historical 600×600 examples. Their original author credit remains with the dataset creators. Exact source paths and hashes are in [input provenance](../assets/demo/input-provenance.json).
- `assets/demo/capture`: responses and annotation JPEGs produced by this project's released model through its current FastAPI service.
- Demo video/poster: compositions of the above images and recorded responses; typography/layout added for explanation. No generated road imagery or invented predictions.

The mirror's own train/validation directory names are not adopted as this project's split. Demo fixtures are not a new held-out dataset and do not modify the published accuracy figures.

## Software and weights

Inference/training use [Ultralytics](https://github.com/ultralytics/ultralytics), [PyTorch](https://pytorch.org/), [OpenCV](https://opencv.org/) and [ONNX Runtime](https://onnxruntime.ai/). Check their respective license terms when distributing a deployment. The released fine-tuned weights derive from a pretrained YOLOv8n model; their provenance is documented separately from the MIT license for this repository's original code.
