"""FastAPI inference, separate liveness/readiness, and a local demo page."""
from __future__ import annotations

import io
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from PIL.JpegImagePlugin import JpegImageFile
from PIL.PngImagePlugin import PngImageFile

from .config import Config
from .utils.io import decode_image, to_base64
from .utils.logging import get_logger
from .visualize import draw_detections

logger = get_logger(__name__)


def _load_model(config: Config):
    # Keeping the ML import here lets contract tests use an injected predictor.
    from .model import RoadDamageModel
    return RoadDamageModel(config.model.checkpoint, device=config.inference.device)


def create_app(config: Config, model_factory: Callable = _load_model) -> FastAPI:
    app = FastAPI(title="Road Damage Detection API", version="0.1.0")
    model = None
    lock = threading.Lock()

    def get_model():
        nonlocal model
        if model is None:
            try:
                candidate = model_factory(config)
                candidate.predict_image(
                    np.zeros((64, 64, 3), dtype=np.uint8),
                    conf=config.inference.conf, iou=config.inference.iou,
                    imgsz=config.inference.imgsz,
                )
                model = candidate
            except Exception as exc:
                logger.exception("Model initialization failed")
                raise HTTPException(
                    503, "Model unavailable. Check checkpoint and configuration."
                ) from exc
        return model

    @app.get("/health")
    def health() -> dict:
        """Liveness only; does not claim the model is available."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        """Load and validate model availability before sending predictions."""
        with lock:
            get_model()
        return {"status": "ready", "checkpoint": Path(config.model.checkpoint).name,
                "device": str(config.inference.device)}

    @app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
    def demo() -> str:
        return Path(__file__).with_name("demo.html").read_text(encoding="utf-8")

    @app.post("/predict")
    def predict(
        file: Annotated[UploadFile, File(...)],
        annotate: Annotated[bool, Query(description="Include base64 annotated image.")] = False,
    ) -> JSONResponse:
        # A sync route runs blocking inference in FastAPI's worker thread pool.
        if file.content_type not in {"image/jpeg", "image/png"}:
            raise HTTPException(415, "Only JPEG/PNG are supported.")
        max_bytes = int(config.api.max_upload_mb * 1024 * 1024)
        raw = file.file.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise HTTPException(413, "Uploaded file too large.")
        if not raw:
            raise HTTPException(400, "Image data is empty.")
        try:
            # Check dimensions before allocating a decoded pixel buffer.
            # Use explicit JPEG/PNG decoders: ML libraries may monkey-patch
            # Image.open to install optional format plugins on invalid input.
            if raw.startswith(b"\x89PNG\r\n\x1a\n"):
                header = PngImageFile(io.BytesIO(raw))
            elif raw.startswith(b"\xff\xd8"):
                header = JpegImageFile(io.BytesIO(raw))
            else:
                raise ValueError("Not a JPEG or PNG image")
            with header:
                if header.width * header.height > config.api.max_image_pixels:
                    raise HTTPException(413, "Image dimensions exceed the pixel limit.")
                if header.format not in {"JPEG", "PNG"}:
                    raise HTTPException(415, "Only JPEG/PNG are supported.")
            image = decode_image(raw)
        except Image.DecompressionBombError as exc:
            raise HTTPException(413, "Image dimensions exceed the pixel limit.") from exc
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            raise HTTPException(400, "Invalid or unsupported image data.") from exc

        # The shared Ultralytics predictor is serialized, including lazy loading.
        with lock:
            predictor = get_model()
            start = time.perf_counter()
            detections = predictor.predict_image(
                image, conf=config.inference.conf, iou=config.inference.iou,
                imgsz=config.inference.imgsz,
            )
            elapsed = (time.perf_counter() - start) * 1000
        payload = {"filename": file.filename, "count": len(detections),
                   "detections": [d.to_dict() for d in detections],
                   "inference_ms": round(elapsed, 3)}
        if annotate:
            payload["annotated_image_b64"] = to_base64(draw_detections(image, detections))
        return JSONResponse(payload)

    return app


config = Config.load(os.environ.get("RDD_CONFIG", "configs/config.yaml"))
app = create_app(config)


def main() -> None:
    import uvicorn
    uvicorn.run("road_damage.api:app", host=config.api.host, port=config.api.port)


if __name__ == "__main__":
    main()
