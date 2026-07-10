"""FastAPI service exposing health and prediction endpoints.

The model is lazy-loaded on the first prediction so the process starts fast and
the /health probe responds even before weights are in memory.
"""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from .config import Config
from .model import RoadDamageModel
from .utils.io import decode_image, to_base64
from .utils.logging import get_logger
from .visualize import draw_detections

logger = get_logger(__name__)
config = Config.load("configs/config.yaml")

app = FastAPI(title="Road Damage Detection API", version="0.1.0")

_model: RoadDamageModel | None = None


def get_model() -> RoadDamageModel:
    global _model
    if _model is None:
        logger.info("Loading model checkpoint: %s", config.model.checkpoint)
        _model = RoadDamageModel(config.model.checkpoint, device=config.inference.device)
    return _model


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    annotate: bool = Query(False, description="Include base64 annotated image."),
) -> JSONResponse:
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Only JPEG/PNG are supported.")

    raw = await file.read()
    max_bytes = config.api.max_upload_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail="Uploaded file too large.")

    try:
        image = decode_image(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model = get_model()
    detections = model.predict_image(
        image,
        conf=config.inference.conf,
        iou=config.inference.iou,
        imgsz=config.inference.imgsz,
    )

    payload: dict = {
        "filename": file.filename,
        "count": len(detections),
        "detections": [d.to_dict() for d in detections],
    }
    if annotate:
        payload["annotated_image_b64"] = to_base64(draw_detections(image, detections))

    return JSONResponse(payload)


def main() -> None:
    import uvicorn

    uvicorn.run("road_damage.api:app", host=config.api.host, port=config.api.port)


if __name__ == "__main__":
    main()
