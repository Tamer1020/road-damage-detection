import base64
import concurrent.futures
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from road_damage.api import create_app
from road_damage.config import Config
from road_damage.detection import Detection
from road_damage.utils.io import decode_image, encode_image


class Predictor:
    def predict_image(self, image, **kwargs):
        return [Detection(0, "D00", 0.8, (1, 1, 8, 8))]


def upload(content=None, kind="image/png"):
    if content is None:
        content = encode_image(np.zeros((16, 16, 3), dtype=np.uint8), ".png")
    return {"file": ("example.png", content, kind)}


def test_health_is_liveness_and_readiness_detects_missing_weights():
    def missing(config):
        raise FileNotFoundError("private internal path")
    client = TestClient(create_app(Config(), missing))
    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/ready")
    assert response.status_code == 503
    assert "private internal path" not in response.text
    assert client.post("/predict", files=upload()).status_code == 503


def test_ready_loads_once_and_predict_returns_contract():
    loads = []
    def factory(config):
        loads.append(config)
        return Predictor()
    client = TestClient(create_app(Config(), factory))
    assert client.get("/health").status_code == 200
    assert not loads
    assert client.get("/ready").status_code == 200
    payload = client.post("/predict?annotate=true", files=upload()).json()
    assert len(loads) == 1
    assert payload["count"] == 1
    assert payload["detections"][0]["label"] == "D00"
    assert payload["inference_ms"] >= 0
    assert decode_image(base64.b64decode(payload["annotated_image_b64"])).shape == (16, 16, 3)


def test_no_detection_is_a_valid_empty_response():
    class Empty:
        def predict_image(self, *args, **kwargs):
            return []
    client = TestClient(create_app(Config(), lambda _: Empty()))
    payload = client.post("/predict", files=upload()).json()
    assert payload["count"] == 0 and payload["detections"] == []
    assert "annotated_image_b64" not in payload


@pytest.mark.parametrize("body,kind,status", [(b"", "image/png", 400),
                                              (b"bad", "image/png", 400),
                                              (b"text", "text/plain", 415)])
def test_invalid_uploads_do_not_load_model(body, kind, status):
    def forbidden(config):
        pytest.fail("Invalid request must not load the model")
    client = TestClient(create_app(Config(), forbidden))
    assert client.post("/predict", files=upload(body, kind)).status_code == status


def test_byte_and_pixel_limits_are_enforced():
    config = Config()
    config.api.max_upload_mb = 1
    config.api.max_image_pixels = 100
    client = TestClient(create_app(config, lambda _: Predictor()))
    assert client.post("/predict", files=upload(b"x" * (1024 * 1024 + 1))).status_code == 413
    assert client.post("/predict", files=upload()).status_code == 413


def test_demo_page_and_openapi_are_available_without_weights():
    client = TestClient(create_app(Config(), lambda _: Predictor()))
    assert "Road Damage" in client.get("/demo").text
    assert "/predict" in client.get("/openapi.json").json()["paths"]


def test_concurrent_requests_share_one_serialized_predictor():
    state = {"loads": 0, "active": 0, "peak": 0}
    guard = threading.Lock()
    class Concurrent:
        def predict_image(self, *args, **kwargs):
            with guard:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.02)
            with guard:
                state["active"] -= 1
            return []
    def factory(config):
        state["loads"] += 1
        return Concurrent()
    app = create_app(Config(), factory)
    def request(_):
        with TestClient(app) as client:
            return client.post("/predict", files=upload()).status_code
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(request, range(4))) == [200] * 4
    assert state["loads"] == 1 and state["peak"] == 1
