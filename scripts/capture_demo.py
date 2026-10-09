"""Capture real local HTTP responses using the released model and repository images."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from road_damage.config import Config

CASES = {
    "crack": "assets/demo/inputs/Czech_000047.jpg",
    "missed_damage": "assets/bench_images/Czech_000006.jpg",
    "class_confusion": "assets/demo/inputs/Czech_000113.jpg",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(config_path: str, output: Path):
    config = Config.load(config_path)
    output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "RDD_CONFIG": config_path}
    base = f"http://127.0.0.1:{port}"
    session = requests.Session()
    session.trust_env = False  # Local loopback; do not send requests through a web proxy.
    log = (output / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "road_damage.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited; inspect {output / 'server.log'}")
            try:
                response = session.get(base + "/health", timeout=1)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError("Server startup timed out")
        ready = session.get(base + "/ready", timeout=120)
        ready.raise_for_status()
        cases = []
        for name, source in CASES.items():
            path = Path(source)
            response = session.post(
                base + "/predict?annotate=true",
                files={"file": (path.name, path.read_bytes(), "image/jpeg")},
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            annotated = base64.b64decode(payload.pop("annotated_image_b64"))
            (output / f"{name}.jpg").write_bytes(annotated)
            (output / f"{name}.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            cases.append(
                {
                    "name": name,
                    "input": source,
                    "input_sha256": sha256(path),
                    "http_status": response.status_code,
                    "response": f"{name}.json",
                    "annotation": f"{name}.jpg",
                    "annotation_sha256": hashlib.sha256(annotated).hexdigest(),
                }
            )
            print(name, response.status_code, payload["count"], "detections", flush=True)
        invalid = session.post(
            base + "/predict",
            files={"file": ("invalid.jpg", b"not an image", "image/jpeg")},
            timeout=10,
        )
        if invalid.status_code != 400:
            raise RuntimeError("Invalid-image HTTP contract failed")
        (output / "invalid_upload.json").write_text(
            json.dumps({"http_status": invalid.status_code, "response": invalid.json()}, indent=2)
            + "\n"
        )
        manifest = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "capture": "actual HTTP requests to a local uvicorn/FastAPI process",
            "video": "replay of captured responses, edited chapter timing; not a speed benchmark",
            "ready_response": ready.json(),
            "cases": cases,
            "json_note": "annotated_image_b64 decoded to the adjacent JPEG; other fields unchanged",
            "model": {"file": config.model.checkpoint, "sha256": sha256(config.model.checkpoint)},
            "config": {"file": config_path, "sha256": sha256(config_path)},
            "environment": {
                "python": platform.python_version(),
                "system": platform.system(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "ultralytics", "fastapi", "uvicorn")
                },
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        session.close()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="assets/demo/capture")
    args = parser.parse_args()
    capture(args.config, Path(args.output))


if __name__ == "__main__":
    main()
