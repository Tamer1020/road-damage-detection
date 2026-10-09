"""Download the published v0.1.0 weights and verify their SHA-256 hashes."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

BASE = "https://github.com/Tamer1020/road-damage-detection/releases/download/v0.1.0/"
MODELS = {
    "best.pt": "12f65922fe28fbc963ba3bafb95390d850e96ca8fd99058f40f4ce00e54feddb",
    "road_damage_640.onnx": "d898d2d4860ab44949276d50f01610f350244d378042659072b4110cd9d677da",
}


def download(name: str, directory: Path) -> Path:
    expected = MODELS[name]
    target = directory / name
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{target} contains different weights. Use another --output-dir.")
        print(f"Verified existing {target}")
        return target
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False, suffix=".download") as out:
            temporary = Path(out.name)
            digest = hashlib.sha256()
            with urllib.request.urlopen(BASE + name, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch for {name}; downloaded file was not installed")
        if target.exists():
            raise FileExistsError(f"Destination appeared during download: {target}")
        temporary.rename(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"Downloaded and verified {target}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    args = parser.parse_args()
    try:
        for name in MODELS if args.model == "all" else [args.model]:
            download(name, Path(args.output_dir))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Download failed: {exc}\n")


if __name__ == "__main__":
    main()
