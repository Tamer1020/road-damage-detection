"""Dataset helpers.

`build_data_yaml` writes the Ultralytics dataset descriptor (`data.yaml`) that
points to the image folders and lists class names. Training/eval call this so
the descriptor always matches `config.yaml` — no manual editing.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import Config


def build_data_yaml(config: Config) -> Path:
    """Generate the Ultralytics data.yaml and return its path."""
    data_root = Path(config.data.root).resolve()
    names = {i: c for i, c in enumerate(config.data.classes)}

    descriptor = {
        "path": str(data_root),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }

    out_path = Path(config.data.yaml)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(descriptor, fh, sort_keys=False)
    return out_path
