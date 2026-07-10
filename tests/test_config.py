from pathlib import Path

import yaml

from road_damage.config import Config


def test_defaults():
    cfg = Config.from_dict({})
    assert cfg.project.seed == 42
    assert cfg.data.classes[0] == "D00"
    assert cfg.inference.conf == 0.25


def test_override_from_yaml(tmp_path: Path):
    data = {
        "project": {"seed": 7},
        "inference": {"conf": 0.5, "iou": 0.6},
        "data": {"classes": ["D00", "D40"]},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    cfg = Config.load(path)
    assert cfg.project.seed == 7
    assert cfg.inference.conf == 0.5
    assert cfg.data.classes == ["D00", "D40"]


def test_missing_file_raises(tmp_path: Path):
    try:
        Config.load(tmp_path / "does_not_exist.yaml")
    except FileNotFoundError:
        return
    raise AssertionError("Expected FileNotFoundError")
