import yaml

from road_damage.config import Config
from road_damage.dataset import build_data_yaml


def test_build_data_yaml(tmp_path):
    cfg = Config.from_dict(
        {"data": {"root": str(tmp_path / "ds"), "yaml": str(tmp_path / "ds/data.yaml")}}
    )
    out = build_data_yaml(cfg)
    assert out.exists()

    content = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert content["train"] == "images/train"
    assert content["val"] == "images/val"
    assert content["names"][0] == "D00"
    assert len(content["names"]) == 4
