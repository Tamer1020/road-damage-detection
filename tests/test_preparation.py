import hashlib
import json

import pytest

from road_damage.preparation import convert_annotation, prepare


def annotation(name="D00", box=(10, 20, 50, 60), width=100, height=100):
    fields = "".join(f"<{k}>{v}</{k}>" for k, v in
                     zip(("xmin", "ymin", "xmax", "ymax"), box, strict=True))
    return (f"<annotation><size><width>{width}</width><height>{height}</height></size>"
            f"<object><name>{name}</name><bndbox>{fields}</bndbox></object></annotation>")


@pytest.fixture
def raw(tmp_path):
    root = tmp_path / "raw"
    (root / "annotations/xmls").mkdir(parents=True)
    (root / "images").mkdir()
    for i in range(10):
        (root / "annotations/xmls" / f"image{i:02}.xml").write_text(annotation())
        (root / "images" / f"image{i:02}.jpg").write_bytes(f"image fixture {i}".encode())
    return root


def test_coordinates_are_normalized(tmp_path):
    path = tmp_path / "valid.xml"
    path.write_text(annotation())
    assert convert_annotation(path) == ["0 0.300000 0.400000 0.400000 0.400000"]


@pytest.mark.parametrize("box", [(50, 20, 10, 60), (-1, 20, 50, 60), (0, 0, 101, 60),
                                 (0, 0, 10, 0), (0, 0, "nan", 10)])
def test_invalid_target_boxes_fail(tmp_path, box):
    path = tmp_path / "invalid.xml"
    path.write_text(annotation(box=box))
    with pytest.raises(ValueError, match="invalid.xml"):
        convert_annotation(path)


@pytest.mark.parametrize("content", [annotation(width=0), "<annotation>",
                                     "<annotation><size/></annotation>"])
def test_invalid_annotation_structure_fails(tmp_path, content):
    path = tmp_path / "invalid.xml"
    path.write_text(content)
    with pytest.raises(ValueError, match="Invalid annotation"):
        convert_annotation(path)


def test_unknown_classes_become_background(tmp_path):
    path = tmp_path / "background.xml"
    path.write_text(annotation(name="D99"))
    assert convert_annotation(path) == []


def test_split_is_reproducible_disjoint_and_hashes_match(raw, tmp_path):
    first = prepare(raw, tmp_path / "first", 0.2, 42)
    assert first == prepare(raw, tmp_path / "second", 0.2, 42)
    assert first["counts"] == {"train": 8, "val": 2}
    assert {r["id"] for r in first["splits"]["train"]}.isdisjoint(
        r["id"] for r in first["splits"]["val"])
    root = tmp_path / "first"
    assert json.loads((root / "split_manifest.json").read_text()) == first
    for split, items in first["splits"].items():
        for item in items:
            assert hashlib.sha256((root / item["image"]).read_bytes()).hexdigest() == (
                item["image_sha256"])
            label = root / "labels" / split / f"{item['id']}.txt"
            assert hashlib.sha256(label.read_bytes()).hexdigest() == item["label_sha256"]


def test_rerun_with_different_seed_cannot_mix_existing_splits(raw, tmp_path):
    dst = tmp_path / "prepared"
    prepare(raw, dst, 0.2, 42)
    before = {str(p.relative_to(dst)): p.read_bytes() for p in dst.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError, match="Choose a new --dst"):
        prepare(raw, dst, 0.2, 7)
    after = {str(p.relative_to(dst)): p.read_bytes() for p in dst.rglob("*") if p.is_file()}
    assert before == after


def test_exact_duplicates_across_splits_fail(raw, tmp_path):
    for path in (raw / "images").glob("*.jpg"):
        path.write_bytes(b"same image under different names")
    with pytest.raises(ValueError, match="Identical image content crosses splits"):
        prepare(raw, tmp_path / "out", 0.2, 42)
    assert not (tmp_path / "out").exists()


def test_missing_image_fails_before_publication(raw, tmp_path):
    (raw / "images/image00.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="Missing image"):
        prepare(raw, tmp_path / "out", 0.2, 42)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("ratio", [0, 1, -0.2, float("nan"), 0.01])
def test_invalid_or_empty_split_rejected(raw, tmp_path, ratio):
    with pytest.raises(ValueError):
        prepare(raw, tmp_path / "out", ratio, 42)


def test_source_destination_overlap_rejected(raw):
    with pytest.raises(ValueError, match="must not contain each other"):
        prepare(raw, raw / "prepared", 0.2, 42)


def test_existing_empty_destination_is_allowed(raw, tmp_path):
    dst = tmp_path / "prepared"
    dst.mkdir()
    assert prepare(raw, dst, 0.2, 42)["counts"]["val"] == 2
