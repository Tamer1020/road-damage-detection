import numpy as np

from road_damage.detection import Detection
from road_damage.visualize import draw_detections


def test_draw_keeps_shape():
    image = np.zeros((100, 100, 3), dtype="uint8")
    dets = [Detection(class_id=0, label="D00", confidence=0.9, bbox=(10, 10, 50, 50))]
    out = draw_detections(image, dets)
    assert out.shape == image.shape
    # Something was drawn, so the output differs from the blank input.
    assert not np.array_equal(out, image)


def test_draw_empty_returns_copy():
    image = np.zeros((20, 20, 3), dtype="uint8")
    out = draw_detections(image, [])
    assert np.array_equal(out, image)
    assert out is not image  # must be a copy, not the same buffer
