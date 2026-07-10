import numpy as np

from road_damage.utils.io import decode_image, encode_image, to_base64


def _sample_image():
    return (np.random.rand(32, 32, 3) * 255).astype("uint8")


def test_encode_decode_roundtrip():
    image = _sample_image()
    encoded = encode_image(image, ".png")  # lossless for exact roundtrip
    decoded = decode_image(encoded)
    assert decoded.shape == image.shape
    assert np.array_equal(decoded, image)


def test_to_base64_is_str():
    b64 = to_base64(_sample_image())
    assert isinstance(b64, str)
    assert len(b64) > 0


def test_decode_invalid_raises():
    try:
        decode_image(b"not-an-image")
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
