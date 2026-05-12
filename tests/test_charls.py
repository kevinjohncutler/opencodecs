"""CharLS / JPEG-LS tests.

JPEG-LS is the predictive lossless / near-lossless JPEG variant used
heavily in DICOM medical imaging. Verified by round-trip + by cross-
decode against imagecodecs.jpegls_decode when present.
"""

from __future__ import annotations

import numpy as np
import pytest

mod = pytest.importorskip("opencodecs.codecs._charls")
encode = mod.encode
decode = mod.decode


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_charls_lossless_grayscale(dtype):
    rng = np.random.default_rng(0)
    if dtype is np.uint8:
        arr = rng.integers(0, 256, size=(64, 96), dtype=dtype)
    else:
        arr = rng.integers(0, 4000, size=(64, 96), dtype=dtype)
    enc = encode(arr)
    back = decode(enc)
    assert back.dtype == arr.dtype
    assert back.shape == arr.shape
    np.testing.assert_array_equal(back, arr)


def test_charls_lossless_rgb_u8():
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    enc = encode(arr)
    back = decode(enc)
    np.testing.assert_array_equal(back, arr)


def test_charls_lossless_rgba_u8():
    rng = np.random.default_rng(2)
    arr = rng.integers(0, 256, size=(32, 48, 4), dtype=np.uint8)
    enc = encode(arr)
    back = decode(enc)
    np.testing.assert_array_equal(back, arr)


@pytest.mark.parametrize("near_lossless", [1, 3, 5])
def test_charls_near_lossless_error_bound(near_lossless):
    """Near-lossless mode guarantees per-sample error <= near_lossless."""
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 4000, size=(64, 96), dtype=np.uint16)
    enc = encode(arr, near_lossless=near_lossless)
    back = decode(enc)
    diff = np.abs(back.astype(int) - arr.astype(int)).max()
    assert diff <= near_lossless, (
        f"near_lossless={near_lossless} but max diff was {diff}"
    )


def test_charls_near_lossless_shrinks_file():
    """Bounded-error mode produces smaller files than lossless."""
    rng = np.random.default_rng(4)
    arr = rng.integers(0, 4000, size=(96, 128), dtype=np.uint16)
    lossless = len(encode(arr, near_lossless=0))
    nl3 = len(encode(arr, near_lossless=3))
    assert nl3 < lossless, f"nl3={nl3} should be < lossless={lossless}"


def test_charls_imagecodecs_cross_decode():
    """Output decodable by imagecodecs.jpegls_decode (the reference
    JPEG-LS decoder)."""
    imagecodecs = pytest.importorskip("imagecodecs")
    if not hasattr(imagecodecs, "jpegls_decode"):
        pytest.skip("imagecodecs has no jpegls_decode")
    arr = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) * 17 % 4000
    arr = arr.astype(np.uint16)
    enc = encode(arr)
    via_ic = imagecodecs.jpegls_decode(enc)
    np.testing.assert_array_equal(via_ic.reshape(arr.shape), arr)


def test_charls_rejects_unsupported_dtype():
    arr = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(Exception):
        encode(arr)


def test_charls_rejects_5_channel():
    arr = np.zeros((16, 16, 5), dtype=np.uint8)
    with pytest.raises(Exception):
        encode(arr)
