"""OpenJPH / HTJ2K tests.

HTJ2K (High-Throughput JPEG-2000, ISO/IEC 15444-15) replaces JPEG-2000
Part-1's EBCOT block coder with a fast block coder while keeping the
DWT front end. We exercise lossless + irreversible (lossy) modes
across all supported dtypes / channel counts and cross-decode against
imagecodecs.htj2k_decode when present.
"""

from __future__ import annotations

import numpy as np
import pytest

mod = pytest.importorskip("opencodecs.codecs._openjph")
encode = mod.encode
decode = mod.decode
decode_info = mod.decode_info


@pytest.mark.parametrize(
    "shape, dtype, hi",
    [
        ((48, 64), np.uint8, 256),
        ((48, 64, 3), np.uint8, 256),
        ((48, 64, 4), np.uint8, 256),
        ((40, 56), np.uint16, 4000),
        ((40, 56, 3), np.uint16, 4000),
        ((32, 48), np.int8, None),
        ((32, 48), np.int16, None),
    ],
)
def test_openjph_lossless_roundtrip(shape, dtype, hi):
    rng = np.random.default_rng(0)
    if dtype is np.int8:
        arr = rng.integers(-100, 100, size=shape, dtype=dtype)
    elif dtype is np.int16:
        arr = rng.integers(-2000, 2000, size=shape, dtype=dtype)
    else:
        arr = rng.integers(0, hi, size=shape, dtype=dtype)
    enc = encode(arr)
    back = decode(enc)
    assert back.dtype == arr.dtype
    assert back.shape == arr.shape
    np.testing.assert_array_equal(back, arr)


def test_openjph_decode_info_matches_image():
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 4000, size=(40, 56, 3), dtype=np.uint16)
    enc = encode(arr)
    info = decode_info(enc)
    assert info["width"] == 56
    assert info["height"] == 40
    assert info["components"] == 3
    assert info["bit_depth"] == 16
    assert info["signed"] is False


def test_openjph_lossy_shrinks_file():
    """Irreversible mode at a sensible delta beats lossless on bytes."""
    rng = np.random.default_rng(2)
    arr = rng.integers(0, 4000, size=(96, 128), dtype=np.uint16)
    lossless = len(encode(arr))
    lossy = len(encode(arr, level=0.01))
    assert lossy < lossless, (
        f"lossy {lossy} should be smaller than lossless {lossless}"
    )


def test_openjph_lossy_error_decreases_with_smaller_delta():
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 4000, size=(96, 128), dtype=np.uint16)
    a = decode(encode(arr, level=0.01))
    b = decode(encode(arr, level=0.001))
    err_coarse = float(np.abs(a.astype(int) - arr.astype(int)).mean())
    err_fine = float(np.abs(b.astype(int) - arr.astype(int)).mean())
    assert err_fine < err_coarse, (
        f"smaller delta should reduce mean error: "
        f"coarse={err_coarse}, fine={err_fine}"
    )


def test_openjph_imagecodecs_cross_decode():
    """Output decodable by imagecodecs.htj2k_decode if that backend exists."""
    imagecodecs = pytest.importorskip("imagecodecs")
    if not hasattr(imagecodecs, "htj2k_decode"):
        pytest.skip("imagecodecs has no htj2k_decode")
    try:
        # Calling once forces the lazy backend probe to either succeed
        # or raise DelayedImportError if HTJ2K wasn't built in.
        imagecodecs.htj2k_decode(b"\x00")
    except Exception as e:
        if "could not import" in str(e):
            pytest.skip(f"imagecodecs.htj2k_decode unavailable: {e}")
        # Other exceptions just mean the empty input failed parsing —
        # that's expected; the decoder is reachable.
    arr = (np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) * 17 % 4000)
    arr = arr.astype(np.uint16)
    enc = encode(arr)
    via_ic = imagecodecs.htj2k_decode(enc)
    np.testing.assert_array_equal(via_ic.reshape(arr.shape), arr)


def test_openjph_rejects_unsupported_dtype():
    arr = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(Exception):
        encode(arr)


def test_openjph_rejects_5_channel():
    arr = np.zeros((16, 16, 5), dtype=np.uint8)
    with pytest.raises(Exception):
        encode(arr)


def test_openjph_lossy_requires_positive_level():
    arr = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(Exception):
        encode(arr, level=0.0)
    with pytest.raises(Exception):
        encode(arr, level=-0.001)
