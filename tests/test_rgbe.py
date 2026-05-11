"""RGBE / Radiance HDR codec tests.

Validates against:

* Round-trip: encode + decode our own output, confirm pixel-equal
  within RGBE's quantization (~0.4% of channel magnitude).
* imagecodecs reference reader where available: imagecodecs.rgbe_decode
  reads our encoded bytes and produces matching pixels.
"""

from __future__ import annotations

import numpy as np
import pytest

from opencodecs._rgbe import encode, decode, RgbeError


def _make_hdr(shape=(48, 96, 3), seed=0):
    """Synthesize an HDR-ish RGB float32 array."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.05, 8.0, size=shape).astype(np.float32)
    # Hotspots up to ~100 in the upper-left quadrant — exercises the
    # high-exponent path.
    base[: shape[0] // 4, : shape[1] // 4, :] *= 12
    # Some pixels exactly zero — exercises the all-zero exponent path.
    base[-1, -1] = 0.0
    return base


def test_rgbe_basic_round_trip():
    arr = _make_hdr()
    raw = encode(arr)
    back = decode(raw)
    # RGBE quantization error: 8-bit mantissa shared across channels,
    # so per-pixel relative error can be up to ~1% on channels far
    # smaller than the per-pixel max (the shared exponent quantizes
    # to the brightest channel). Compare relative max-diff per channel.
    pixel_max = np.maximum(np.abs(arr), np.abs(back)).max(axis=-1, keepdims=True)
    rel = np.abs(arr - back) / np.maximum(pixel_max, 1e-6)
    assert rel.max() < 0.01, f"max rel diff = {rel.max()}"


def test_rgbe_zero_pixel_preserved():
    arr = _make_hdr()
    arr[5, 7] = 0.0
    back = decode(encode(arr))
    assert np.all(back[5, 7] == 0.0)


def test_rgbe_high_dynamic_range():
    """Values across many stops of magnitude all encode correctly."""
    arr = np.zeros((4, 4, 3), dtype=np.float32)
    arr[0, 0] = 1e-3
    arr[0, 1] = 1.0
    arr[0, 2] = 1e3
    arr[0, 3] = 1e5
    back = decode(encode(arr))
    for col in range(4):
        rel = np.abs(arr[0, col] - back[0, col]).max() / max(arr[0, col].max(), 1e-6)
        assert rel < 0.01, f"col {col}: arr={arr[0, col]} back={back[0, col]}"


def test_rgbe_imagecodecs_can_read_our_output():
    """imagecodecs.rgbe_decode reads what we wrote (it's the
    well-tested reference reader)."""
    imagecodecs = pytest.importorskip("imagecodecs")
    if not hasattr(imagecodecs, "rgbe_decode"):
        pytest.skip("imagecodecs has no rgbe_decode in this build")
    arr = _make_hdr(shape=(32, 64, 3), seed=1)
    raw = encode(arr)
    via_ic = imagecodecs.rgbe_decode(raw)
    # imagecodecs's rgbe_decode also returns float32 (H, W, 3).
    assert via_ic.shape == arr.shape
    # Compare against our own decoder (both should give the same
    # RGBE-quantized pixels).
    via_us = decode(raw)
    np.testing.assert_array_equal(via_us, via_ic)


def test_rgbe_we_can_read_imagecodecs_output():
    """And the reverse: we decode what imagecodecs encoded."""
    imagecodecs = pytest.importorskip("imagecodecs")
    if not hasattr(imagecodecs, "rgbe_encode"):
        pytest.skip("imagecodecs has no rgbe_encode in this build")
    arr = _make_hdr(shape=(40, 80, 3), seed=2)
    raw = imagecodecs.rgbe_encode(arr)
    back = decode(raw)
    via_ic = imagecodecs.rgbe_decode(raw)
    np.testing.assert_array_equal(back, via_ic)


def test_rgbe_short_width_uncompressed_path():
    """Widths < 8 can't use new-style RLE (the RLE header would clash
    with a literal byte). We fall back to uncompressed; both encode
    and decode must handle that."""
    rng = np.random.default_rng(3)
    arr = rng.uniform(0.0, 5.0, size=(4, 4, 3)).astype(np.float32)
    back = decode(encode(arr))
    rel = np.abs(arr - back).max() / arr.max()
    assert rel < 0.01


def test_rgbe_invalid_header_raises():
    with pytest.raises(RgbeError):
        decode(b"not a hdr file")
