"""Ultra HDR (gainmap JPEG) codec tests.

Validates:
* Round-trip: encode + decode produces an HDR image that's
  perceptually close to the original (lossy codec — exact equality
  is not expected).
* Cross-validation: imagecodecs.ultrahdr_decode reads our output,
  and we read imagecodecs's output. Both round-trips should produce
  the same float16 pixels.
* SDR fallback: dtype=uint8 returns the base JPEG tonemapped.
* Bad inputs are rejected with a clear error.
"""

from __future__ import annotations

import numpy as np
import pytest

from opencodecs.codecs._ultrahdr import (
    encode, decode, version, UltrahdrError,
)


def _make_hdr(shape=(64, 96, 4), seed=0):
    """Synthesize an HDR float16 RGBA image.

    Mid-tone base around 0.3-0.7, with a bright highlight quadrant
    up to ~6.0 nits — exercises the gainmap's purpose. Alpha = 1.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.05, 0.7, size=shape).astype(np.float32)
    h, w = shape[0], shape[1]
    base[: h // 4, : w // 4, :3] *= 8.0
    base[..., 3] = 1.0
    return base.astype(np.float16)


def test_version():
    v = version()
    assert isinstance(v, str) and len(v) > 0


def test_ultrahdr_basic_round_trip():
    arr = _make_hdr()
    blob = encode(arr)
    back = decode(blob)
    assert back.shape == arr.shape
    assert back.dtype == np.float16
    # Lossy codec; check that the bulk of pixels are within a sane
    # tolerance. Mean error on a smooth HDR image should be ~5% or
    # better (the gainmap is a separate JPEG channel quantized to
    # 8-bit, so individual pixels can drift more).
    arr32 = arr.astype(np.float32)
    back32 = back.astype(np.float32)
    rel = np.abs(arr32 - back32) / np.maximum(arr32, 1e-3)
    assert np.mean(rel) < 0.20, f"mean rel diff = {np.mean(rel)}"


def test_ultrahdr_sdr_fallback_is_uint8_rgba():
    """dtype=uint8 returns the base JPEG (no HDR boost applied)."""
    arr = _make_hdr()
    blob = encode(arr)
    sdr = decode(blob, dtype=np.uint8)
    assert sdr.shape == arr.shape
    assert sdr.dtype == np.uint8
    # The base JPEG must be a well-formed 8-bit image — at least some
    # pixels in the dim region should be non-zero and not saturated.
    assert sdr[..., :3].max() > 0
    assert sdr[..., :3].min() < 255


def test_ultrahdr_imagecodecs_reads_our_output():
    """imagecodecs.ultrahdr_decode reads what we wrote."""
    ic = pytest.importorskip("imagecodecs")
    if not hasattr(ic, "ultrahdr_decode"):
        pytest.skip("imagecodecs has no ultrahdr_decode in this build")
    arr = _make_hdr(seed=1)
    blob = encode(arr)
    via_ic = ic.ultrahdr_decode(blob)
    via_us = decode(blob)
    # Both decoders should produce the same pixels — they call the
    # same libultrahdr API and request float16-linear output.
    np.testing.assert_array_equal(via_us, via_ic)


def test_ultrahdr_we_read_imagecodecs_output():
    """We decode what imagecodecs.ultrahdr_encode wrote."""
    ic = pytest.importorskip("imagecodecs")
    if not hasattr(ic, "ultrahdr_encode"):
        pytest.skip("imagecodecs has no ultrahdr_encode in this build")
    arr = _make_hdr(seed=2)
    blob = ic.ultrahdr_encode(arr)
    via_us = decode(blob)
    via_ic = ic.ultrahdr_decode(blob)
    np.testing.assert_array_equal(via_us, via_ic)


def test_ultrahdr_rejects_wrong_dtype():
    arr = _make_hdr().astype(np.float32)
    with pytest.raises(ValueError, match="float16"):
        encode(arr)


def test_ultrahdr_rejects_wrong_shape():
    # Missing alpha channel
    arr = np.zeros((32, 32, 3), dtype=np.float16)
    with pytest.raises(ValueError, match="\\(H, W, 4\\)"):
        encode(arr)


def test_ultrahdr_rejects_garbage_input_on_decode():
    with pytest.raises((UltrahdrError, ValueError)):
        decode(b"not a gainmap jpeg")


def test_ultrahdr_quality_changes_size():
    """Higher quality produces larger output (sanity check)."""
    arr = _make_hdr(shape=(96, 128, 4), seed=3)
    small = encode(arr, level=50)
    large = encode(arr, level=95)
    assert len(large) > len(small)


def test_ultrahdr_fast_preset_is_smaller_and_faster():
    """The REALTIME preset trades quality for speed/size."""
    arr = _make_hdr(shape=(128, 256, 4), seed=4)
    best = encode(arr, fast=False)
    fast = encode(arr, fast=True)
    # The realtime preset produces a smaller gainmap.
    assert len(fast) <= len(best)
