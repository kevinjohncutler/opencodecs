"""Tests for the Phase-6 new codec wrappers (bcn / rcomp / dicomrle).

Phase 6 also lists apng / ultrahdr — those are deferred:

* apng — needs libspng APNG (animation) support, which lives behind
  a build-time flag we haven't enabled. Tracked as a follow-up.
* ultrahdr — needs Google's libultrahdr, which we don't yet link.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc


# ---------------------------------------------------------------------------
# bcn — top-level BC1-7 codec
# ---------------------------------------------------------------------------


def test_bcn_codec_registered():
    assert oc.has_codec("bcn")
    entry = next(c for c in oc.list_codecs() if c["name"] == "bcn")
    assert entry["decode"] is True
    # We don't yet ship a BCn encoder.
    assert entry["encode"] is False


@pytest.mark.parametrize("fmt,block_bytes", [
    ("bc1", 8), ("bc2", 16), ("bc3", 16), ("bc7", 16),
])
def test_bcn_codec_rgba_variants(fmt, block_bytes):
    c = oc.get_codec("bcn")
    rng = np.random.default_rng(0)
    w, h = 16, 12   # both multiples of 4
    n_blocks = (w // 4) * (h // 4)
    blob = rng.integers(0, 256, size=n_blocks * block_bytes,
                        dtype=np.uint8).tobytes()
    out = c.decode(blob, format=fmt, width=w, height=h)
    assert out.shape == (h, w, 4)
    assert out.dtype == np.uint8


def test_bcn_codec_bc4_grayscale():
    c = oc.get_codec("bcn")
    rng = np.random.default_rng(1)
    blob = rng.integers(0, 256, size=4 * 8, dtype=np.uint8).tobytes()
    out = c.decode(blob, format="bc4", width=8, height=8)
    assert out.shape == (8, 8)
    assert out.dtype == np.uint8


def test_bcn_codec_bc6h_float():
    c = oc.get_codec("bcn")
    rng = np.random.default_rng(2)
    blob = rng.integers(0, 256, size=4 * 16, dtype=np.uint8).tobytes()
    out = c.decode(blob, format="bc6h", width=8, height=8)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.float32
    out_half = c.decode(blob, format="bc6h", width=8, height=8, fp16=True)
    assert out_half.dtype == np.float16


def test_bcn_codec_aliases():
    """``dxt1`` / ``dxt3`` / ``dxt5`` are the Direct3D legacy names
    for BC1 / BC2 / BC3. Same decoder."""
    c = oc.get_codec("bcn")
    rng = np.random.default_rng(3)
    blob = rng.integers(0, 256, size=4 * 8, dtype=np.uint8).tobytes()
    a = c.decode(blob, format="bc1", width=8, height=8)
    b = c.decode(blob, format="dxt1", width=8, height=8)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# rcomp — Rice compression (FITS-style)
# ---------------------------------------------------------------------------


def test_rcomp_codec_registered():
    assert oc.has_codec("rcomp")
    assert oc.has_codec("rice")          # alias
    assert oc.has_codec("ricecomp")      # alias


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32])
def test_rcomp_signed_roundtrip(dtype):
    c = oc.get_codec("rcomp")
    rng = np.random.default_rng(0)
    info = np.iinfo(dtype)
    arr = rng.integers(max(info.min, -1000), min(info.max, 1000),
                        size=4096, dtype=dtype)
    blob = c.encode(arr)
    back = c.decode(blob).astype(arr.dtype)
    np.testing.assert_array_equal(back, arr)


def test_rcomp_constant_compresses_well():
    """A constant stream should hit very high compression ratio."""
    c = oc.get_codec("rcomp")
    arr = np.zeros(8192, dtype=np.int16)
    blob = c.encode(arr)
    # Header (12 bytes) + a few bytes per block (8192/32 = 256 blocks
    # × ~1.5 bytes each). Should be well under 10% of raw.
    assert len(blob) * 10 < arr.nbytes, (
        f"rcomp on zeros: {len(blob)} bytes for {arr.nbytes} raw "
        "(expected <10%)")


def test_rcomp_unsigned_roundtrip_via_dtype_kwarg():
    c = oc.get_codec("rcomp")
    arr = np.random.default_rng(1).integers(
        0, 1000, size=2048, dtype=np.uint16)
    blob = c.encode(arr)
    back = c.decode(blob, dtype=np.uint16)
    np.testing.assert_array_equal(back, arr)


# ---------------------------------------------------------------------------
# dicomrle — DICOM RLE Image Compression
# ---------------------------------------------------------------------------


def test_dicomrle_codec_registered():
    assert oc.has_codec("dicomrle")
    assert oc.has_codec("dcmrle")        # alias


def test_dicomrle_8bit_gray_roundtrip():
    c = oc.get_codec("dicomrle")
    arr = np.random.default_rng(0).integers(0, 256, (32, 64),
                                             dtype=np.uint8)
    blob = c.encode(arr)
    back = c.decode(blob, shape=arr.shape, dtype=arr.dtype)
    np.testing.assert_array_equal(back, arr)


def test_dicomrle_16bit_gray_roundtrip():
    c = oc.get_codec("dicomrle")
    arr = np.random.default_rng(1).integers(0, 65536, (32, 64),
                                             dtype=np.uint16)
    blob = c.encode(arr)
    back = c.decode(blob, shape=arr.shape, dtype=arr.dtype)
    np.testing.assert_array_equal(back, arr)


def test_dicomrle_8bit_rgb_roundtrip():
    c = oc.get_codec("dicomrle")
    arr = np.random.default_rng(2).integers(0, 256, (16, 32, 3),
                                             dtype=np.uint8)
    blob = c.encode(arr)
    back = c.decode(blob, shape=arr.shape, dtype=arr.dtype)
    np.testing.assert_array_equal(back, arr)


def test_dicomrle_constant_image_packs_tight():
    """RLE on a constant should beat raw by a large margin. PackBits
    caps replicate runs at 128 bytes, so a 128×128 image of one value
    encodes as 16384/128 = 128 replicate runs × 2 bytes each + 64-byte
    header = 320 bytes — still ~50× smaller than raw."""
    c = oc.get_codec("dicomrle")
    arr = np.full((128, 128), 42, dtype=np.uint8)
    blob = c.encode(arr)
    assert len(blob) * 20 < arr.nbytes, (
        f"constant 128×128 should beat raw by 20× minimum, "
        f"got {len(blob)} / {arr.nbytes}")
