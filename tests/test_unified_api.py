"""Tests for opencodecs's unified Codec/Reader/Writer API.

Covers:
- Codec registry: list_codecs, get_codec, has_codec, aliases
- Top-level read/write/open with format= override
- Auto-detection by path extension and by magic bytes
- Round-trips through delegate codecs (PNG, JPEG, BMP, etc.)
- Native JXL via the same Codec interface
- Frame-count and parallel multi-frame decode
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest

import opencodecs as oc


# This file's tests assume the JXL and PNG codecs are registered (most use
# them directly via oc.read/write/get_codec). On platforms where libjxl /
# libspng didn't build, skip the whole file rather than letting each test
# explode with KeyError.
pytestmark = pytest.mark.skipif(
    not (oc.has_codec("jxl") and oc.has_codec("png")),
    reason="this file requires both jxl and png codecs",
)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


def _u8_rgb(h=64, w=64):
    return ((np.arange(h * w * 3) % 256).astype(np.uint8).reshape(h, w, 3))


def _u8_gray(h=64, w=64):
    return ((np.arange(h * w) % 256).astype(np.uint8).reshape(h, w))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_list_codecs_includes_native_jxl():
    names = {c["name"] for c in oc.list_codecs()}
    assert "jxl" in names
    info = next(c for c in oc.list_codecs() if c["name"] == "jxl")
    assert info["native"] is True
    assert info["encode"] and info["decode"]
    assert info["multi_frame"] is True
    assert ".jxl" in info["extensions"]


def test_aliases_resolve_to_same_codec():
    a = oc.get_codec("jxl")
    b = oc.get_codec("jpegxl")
    c = oc.get_codec("jpeg-xl")
    assert a is b is c


def test_has_codec():
    assert oc.has_codec("jxl") is True
    assert oc.has_codec("png") is True
    assert oc.has_codec("nope") is False


def test_get_codec_unknown_raises():
    with pytest.raises(KeyError):
        oc.get_codec("nope")


# ---------------------------------------------------------------------------
# Top-level read/write/open dispatch
# ---------------------------------------------------------------------------


def test_top_level_read_write_format_explicit():
    arr = _u8_rgb()
    data = oc.write(None, arr, format="jxl", lossless=True)
    out = oc.read(data, format="jxl")
    assert np.array_equal(out, arr)


def test_top_level_read_write_path_autodetect(tmp_path):
    arr = _u8_rgb()
    # Native JXL through the unified API by extension
    p = tmp_path / "img.jxl"
    oc.write(p, arr, lossless=True)
    out = oc.read(p)
    assert np.array_equal(out, arr)


@pytest.mark.parametrize("ext,kwargs", [
    (".jxl", {"lossless": True}),  # JXL default is lossy
    (".png", {}),
    (".bmp", {}),
])
def test_path_round_trip_lossless_formats(ext, kwargs, tmp_path):
    """Lossless formats should byte-equal round-trip via path autodetect."""
    arr = _u8_rgb()
    p = tmp_path / f"x{ext}"
    oc.write(p, arr, **kwargs)
    out = oc.read(p)
    assert out.shape == arr.shape
    assert out.dtype == arr.dtype
    assert np.array_equal(out, arr), f"{ext}: {out.dtype} mismatch"


@pytest.mark.skip(
    reason="lossy native codecs (jpeg/webp) not implemented yet")
def test_path_round_trip_lossy_formats_close(tmp_path):
    pass


def test_bytes_autodetect_jxl():
    arr = _u8_rgb()
    data = oc.write(None, arr, format="jxl", lossless=True)
    # No format hint — should sniff JXL signature from the bytes
    out = oc.read(data)
    assert np.array_equal(out, arr)


def test_bytes_autodetect_png():
    arr = _u8_rgb()
    data = oc.write(None, arr, format="png")
    out = oc.read(data)
    assert np.array_equal(out, arr)


def test_open_returns_reader():
    arr = _u8_rgb()
    data = oc.write(None, arr, format="jxl", lossless=True)
    with oc.open(data, format="jxl") as r:
        assert r.shape == arr.shape
        assert r.dtype == arr.dtype
        frames = list(r)
        assert len(frames) == 1
        assert np.array_equal(frames[0], arr)


# ---------------------------------------------------------------------------
# Native JXL through the new interface
# ---------------------------------------------------------------------------


def test_jxl_codec_object_directly():
    arr = _u8_rgb()
    codec = oc.get_codec("jxl")
    data = codec.encode(arr, lossless=True)
    out = codec.decode(data)
    assert np.array_equal(out, arr)


def test_jxl_signature_check():
    arr = _u8_rgb()
    data = oc.write(None, arr, format="jxl", lossless=True)
    assert oc.get_codec("jxl").signature(data[:32]) is True
    # PNG bytes shouldn't match JXL signature
    png = oc.write(None, arr, format="png")
    assert oc.get_codec("jxl").signature(png[:32]) is False


def test_write_to_bytesio_via_top_api():
    arr = _u8_rgb()
    buf = io.BytesIO()
    oc.write(buf, arr, format="jxl", lossless=True)
    buf.seek(0)
    out = oc.read(buf.getvalue(), format="jxl")
    assert np.array_equal(out, arr)


# ---------------------------------------------------------------------------
# Frame-index / parallel multi-frame decode
# ---------------------------------------------------------------------------


def test_frame_count_helper():
    from opencodecs.parallel import frame_count
    n = 5
    stack = np.stack([_u8_rgb(32, 32) + i for i in range(n)], axis=0)
    data = oc.write(None, stack, format="jxl", lossless=True, animation=True)
    assert frame_count(data) == n


def test_decode_frame_by_index():
    from opencodecs.codecs._jxl import decode as jxl_decode
    n = 4
    stack = np.stack([_u8_rgb(32, 32) + i for i in range(n)], axis=0)
    data = oc.write(None, stack, format="jxl", lossless=True, animation=True)
    for i in range(n):
        f = jxl_decode(data, index=i)
        assert np.array_equal(f, stack[i]), f"frame {i} mismatch"


def test_decode_frames_parallel():
    from opencodecs.parallel import decode_frames_parallel
    n = 8
    stack = np.stack([_u8_rgb(64, 64) + i for i in range(n)], axis=0)
    data = oc.write(None, stack, format="jxl", lossless=True, animation=True)

    # Default: all frames in order
    frames = decode_frames_parallel(data, n_workers=4)
    assert len(frames) == n
    for i, f in enumerate(frames):
        assert np.array_equal(f, stack[i]), f"frame {i} mismatch"

    # Explicit indices, out of order
    frames = decode_frames_parallel(data, indices=[3, 0, 7], n_workers=2)
    assert len(frames) == 3
    assert np.array_equal(frames[0], stack[3])
    assert np.array_equal(frames[1], stack[0])
    assert np.array_equal(frames[2], stack[7])


def test_decode_frames_parallel_n_workers_1_fast_path():
    """n_workers=1 should use sequential iter_frames (avoiding O(N^2) skips)."""
    from opencodecs.parallel import decode_frames_parallel
    n = 6
    stack = np.stack([_u8_rgb(32, 32) + i for i in range(n)], axis=0)
    data = oc.write(None, stack, format="jxl", lossless=True, animation=True)
    frames = decode_frames_parallel(data, n_workers=1)
    assert len(frames) == n
    for i, f in enumerate(frames):
        assert np.array_equal(f, stack[i])


# ---------------------------------------------------------------------------
# Reader interface uniformity
# ---------------------------------------------------------------------------


def test_reader_supplies_shape_dtype_eagerly():
    arr = _u8_rgb(80, 100)
    data = oc.write(None, arr, format="jxl", lossless=True)
    with oc.open(data, format="jxl") as r:
        # Shape/dtype available before decoding any pixels
        assert r.shape == (80, 100, 3)
        assert r.dtype == np.uint8


def test_reader_random_access_jxl():
    n = 4
    stack = np.stack([_u8_rgb(32, 32) + i for i in range(n)], axis=0)
    data = oc.write(None, stack, format="jxl", lossless=True, animation=True)
    with oc.open(data, format="jxl") as r:
        # Default Reader.__getitem__ is O(N) — works but doesn't beat
        # parallel decode. We just verify it returns the right frames.
        f0 = r[0]
        assert np.array_equal(f0, stack[0])


# ---------------------------------------------------------------------------
# Native PNG (libspng) smoke test
# ---------------------------------------------------------------------------


def test_png_codec_repr():
    c = oc.get_codec("png")
    assert "png" in repr(c)
    assert "native" in repr(c)


def test_png_codec_encode_decode_bytes():
    arr = _u8_rgb()
    codec = oc.get_codec("png")
    data = codec.encode(arr)
    out = codec.decode(data)
    assert np.array_equal(out, arr)
