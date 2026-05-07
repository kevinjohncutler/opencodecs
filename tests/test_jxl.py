"""Tests for the streaming JPEG XL codec.

Covers:
- Round-trip for uint8 / uint16 / float16 / float32, in L / RGB / RGBA layouts
- Lossless and lossy modes
- Header parsing without pixel decode (streaming reader exposes shape/dtype eagerly)
- Multi-frame animation: encode N frames, iter_frames yields N arrays
- Color encoding: sRGB default, Display P3, BT.2100 PQ (HDR)
- Path / file-like / bytes / BytesIO destinations
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

import opencodecs.jxl as jxl
from opencodecs.core.color import (
    DISPLAY_P3,
    REC2020_PQ,
    REC2020_HLG,
    SRGB,
    JXL_PRIMARIES_P3,
    JXL_PRIMARIES_2100,
    JXL_TF_PQ,
    JXL_TF_HLG,
    parse_color,
)

# Skip the entire module on platforms without the libjxl Cython extension
# built (Windows / Linux without libjxl-dev). The optional-backend pattern
# means `import opencodecs.jxl` itself succeeds, but JxlReader/JxlWriter
# only resolve when the backend is present.
pytestmark = pytest.mark.skipif(
    not jxl._HAVE_BACKEND,
    reason="libjxl backend not available",
)

if jxl._HAVE_BACKEND:
    from opencodecs.codecs._jxl import JxlReader, JxlWriter
else:  # pragma: no cover - skip path
    JxlReader = JxlWriter = None  # type: ignore[assignment]


# ----------------------------- helpers --------------------------------------


def _grad_uint8(h=64, w=64, c=3):
    return ((np.arange(h * w * c) % 256).astype(np.uint8).reshape(h, w, c)
            if c > 1 else (np.arange(h * w) % 256).astype(np.uint8).reshape(h, w))


def _grad_uint16(h=64, w=64, c=3):
    return ((np.arange(h * w * c) % 65535).astype(np.uint16).reshape(h, w, c)
            if c > 1 else (np.arange(h * w) % 65535).astype(np.uint16).reshape(h, w))


def _grad_float32(h=64, w=64, c=3):
    return (np.arange(h * w * c).astype(np.float32) / (h * w * c)).reshape(h, w, c) \
        if c > 1 else (np.arange(h * w).astype(np.float32) / (h * w)).reshape(h, w)


# ----------------------------- basics ---------------------------------------


def test_libjxl_version_string():
    v = jxl.libjxl_version()
    parts = v.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_check_signature_negative():
    assert jxl.check_signature(b"") is False
    assert jxl.check_signature(b"\x00" * 16) is False


def test_check_signature_positive():
    arr = _grad_uint8()
    data = jxl.write(None, arr, lossless=True)
    assert jxl.check_signature(data) is True


# ----------------------------- round-trip dtypes ----------------------------


@pytest.mark.parametrize("dtype_kind, channels", [
    ("uint8", 1),
    ("uint8", 3),
    ("uint8", 4),
    ("uint16", 1),
    ("uint16", 3),
    ("float16", 3),
    ("float32", 3),
])
def test_lossless_roundtrip(dtype_kind, channels):
    h, w = 48, 64
    if dtype_kind == "uint8":
        arr = _grad_uint8(h, w, channels) if channels > 1 else _grad_uint8(h, w, 1).reshape(h, w)
    elif dtype_kind == "uint16":
        arr = _grad_uint16(h, w, channels) if channels > 1 else _grad_uint16(h, w, 1).reshape(h, w)
    elif dtype_kind == "float16":
        arr = _grad_float32(h, w, channels).astype(np.float16)
    else:
        arr = _grad_float32(h, w, channels)

    data = jxl.write(None, arr, lossless=True)
    out = jxl.read(data)

    assert out.shape == arr.shape
    assert out.dtype == arr.dtype
    if dtype_kind in ("uint8", "uint16"):
        assert np.array_equal(out, arr)
    else:
        # Float lossless should still be exact
        assert np.allclose(out, arr, rtol=0, atol=0)


def test_grayscale_2d_roundtrip():
    """A (Y, X) grayscale image should round-trip with the same shape (no
    trailing channel axis)."""
    arr = _grad_uint8(40, 50, 1).reshape(40, 50)
    data = jxl.write(None, arr, lossless=True)
    out = jxl.read(data)
    assert out.shape == arr.shape
    assert np.array_equal(out, arr)


def test_lossy_roundtrip_close():
    arr = _grad_uint8(64, 64, 3)
    data = jxl.write(None, arr, distance=1.0, effort=3)
    out = jxl.read(data)
    assert out.shape == arr.shape
    # distance 1.0 is "visually lossless" — we just check it's close
    diff = np.abs(out.astype(int) - arr.astype(int))
    assert diff.mean() < 5.0


# ----------------------------- header / streaming reader --------------------


def test_reader_parses_header_eagerly():
    arr = _grad_uint8(80, 100, 3)
    data = jxl.write(None, arr, lossless=True)

    with jxl.open(data) as r:
        # No pixels decoded yet, but shape/dtype known
        assert r.xsize == 100
        assert r.ysize == 80
        assert r.samples == 3
        assert r.frame_shape == (80, 100, 3)
        assert r.dtype == np.uint8
        assert r.is_animation is False

    # After context exit, .close() was called; properties still work in place
    # but we shouldn't decode further.


def test_reader_iter_frames_single():
    arr = _grad_uint8(48, 48, 3)
    data = jxl.write(None, arr, lossless=True)

    with jxl.open(data) as r:
        frames = list(r.iter_frames())

    assert len(frames) == 1
    assert np.array_equal(frames[0], arr)


def test_iter_frames_helper():
    """`jxl.iter_frames(...)` is a stand-alone generator function."""
    arr = _grad_uint8(16, 16, 3)
    data = jxl.write(None, arr, lossless=True)
    frames = list(jxl.iter_frames(data))
    assert len(frames) == 1
    assert np.array_equal(frames[0], arr)


# ----------------------------- multi-frame ----------------------------------


def test_multi_frame_animation_roundtrip():
    """Encode N independent frames as an animation, iterate them back."""
    n = 4
    stack = np.stack([_grad_uint8(32, 40, 3) + i for i in range(n)], axis=0)

    # Use the streaming writer directly so the test exercises write_frame()
    w = JxlWriter(None, lossless=True, animation=True)
    for i in range(n):
        w.write_frame(stack[i], is_last=(i == n - 1))
    data = w.close()

    assert isinstance(data, bytes) and len(data) > 0

    # Decode back
    with jxl.open(data) as r:
        assert r.is_animation
        frames = list(r.iter_frames())

    assert len(frames) == n
    for i, frame in enumerate(frames):
        assert np.array_equal(frame, stack[i]), f"frame {i} mismatch"


def test_multi_frame_via_encode_helper():
    """The encode() helper accepts a (T, Y, X, C) stack with animation=True."""
    n = 3
    stack = np.stack([_grad_uint8(24, 32, 3) + i for i in range(n)], axis=0)
    data = jxl.write(None, stack, lossless=True, animation=True)
    out = jxl.read(data)
    # Multi-frame read returns (T, Y, X, C)
    assert out.shape == stack.shape
    assert np.array_equal(out, stack)


def test_writer_rejects_extra_frame_when_not_animation():
    arr = _grad_uint8(16, 16, 3)
    w = JxlWriter(None, lossless=True, animation=False)
    w.write_frame(arr)
    with pytest.raises(RuntimeError, match="animation"):
        w.write_frame(arr)
    w.close()


# ----------------------------- color: P3 / HDR ------------------------------


def test_display_p3_color_roundtrip():
    arr = _grad_uint8(32, 32, 3)
    data = jxl.write(None, arr, color="display-p3", lossless=True)

    with jxl.open(data) as r:
        # Color enum should reflect P3 primaries
        assert r.color is not None
        assert r.color["primaries"] == JXL_PRIMARIES_P3

    out = jxl.read(data)
    assert np.array_equal(out, arr)


def test_rec2020_pq_hdr_roundtrip():
    arr = _grad_float32(32, 32, 3)
    data = jxl.write(None, arr, color="rec2020-pq", lossless=True)

    with jxl.open(data) as r:
        assert r.color is not None
        assert r.color["primaries"] == JXL_PRIMARIES_2100
        assert r.color["transfer_function"] == JXL_TF_PQ
        # HDR streams set uses_original_profile so the transfer is preserved
        assert r.basic_info["uses_original_profile"] is True

    out = jxl.read(data)
    assert np.allclose(out, arr)


def test_rec2020_hlg_color_tag():
    arr = _grad_float32(32, 32, 3)
    data = jxl.write(None, arr, color=REC2020_HLG, lossless=True)
    with jxl.open(data) as r:
        assert r.color["primaries"] == JXL_PRIMARIES_2100
        assert r.color["transfer_function"] == JXL_TF_HLG


def test_color_alias_resolution():
    assert parse_color("p3") == DISPLAY_P3
    assert parse_color("display-p3") == DISPLAY_P3
    assert parse_color("rec2020-pq") == REC2020_PQ
    assert parse_color("bt2020-pq") == REC2020_PQ
    assert parse_color("srgb") == SRGB
    assert parse_color(None) is None
    assert parse_color(DISPLAY_P3) is DISPLAY_P3
    with pytest.raises(ValueError):
        parse_color("not-a-color")


# ----------------------------- destinations ---------------------------------


def test_write_to_path(tmp_path):
    arr = _grad_uint8(40, 40, 3)
    out_path: Path = tmp_path / "out.jxl"
    result = jxl.write(out_path, arr, lossless=True)
    assert result is None  # streaming to file: no bytes returned
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    out = jxl.read(out_path)
    assert np.array_equal(out, arr)


def test_write_to_bytesio():
    arr = _grad_uint8(40, 40, 3)
    buf = io.BytesIO()
    jxl.write(buf, arr, lossless=True)
    buf.seek(0)
    out = jxl.read(buf.getvalue())
    assert np.array_equal(out, arr)


def test_open_from_path(tmp_path):
    arr = _grad_uint8(24, 24, 3)
    out_path = tmp_path / "x.jxl"
    jxl.write(out_path, arr, lossless=True)
    with jxl.open(out_path) as r:
        frames = list(r.iter_frames())
    assert np.array_equal(frames[0], arr)


# ----------------------------- error cases ----------------------------------


def test_invalid_input_bytes():
    with pytest.raises(ValueError):
        jxl.read(b"not a jxl stream at all")


def test_dtype_mismatch_between_frames():
    w = JxlWriter(None, lossless=True, animation=True)
    w.write_frame(_grad_uint8(16, 16, 3))
    with pytest.raises(ValueError):
        w.write_frame(_grad_uint16(16, 16, 3))
    w.close()


def test_quality_and_distance_are_mutually_exclusive():
    with pytest.raises(ValueError):
        JxlWriter(None, quality=80, distance=1.0)
