"""Tests for the libultrahdr-backed Ultra-HDR / ISO 21496-1 codec.

The binding at ``opencodecs.codecs._uhdr`` is a thin wrapper around
libuhdr's own encode / decode entry points — no Python-side gain-map
reimplementation. These tests exercise the round-trip:

  HDR float ndarray → uhdr.encode → Ultra-HDR JPEG bytes
  → uhdr.decode → fp16 HDR + (optional) gain map + SDR base

plus the trivial probes (version string, signature detection) and the
file-on-disk read / write helpers in ``opencodecs.uhdr``.

Skipped at module level if the Cython extension isn't built (e.g. the
host doesn't have libultrahdr installed, which is the typical CI miss
case).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

_uhdr_codecs_mod = pytest.importorskip("opencodecs.codecs._uhdr")

from opencodecs import uhdr as oc_uhdr
from opencodecs.codecs._uhdr import (
    decode,
    encode,
    encode_assembled,
    is_uhdr,
    libuhdr_version,
    UhdrError,
)


# ---------------------------------------------------------------------------
# Trivial probes
# ---------------------------------------------------------------------------


def test_libuhdr_version_returns_dotted_string():
    v = libuhdr_version()
    assert isinstance(v, str)
    parts = v.split(".")
    assert len(parts) == 3
    # libultrahdr 1.x is the published series; allow forward jumps.
    assert all(p.isdigit() for p in parts)


def test_is_uhdr_false_on_garbage():
    assert is_uhdr(b"") is False
    assert is_uhdr(b"abc") is False
    assert is_uhdr(b"\x00" * 32) is False


def test_is_uhdr_false_on_plain_jpeg():
    """A plain (non-Ultra-HDR) JPEG should not register as Ultra-HDR
    — libuhdr's signature check looks for the gain-map metadata box."""
    plain = pytest.importorskip("imagecodecs")
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    jpg = plain.jpeg_encode(arr, level=95)
    assert is_uhdr(jpg) is False


# ---------------------------------------------------------------------------
# Encode → is_uhdr → decode round-trip
# ---------------------------------------------------------------------------


def _synthetic_hdr_rgb(h=64, w=64) -> np.ndarray:
    """A small linear-light Display-P3 HDR raster with a bright corner
    so libuhdr has actual HDR headroom to encode. 1.0 = SDR white."""
    arr = np.zeros((h, w, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    arr[..., 0] = (xx / w) * 4.0    # 0 → 4× SDR
    arr[..., 1] = (yy / h) * 4.0
    arr[..., 2] = ((xx + yy) / (w + h)) * 2.0
    return arr


def test_encode_produces_uhdr_jpeg():
    """Round-trip just the encode → signature check path. Confirms
    libuhdr emitted a real ISO 21496-1 container, not a plain JPEG."""
    hdr = _synthetic_hdr_rgb()
    data = encode(hdr, gamut="display-p3", quality=80)
    assert isinstance(data, (bytes, bytearray))
    assert len(data) > 0
    assert is_uhdr(data) is True


def test_decode_full_roundtrip():
    """Encode then decode; verify shape + dtype + bounded reconstruction
    error. We can't bit-exact-equal because libuhdr is lossy (JPEG base
    + lossy gain-map encoding), but the decoded HDR should track the
    input rasters' coarse structure."""
    hdr = _synthetic_hdr_rgb(64, 64)
    data = encode(hdr, gamut="display-p3", quality=95)

    info = decode(data, want_hdr=True)
    assert set(info.keys()) >= {"hdr_fp16", "width", "height"}
    rec = info["hdr_fp16"]
    assert rec.dtype == np.float16
    # Decoder returns (H, W, 4) RGBA in canonical linear-light units.
    assert rec.shape[:2] == hdr.shape[:2]
    assert rec.shape[2] in (3, 4)
    # Coarse sanity: per-channel mean tracks the input. Ultra-HDR is
    # double-lossy (JPEG base + quantised gain map), so the budget has
    # to be generous — exact-equal is a much stricter contract than
    # the codec was designed to deliver. We just confirm the output
    # isn't pinned to 0 / constant / wildly off.
    rec_rgb = rec[..., :3].astype(np.float32)
    in_mean = hdr.mean(axis=(0, 1))
    out_mean = rec_rgb.mean(axis=(0, 1))
    # All channels actually have signal (not NaN, not zero).
    assert np.all(np.isfinite(out_mean))
    assert np.all(out_mean > 0.05 * in_mean)
    # And within a factor of 4x of input (HDR signals re-quantised
    # through the gain-map round-trip can lose / gain ~2x depending
    # on the encoder's content-boost choice).
    ratio = out_mean / np.maximum(in_mean, 1e-6)
    assert (ratio > 0.25).all() and (ratio < 4.0).all(), (
        f"per-channel ratio out-of-bounds: in={in_mean}, out={out_mean}"
    )


def test_decode_with_gainmap_and_base():
    """want_gainmap=True returns the raw gain map; want_base=True returns
    the embedded SDR JPEG. Both are standalone artefacts the user can
    introspect or save independently."""
    hdr = _synthetic_hdr_rgb()
    data = encode(hdr, quality=95)
    info = decode(
        data,
        want_hdr=False,
        want_gainmap=True,
        want_base=True,
    )
    assert "gainmap_u8" in info
    assert "gainmap_metadata" in info
    assert "base_compressed" in info
    gm = info["gainmap_u8"]
    assert gm.dtype == np.uint8
    assert gm.ndim == 3 and gm.shape[2] in (3, 4)
    meta = info["gainmap_metadata"]
    assert isinstance(meta, dict)
    assert "max_content_boost" in meta
    # base_compressed is a valid standalone JPEG: starts with SOI
    # marker (0xFFD8). HDR-unaware viewers read just this.
    base = info["base_compressed"]
    assert isinstance(base, (bytes, bytearray))
    assert bytes(base[:2]) == b"\xff\xd8"


# ---------------------------------------------------------------------------
# Public file-on-disk wrappers
# ---------------------------------------------------------------------------


def test_write_then_read_roundtrip(tmp_path: Path):
    hdr = _synthetic_hdr_rgb()
    p = tmp_path / "synth.jpg"
    oc_uhdr.write(str(p), hdr, quality=90)
    assert p.exists()
    info = oc_uhdr.read(str(p), want_hdr=True)
    assert info["hdr_fp16"].shape[:2] == hdr.shape[:2]


# ---------------------------------------------------------------------------
# encode_assembled: pre-encoded layers in → libuhdr container out
# ---------------------------------------------------------------------------


def test_encode_assembled_minimal():
    """encode_assembled takes pre-encoded SDR + gain-map JPEGs plus a
    metadata dict and emits an Ultra-HDR container. Confirms the
    direct-binding entry point works without going through encode()."""
    imagecodecs = pytest.importorskip("imagecodecs")
    # Tiny SDR base + tiny gain map, both as plain JPEGs.
    H, W = 32, 32
    sdr = (np.linspace(0, 255, H * W * 3, dtype=np.uint8)
           .reshape(H, W, 3))
    gain = np.full((H, W, 3), 128, dtype=np.uint8)
    sdr_jpeg = imagecodecs.jpeg_encode(sdr, level=85)
    gain_jpeg = imagecodecs.jpeg_encode(gain, level=85)

    metadata = {
        "max_content_boost": 4.0,
        "min_content_boost": 1.0,
        "gamma": 1.0,
        "offset_sdr": 0.0,
        "offset_hdr": 0.0,
        "hdr_capacity_min": 1.0,
        "hdr_capacity_max": 4.0,
        "use_base_cg": True,
    }
    blob = encode_assembled(
        base_jpeg=sdr_jpeg, gainmap_jpeg=gain_jpeg, metadata=metadata,
    )
    assert is_uhdr(blob) is True


# ---------------------------------------------------------------------------
# UhdrError surfaces on malformed input
# ---------------------------------------------------------------------------


def test_decode_garbage_raises():
    with pytest.raises((UhdrError, ValueError, RuntimeError)):
        decode(b"not an ultra-hdr blob, not even close")
