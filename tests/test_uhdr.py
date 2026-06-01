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
    compute_gain_map_u8,
    compute_sdr_base_u8,
    decode,
    encode,
    encode_assembled,
    is_uhdr,
    libuhdr_version,
    UhdrError,
)
from opencodecs.uhdr import encode_native, decode_native, encode_to


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
# encode_native: fused-Cython fast path
# ---------------------------------------------------------------------------


def test_encode_native_produces_uhdr():
    """The native fast path computes SDR + gain map outside libuhdr and
    hands pre-encoded JPEGs to libuhdr's container-assembly api. Output
    must still be a valid Ultra-HDR JPEG."""
    hdr = _synthetic_hdr_rgb()
    data = encode_native(hdr, gamut="display-p3", quality=95)
    assert isinstance(data, (bytes, bytearray))
    assert len(data) > 0
    assert is_uhdr(data) is True


def test_encode_native_roundtrip_matches_libuhdr():
    """encode_native and encode should produce decoded HDR rasters in
    roughly the same neighbourhood — gain-map quantization + JPEG-q95
    noise is the dominant error, not the kernel implementation choice."""
    hdr = _synthetic_hdr_rgb(64, 64)
    data_n = encode_native(hdr, quality=95)
    data_r = encode(hdr, quality=95)
    rec_n = decode(data_n, want_hdr=True)["hdr_fp16"][..., :3].astype(
        np.float32)
    rec_r = decode(data_r, want_hdr=True)["hdr_fp16"][..., :3].astype(
        np.float32)
    # Mean per-channel ratio should be within 2x in either direction
    # (both encoders normalize by ~peak, so they share the same coarse
    # scale; q95 + 8-bit gain-map quantisation contributes the residual).
    mn = rec_n.mean(axis=(0, 1))
    mr = rec_r.mean(axis=(0, 1))
    assert np.all(np.isfinite(mn)) and np.all(np.isfinite(mr))
    ratio = mn / np.maximum(mr, 1e-6)
    assert (ratio > 0.5).all() and (ratio < 2.0).all(), (
        f"native vs ref decoded means diverge: native={mn}, ref={mr}")


def test_encode_native_accepts_user_sdr_base():
    """Caller-supplied SDR base path: encode_native skips its own
    peak-normalize and uses the user's SDR exactly. Confirms the
    'I want my own base supplied' contract — useful for tilescan
    callers that already have a tonemapped SDR."""
    hdr = _synthetic_hdr_rgb(64, 64)
    # Simple peak-normalize SDR base.
    peak = float(hdr.max())
    sdr = compute_sdr_base_u8(hdr, peak=peak)
    data = encode_native(
        hdr, sdr=sdr, quality=95, max_content_boost=peak,
    )
    assert is_uhdr(data) is True
    info = decode(data, want_hdr=True)
    assert info["hdr_fp16"].shape[:2] == hdr.shape[:2]


def test_encode_native_parallel_matches_serial():
    """parallel=True overlaps SDR JPEG encode + gain-map compute +
    gain-map JPEG encode via a ThreadPoolExecutor. Result must be
    byte-identical to the serial path (same kernels, same inputs)."""
    hdr = _synthetic_hdr_rgb(64, 64)
    data_p = encode_native(hdr, parallel=True, quality=90)
    data_s = encode_native(hdr, parallel=False, quality=90)
    assert data_p == data_s


def test_encode_native_rejects_bad_shape():
    with pytest.raises(ValueError):
        encode_native(np.zeros((64, 64), dtype=np.float32))
    with pytest.raises(ValueError):
        encode_native(np.zeros((64, 64, 4), dtype=np.float32))


def test_compute_gain_map_and_sdr_helpers():
    """The standalone helpers (compute_sdr_base_u8 +
    compute_gain_map_u8) should produce shape-correct uint8 outputs
    plus a metadata dict with all libuhdr-required keys."""
    hdr = _synthetic_hdr_rgb(32, 32)
    sdr = compute_sdr_base_u8(hdr)
    assert sdr.dtype == np.uint8
    assert sdr.shape == hdr.shape
    gain, meta = compute_gain_map_u8(hdr, sdr)
    assert gain.dtype == np.uint8
    assert gain.shape == hdr.shape
    for key in ("max_content_boost", "min_content_boost", "gamma",
                "hdr_capacity_min", "hdr_capacity_max"):
        assert key in meta


# ---------------------------------------------------------------------------
# decode_native: fused-Cython fast-path decoder
# ---------------------------------------------------------------------------


def test_decode_native_returns_expected_keys():
    """decode_native returns hdr + sdr_u8 + gainmap_u8 + metadata."""
    hdr = _synthetic_hdr_rgb(64, 64)
    data = encode_native(hdr, quality=95)
    info = decode_native(data)
    assert set(info.keys()) >= {
        "hdr", "sdr_u8", "gainmap_u8", "gainmap_metadata",
        "width", "height",
    }
    assert info["hdr"].dtype == np.float16
    assert info["hdr"].shape == (64, 64, 3)
    assert info["sdr_u8"].dtype == np.uint8
    assert info["gainmap_u8"].dtype == np.uint8
    assert info["width"] == 64 and info["height"] == 64


def test_decode_native_matches_libuhdr_at_sdr_boost():
    """At display_boost=1.0 decode_native should match libuhdr's
    default decode (also display_boost=1.0) within JPEG-q95 + 8-bit
    gain-quantisation noise."""
    hdr = _synthetic_hdr_rgb(64, 64)
    data = encode_native(hdr, quality=95)

    lib = decode(data, want_hdr=True)["hdr_fp16"][..., :3].astype(np.float32)
    nat = decode_native(data, dtype=np.float32, display_boost=1.0)["hdr"]

    # Per-channel mean diff should be small (~JPEG q95 noise).
    diff = np.abs(lib - nat)
    assert diff.mean() < 0.05, (
        f"native vs libuhdr decode mean diff {diff.mean():.4f} > 0.05"
    )
    # Channel-wise means should track each other within ~25%.
    ratio = nat.mean(axis=(0, 1)) / np.maximum(lib.mean(axis=(0, 1)), 1e-6)
    assert (ratio > 0.75).all() and (ratio < 1.25).all(), (
        f"per-channel ratio out of bounds: native={nat.mean(axis=(0,1))}, "
        f"lib={lib.mean(axis=(0,1))}"
    )


def test_decode_native_full_boost_is_brighter_than_sdr():
    """display_boost=hdr_capacity_max should produce a brighter HDR
    raster than display_boost=1.0 (the SDR-equivalent)."""
    hdr = _synthetic_hdr_rgb(64, 64)
    data = encode_native(hdr, quality=95)

    sdr_view = decode_native(data, dtype=np.float32, display_boost=1.0)["hdr"]
    hdr_view = decode_native(data, dtype=np.float32)["hdr"]  # default = full
    # Full-HDR view should have strictly more energy than SDR view.
    assert hdr_view.mean() > sdr_view.mean() * 1.5


def test_decode_native_parallel_matches_serial():
    """parallel=True kicks the two JPEG decodes into threads. Result
    must be bit-identical to the serial path (same kernel inputs)."""
    hdr = _synthetic_hdr_rgb(64, 64)
    data = encode_native(hdr, quality=95)
    a = decode_native(data, parallel=True)["hdr"]
    b = decode_native(data, parallel=False)["hdr"]
    np.testing.assert_array_equal(a, b)


def test_decode_native_dtype_fp32_skips_cast():
    hdr = _synthetic_hdr_rgb(32, 32)
    data = encode_native(hdr, quality=95)
    info = decode_native(data, dtype=np.float32)
    assert info["hdr"].dtype == np.float32


# ---------------------------------------------------------------------------
# encode_to: streaming write to file-like
# ---------------------------------------------------------------------------


def test_encode_to_writes_to_bytesio():
    """encode_to mirrors encode_native but emits bytes to a file-like
    instead of returning them. Bytes match the encode_native output."""
    hdr = _synthetic_hdr_rgb(64, 64)
    direct = encode_native(hdr, quality=95)
    buf = io.BytesIO()
    n = encode_to(buf, hdr, quality=95)
    assert n == len(direct)
    assert buf.getvalue() == direct
    assert is_uhdr(buf.getvalue()) is True


# ---------------------------------------------------------------------------
# UhdrError surfaces on malformed input
# ---------------------------------------------------------------------------


def test_decode_garbage_raises():
    with pytest.raises((UhdrError, ValueError, RuntimeError)):
        decode(b"not an ultra-hdr blob, not even close")
