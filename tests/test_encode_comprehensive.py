"""Comprehensive encode-side parity tests for every encode-capable codec.

Three coverage layers:

  1. **Round-trip at min, default, max levels.** Every codec round-trips
     through encode + decode exactly. Catches off-by-one level handling,
     underflow/overflow on level clamping, and silent corruption at
     extreme settings.

  2. **Byte-equality with imagecodecs at default level** for the
     deterministic lossless codecs. These should produce the *exact same
     bytes* as the imagecodecs reference encoder, which is the strongest
     possible parity guarantee.

  3. **Cross-decode equivalence** for lossy codecs. Encoded output must
     decode to the same array via either opencodecs's or imagecodecs's
     decoder.

Plus per-dtype and per-color-format coverage where the codec supports
multiple of each.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import opencodecs as oc

imagecodecs = pytest.importorskip("imagecodecs")


def _need(codec_name: str) -> None:
    if not oc.has_codec(codec_name):
        pytest.skip(f"codec {codec_name!r} not registered on this host")


# ---------------------------------------------------------------------------
# Compression codecs — bytes-in / bytes-out
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def random_payload() -> bytes:
    return os.urandom(64 * 1024)


@pytest.fixture(scope="module")
def compressible_payload() -> bytes:
    return (b"the quick brown fox jumps over the lazy dog\n" * 2048)


@pytest.fixture(scope="module")
def incompressible_payload() -> bytes:
    return os.urandom(256 * 1024)


# zstd levels: 1 to 22 (0 = default ~3)
@pytest.mark.parametrize("level", [1, 3, 22])
def test_zstd_roundtrip_levels(random_payload, level):
    _need("zstd")
    enc = oc.write(None, random_payload, format="zstd", level=level)
    assert oc.read(enc, format="zstd") == random_payload
    # Cross-decode with imagecodecs
    assert imagecodecs.zstd_decode(enc) == random_payload


def test_zstd_byte_equal_imagecodecs_default(random_payload):
    """Same level, same system libzstd → byte-identical output."""
    _need("zstd")
    oc_enc = oc.write(None, random_payload, format="zstd")
    ic_enc = imagecodecs.zstd_encode(random_payload)
    assert oc_enc == ic_enc, (
        f"zstd default-level encode should be byte-identical "
        f"(got {len(oc_enc)} vs {len(ic_enc)} bytes)")


@pytest.mark.parametrize("level", [0, 5, 11])
def test_brotli_roundtrip_levels(random_payload, level):
    _need("brotli")
    enc = oc.write(None, random_payload, format="brotli", level=level)
    assert oc.read(enc, format="brotli") == random_payload
    assert imagecodecs.brotli_decode(enc) == random_payload


def test_brotli_byte_equal_imagecodecs(random_payload):
    _need("brotli")
    oc_enc = oc.write(None, random_payload, format="brotli", level=11)
    ic_enc = imagecodecs.brotli_encode(random_payload, level=11)
    assert oc_enc == ic_enc


@pytest.mark.parametrize("level", [0, 5, 12])
def test_lz4_roundtrip_levels(random_payload, level):
    _need("lz4")
    enc = oc.write(None, random_payload, format="lz4", level=level)
    assert oc.read(enc, format="lz4") == random_payload
    assert imagecodecs.lz4f_decode(enc) == random_payload


@pytest.mark.parametrize("level", [0, 5, 9])
def test_deflate_roundtrip_levels(random_payload, level):
    _need("deflate")
    enc = oc.write(None, random_payload, format="deflate", level=level)
    assert oc.read(enc, format="deflate") == random_payload
    assert imagecodecs.zlib_decode(enc) == random_payload


def test_deflate_cross_decode_imagecodecs(random_payload):
    """deflate at the same level → not necessarily byte-identical
    (we link zlib-ng-compat when available, which emits a slightly
    different but standard-conformant deflate stream), but each
    library MUST be able to decode the other's output back to the
    same bytes.
    """
    _need("deflate")
    oc_enc = oc.write(None, random_payload, format="deflate", level=6)
    ic_enc = imagecodecs.zlib_encode(random_payload, level=6)
    # Round-trip via the OTHER library
    assert imagecodecs.zlib_decode(oc_enc) == random_payload
    assert oc.read(ic_enc, format="deflate") == random_payload


@pytest.mark.parametrize("level", [0, 5, 9])
def test_blosc2_roundtrip_levels(random_payload, level):
    _need("blosc2")
    enc = oc.write(None, random_payload, format="blosc2", level=level)
    assert oc.read(enc, format="blosc2") == random_payload
    assert imagecodecs.blosc2_decode(enc) == random_payload


def test_compressible_payload_actually_compresses(compressible_payload):
    """Sanity: compressible data should produce smaller-than-input output."""
    _need("zstd")
    enc = oc.write(None, compressible_payload, format="zstd")
    assert len(enc) < len(compressible_payload) // 5, (
        f"highly-compressible data should compress hard "
        f"(got {len(enc)} from {len(compressible_payload)})")


def test_incompressible_payload_does_not_explode(incompressible_payload):
    """Sanity: incompressible data should not blow up size beyond a reasonable
    overhead (zstd guarantees worst-case ~1% expansion)."""
    _need("zstd")
    enc = oc.write(None, incompressible_payload, format="zstd")
    overhead = len(enc) - len(incompressible_payload)
    assert overhead < len(incompressible_payload) // 100 + 64, (
        f"zstd encoded {len(incompressible_payload)} bytes to {len(enc)} "
        f"(overhead {overhead}); expected near-no-op for incompressible input")


# ---------------------------------------------------------------------------
# QOI — lossless RGB / RGBA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(64, 96, 3), (64, 96, 4)])
def test_qoi_roundtrip_shapes(shape):
    _need("qoi")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, shape, dtype=np.uint8)
    enc = oc.write(None, arr, format="qoi")
    np.testing.assert_array_equal(oc.read(enc, format="qoi"), arr)
    np.testing.assert_array_equal(imagecodecs.qoi_decode(enc), arr)


def test_qoi_byte_equal_imagecodecs_rgb():
    """QOI is a deterministic format; same input → same bytes."""
    _need("qoi")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    oc_enc = oc.write(None, arr, format="qoi")
    ic_enc = imagecodecs.qoi_encode(arr)
    assert oc_enc == ic_enc


# ---------------------------------------------------------------------------
# BMP — lossless gray / RGB / RGBA, 8-bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(16, 24), (16, 24, 3), (16, 24, 4)])
def test_bmp_roundtrip_all_modes(shape):
    _need("bmp")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, shape, dtype=np.uint8)
    enc = oc.write(None, arr, format="bmp")
    np.testing.assert_array_equal(oc.read(enc, format="bmp"), arr)


# ---------------------------------------------------------------------------
# PNG — lossless gray / GA / RGB / RGBA at 8 and 16 bits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((24, 32), np.uint8),
        ((24, 32, 2), np.uint8),
        ((24, 32, 3), np.uint8),
        ((24, 32, 4), np.uint8),
        ((24, 32), np.uint16),
        ((24, 32, 2), np.uint16),
        ((24, 32, 3), np.uint16),
        ((24, 32, 4), np.uint16),
    ],
)
def test_png_roundtrip_all_modes(shape, dtype):
    _need("png")
    rng = np.random.default_rng(0)
    high = 65536 if dtype is np.uint16 else 256
    arr = rng.integers(0, high, shape, dtype=dtype)
    enc = oc.write(None, arr, format="png")
    decoded = oc.read(enc, format="png")
    np.testing.assert_array_equal(np.squeeze(decoded), np.squeeze(arr))
    # Cross-decode with imagecodecs
    ic_decoded = imagecodecs.png_decode(enc)
    np.testing.assert_array_equal(np.squeeze(ic_decoded), np.squeeze(arr))


@pytest.mark.parametrize("level", [0, 1, 6, 9])
def test_png_roundtrip_levels(level):
    _need("png")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    enc = oc.write(None, arr, format="png", level=level)
    np.testing.assert_array_equal(oc.read(enc, format="png"), arr)


def test_png_higher_level_compresses_compressible_data_better():
    """Sanity: level 9 ≤ level 0 on actually-compressible data.

    On purely random pixels the level-9 effort can't find patterns, so
    the dictionary overhead can make it bigger. Use a smooth gradient
    (fully predictable) so we're testing what the level knob does
    when there IS signal to compress.
    """
    _need("png")
    arr = np.tile(
        np.arange(256, dtype=np.uint8)[None, :, None],
        (256, 1, 3),
    )
    fast = oc.write(None, arr, format="png", level=0)
    slow = oc.write(None, arr, format="png", level=9)
    assert len(slow) <= len(fast), (
        f"level=9 produced {len(slow)} bytes, level=0 produced {len(fast)} "
        "— higher level should compress smooth data at least as well")


# ---------------------------------------------------------------------------
# JPEG — lossy gray / RGB
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(48, 64), (48, 64, 3)])
@pytest.mark.parametrize("level", [10, 50, 95])
def test_jpeg_roundtrip_levels(shape, level):
    _need("jpeg")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, shape, dtype=np.uint8)
    enc = oc.write(None, arr, format="jpeg", level=level)
    decoded = oc.read(enc, format="jpeg")
    assert decoded.shape == arr.shape
    assert decoded.dtype == arr.dtype
    # Cross-decode produces the exact same bytes (deterministic decode of
    # a fixed JPEG stream by both turbojpeg and libjpeg).
    np.testing.assert_array_equal(decoded, imagecodecs.jpeg_decode(enc))


def test_jpeg_higher_level_better_quality():
    """Sanity: q=95 should match the input more closely than q=10."""
    _need("jpeg")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    low = oc.read(oc.write(None, arr, format="jpeg", level=10), format="jpeg")
    high = oc.read(oc.write(None, arr, format="jpeg", level=95), format="jpeg")
    err_low = np.abs(low.astype(int) - arr).mean()
    err_high = np.abs(high.astype(int) - arr).mean()
    assert err_high < err_low, (
        f"q=95 should be more faithful than q=10 "
        f"(got mean abs error {err_high:.2f} vs {err_low:.2f})")


# ---------------------------------------------------------------------------
# WebP — lossy + lossless RGB / RGBA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [3, 4])
def test_webp_lossless_byte_perfect(channels):
    _need("webp")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (32, 48, channels), dtype=np.uint8)
    if channels == 4:
        arr[..., 3] = rng.integers(1, 256, (32, 48), dtype=np.uint8)
    enc = oc.write(None, arr, format="webp", lossless=True)
    np.testing.assert_array_equal(oc.read(enc, format="webp"), arr)
    np.testing.assert_array_equal(imagecodecs.webp_decode(enc), arr)


@pytest.mark.parametrize("level", [10, 50, 95])
def test_webp_lossy_cross_decode(level):
    _need("webp")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    oc_enc = oc.write(None, arr, format="webp", level=level)
    np.testing.assert_array_equal(
        oc.read(oc_enc, format="webp"),
        imagecodecs.webp_decode(oc_enc),
    )


# ---------------------------------------------------------------------------
# JPEG-2000 — lossless + lossy, 8/16-bit, 1/3/4 channels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((32, 48), np.uint8),
        ((32, 48, 3), np.uint8),
        ((32, 48, 4), np.uint8),
        ((32, 48), np.uint16),
        ((32, 48, 3), np.uint16),
    ],
)
def test_jpeg2k_lossless_all_modes(shape, dtype):
    _need("jpeg2k")
    rng = np.random.default_rng(0)
    high = 65536 if dtype is np.uint16 else 256
    arr = rng.integers(0, high, shape, dtype=dtype)
    enc = oc.write(None, arr, format="jpeg2k", lossless=True)
    np.testing.assert_array_equal(oc.read(enc, format="jpeg2k"), arr)
    np.testing.assert_array_equal(imagecodecs.jpeg2k_decode(enc), arr)


@pytest.mark.parametrize("level", [10, 50, 100])
def test_jpeg2k_lossy_cross_decode(level):
    _need("jpeg2k")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    enc = oc.write(None, arr, format="jpeg2k", level=level)
    np.testing.assert_array_equal(
        oc.read(enc, format="jpeg2k"),
        imagecodecs.jpeg2k_decode(enc),
    )


# ---------------------------------------------------------------------------
# AVIF — lossless + lossy
# ---------------------------------------------------------------------------


def _skip_if_no_avif_encoder(exc):
    """Decode-only libavif (e.g. manylinux wheel: dav1d only, no aom)
    raises on encode with 'No codec available' / similar."""
    msg = str(exc).lower()
    if any(s in msg for s in ("no codec available", "unsupported", "encoder")):
        pytest.skip(f"libavif build has no AV1 encoder: {exc}")
    raise exc


@pytest.mark.parametrize("channels", [3, 4])
def test_avif_lossless_byte_perfect(channels):
    _need("avif")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (32, 48, channels), dtype=np.uint8)
    try:
        enc = oc.write(None, arr, format="avif", lossless=True)
    except Exception as exc:
        _skip_if_no_avif_encoder(exc)
    np.testing.assert_array_equal(oc.read(enc, format="avif"), arr)


@pytest.mark.parametrize("level", [40, 70, 95])
def test_avif_lossy_levels(level):
    _need("avif")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    try:
        enc = oc.write(None, arr, format="avif", level=level, speed=10)
    except Exception as exc:
        _skip_if_no_avif_encoder(exc)
    decoded = oc.read(enc, format="avif")
    assert decoded.shape == arr.shape
    assert decoded.dtype == arr.dtype


# ---------------------------------------------------------------------------
# HEIF — lossy round-trip (imagecodecs may lack heif support; smoke-only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [50, 80, 95])
def test_heif_lossy_levels(level):
    _need("heif")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    try:
        enc = oc.write(None, arr, format="heif", level=level)
    except Exception as exc:
        msg = str(exc).lower()
        if any(s in msg for s in (
            "encoder", "unsupported", "null error text", "heif_writer",
        )):
            pytest.skip(f"no HEVC encoder available: {exc}")
        raise
    decoded = oc.read(enc, format="heif")
    assert decoded.shape == arr.shape
    assert decoded.dtype == arr.dtype


# ---------------------------------------------------------------------------
# JXL — lossless + lossy, multi-frame
# ---------------------------------------------------------------------------


def test_jxl_lossless_byte_perfect_rgb():
    _need("jxl")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    enc = oc.write(None, arr, format="jxl", lossless=True)
    np.testing.assert_array_equal(oc.read(enc, format="jxl"), arr)


@pytest.mark.parametrize("distance", [0.5, 1.0, 5.0])
def test_jxl_lossy_distance(distance):
    _need("jxl")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    # JXL uses Butteraugli "distance" (lower = better quality, 1.0 ≈ visually
    # lossless), not 0-100 quality. opencodecs accepts ``distance=`` directly.
    enc = oc.write(None, arr, format="jxl", distance=distance)
    decoded = oc.read(enc, format="jxl")
    assert decoded.shape == arr.shape
    assert decoded.dtype == arr.dtype


# ---------------------------------------------------------------------------
# Cross-codec: encode-then-decode-then-encode-again should be stable
# (lossless codecs only, so we know we're hitting the same bytes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["zstd", "brotli", "deflate", "blosc2"])
def test_compression_re_encode_stable(fmt, random_payload):
    """encode → decode → encode should produce the same bytes the second
    time. Catches any non-determinism in the encoder (random padding,
    timestamps, etc.)."""
    _need(fmt)
    enc1 = oc.write(None, random_payload, format=fmt)
    decoded = oc.read(enc1, format=fmt)
    enc2 = oc.write(None, decoded, format=fmt)
    assert enc1 == enc2, f"{fmt} encoder is not deterministic"
