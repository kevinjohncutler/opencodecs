"""Edge-case tests: empty, tiny, huge, prime sizes, boundary values.

Catches the classes of bugs that don't show up on round numbers or
random data:

  * **Off-by-one** in row stride / padding (prime widths)
  * **Header-only** edge of compression formats (1 byte input)
  * **Boundary values** (uint8 0 and 255, uint16 0 and 65535)
  * **Highly-compressible** input (constant arrays — dictionary-stress
    for compression codecs, predictor-stress for image codecs)
  * **Larger-than-typical** images (ensures we don't quietly blow up
    on 100+ MB inputs)
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import opencodecs as oc


def _need(codec_name: str) -> None:
    if not oc.has_codec(codec_name):
        pytest.skip(f"codec {codec_name!r} not registered on this host")


# ---------------------------------------------------------------------------
# Compression: empty / 1-byte / huge / constant
# ---------------------------------------------------------------------------


_COMPRESSION_CODECS = ["zstd", "lz4", "brotli", "blosc2", "deflate"]


@pytest.mark.parametrize("fmt", _COMPRESSION_CODECS)
def test_compression_empty_roundtrips_to_empty(fmt):
    _need(fmt)
    enc = oc.write(None, b"", format=fmt)
    assert oc.read(enc, format=fmt) == b""


@pytest.mark.parametrize("fmt", _COMPRESSION_CODECS)
def test_compression_one_byte_roundtrips(fmt):
    _need(fmt)
    enc = oc.write(None, b"\xab", format=fmt)
    assert oc.read(enc, format=fmt) == b"\xab"


# Per-codec ceilings for "16 MB of zeros encoded". Each is calibrated to
# what the codec's framing overhead allows: lz4 frame format adds a
# block header per ~64 KB chunk so it can't go below ~70 KB; deflate
# does proper RLE-like LZ77 and lands ~16 KB; zstd / brotli / blosc2
# all detect the run trivially and produce hundreds of bytes or less.
_CONSTANT_DATA_CEILINGS = {
    "zstd": 4096,        # measured ~530 bytes
    "lz4": 96 * 1024,    # measured ~70 KB; frame format block overhead
    "brotli": 1024,      # measured ~30 bytes
    "blosc2": 1024,      # measured ~30 bytes
    "deflate": 32 * 1024, # measured ~16 KB
}


@pytest.mark.parametrize("fmt", _COMPRESSION_CODECS)
def test_compression_constant_data_compresses_significantly(fmt):
    """16 MB of zeros should compress to far less than the input.

    Per-codec ceilings reflect each format's framing overhead — the
    point is to catch regressions where (e.g.) the encoder silently
    falls back to "store uncompressed". A 16 MB input that produces
    16 MB output means we hit the bug we found in deflate where
    Z_DEFAULT_COMPRESSION was getting clamped to 0.
    """
    _need(fmt)
    payload = b"\x00" * (16 * 1024 * 1024)
    enc = oc.write(None, payload, format=fmt)
    ceiling = _CONSTANT_DATA_CEILINGS[fmt]
    assert len(enc) < ceiling, (
        f"{fmt}: 16 MB of zeros encoded to {len(enc)} bytes "
        f"(expected < {ceiling}); encoder may be falling back to "
        f"uncompressed")
    assert oc.read(enc, format=fmt) == payload


@pytest.mark.parametrize("fmt", _COMPRESSION_CODECS)
def test_compression_large_random(fmt):
    """64 MB random round-trip: catches bugs where encoders use 32-bit
    size accumulators, output buffers that can't grow past 4 GB, etc."""
    _need(fmt)
    payload = os.urandom(64 * 1024 * 1024)
    enc = oc.write(None, payload, format=fmt)
    assert oc.read(enc, format=fmt) == payload


# ---------------------------------------------------------------------------
# Image codecs: tiny / boundary / prime / huge dimensions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["qoi", "bmp", "png"])
def test_image_1x1_pixel(fmt):
    """The smallest non-degenerate image. Catches divide-by-zero and
    "image must have at least N pixels" assumptions."""
    _need(fmt)
    arr = np.array([[[128, 64, 200]]], dtype=np.uint8)
    enc = oc.write(None, arr, format=fmt)
    decoded = oc.read(enc, format=fmt)
    np.testing.assert_array_equal(np.squeeze(decoded), np.squeeze(arr))


@pytest.mark.parametrize("fmt", ["qoi", "bmp", "png"])
def test_image_1xN_strip(fmt):
    """One-pixel-tall image. Catches off-by-one stride bugs."""
    _need(fmt)
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (1, 256, 3), dtype=np.uint8)
    enc = oc.write(None, arr, format=fmt)
    decoded = oc.read(enc, format=fmt)
    np.testing.assert_array_equal(np.squeeze(decoded), np.squeeze(arr))


@pytest.mark.parametrize("fmt", ["qoi", "bmp", "png"])
def test_image_Nx1_column(fmt):
    """One-pixel-wide image. Catches off-by-one stride bugs."""
    _need(fmt)
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (256, 1, 3), dtype=np.uint8)
    enc = oc.write(None, arr, format=fmt)
    decoded = oc.read(enc, format=fmt)
    np.testing.assert_array_equal(np.squeeze(decoded), np.squeeze(arr))


@pytest.mark.parametrize("fmt", ["qoi", "bmp", "png"])
def test_image_prime_dimensions(fmt):
    """Prime-sized width/height — exercises non-aligned row strides
    (e.g. BMP's 4-byte row padding kicks in interestingly here)."""
    _need(fmt)
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (37, 41, 3), dtype=np.uint8)
    enc = oc.write(None, arr, format=fmt)
    decoded = oc.read(enc, format=fmt)
    np.testing.assert_array_equal(np.squeeze(decoded), np.squeeze(arr))


@pytest.mark.parametrize("fmt", ["png"])
def test_image_all_zero(fmt):
    """All-zero input — minimum-entropy case for predictor-based codecs."""
    _need(fmt)
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    enc = oc.write(None, arr, format=fmt)
    np.testing.assert_array_equal(oc.read(enc, format=fmt), arr)


@pytest.mark.parametrize("fmt", ["png"])
def test_image_all_max(fmt):
    """All-255 input — boundary value test."""
    _need(fmt)
    arr = np.full((64, 64, 3), 255, dtype=np.uint8)
    enc = oc.write(None, arr, format=fmt)
    np.testing.assert_array_equal(oc.read(enc, format=fmt), arr)


def test_png_uint16_full_range():
    """uint16 PNG with min and max values present."""
    _need("png")
    arr = np.zeros((16, 16), dtype=np.uint16)
    arr[0, 0] = 0
    arr[0, 1] = 0xFFFF
    arr[1, 0] = 0x8000
    enc = oc.write(None, arr, format="png")
    decoded = oc.read(enc, format="png")
    np.testing.assert_array_equal(np.squeeze(decoded), arr)


@pytest.mark.parametrize("fmt", ["qoi", "bmp", "png"])
def test_image_large(fmt):
    """4096×4096 image — catches anything that quietly assumes a
    dimension fits in uint16."""
    _need(fmt)
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (4096, 4096, 3), dtype=np.uint8)
    enc = oc.write(None, arr, format=fmt)
    decoded = oc.read(enc, format=fmt)
    np.testing.assert_array_equal(np.squeeze(decoded), np.squeeze(arr))


# ---------------------------------------------------------------------------
# JPEG-2000: integer dtype boundary (the 8/16-bit precision edge)
# ---------------------------------------------------------------------------


def test_jpeg2k_uint16_max_value():
    """uint16 JPEG-2000 with the maximum value at every pixel."""
    _need("jpeg2k")
    arr = np.full((32, 32), 0xFFFF, dtype=np.uint16)
    enc = oc.write(None, arr, format="jpeg2k", lossless=True)
    np.testing.assert_array_equal(np.squeeze(oc.read(enc, format="jpeg2k")), arr)


# ---------------------------------------------------------------------------
# Buffer-protocol input: bytes / bytearray / memoryview / mmap / numpy uint8
# all reach the codec without copies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["zstd", "lz4", "brotli", "deflate"])
def test_compression_accepts_memoryview(fmt):
    _need(fmt)
    payload = os.urandom(8 * 1024)
    enc = oc.write(None, memoryview(payload), format=fmt)
    assert oc.read(memoryview(enc), format=fmt) == payload


@pytest.mark.parametrize("fmt", ["zstd", "lz4", "brotli", "deflate"])
def test_compression_accepts_bytearray(fmt):
    _need(fmt)
    payload = os.urandom(8 * 1024)
    enc = oc.write(None, bytearray(payload), format=fmt)
    assert oc.read(bytearray(enc), format=fmt) == payload


@pytest.mark.parametrize("fmt", ["zstd", "lz4", "brotli", "deflate"])
def test_compression_accepts_numpy_uint8(fmt):
    _need(fmt)
    payload = np.frombuffer(os.urandom(8 * 1024), dtype=np.uint8)
    enc = oc.write(None, payload, format=fmt)
    assert oc.read(np.frombuffer(enc, dtype=np.uint8), format=fmt) == payload.tobytes()


# ---------------------------------------------------------------------------
# CZI metadata edge cases
# ---------------------------------------------------------------------------


_LAB_CZI = (
    "/Volumes/HiprDrive/2024_02_02_GNE_synthetic_community/"
    "2024_02_02_GNEPanelTest_slide1_B1_GNE0001_cellmix01_200nMENC_"
    "20nMCOMP_quarterpower_fov_4_561.czi"
)


@pytest.mark.skipif(
    not os.path.isfile(_LAB_CZI),
    reason="reference lab CZI file not available",
)
def test_czi_subblock_index_out_of_range():
    """Out-of-range sub-block index raises IndexError, not silent garbage."""
    _need("czi")
    with oc.get_codec("czi").open(_LAB_CZI) as r:
        with pytest.raises(IndexError):
            _ = r[r.n_frames + 100]
        with pytest.raises(IndexError):
            _ = r.subblock_metadata_bytes(r.n_frames + 100)


@pytest.mark.skipif(
    not os.path.isfile(_LAB_CZI),
    reason="reference lab CZI file not available",
)
def test_czi_negative_index_works():
    """Python convention: r[-1] returns the last frame."""
    _need("czi")
    with oc.get_codec("czi").open(_LAB_CZI) as r:
        last = r[-1]
        same = r[r.n_frames - 1]
        np.testing.assert_array_equal(last, same)


@pytest.mark.skipif(
    not os.path.isfile(_LAB_CZI),
    reason="reference lab CZI file not available",
)
def test_czi_metadata_cache_returns_same_object():
    """Repeat calls to metadata_bytes / metadata_xml return the cached
    object (verified with `is`), so downstream parsers can memoize."""
    _need("czi")
    with oc.get_codec("czi").open(_LAB_CZI) as r:
        assert r.metadata_bytes is r.metadata_bytes
        assert r.metadata_xml is r.metadata_xml
