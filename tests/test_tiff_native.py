"""Native TIFF reader (Tier 5) parity tests against tifffile.

The native reader is built on top of opencodecs.codecs._tiff (Cython)
and is exposed as a Codec named "tiff". Session 1 covers the full IFD
walk + tile/strip layout extraction; tile decoding only handles
``compression=NONE`` so far. Compressed-tile tests will land in
session 2 once the deflate / LZW / packbits / JPEG dispatchers are
wired in.
"""

from __future__ import annotations

import io
import os
import struct

import numpy as np
import pytest

import opencodecs as oc

tifffile = pytest.importorskip("tifffile")


def _need_tiff():
    if not oc.has_codec("tiff"):
        pytest.skip("native TIFF reader not built")


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


def test_tiff_signature_classic_le():
    _need_tiff()
    codec = oc.get_codec("tiff")
    assert codec.signature(b"II*\x00") is True
    assert codec.signature(b"II+\x00") is True
    assert codec.signature(b"MM\x00*") is True
    assert codec.signature(b"MM\x00+") is True
    assert codec.signature(b"") is False
    assert codec.signature(b"PNG\x89") is False


# ---------------------------------------------------------------------------
# Single-page striped uncompressed (the simplest happy path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [
    np.uint8, np.int8,
    np.uint16, np.int16,
    np.uint32, np.int32,
    np.uint64, np.int64,
    np.float32, np.float64,
])
def test_tiff_single_page_dtype_roundtrip(dtype):
    _need_tiff()
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        arr = rng.integers(info.min // 2, info.max // 2, size=(50, 70)).astype(dtype)
    else:
        arr = rng.random((50, 70)).astype(dtype)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, compression=None)
    with oc.get_codec("tiff").open(buf.getvalue()) as r:
        assert r.n_frames == 1
        assert r.dtype == np.dtype(dtype)
        page = r.page(0)
        assert page.compression == 1   # NONE
        assert page.is_tiled is False
        np.testing.assert_array_equal(page.asarray(), arr)


# ---------------------------------------------------------------------------
# Endianness (big-endian uncompressed must byte-swap multi-byte samples)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.float32, np.float64])
def test_tiff_big_endian_roundtrip(dtype):
    _need_tiff()
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.integer):
        arr = rng.integers(0, 100, size=(40, 60)).astype(dtype)
    else:
        arr = rng.random((40, 60)).astype(dtype)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, compression=None, byteorder=">")
    with oc.get_codec("tiff").open(buf.getvalue()) as r:
        np.testing.assert_array_equal(r.page(0).asarray(), arr)


# ---------------------------------------------------------------------------
# Multi-strip (last strip partial) — uncompressed
# ---------------------------------------------------------------------------


def test_tiff_multistrip_partial_last_strip():
    _need_tiff()
    arr = np.arange(200 * 128, dtype=np.uint16).reshape(200, 128)
    buf = io.BytesIO()
    # 200 rows / 32 rows per strip = 6 full + 1 partial (8 rows)
    tifffile.imwrite(buf, arr, compression=None, rowsperstrip=32)
    with oc.get_codec("tiff").open(buf.getvalue()) as r:
        page = r.page(0)
        assert page.is_tiled is False
        assert page.tile_height == 32
        assert len(page.offsets) == 7
        np.testing.assert_array_equal(page.asarray(), arr)


# ---------------------------------------------------------------------------
# Tiled (COG-style)
# ---------------------------------------------------------------------------


def test_tiff_tiled_layout():
    _need_tiff()
    arr = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, compression=None, tile=(128, 128))
    with oc.get_codec("tiff").open(buf.getvalue()) as r:
        page = r.page(0)
        assert page.is_tiled is True
        assert page.tile_width == 128
        assert page.tile_height == 128
        assert page.tiles_x == 2
        assert page.tiles_y == 2
        assert len(page.offsets) == 4
        np.testing.assert_array_equal(page.asarray(), arr)


def test_tiff_tiled_padded_edges():
    """Last column / row of tiles is partial — verify cropping logic."""
    _need_tiff()
    arr = np.arange(300 * 200, dtype=np.uint16).reshape(300, 200)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, compression=None, tile=(128, 128))
    with oc.get_codec("tiff").open(buf.getvalue()) as r:
        page = r.page(0)
        # 300 / 128 = 2.34 -> 3 tile rows; 200 / 128 = 1.56 -> 2 tile cols
        assert page.tiles_y == 3
        assert page.tiles_x == 2
        np.testing.assert_array_equal(page.asarray(), arr)


# ---------------------------------------------------------------------------
# RGB / multi-channel
# ---------------------------------------------------------------------------


def test_tiff_rgb_uint8():
    _need_tiff()
    rgb = np.random.default_rng(0).integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
    buf = io.BytesIO()
    tifffile.imwrite(buf, rgb, compression=None)
    with oc.get_codec("tiff").open(buf.getvalue()) as r:
        page = r.page(0)
        assert page.samples_per_pixel == 3
        assert page.shape == (64, 96, 3)
        np.testing.assert_array_equal(page.asarray(), rgb)


# ---------------------------------------------------------------------------
# Multi-page TIFF
# ---------------------------------------------------------------------------


def test_tiff_multipage_iter_and_random_access():
    _need_tiff()
    base = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64)
    pages = [base + i * 1000 for i in range(5)]
    buf = io.BytesIO()
    with tifffile.TiffWriter(buf) as tw:
        for p in pages:
            tw.write(p, compression=None)
    with oc.get_codec("tiff").open(buf.getvalue()) as r:
        assert r.n_frames == 5
        for i, frame in enumerate(r.iter_frames()):
            np.testing.assert_array_equal(frame, pages[i])
        # Random access
        np.testing.assert_array_equal(r[3], pages[3])
        np.testing.assert_array_equal(r[-1], pages[-1])
        with pytest.raises(IndexError):
            r[5]


# ---------------------------------------------------------------------------
# BigTIFF
# ---------------------------------------------------------------------------


def test_tiff_bigtiff_classic_features():
    _need_tiff()
    arr = np.arange(96 * 128, dtype=np.uint16).reshape(96, 128)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, compression=None, bigtiff=True)
    with oc.get_codec("tiff").open(buf.getvalue()) as r:
        assert r._is_bigtiff is True
        np.testing.assert_array_equal(r.page(0).asarray(), arr)


def test_tiff_bigtiff_multipage():
    _need_tiff()
    pages = [
        np.full((32, 48), v, dtype=np.uint8) for v in (10, 20, 30)
    ]
    buf = io.BytesIO()
    with tifffile.TiffWriter(buf, bigtiff=True) as tw:
        for p in pages:
            tw.write(p, compression=None)
    with oc.get_codec("tiff").open(buf.getvalue()) as r:
        assert r._is_bigtiff is True
        assert r.n_frames == 3
        for i, frame in enumerate(r.iter_frames()):
            np.testing.assert_array_equal(frame, pages[i])


# ---------------------------------------------------------------------------
# Reader contract: file path, file-like, bytes, custom read_at
# ---------------------------------------------------------------------------


def test_tiff_reader_from_filesystem_path(tmp_path):
    _need_tiff()
    arr = np.arange(40 * 60, dtype=np.uint16).reshape(40, 60)
    path = tmp_path / "x.tif"
    tifffile.imwrite(str(path), arr, compression=None)
    with oc.get_codec("tiff").open(str(path)) as r:
        np.testing.assert_array_equal(r.page(0).asarray(), arr)


def test_tiff_reader_from_filelike():
    _need_tiff()
    arr = np.arange(40 * 60, dtype=np.uint16).reshape(40, 60)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, compression=None)
    buf.seek(0)
    with oc.get_codec("tiff").open(buf) as r:
        np.testing.assert_array_equal(r.page(0).asarray(), arr)


def test_tiff_reader_custom_read_at():
    """The constructor accepts a read_at(offset, n) callable for
    non-seekable / HTTP-range / S3 data sources."""
    _need_tiff()
    from opencodecs._tiff_codec import TiffStream

    arr = np.arange(40 * 60, dtype=np.uint16).reshape(40, 60)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, compression=None)
    data = buf.getvalue()

    n_calls = {"count": 0}

    def read_at(off: int, n: int) -> bytes:
        n_calls["count"] += 1
        return data[off:off + n]

    with TiffStream(None, read_at=read_at) as r:
        np.testing.assert_array_equal(r.page(0).asarray(), arr)
    assert n_calls["count"] > 0  # parser actually used our callable


# ---------------------------------------------------------------------------
# Compressed paths return NotImplementedError (proves dispatch logic
# is wired but not yet implemented in session 1).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compression,dtype", [
    ("deflate", np.uint16),
    ("zstd",    np.uint16),
    ("packbits", np.uint8),
    ("lzw",     np.uint8),
    ("lzw",     np.uint16),
])
def test_tiff_byte_stream_compression_roundtrip(compression, dtype):
    """Compressed strip/tile decode through the native dispatcher."""
    _need_tiff()
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        arr = rng.integers(0, info.max // 2, size=(48, 64)).astype(dtype)
    else:
        arr = rng.random((48, 64)).astype(dtype)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, compression=compression)
    with oc.get_codec("tiff").open(buf.getvalue()) as r:
        np.testing.assert_array_equal(r.page(0).asarray(), arr)


@pytest.mark.parametrize("compression", ["deflate", "zstd", "lzw"])
def test_tiff_compressed_tiled(compression):
    """Compressed + tiled (the COG path)."""
    _need_tiff()
    arr = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, compression=compression, tile=(128, 128))
    with oc.get_codec("tiff").open(buf.getvalue()) as r:
        page = r.page(0)
        assert page.is_tiled is True
        np.testing.assert_array_equal(page.asarray(), arr)


@pytest.mark.parametrize("compression", ["deflate", "zstd", "lzw"])
def test_tiff_compressed_with_horizontal_predictor(compression):
    """Predictor 2 (horizontal differencing) — common with deflate/lzw."""
    _need_tiff()
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 4096, size=(64, 96), dtype=np.uint16)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, compression=compression, predictor=True)
    with oc.get_codec("tiff").open(buf.getvalue()) as r:
        page = r.page(0)
        assert page.predictor == 2
        np.testing.assert_array_equal(page.asarray(), arr)


@pytest.mark.skip(reason=(
    "tifffile-emitted LERC TIFFs use a two-level compression (LERC + "
    "secondary deflate/zstd indicated by the LercParameters tag 50674); "
    "Tier 5 session 2 dispatcher only handles the bare-LERC variant. "
    "Track in Tier 5 session 2 follow-up: parse LercParameters and "
    "chain through the appropriate inner deflate/zstd before LERC."))
def test_tiff_lerc_compression():
    """LERC: dispatches to opencodecs._lerc."""
    _need_tiff()
    if not oc.has_codec("lerc"):
        pytest.skip("opencodecs._lerc backend not available")
    arr = np.arange(64 * 96, dtype=np.uint16).reshape(64, 96)
    buf = io.BytesIO()
    try:
        tifffile.imwrite(buf, arr, compression="lerc")
    except Exception as exc:
        pytest.skip(f"tifffile cannot write LERC TIFF: {exc}")
    with oc.get_codec("tiff").open(buf.getvalue()) as r:
        np.testing.assert_array_equal(r.page(0).asarray(), arr)


def test_tiff_jpeg_compression_grayscale():
    """JPEG-in-TIFF (with stitched JPEGTables)."""
    _need_tiff()
    if not oc.has_codec("jpeg"):
        pytest.skip("opencodecs._jpeg backend not available")
    arr = (np.indices((64, 96)).sum(0) * 2).astype(np.uint8)
    buf = io.BytesIO()
    try:
        tifffile.imwrite(buf, arr, compression="jpeg")
    except Exception as exc:
        pytest.skip(f"tifffile cannot write JPEG TIFF: {exc}")
    with oc.get_codec("tiff").open(buf.getvalue()) as r:
        page = r.page(0)
        out = page.asarray()
        # JPEG is lossy; check shape + dtype + bounded error.
        assert out.shape == arr.shape
        assert out.dtype == np.uint8
        assert np.abs(out.astype(np.int32) - arr.astype(np.int32)).max() < 32
