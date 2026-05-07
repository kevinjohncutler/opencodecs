"""Second wave of targeted coverage tests aimed at the remaining gaps:

  * ``_bmp_codec.py``    — 32-bit BI_RGB (BGRX) decode, BI_BITFIELDS at
                           info_size < 108, 16-bit BI_BITFIELDS with
                           alpha mask, padded 8-bit encode, _shift_for_mask
                           zero-mask early return, unsupported bpp error
  * ``_czi_reader.py``   — _pixel_type_dtype on unsupported value,
                           CziError paths in directory parsing, ZSTDHDR
                           error branches
  * ``core/io.py``       — error-path branches in BackgroundChunkReader
  * ``zarr.py``          — JxlCodec decode-without-out branches
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

import opencodecs as oc


# ---------------------------------------------------------------------------
# _bmp_codec.py — fill remaining gaps
# ---------------------------------------------------------------------------


def _bmp_header_only(info_size: int = 108) -> bytes:
    """Build a BMP that promises an info_size larger than what's present."""
    file_hdr = struct.pack("<2sIHHI", b"BM", 14 + info_size, 0, 0, 14 + info_size)
    info_hdr_size_only = struct.pack("<I", info_size)
    return file_hdr + info_hdr_size_only  # truncated DIB after the size word


def test_bmp_truncated_dib_after_size_field_raises():
    """Header claims info_size=108 but file ends after the 4-byte size."""
    from opencodecs._bmp_codec import BmpError
    bogus = _bmp_header_only(108)
    with pytest.raises(BmpError, match="DIB header"):
        oc.read(bogus, format="bmp")


def test_bmp_decode_32bit_bi_rgb_no_alpha():
    """BI_RGB at 32 bpp = BGRX in memory; output is 3-channel RGB."""
    width, height = 1, 1
    # Single pixel: B=10, G=20, R=30, X=255 → little-endian bytes
    pix = struct.pack("<BBBB", 10, 20, 30, 255)
    # info_size=40 (BITMAPINFOHEADER), compression=BI_RGB
    file_hdr = struct.pack("<2sIHHI", b"BM", 14 + 40 + len(pix), 0, 0, 14 + 40)
    info_hdr = struct.pack(
        "<IiiHHIIiiII",
        40, width, height, 1, 32, 0, len(pix), 3780, 3780, 0, 0,
    )
    bmp = file_hdr + info_hdr + pix
    arr = oc.read(bmp, format="bmp")
    assert arr.shape == (1, 1, 3)
    np.testing.assert_array_equal(arr[0, 0], [30, 20, 10])  # RGB


def test_bmp_decode_16bit_bi_bitfields_with_alpha():
    """16-bit BI_BITFIELDS via BITMAPV4HEADER with a non-zero alpha mask
    returns RGBA. Inline-mask path (info_size=40) only reads 3 masks,
    so the alpha-bearing 16-bit format requires the V4 header at info_size=108."""
    width, height = 1, 1
    # Pixel: R=15, G=15, B=15, A=15 → 0xFFFF
    pix = struct.pack("<H", 0xFFFF) + b"\x00\x00"  # pad to 4-byte stride
    info_size = 108
    base = struct.pack(
        "<IiiHHIIiiII",
        info_size, width, height, 1, 16, 3, len(pix), 3780, 3780, 0, 0,
    )
    # V4 mask block (R, G, B, A) at offset 54
    mask_block = struct.pack("<IIII", 0xF000, 0x0F00, 0x00F0, 0x000F)
    # 4-byte CS + 36 endpoint + 12 gamma = 52 bytes filler to reach 108
    rest = b"\x00" * (info_size - 40 - 16)
    info_hdr = base + mask_block + rest
    pix_offset = 14 + info_size
    file_hdr = struct.pack(
        "<2sIHHI", b"BM", pix_offset + len(pix), 0, 0, pix_offset,
    )
    bmp = file_hdr + info_hdr + pix
    arr = oc.read(bmp, format="bmp")
    assert arr.shape == (1, 1, 4)
    # 4-bit channels expanded to 8 by replication: 0xF→0xFF
    np.testing.assert_array_equal(arr[0, 0], [255, 255, 255, 255])


def test_bmp_decode_32bit_bi_bitfields_info_size_40():
    """32-bit BI_BITFIELDS where info_size < 108: masks are inline after DIB."""
    width, height = 1, 1
    # Use ARGB layout; pixel value = (A=255, R=10, G=20, B=30) packed LE.
    pix = struct.pack("<I", (255 << 24) | (10 << 16) | (20 << 8) | 30)
    info_size = 40
    info_hdr = struct.pack(
        "<IiiHHIIiiII",
        info_size, width, height, 1, 32, 3, len(pix), 3780, 3780, 0, 0,
    )
    # 16 bytes of inline masks (R, G, B, A)
    masks = struct.pack(
        "<IIII", 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000,
    )
    pix_offset = 14 + info_size + 16
    file_hdr = struct.pack(
        "<2sIHHI", b"BM", pix_offset + len(pix), 0, 0, pix_offset,
    )
    bmp = file_hdr + info_hdr + masks + pix
    arr = oc.read(bmp, format="bmp")
    assert arr.shape == (1, 1, 4)
    np.testing.assert_array_equal(arr[0, 0], [10, 20, 30, 255])


def test_bmp_decode_unsupported_bpp_raises():
    """64-bit BMP isn't in the dispatch and should raise. (64 because
    bpp <= 8 triggers palette handling; we want a value that bypasses
    the palette path AND isn't in {16, 24, 32}.)"""
    from opencodecs._bmp_codec import BmpError
    width, height = 1, 1
    pix = b"\x00" * 8  # 64 bpp = 8 bytes per pixel
    file_hdr = struct.pack("<2sIHHI", b"BM", 14 + 40 + len(pix), 0, 0, 14 + 40)
    info_hdr = struct.pack(
        "<IiiHHIIiiII",
        40, width, height, 1, 64, 0, len(pix), 3780, 3780, 0, 0,
    )
    bmp = file_hdr + info_hdr + pix
    with pytest.raises(BmpError, match="bpp"):
        oc.read(bmp, format="bmp")


def test_bmp_encode_paletted_with_padding():
    """Paletted 8-bit width=3 → row stride=4 → triggers the padding branch."""
    arr = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    enc = oc.write(None, arr, format="bmp")
    dec = oc.read(enc, format="bmp")
    np.testing.assert_array_equal(np.squeeze(dec), arr)


def test_bmp_shift_for_mask_zero():
    """_shift_for_mask(0) returns (0, 0)."""
    from opencodecs._bmp_codec import _shift_for_mask
    assert _shift_for_mask(0) == (0, 0)


# ---------------------------------------------------------------------------
# _czi_reader.py — pure-function error paths (no real CZI needed)
# ---------------------------------------------------------------------------


def test_czi_pixel_type_dtype_unsupported_raises():
    from opencodecs._czi_reader import _pixel_type_dtype
    with pytest.raises(ValueError, match="unsupported CZI pixel type"):
        _pixel_type_dtype(99999)


def test_czi_subblock_entry_dtype_and_samples():
    """CziSubBlockEntry.dtype / .samples properties."""
    from opencodecs._czi_reader import CziSubBlockEntry
    e = CziSubBlockEntry(
        file_position=0, pixel_type=0, compression=0, dimensions_count=2,
        dims=("Y", "X", "S"), shape=(8, 8, 1), stored_shape=(8, 8, 1),
        start=(0, 0, 0), mosaic_index=-1, scene_index=-1, storage_size=72,
    )
    assert e.dtype == np.dtype("u1")
    assert e.samples == 1


def test_czi_reader_open_nonexistent_raises(tmp_path):
    """File that doesn't exist: open(2) errors out cleanly."""
    from opencodecs._czi_reader import CziReader
    with pytest.raises(FileNotFoundError):
        CziReader(str(tmp_path / "does_not_exist.czi"))


def test_czi_reader_bad_magic_raises(tmp_path):
    """First 16 bytes don't start with ZISRAWFILE magic."""
    p = tmp_path / "fake.czi"
    p.write_bytes(b"NOTACZIFILE\x00\x00\x00\x00\x00" + b"\x00" * 4096)
    from opencodecs._czi_reader import CziReader, CziError
    with pytest.raises(CziError, match="not a CZI"):
        CziReader(str(p))


# ---------------------------------------------------------------------------
# zarr.py — JxlCodec decode without out
# ---------------------------------------------------------------------------


def test_jxlcodec_decode_no_out_returns_array():
    pytest.importorskip("numcodecs")
    if not oc.has_codec("jxl"):
        pytest.skip("jxl not available")
    from opencodecs.zarr import JxlCodec
    rng = np.random.default_rng(0)
    chunk = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    c = JxlCodec(lossless=True)
    enc = c.encode(chunk)
    dec = c.decode(enc)
    assert isinstance(dec, np.ndarray)
    np.testing.assert_array_equal(np.squeeze(dec), chunk)


# ---------------------------------------------------------------------------
# core/io.py — BackgroundChunkReader edge cases
# ---------------------------------------------------------------------------


def test_chunked_reader_chunk_size_too_large_clamped(tmp_path):
    """Very large chunk_size still works (no explicit upper bound)."""
    from opencodecs.core.io import BackgroundChunkReader
    p = tmp_path / "data.bin"
    p.write_bytes(b"x" * 1024)
    with BackgroundChunkReader(p, chunk_size=64 * 1024 * 1024) as r:
        chunks = list(r)
    assert b"".join(chunks) == b"x" * 1024


def test_chunked_reader_close_idempotent_via_context_manager(tmp_path):
    """Exiting the with-block twice (impossible in practice, but
    close() must remain idempotent across all exit paths)."""
    from opencodecs.core.io import BackgroundChunkReader
    p = tmp_path / "data.bin"
    p.write_bytes(b"hello")
    r = BackgroundChunkReader(p)
    r.close()
    # Calling close after enter/exit should also be safe.
    with BackgroundChunkReader(p) as r2:
        pass
    r2.close()  # post-exit close


# ---------------------------------------------------------------------------
# tiff_reader.py — empty page short-circuit
# ---------------------------------------------------------------------------


def test_tiff_reader_imread_uncompressed_strip(tmp_path):
    """Strip-based TIFF (not tiled) — currently goes through page.asarray
    fallback in _read_one_page_parallel because dataoffsets has 1 entry."""
    tifffile = pytest.importorskip("tifffile")
    from opencodecs.tiff_reader import imread
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    p = tmp_path / "strip.tif"
    tifffile.imwrite(str(p), arr)  # default = strip-based, not tiled
    out = imread(p)
    np.testing.assert_array_equal(out, arr)


# ---------------------------------------------------------------------------
# core/codec.py — file-like resolution failure
# ---------------------------------------------------------------------------


def test_resolve_codec_file_like_unrecognized_raises():
    """A file-like with unrecognized magic should raise KeyError."""
    import io
    from opencodecs.core.codec import _resolve_codec
    bio = io.BytesIO(b"\x00" * 64)
    with pytest.raises(KeyError):
        _resolve_codec(bio)
