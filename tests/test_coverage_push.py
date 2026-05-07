"""Targeted tests that exercise paths uncovered by the broader test
modules. Goal: drive package coverage from ~87% to >=95%.

Each block below targets a specific module's missing lines (identified
from ``pytest --cov=src/opencodecs --cov-report=term-missing``):

  * ``_bmp_codec.py``    — bitfield decoders, top-down rows, errors
  * ``_czi_reader.py``   — slice access, repr, sub-block metadata
  * ``_czi_codec.py``    — bytes / memoryview / file-like input paths
  * ``_hdf5_codec.py``   — Codec.decode + open() with bytes/file-like
  * ``_zarr_codecs.py``  — zarr v3 wrappers' missing branches
  * ``parallel.py``      — file-like input paths to frame_count + decode_frames_parallel
  * ``tifffile_patch.py`` — tobytes() fallback for non-bytes-coercible inputs
  * ``core/codec.py``    — ``codec_for_path``-fallback when reading magic fails
"""

from __future__ import annotations

import io
import os
import struct
from pathlib import Path

import numpy as np
import pytest

import opencodecs as oc


# ===========================================================================
# _bmp_codec.py — top-down rows, 16-bit RGB555, V4 BI_BITFIELDS, error paths
# ===========================================================================


def _build_bmp(width: int, height_signed: int, bpp: int, pixel_rows: bytes,
               compression: int = 0, palette: bytes | None = None,
               masks: tuple[int, int, int, int] | None = None) -> bytes:
    """Build a minimal BMP file header + DIB header + pixels for tests."""
    info_size = 40 if masks is None else 108
    palette_bytes = palette or b""
    extra_mask_bytes = b""
    if masks is not None:
        extra_mask_bytes = struct.pack("<IIII", *masks)
        info_size = 108  # BITMAPV4HEADER
    pix_offset = 14 + info_size + len(palette_bytes)
    file_size = pix_offset + len(pixel_rows)
    file_hdr = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, pix_offset)
    if info_size == 40:
        info_hdr = struct.pack(
            "<IiiHHIIiiII",
            info_size, width, height_signed, 1, bpp, compression,
            len(pixel_rows), 3780, 3780, 0, 0,
        )
    else:
        info_hdr = struct.pack(
            "<IiiHHIIiiII",
            info_size, width, height_signed, 1, bpp, compression,
            len(pixel_rows), 3780, 3780, 0, 0,
        ) + extra_mask_bytes + b"\x00" * (108 - 40 - 16)
    return file_hdr + info_hdr + palette_bytes + pixel_rows


def test_bmp_decode_top_down_24bit():
    """height < 0 means top-down rows; pixels should NOT be flipped."""
    width, height = 4, 2
    pixel_row = bytes([10, 20, 30,  40, 50, 60,  70, 80, 90,  100, 110, 120])
    # 24-bit row stride is 4-byte aligned; 4 px * 3 bytes = 12 already aligned.
    pix = pixel_row * height
    bmp = _build_bmp(width, -height, 24, pix)
    arr = oc.read(bmp, format="bmp")
    assert arr.shape == (2, 4, 3)
    # First row in file == first row in output (RGB-converted from BGR).
    np.testing.assert_array_equal(
        arr[0], np.array([[30, 20, 10], [60, 50, 40], [90, 80, 70], [120, 110, 100]])
    )


def test_bmp_decode_16bit_rgb555_implicit_masks():
    """16-bit BI_RGB uses implicit RGB555 masks."""
    # 1 row of 2 pixels, 16bpp; row stride 4-byte aligned: 2*2=4, no pad.
    # RGB555: 0x7C00 R, 0x03E0 G, 0x001F B; build a red pixel and blue pixel.
    red = struct.pack("<H", 0x7C00)
    blue = struct.pack("<H", 0x001F)
    pix = red + blue
    # Need 4-byte stride; 2*2=4 already aligned.
    bmp = _build_bmp(2, 1, 16, pix)
    arr = oc.read(bmp, format="bmp")
    assert arr.shape == (1, 2, 3)
    # 5-bit channels expanded to 8: max=255.
    assert arr[0, 0, 0] == 255  # Red full
    assert arr[0, 1, 2] == 255  # Blue full


def test_bmp_decode_32bit_bitfields_with_alpha():
    """V4 header with BI_BITFIELDS and a non-zero alpha mask returns RGBA."""
    masks = (0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)  # ARGB layout
    width, height = 1, 1
    # Single pixel: A=255, R=10, G=20, B=30 → 0xFF0A141E (little-endian)
    pix = struct.pack("<I", 0xFF0A141E)
    bmp = _build_bmp(width, height, 32, pix, compression=3, masks=masks)
    arr = oc.read(bmp, format="bmp")
    assert arr.shape == (1, 1, 4)
    # Bottom-up storage means the only row is the pixel above.
    np.testing.assert_array_equal(arr[0, 0], [10, 20, 30, 255])


def test_bmp_decode_invalid_magic_raises():
    """Wrong magic bytes raise."""
    from opencodecs._bmp_codec import BmpError
    bogus = b"XX" + b"\x00" * 100
    with pytest.raises(BmpError, match="not a BMP"):
        oc.read(bogus, format="bmp")


def test_bmp_decode_truncated_dib_raises():
    from opencodecs._bmp_codec import BmpError
    # Just the file header — DIB header missing.
    short = struct.pack("<2sIHHI", b"BM", 14, 0, 0, 14)
    with pytest.raises(BmpError, match="DIB"):
        oc.read(short, format="bmp")


def test_bmp_decode_unsupported_dib_size_raises():
    """info_size < 40 is a (deprecated) BITMAPCOREHEADER — we don't support it."""
    from opencodecs._bmp_codec import BmpError
    file_hdr = struct.pack("<2sIHHI", b"BM", 30, 0, 0, 30)
    info_hdr = struct.pack("<I", 12) + b"\x00" * 8  # info_size=12
    with pytest.raises(BmpError, match="DIB header size"):
        oc.read(file_hdr + info_hdr, format="bmp")


def test_bmp_decode_unsupported_compression_raises():
    """Compression types other than BI_RGB / BI_BITFIELDS raise."""
    from opencodecs._bmp_codec import BmpError
    pix = b"\x00" * 4
    bmp = _build_bmp(1, 1, 24, pix, compression=1)  # BI_RLE8
    with pytest.raises(BmpError, match="compression"):
        oc.read(bmp, format="bmp")


def test_bmp_decode_invalid_dimensions_raises():
    from opencodecs._bmp_codec import BmpError
    pix = b"\x00" * 4
    bmp = _build_bmp(0, 1, 24, pix)
    with pytest.raises(BmpError, match="dimensions"):
        oc.read(bmp, format="bmp")


def test_bmp_encode_unsupported_dtype_raises():
    from opencodecs._bmp_codec import BmpError
    arr = np.zeros((4, 4), dtype=np.uint16)
    with pytest.raises(BmpError, match="dtype"):
        oc.write(None, arr, format="bmp")


def test_bmp_encode_unsupported_shape_raises():
    """Non-2D / non-(H,W,3|4) shapes raise."""
    from opencodecs._bmp_codec import BmpError
    arr = np.zeros((4, 4, 2), dtype=np.uint8)
    with pytest.raises(BmpError, match="array shape"):
        oc.write(None, arr, format="bmp")


def test_bmp_decode_rgba_roundtrip():
    """BGRA32 encode → decode round-trip."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (8, 8, 4), dtype=np.uint8)
    enc = oc.write(None, arr, format="bmp")
    dec = oc.read(enc, format="bmp")
    np.testing.assert_array_equal(np.squeeze(dec), arr)


def test_bmp_decode_paletted_rgb_palette():
    """Paletted 8-bit BMP with a non-grayscale palette returns 3-channel RGB."""
    width, height = 2, 2
    # row stride 4-byte aligned: 2 → 4; pad 2.
    rows = bytes([0, 1, 0, 0,  2, 3, 0, 0])  # bottom-up (rows 1, then 0)
    palette = bytearray()
    # 256 BGRA palette entries; first 4 set to distinguishable colors.
    palette += bytes([0, 0, 255, 0])      # idx 0: RED   (B=0, G=0, R=255)
    palette += bytes([0, 255, 0, 0])      # idx 1: GREEN
    palette += bytes([255, 0, 0, 0])      # idx 2: BLUE
    palette += bytes([255, 255, 255, 0])  # idx 3: WHITE
    palette += bytes(252 * 4)             # rest zero
    bmp = _build_bmp(width, height, 8, rows, compression=0, palette=bytes(palette))
    arr = oc.read(bmp, format="bmp")
    assert arr.shape == (2, 2, 3)
    # Bottom-up storage: file row 0 = image bottom row.
    #   File row 0 = [0=RED, 1=GREEN]   → output row 1 (image bottom)
    #   File row 1 = [2=BLUE, 3=WHITE]  → output row 0 (image top)
    np.testing.assert_array_equal(arr[0, 0], [0, 0, 255])   # BLUE
    np.testing.assert_array_equal(arr[0, 1], [255, 255, 255])  # WHITE
    np.testing.assert_array_equal(arr[1, 0], [255, 0, 0])   # RED
    np.testing.assert_array_equal(arr[1, 1], [0, 255, 0])   # GREEN


# ===========================================================================
# _czi_reader.py / _czi_codec.py — uses real lab CZI when present
# ===========================================================================


_LAB_CZI = (
    "/Volumes/HiprDrive/2024_02_02_GNE_synthetic_community/"
    "2024_02_02_GNEPanelTest_slide1_B1_GNE0001_cellmix01_200nMENC_"
    "20nMCOMP_quarterpower_fov_4_561.czi"
)


pytestmark_czi = pytest.mark.skipif(
    not os.path.isfile(_LAB_CZI), reason="lab CZI not mounted"
)


@pytestmark_czi
def test_czi_reader_repr_with_entries():
    """__repr__ should include sub-block count, dtype, tile shape."""
    with oc.get_codec("czi").open(_LAB_CZI) as r:
        s = repr(r)
        assert "CziReader" in s
        assert "sub-blocks" in s


@pytestmark_czi
def test_czi_iter_tiles():
    """iter_tiles() yields one ndarray per sub-block in order."""
    with oc.get_codec("czi").open(_LAB_CZI) as r:
        seen = 0
        for tile in r.iter_tiles():
            assert isinstance(tile, np.ndarray)
            seen += 1
            if seen >= 3:
                break
        assert seen >= 1


@pytestmark_czi
def test_czi_slice_access():
    """r[slice] returns a stacked ndarray of the requested sub-blocks."""
    with oc.get_codec("czi").open(_LAB_CZI) as r:
        if r.n_frames < 3:
            pytest.skip("file has too few sub-blocks for slice test")
        stack = r[0:3]
        assert stack.shape[0] == 3


@pytestmark_czi
def test_czi_subblock_metadata_bytes():
    """subblock_metadata_bytes returns bytes for valid index."""
    with oc.get_codec("czi").open(_LAB_CZI) as r:
        m = r.subblock_metadata_bytes(0)
        assert isinstance(m, bytes)


@pytestmark_czi
def test_czi_subblock_metadata_negative_index():
    """subblock_metadata_bytes(-1) selects last sub-block."""
    with oc.get_codec("czi").open(_LAB_CZI) as r:
        m_last = r.subblock_metadata_bytes(-1)
        m_same = r.subblock_metadata_bytes(r.n_frames - 1)
        assert m_last == m_same


@pytestmark_czi
def test_czi_negative_subblock_metadata_out_of_range():
    """Negative index past start raises IndexError."""
    with oc.get_codec("czi").open(_LAB_CZI) as r:
        with pytest.raises(IndexError):
            r.subblock_metadata_bytes(-(r.n_frames + 100))


@pytestmark_czi
def test_czi_read_explicit_n_workers():
    """read(n_workers=2) takes the explicit-pool branch (not the persistent one)."""
    with oc.get_codec("czi").open(_LAB_CZI) as r:
        if r.n_frames < 2:
            pytest.skip("need >=2 sub-blocks")
        arr = r.read(n_workers=2)
        assert arr.shape[0] == r.n_frames or arr.ndim >= 1


@pytestmark_czi
def test_czi_read_squeeze_false():
    """squeeze=False keeps singleton axes in tile_shape."""
    with oc.get_codec("czi").open(_LAB_CZI) as r:
        arr = r.read(squeeze=False, n_workers=1)
        assert arr.ndim >= 2


@pytestmark_czi
def test_czi_codec_decode_from_bytes():
    """CziCodec.decode() with bytes input uses the temp-file path."""
    data = Path(_LAB_CZI).read_bytes()
    arr = oc.get_codec("czi").decode(data)
    assert arr.size > 0


@pytestmark_czi
def test_czi_codec_decode_from_file_like():
    """CziCodec.decode() with file-like input uses the temp-file path."""
    with open(_LAB_CZI, "rb") as f:
        arr = oc.get_codec("czi").decode(f)
    assert arr.size > 0


@pytestmark_czi
def test_czi_codec_decode_from_memoryview():
    data = Path(_LAB_CZI).read_bytes()
    arr = oc.get_codec("czi").decode(memoryview(data))
    assert arr.size > 0


@pytest.mark.skipif(not oc.has_codec("czi"), reason="czi codec not registered")
def test_czi_codec_unsupported_source_raises():
    with pytest.raises(TypeError, match="unsupported CZI source"):
        oc.get_codec("czi").decode(42)


@pytest.mark.skipif(not oc.has_codec("czi"), reason="czi codec not registered")
def test_czi_signature_negative():
    """signature() returns False for non-CZI bytes."""
    assert oc.get_codec("czi").signature(b"NOTCZI" + b"\x00" * 32) is False
    assert oc.get_codec("czi").signature(b"") is False


# ===========================================================================
# _hdf5_codec.py — Codec.decode, open() with bytes / file-like
# ===========================================================================


h5py = pytest.importorskip("h5py")


def _make_h5_file(path: Path, dataset_name: str = "data",
                  shape: tuple = (4, 8, 8), dtype=np.uint8) -> np.ndarray:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, shape, dtype=dtype)
    with h5py.File(str(path), "w") as f:
        f.create_dataset(dataset_name, data=arr)
    return arr


def test_hdf5_signature_positive(tmp_path):
    p = tmp_path / "x.h5"
    _make_h5_file(p)
    head = p.read_bytes()[:32]
    assert oc.get_codec("hdf5").signature(head) is True


def test_hdf5_signature_negative():
    assert oc.get_codec("hdf5").signature(b"NOTHDF" + b"\x00" * 32) is False


def test_hdf5_codec_decode(tmp_path):
    p = tmp_path / "x.h5"
    src = _make_h5_file(p, shape=(2, 4, 4))
    arr = oc.get_codec("hdf5").decode(p)
    np.testing.assert_array_equal(arr, src)


def test_hdf5_codec_open_with_bytes(tmp_path):
    p = tmp_path / "x.h5"
    src = _make_h5_file(p)
    data = p.read_bytes()
    with oc.get_codec("hdf5").open(data) as r:
        np.testing.assert_array_equal(r.read(), src)


def test_hdf5_codec_open_with_file_like(tmp_path):
    p = tmp_path / "x.h5"
    src = _make_h5_file(p)
    with open(p, "rb") as f:
        with oc.get_codec("hdf5").open(f) as r:
            np.testing.assert_array_equal(r.read(), src)


def test_hdf5_codec_open_unsupported_source_raises():
    with pytest.raises(TypeError, match="unsupported HDF5 source"):
        oc.get_codec("hdf5").open(42)


def test_hdf5_reader_select_dataset(tmp_path):
    p = tmp_path / "multi.h5"
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (4, 4), dtype=np.uint8)
    b = rng.integers(0, 256, (4, 4), dtype=np.uint8)
    with h5py.File(str(p), "w") as f:
        f.create_dataset("a", data=a)
        f.create_dataset("b", data=b)

    from opencodecs._hdf5_codec import HdfReader
    r = HdfReader(p)
    try:
        names = r.dataset_names
        assert set(names) >= {"a", "b"}
        r.select("b")
        np.testing.assert_array_equal(r.read(), b)
    finally:
        r.close()


def test_hdf5_reader_no_image_dataset_raises(tmp_path):
    """File with only string/non-numeric datasets → ValueError."""
    p = tmp_path / "only_str.h5"
    with h5py.File(str(p), "w") as f:
        f.create_dataset("note", data="hello")
    from opencodecs._hdf5_codec import HdfReader
    with pytest.raises(ValueError, match="image-like"):
        HdfReader(p)


def test_hdf5_reader_2d_dataset_iter_frames(tmp_path):
    """A 2D dataset yields exactly one frame (the whole thing)."""
    p = tmp_path / "flat.h5"
    src = _make_h5_file(p, shape=(8, 8))
    from opencodecs._hdf5_codec import HdfReader
    with HdfReader(p) as r:
        frames = list(r.iter_frames())
    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0], src)


def test_hdf5_reader_getitem(tmp_path):
    """[idx] reads a single chunk along axis 0."""
    p = tmp_path / "x.h5"
    src = _make_h5_file(p, shape=(4, 4, 4))
    from opencodecs._hdf5_codec import HdfReader
    with HdfReader(p) as r:
        np.testing.assert_array_equal(r[2], src[2])


# ===========================================================================
# _zarr_codecs.py — missing branches
# ===========================================================================


@pytest.mark.skipif(not oc.has_codec("zstd"), reason="zstd codec not registered")
def test_zarr_v3_wrapper_handles_plain_bytes():
    """The wrapper's `else` branch (plain bytes that need bytes() coerce)."""
    pytest.importorskip("zarr")
    from opencodecs._zarr_codecs import OcZstd
    inst = OcZstd()
    payload = b"raw bytes path"
    enc = inst._encode_bytes(payload)
    assert inst._decode_bytes(enc) == payload


@pytest.mark.skipif(not oc.has_codec("zstd"), reason="zstd codec not registered")
def test_zarr_v3_wrapper_handles_memoryview():
    """memoryview goes through the bytes() coerce path."""
    pytest.importorskip("zarr")
    from opencodecs._zarr_codecs import OcZstd
    inst = OcZstd()
    payload = b"memoryview path test" * 20
    enc = inst._encode_bytes(memoryview(payload))
    assert inst._decode_bytes(memoryview(enc)) == payload


# ===========================================================================
# parallel.py — file-like input paths
# ===========================================================================


def _have_jxl():
    return oc.has_codec("jxl")


def test_parallel_frame_count_from_file_like(tmp_path):
    """frame_count(file_like) must call .read() (not pass the file as path)."""
    if not _have_jxl():
        pytest.skip("jxl not available")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    p = tmp_path / "x.jxl"
    oc.write(str(p), arr, format="jxl", lossless=True)

    from opencodecs import parallel
    with open(p, "rb") as f:
        assert parallel.frame_count(f) == 1


def test_parallel_decode_frames_from_file_like(tmp_path):
    if not _have_jxl():
        pytest.skip("jxl not available")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    p = tmp_path / "x.jxl"
    oc.write(str(p), arr, format="jxl", lossless=True)

    from opencodecs import parallel
    with open(p, "rb") as f:
        frames = parallel.decode_frames_parallel(f, n_workers=1)
    assert len(frames) == 1


# ===========================================================================
# tifffile_patch.py — tobytes() fallback when bytes() coerce fails
# ===========================================================================


class _NumpyLikeNotBytesCoerceable:
    """Simulates a numpy-array-like that fails ``bytes(obj)`` but exposes
    ``tobytes()``. tifffile sometimes hands such objects to encode()."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def tobytes(self) -> bytes:
        return self._payload

    # Force bytes(obj) to fail without triggering buffer protocol.
    def __bytes__(self):  # noqa: D401
        raise TypeError("simulated: cannot bytes() me")


@pytest.mark.skipif(not oc.has_codec("zstd"), reason="zstd codec not available")
def test_tifffile_patch_zstd_encode_tobytes_fallback():
    from opencodecs import tifffile_patch as patch
    payload = b"zstd encode tobytes fallback" * 50
    obj = _NumpyLikeNotBytesCoerceable(payload)
    enc = patch.zstd_encode(obj)
    assert patch.zstd_decode(enc) == payload


@pytest.mark.skipif(not oc.has_codec("deflate"), reason="deflate codec not available")
def test_tifffile_patch_deflate_encode_tobytes_fallback():
    from opencodecs import tifffile_patch as patch
    payload = b"deflate tobytes fallback" * 50
    obj = _NumpyLikeNotBytesCoerceable(payload)
    enc = patch.deflate_encode(obj)
    assert patch.deflate_decode(enc) == payload


@pytest.mark.skipif(not oc.has_codec("lz4"), reason="lz4 codec not available")
def test_tifffile_patch_lz4_encode_tobytes_fallback():
    from opencodecs import tifffile_patch as patch
    payload = b"lz4 tobytes fallback" * 50
    obj = _NumpyLikeNotBytesCoerceable(payload)
    enc = patch.lz4_encode(obj)
    assert patch.lz4_decode(enc) == payload


# ===========================================================================
# core/codec.py — codec_for_path fallback when reading magic fails
# ===========================================================================


def test_resolve_codec_path_unreadable_raises():
    """Path with unknown extension, can't open → KeyError (not OSError)."""
    from opencodecs.core.codec import _resolve_codec
    with pytest.raises(KeyError):
        _resolve_codec("/nonexistent/path/to/file.xyz")


# ===========================================================================
# zarr.py — JxlCodec missing branches (decode `out` reshape paths)
# ===========================================================================


def test_jxlcodec_decode_into_out_with_leading_singletons():
    """When zarr's `out` has extra leading length-1 axes vs decoded shape,
    decode() must reshape/squeeze the target view."""
    pytest.importorskip("numcodecs")
    if not oc.has_codec("jxl"):
        pytest.skip("jxl not available")
    from opencodecs.zarr import JxlCodec
    rng = np.random.default_rng(0)
    chunk = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    c = JxlCodec(lossless=True)
    enc = c.encode(chunk)

    # zarr-style out: (1, 1, H, W, C) with the same backing dtype.
    out = np.empty((1, 1, 32, 48, 3), dtype=np.uint8)
    ret = c.decode(enc, out=out)
    assert ret is out
    np.testing.assert_array_equal(np.squeeze(out), chunk)
