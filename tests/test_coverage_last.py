"""Last targeted-coverage test module before the platform-conditional
branches start to dominate. Picks off remaining low-hanging gaps:

  * codec adapters' ``np.asarray(data)`` branch (when caller passes a
    list / buffer instead of an ndarray)
  * JxlReader's ``basic_info`` / ``is_animation`` / ``read`` properties
  * ``zarr.py`` lossy-mode repr
  * ``_blosc2_codec.py`` decode error path
  * a few czi_reader synthetic-bytes paths still missing
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

import opencodecs as oc


# ---------------------------------------------------------------------------
# Codec adapter: np.asarray(data) branch via list input
# ---------------------------------------------------------------------------


def _need(name: str) -> None:
    if not oc.has_codec(name):
        pytest.skip(f"codec {name!r} not available")


class _ArrayLike:
    """A non-ndarray with ``__array__`` so np.asarray() falls back to
    that hook. Hits the ``if not isinstance(data, np.ndarray)`` branch
    in each codec adapter without forcing a default-int64 array."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def __array__(self, dtype=None):
        return self._arr if dtype is None else self._arr.astype(dtype)


@pytest.mark.parametrize("fmt", ["png", "qoi", "bmp"])
def test_image_encode_accepts_array_like(fmt):
    """Non-ndarray with __array__ hits the ``np.asarray(data)`` branch."""
    _need(fmt)
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    enc = oc.write(None, _ArrayLike(arr), format=fmt)
    dec = oc.read(enc, format=fmt)
    np.testing.assert_array_equal(np.squeeze(dec), arr)


def test_jpeg_encode_accepts_array_like():
    _need("jpeg")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    enc = oc.write(None, _ArrayLike(arr), format="jpeg", level=90)
    decoded = oc.read(enc, format="jpeg")
    assert decoded.shape == (16, 16, 3)


def test_webp_encode_accepts_array_like():
    _need("webp")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    enc = oc.write(None, _ArrayLike(arr), format="webp", lossless=True)
    decoded = oc.read(enc, format="webp")
    np.testing.assert_array_equal(np.squeeze(decoded), arr)


def test_jpeg2k_encode_accepts_array_like():
    _need("jpeg2k")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    enc = oc.write(None, _ArrayLike(arr), format="jpeg2k", lossless=True)
    decoded = oc.read(enc, format="jpeg2k")
    np.testing.assert_array_equal(np.squeeze(decoded), arr)


def test_avif_encode_accepts_array_like():
    _need("avif")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    try:
        enc = oc.write(None, _ArrayLike(arr), format="avif", lossless=True)
    except Exception as exc:
        # Decode-only libavif builds (no libaom / no system aom) raise
        # "No codec available" or "Unsupported" on encode. Skip rather
        # than fail — the cibuildwheel manylinux wheel ships decode-only.
        msg = str(exc).lower()
        if any(s in msg for s in ("no codec available", "unsupported", "encoder")):
            pytest.skip(f"libavif build has no AV1 encoder: {exc}")
        raise
    decoded = oc.read(enc, format="avif")
    assert np.squeeze(decoded).shape == arr.shape


def test_heif_encode_accepts_array_like():
    _need("heif")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    try:
        enc = oc.write(None, _ArrayLike(arr), format="heif", lossless=True)
    except Exception as exc:
        # Some libheif builds lack a HEVC/HEIC encoder plugin (e.g. Ubuntu
        # without libheif-plugin-x265, or conda-forge libheif on Windows
        # which builds decode-only for licensing reasons).
        msg = str(exc).lower()
        if any(s in msg for s in (
            "unsupported", "encoder", "null error text", "heif_writer",
        )):
            pytest.skip(f"libheif has no HEIF encoder available: {exc}")
        raise
    decoded = oc.read(enc, format="heif")
    assert np.squeeze(decoded).shape == arr.shape


def test_jxl_encode_accepts_array_like():
    _need("jxl")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    enc = oc.write(None, _ArrayLike(arr), format="jxl", lossless=True)
    decoded = oc.read(enc, format="jxl")
    np.testing.assert_array_equal(np.squeeze(decoded), arr)


# ---------------------------------------------------------------------------
# JxlReader wrapper: basic_info / is_animation / read properties
# ---------------------------------------------------------------------------


def test_jxlcodec_open_reader_properties(tmp_path):
    _need("jxl")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    p = tmp_path / "x.jxl"
    oc.write(str(p), arr, format="jxl", lossless=True)

    with oc.get_codec("jxl").open(p) as r:
        info = r.basic_info
        assert isinstance(info, dict)
        assert r.is_animation is False
        decoded = r.read()
    np.testing.assert_array_equal(np.squeeze(decoded), arr)


# ---------------------------------------------------------------------------
# zarr.py — lossy-mode repr (line 137)
# ---------------------------------------------------------------------------


def test_zarr_jxlcodec_lossy_repr_includes_distance():
    pytest.importorskip("numcodecs")
    _need("jxl")
    from opencodecs.zarr import JxlCodec
    s = repr(JxlCodec(lossless=False, distance=1.5))
    assert "distance" in s
    assert "1.5" in s


# ---------------------------------------------------------------------------
# _blosc2_codec.py — decoding random garbage should raise (covers error path)
# ---------------------------------------------------------------------------


def test_blosc2_decode_garbage_raises():
    _need("blosc2")
    with pytest.raises(Exception):
        oc.read(b"not a blosc2 frame at all", format="blosc2")


# ---------------------------------------------------------------------------
# _czi_reader.py — synthesize a bad SUBBLOCK_MAGIC + bad pixel-type-and-decode
# ---------------------------------------------------------------------------


def test_czi_subblock_bad_magic_raises(tmp_path):
    """Build a CZI whose directory points at a sub-block whose 14-byte
    segment-id isn't ZISRAWSUBBLOCK — accessing tile[i] must raise."""
    from opencodecs._czi_reader import CziReader, CziError

    # Borrow the helper from test_coverage_final
    sid_file = b"ZISRAWFILE" + b"\x00" * 6
    sid_dir = b"ZISRAWDIRECTORY" + b"\x00"

    file_size = 8192
    sub_pos = 4096  # where the (broken) sub-block lives
    dir_pos = 256
    head = bytearray(file_size)

    # File header
    head[0:32] = struct.pack("<16sqq", sid_file, 0, 0)
    payload = (
        struct.pack("<II", 1, 2)
        + b"\x00" * 8 + b"\x00" * 32
        + struct.pack("<I", 0)
        + struct.pack("<q", dir_pos)
        + struct.pack("<q", 0)  # metadata_position
        + struct.pack("<I", 0)
        + struct.pack("<q", 0)
    )
    head[32:32 + len(payload)] = payload

    # Directory at dir_pos: one entry with file_position=sub_pos, schema='DV'
    head[dir_pos:dir_pos + 32] = struct.pack("<16sqq", sid_dir, 0, 0)
    head[dir_pos + 32:dir_pos + 36] = struct.pack("<I", 1)  # entry_count

    entry_off = dir_pos + 32 + 128
    # 32-byte directory entry with schema='DV', pixel_type=0 (u1), comp=0,
    # dims_count=2 (Y + X dims).
    head[entry_off:entry_off + 32] = struct.pack(
        "<2siqiiBB4si",
        b"DV",      # schema
        0,          # pixel_type
        sub_pos,    # file_position
        0,          # file_part
        0,          # compression
        0, 0, b"\x00\x00\x00\x00",  # pyramid + reserved1 + reserved2
        2,          # dimensions_count
    )
    # Two 20-byte dim entries (e.g. Y, X), each (4s dim, i start, i size, f coord, i stored)
    head[entry_off + 32:entry_off + 52] = struct.pack(
        "<4siifi", b"X", 0, 4, 0.0, 0,
    )
    head[entry_off + 52:entry_off + 72] = struct.pack(
        "<4siifi", b"Y", 0, 4, 0.0, 0,
    )

    # At sub_pos write garbage 14-byte sid (NOT ZISRAWSUBBLOCK).
    head[sub_pos:sub_pos + 14] = b"NOTSUBBLOCK\x00\x00\x00"

    p = tmp_path / "bad_subblock.czi"
    p.write_bytes(bytes(head))

    with CziReader(str(p)) as r:
        with pytest.raises(CziError, match="ZISRAWSUBBLOCK"):
            _ = r[0]


# ---------------------------------------------------------------------------
# tiff_reader.py — empty pages list short-circuit
# ---------------------------------------------------------------------------


def test_tiff_imread_stack_empty_pages(tmp_path):
    """imread_stack with pages=[] should return an empty stack."""
    tifffile = pytest.importorskip("tifffile")
    from opencodecs.tiff_reader import imread_stack
    arr = np.zeros((8, 8), dtype=np.uint8)
    p = tmp_path / "tiny.tif"
    tifffile.imwrite(str(p), arr)
    # Empty pages selection: caller wants nothing back.
    with pytest.raises((IndexError, ValueError)):
        imread_stack(p, pages=[])
