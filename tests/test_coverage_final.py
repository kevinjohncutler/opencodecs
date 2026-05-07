"""Final coverage-pushing tests aimed at the remaining edge paths so we
can land at >=95%. Each block is small and surgical: hit a specific
line that the broader test modules don't touch.

Targets:
  * ``_czi_reader.py``  — synthetic-bytes parsing error paths
                          (bad directory magic, bad subblock magic, bad
                          DV schema, ZSTDHDR header errors)
  * ``_hdf5_codec.py``  — 3+ ndim dataset iter_frames (slices along axis 0)
  * ``_io_helpers.py``  — file-like input/output paths
  * ``core/io.py``      — close-error swallow, file-size-fallback paths
  * ``zarr.py``         — JxlCodec out-buffer reshape fallback
  * ``tiff_reader.py``  — strip-based page (n_segments path)
"""

from __future__ import annotations

import io
import os
import struct
from pathlib import Path

import numpy as np
import pytest

import opencodecs as oc


# ---------------------------------------------------------------------------
# Synthetic CZI-bytes parsing — hit specific error branches without a real file
# ---------------------------------------------------------------------------


def _make_minimal_czi_header(directory_position: int, metadata_position: int,
                             total_size: int = 4096) -> bytes:
    """Build just enough ZISRAWFILE header to reach _parse_directory.

    Layout (CZI 1.2.2):
      0..16   16-byte magic-padded sid (b'ZISRAWFILE\\0\\0\\0\\0\\0\\0')
      16..32  alloc_size (q), used_size (q)
      32..    payload: I major, I minor, I r1, I r2,
              16 primary_guid, 16 file_guid, I file_part,
              q directory_position, q metadata_position,
              I update_pending, q attachment_directory_position
    """
    sid = b"ZISRAWFILE" + b"\x00" * 6
    seg_hdr = struct.pack("<16sqq", sid, 0, 0)
    payload = (
        struct.pack("<II", 1, 2)  # major, minor
        + b"\x00" * 8              # 2x reserved
        + b"\x00" * 32             # 2x 16-byte GUIDs
        + struct.pack("<I", 0)     # file_part
        + struct.pack("<q", directory_position)
        + struct.pack("<q", metadata_position)
        + struct.pack("<I", 0)     # update_pending
        + struct.pack("<q", 0)     # attachment_directory_position
    )
    head = seg_hdr + payload
    return head + b"\x00" * (total_size - len(head))


def test_czi_directory_position_out_of_range_raises(tmp_path):
    """directory_position >= file size → CziError."""
    from opencodecs._czi_reader import CziReader, CziError
    p = tmp_path / "bad_dir.czi"
    # directory_position points way past the end-of-file
    p.write_bytes(_make_minimal_czi_header(directory_position=10**9, metadata_position=0))
    with pytest.raises(CziError, match="directory_position"):
        CziReader(str(p))


def test_czi_directory_position_zero_raises(tmp_path):
    from opencodecs._czi_reader import CziReader, CziError
    p = tmp_path / "zero_dir.czi"
    p.write_bytes(_make_minimal_czi_header(directory_position=0, metadata_position=0))
    with pytest.raises(CziError, match="directory_position"):
        CziReader(str(p))


def test_czi_bad_directory_magic_raises(tmp_path):
    """directory_position is in range but the bytes there aren't a
    ZISRAWDIRECTORY segment header."""
    from opencodecs._czi_reader import CziReader, CziError
    p = tmp_path / "bad_dirmagic.czi"
    dir_pos = 256  # within the file
    head = bytearray(_make_minimal_czi_header(
        directory_position=dir_pos, metadata_position=0, total_size=512,
    ))
    # Overwrite at dir_pos with non-DIR bytes
    head[dir_pos:dir_pos + 16] = b"NOTADIR" + b"\x00" * 9
    p.write_bytes(bytes(head))
    with pytest.raises(CziError, match="ZISRAWDIRECTORY"):
        CziReader(str(p))


def test_czi_bad_directory_entry_schema_raises(tmp_path):
    """The first directory entry has an unknown schema (not 'DV')."""
    from opencodecs._czi_reader import CziReader, CziError
    p = tmp_path / "bad_schema.czi"
    dir_pos = 256
    head = bytearray(_make_minimal_czi_header(
        directory_position=dir_pos, metadata_position=0, total_size=4096,
    ))
    # Build a valid-looking ZISRAWDIRECTORY segment with one entry of bogus schema.
    sid = b"ZISRAWDIRECTORY" + b"\x00"
    head[dir_pos:dir_pos + 32] = struct.pack("<16sqq", sid, 0, 0)
    # Directory payload: entry_count=1, then 124 reserved, then the entry.
    head[dir_pos + 32:dir_pos + 36] = struct.pack("<I", 1)
    # Reserved 124 bytes already zero. Entry starts at dir_pos + 32 + 128.
    entry_off = dir_pos + 32 + 128
    # 32-byte directory entry header with schema='XX' (bogus)
    head[entry_off:entry_off + 2] = b"XX"
    p.write_bytes(bytes(head))
    with pytest.raises(CziError, match="schema"):
        CziReader(str(p))


# ---------------------------------------------------------------------------
# _czi_reader.py — _decode_zstdhdr error paths via direct call
# ---------------------------------------------------------------------------


def test_czi_zstdhdr_data_too_short():
    from opencodecs._czi_reader import CziReader, CziError
    with pytest.raises(CziError, match="too short"):
        CziReader._decode_zstdhdr(memoryview(b"\x01"), np.dtype("u1"), (1,), 1)


def test_czi_zstdhdr_invalid_header_byte_zero():
    from opencodecs._czi_reader import CziReader, CziError
    with pytest.raises(CziError, match="invalid ZSTDHDR header"):
        CziReader._decode_zstdhdr(memoryview(b"\x00\x00"), np.dtype("u1"), (1,), 1)


def test_czi_zstdhdr_header_byte_too_big():
    from opencodecs._czi_reader import CziReader, CziError
    # header_size=10 but view length is 4 → header_size >= len(view)
    with pytest.raises(CziError, match="invalid ZSTDHDR header"):
        CziReader._decode_zstdhdr(memoryview(b"\x0a\x00\x00\x00"),
                                  np.dtype("u1"), (1,), 1)


def test_czi_zstdhdr_truncated_chunk_type_1():
    """chunk_type=1 needs a flag byte after — if header_size truncates
    just before it, raise."""
    from opencodecs._czi_reader import CziReader, CziError
    # header_size=2, len(view)=3 (> header_size, so the upper-bound check
    # passes). Byte at pos 1 = chunk_type 1; pos advances to 2 == header_size,
    # which makes "pos >= header_size" trip the truncate raise.
    with pytest.raises(CziError, match="truncated ZSTDHDR"):
        CziReader._decode_zstdhdr(
            memoryview(b"\x02\x01\x00"), np.dtype("u1"), (1,), 1,
        )


# ---------------------------------------------------------------------------
# _hdf5_codec.py — 3D dataset iter_frames yields per-axis-0 slices
# ---------------------------------------------------------------------------


h5py = pytest.importorskip("h5py")


def test_hdf5_reader_3d_iter_frames_yields_slices(tmp_path):
    """3D dataset → iter_frames yields shape[0] arrays, each shape[1:]."""
    from opencodecs._hdf5_codec import HdfReader
    p = tmp_path / "3d.h5"
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (5, 4, 4), dtype=np.uint8)
    with h5py.File(str(p), "w") as f:
        f.create_dataset("data", data=arr)
    with HdfReader(p) as r:
        frames = list(r.iter_frames())
    assert len(frames) == 5
    for i, frame in enumerate(frames):
        np.testing.assert_array_equal(frame, arr[i])


# ---------------------------------------------------------------------------
# _io_helpers.py — file-like read/write branches
# ---------------------------------------------------------------------------


def test_read_src_from_file_like():
    """File-like input → src.read() branch."""
    from opencodecs.core._io_helpers import read_src
    bio = io.BytesIO(b"hello file-like")
    assert read_src(bio) == b"hello file-like"


def test_write_dest_to_file_like():
    """File-like output → dest.write() branch."""
    from opencodecs.core._io_helpers import write_dest
    bio = io.BytesIO()
    ret = write_dest(b"some bytes", bio)
    assert ret is None
    assert bio.getvalue() == b"some bytes"


def test_codec_round_trip_file_like_io():
    """End-to-end through a real codec with file-like in / file-like out."""
    payload = b"compressible payload " * 50
    encoded_buf = io.BytesIO()
    oc.write(encoded_buf, payload, format="zstd")
    encoded_buf.seek(0)
    assert oc.read(encoded_buf, format="zstd") == payload


# ---------------------------------------------------------------------------
# core/io.py — file_size fallback for in-memory streams + close exception
# ---------------------------------------------------------------------------


def test_chunked_reader_file_size_seek_fallback():
    """A seekable file-like without .fileno() exercises the seek/tell
    fallback path in _try_size."""
    from opencodecs.core.io import BackgroundChunkReader
    bio = io.BytesIO(b"x" * 5000)
    with BackgroundChunkReader(bio) as r:
        assert r.file_size == 5000


def test_chunked_reader_close_swallows_file_close_error(tmp_path):
    """If the owned file's close() raises, BackgroundChunkReader.close()
    swallows the error — never propagates."""
    from opencodecs.core.io import BackgroundChunkReader
    p = tmp_path / "data.bin"
    p.write_bytes(b"hello")
    r = BackgroundChunkReader(p)
    list(r)  # drain so bg thread exits cleanly first

    # Replace internal file with one that raises on close.
    class BoomCloseFile:
        def close(self):
            raise OSError("simulated close failure")

    r._file = BoomCloseFile()
    r._closed = False  # reset so close() actually runs the cleanup branch
    r.close()  # must NOT raise


# ---------------------------------------------------------------------------
# tiff_reader.py — strip-based page (n_segments routes through page.asarray)
# ---------------------------------------------------------------------------


def test_tiff_reader_imread_strip_zero_segments(tmp_path):
    """A TIFF page with no tile/strip dataoffsets falls back to page.asarray.
    This is the n_segments == 0 short-circuit branch."""
    tifffile = pytest.importorskip("tifffile")
    from opencodecs.tiff_reader import imread
    rng = np.random.default_rng(0)
    # A small image; tifffile may or may not write strips. The point is to
    # ensure the path through _read_one_page_parallel handles both.
    arr = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    p = tmp_path / "tiny.tif"
    tifffile.imwrite(str(p), arr)
    out = imread(p)
    np.testing.assert_array_equal(out, arr)


def test_tiff_reader_imread_stack_strip_pages(tmp_path):
    """imread_stack with strip-based (non-tiled) pages still produces
    the right stack."""
    tifffile = pytest.importorskip("tifffile")
    from opencodecs.tiff_reader import imread_stack
    rng = np.random.default_rng(0)
    pages = [rng.integers(0, 256, (32, 32), dtype=np.uint8) for _ in range(3)]
    p = tmp_path / "stack.tif"
    with tifffile.TiffWriter(str(p)) as tw:
        for pg in pages:
            tw.write(pg)
    stack = imread_stack(p)
    assert stack.shape == (3, 32, 32)
    np.testing.assert_array_equal(stack, np.stack(pages))


# ---------------------------------------------------------------------------
# zarr.py — JxlCodec decode out-buffer reshape branch (try/except)
# ---------------------------------------------------------------------------


def test_jxlcodec_decode_into_out_with_extra_singleton_squeeze():
    """When `out` has more singleton axes than the decoded array AND the
    direct reshape would fail, the codec falls back to manual squeeze."""
    pytest.importorskip("numcodecs")
    if not oc.has_codec("jxl"):
        pytest.skip("jxl not available")
    from opencodecs.zarr import JxlCodec
    rng = np.random.default_rng(0)
    chunk = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    c = JxlCodec(lossless=True)
    enc = c.encode(chunk)

    # Deeper nested singleton wrap to force the squeeze fallback path.
    out = np.empty((1, 1, 1, 32, 48, 3), dtype=np.uint8)
    ret = c.decode(enc, out=out)
    assert ret is out
    np.testing.assert_array_equal(np.squeeze(out), chunk)
