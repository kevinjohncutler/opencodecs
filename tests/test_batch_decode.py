"""decode_batch: many chunks per native call, same result as decode per chunk.

Every chunk set is encoded by the codec's own encoder and checked against
a loop of the codec's single-chunk ``decode(chunk, out=...)``: the same
bytes, the same sizes, and on a corrupt chunk the same exception with the
same message, whatever the thread count. The native helpers the zarr
reader builds on (read_ranges, place_windows) are checked against Python
file reads and NumPy slicing.
"""
from __future__ import annotations

import errno
import os

import numpy as np
import pytest

from opencodecs import decode_batch
from opencodecs.codecs import _bytetools

_zstd = pytest.importorskip("opencodecs.codecs._zstd")
_deflate = pytest.importorskip("opencodecs.codecs._deflate")
_lz4 = pytest.importorskip("opencodecs.codecs._lz4")

CHUNK = 4096


def _data(n_chunks=37, seed=0):
    rs = np.random.RandomState(seed)
    # Compressible but not trivial: small noise on a slow ramp.
    ramp = np.arange(n_chunks * CHUNK // 2, dtype=np.uint16) // 7
    return (ramp + rs.randint(0, 9, ramp.size)).astype(np.uint16).view(np.uint8)


def _bare_block(frame):
    """The one LZ4 block inside a single-block LZ4 frame."""
    flg = frame[4]
    pos = 6 + (8 if flg & 0x08 else 0) + 1
    size = int.from_bytes(frame[pos:pos + 4], "little")
    assert not size >> 31
    return frame[pos + 4:pos + 4 + size]


def _cases(raw):
    pieces = [bytes(raw[i:i + CHUNK]) for i in range(0, raw.size, CHUNK)]
    return {
        "zstd": ("zstd", [_zstd.encode(p) for p in pieces], {}, _zstd.decode),
        "deflate": ("deflate", [_deflate.encode(p) for p in pieces], {}, _deflate.decode),
        "deflate_raw": ("deflate", [_deflate.encode(p, raw=True) for p in pieces],
                        {"raw": True}, lambda c, out: _deflate.decode(c, out=out, raw=True)),
        "lz4": ("lz4", [_lz4.encode(p) for p in pieces], {}, _lz4.decode),
        "lz4block": ("lz4block", [_bare_block(_lz4.encode(p, blocksizeid=4)) for p in pieces],
                     {}, lambda c, out: _lz4.block_decode(c, len(out))),
    }


CASES = list(_cases(_data(2)))


@pytest.fixture(scope="module")
def raw():
    return _data()


@pytest.fixture(scope="module")
def cases(raw):
    return _cases(raw)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("numthreads", [None, 1, 3, 16])
def test_matches_per_chunk_decode(raw, cases, case, numthreads):
    codec, chunks, kw, _ = cases[case]
    out = np.zeros(raw.size, np.uint8)
    sizes = decode_batch(codec, chunks, out, numthreads=numthreads, **kw)
    assert sizes.dtype == np.int64 and (sizes == CHUNK).all()
    np.testing.assert_array_equal(out, raw)
    # One destination per chunk.
    outs = [bytearray(CHUNK) for _ in chunks]
    decode_batch(codec, chunks, outs, numthreads=numthreads, **kw)
    assert b"".join(outs) == raw.tobytes()
    # Chunks back to back in one buffer.
    blob = b"".join(chunks)
    edges = np.concatenate([[0], np.cumsum([len(c) for c in chunks])])
    out[:] = 0
    decode_batch(codec, blob, out, chunk_offsets=edges, numthreads=numthreads, **kw)
    np.testing.assert_array_equal(out, raw)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("numthreads", [1, 4])
def test_corrupt_chunk_raises_what_decode_raises(cases, case, numthreads):
    codec, chunks, kw, single = cases[case]
    bad = list(chunks)
    bad[5] = bytes(bad[5][:len(bad[5]) // 2])     # truncated
    bad[20] = b"\x00\x01" * 20                     # garbage, after the first failure
    with pytest.raises(Exception) as want:
        single(bad[5], out=bytearray(CHUNK))
    with pytest.raises(type(want.value)) as got:
        decode_batch(codec, bad, np.zeros(len(bad) * CHUNK, np.uint8),
                     numthreads=numthreads, **kw)
    assert str(got.value) == str(want.value)


def test_uneven_offsets_and_empty_chunks():
    chunks = [_zstd.encode(b"abc"), b"", _zstd.encode(b"defgh"), _zstd.encode(b"")]
    out = bytearray(12)
    sizes = decode_batch("zstd", chunks, out, offsets=[0, 3, 3, 10, 12], numthreads=1)
    assert sizes.tolist() == [3, 0, 5, 0]
    assert bytes(out[:3]) == b"abc" and bytes(out[3:8]) == b"defgh"


def test_short_destination_is_the_decode_error():
    chunk = _zstd.encode(b"x" * 100)
    with pytest.raises(_zstd.ZstdError) as want:
        _zstd.decode(chunk, out=bytearray(50))
    with pytest.raises(_zstd.ZstdError) as got:
        decode_batch("zstd", [chunk, chunk], bytearray(150), offsets=[0, 100, 150])
    assert str(got.value) == str(want.value)


def test_argument_errors():
    with pytest.raises(ValueError, match="not supported"):
        decode_batch("brotli", [b""], bytearray(1))
    with pytest.raises(ValueError, match="raw"):
        decode_batch("zstd", [b""], bytearray(1), raw=True)
    with pytest.raises(ValueError, match="equal parts"):
        decode_batch("zstd", [b"", b""], bytearray(3))
    with pytest.raises(ValueError):
        decode_batch("zstd", [b""], bytearray(4), offsets=[0, 5])
    with pytest.raises(ValueError, match="output buffers"):
        decode_batch("zstd", [b"", b""], [bytearray(1)])
    with pytest.raises((TypeError, BufferError)):
        decode_batch("zstd", [b""], b"read-only")
    assert decode_batch("zstd", [], bytearray(0)).size == 0


# ---------------------------------------------------------------------------
# read_ranges / place_windows
# ---------------------------------------------------------------------------


def _read(paths, offsets, lengths, capacity):
    n = len(paths)
    buf = bytearray(capacity)
    starts = np.empty(n, np.int64)
    sizes = np.empty(n, np.int64)
    errs = np.empty(n, np.int32)
    used = _bytetools.read_ranges(paths, np.asarray(offsets, np.int64),
                                  np.asarray(lengths, np.int64), buf, starts, sizes, errs)
    return buf, used, starts.tolist(), sizes.tolist(), errs.tolist()


def test_read_ranges_matches_python_reads(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(bytes(range(256)) * 4)
    b.write_bytes(b"hello world")
    same = str(a)
    paths = [same, same, str(b), str(b), tmp_path / "missing", str(a), str(b)]
    offsets = [0, 1000, 6, -5, 0, -2000, 20]
    lengths = [-1, 100, 3, -1, -1, 2, 4]
    buf, used, starts, sizes, errs = _read(paths, offsets, lengths, 2048)
    want = [a.read_bytes(), a.read_bytes()[1000:1100], b"wor", b"world", None,
            a.read_bytes()[:2], b""]
    for k, w in enumerate(want):
        if w is None:
            assert sizes[k] == -1 and errs[k] == errno.ENOENT
        else:
            assert bytes(buf[starts[k]:starts[k] + sizes[k]]) == w, k
    assert used == sum(len(w) for w in want if w is not None)


def test_read_ranges_reports_what_does_not_fit(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"x" * 100)
    buf, used, starts, sizes, errs = _read([str(a)] * 3, [0, 0, 0], [-1, 60, 30], 130)
    assert sizes == [100, -2, 30] and used == 130


def test_read_ranges_bad_path_object():
    with pytest.raises(TypeError):
        _read([123], [0], [-1], 10)
    with pytest.raises(ValueError):
        _read(["a\x00b"], [0], [-1], 10)


def test_place_windows_is_numpy_slicing():
    rs = np.random.RandomState(3)
    src = rs.randint(0, 255, 4000).astype(np.uint8)
    out = np.zeros((40, 64), np.uint8)
    want = out.copy()
    windows = np.array([[10, 50, 7, 13, 2, 30], [500, 64, 20, 64, 15, 0],
                        [3000, 9, 0, 5, 0, 0]], np.int64)
    for so, ss, rows, nb, dr, dc in windows.tolist():
        for r in range(rows):
            want[dr + r, dc:dc + nb] = src[so + r * ss:so + r * ss + nb]
    _bytetools.place_windows(src, out, windows)
    np.testing.assert_array_equal(out, want)
    with pytest.raises(ValueError, match="outside"):
        _bytetools.place_windows(src, out, np.array([[0, 64, 41, 1, 0, 0]], np.int64))
    with pytest.raises(ValueError, match="outside"):
        _bytetools.place_windows(src, out, np.array([[3990, 64, 1, 11, 0, 0]], np.int64))
