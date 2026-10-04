"""Path-backed CZI readers fetch payloads with positional reads, and read() takes indices.

A reader opened from a path reads each sub-block payload with one positional read into a per-thread
buffer instead of slicing the mmap (``_PAYLOAD_PREAD``). What has to hold: every decode entry point
returns the same pixels as the mmap path for every compression, concurrent readers on one file do not
see each other's buffers, and a truncated file still raises CziError. ``read(indices=...)`` reads a
chosen set of sub-blocks into one stack, in the order given, through both the pool and bounded paths.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import opencodecs as oc
from opencodecs import _czi_reader

pytestmark = pytest.mark.skipif(
    not oc.has_codec("czi"),
    reason="czi codec requires native zstd + bytetools extensions",
)

from _czi_fixture import czi_bytes  # noqa: E402

COMPRESSIONS = [(0, False), (5, False), (6, False), (6, True)]


def _stack(n=7, shape=(20, 24), seed=0):
    rs = np.random.RandomState(seed)
    return np.stack([rs.randint(0, 65535, shape).astype(np.uint16) for _ in range(n)])


@pytest.fixture(scope="module")
def frames():
    return _stack()


@pytest.fixture(params=COMPRESSIONS, ids=lambda c: f"comp{c[0]}{'-hilo' if c[1] else ''}")
def path(request, tmp_path, frames):
    comp, hilo = request.param
    p = tmp_path / "stack.czi"
    p.write_bytes(czi_bytes(frames, compression=comp, hilo=hilo))
    return p


def _open(path):
    return oc.get_codec("czi").open(str(path))


@pytest.mark.skipif(not _czi_reader._PAYLOAD_PREAD, reason="no positional reads on this platform")
def test_positional_and_mmap_paths_match(path, frames, monkeypatch):
    results = {}
    for flag in (True, False):
        monkeypatch.setattr(_czi_reader, "_PAYLOAD_PREAD", flag)
        with _open(path) as r:
            results[flag] = (r.read(), np.stack([r.read_tile(i) for i in range(len(r))]), r[2:5], r[-1],
                             r.read(indices=[4, 0]))
    for got, want in zip(results[True], results[False]):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(results[True][0], frames)


def test_payload_buffers_are_reused(path):
    _czi_reader._PAYLOAD_FREE.clear()
    with _open(path) as r:
        for _ in range(5):
            r.read()
    assert len(_czi_reader._PAYLOAD_FREE) <= _czi_reader._PAYLOAD_KEEP


def test_payload_location_is_cached(path):
    with _open(path) as r:
        r.read()
        # Every source type computes and keeps its payload ranges, the
        # mapping (no positional reads, e.g. Windows) as well.
        assert len(r._payload_ranges) == len(r)


def test_concurrent_tile_reads_on_one_reader(path, frames):
    order = list(range(len(frames))) * 8
    with _open(path) as r, ThreadPoolExecutor(8) as ex:
        tiles = list(ex.map(r.read_tile, order))
    for i, tile in zip(order, tiles):
        np.testing.assert_array_equal(tile, frames[i])


def test_truncated_file_raises(tmp_path, frames):
    p = tmp_path / "cut.czi"
    data = czi_bytes(frames, compression=6, hilo=True)
    with _open_bytes_path(p, data) as r:
        size = p.stat().st_size
        last = r.entries[-1].file_position
    # cut into the last sub-block's payload while the directory (written first) still describes it
    p.write_bytes(data[:last + 300] if last + 300 < size else data[:-64])
    with pytest.raises((_czi_reader.CziError, ValueError, OSError)):
        with _open(p) as r2:
            r2.read_tile(len(frames) - 1)


def _open_bytes_path(p, data):
    p.write_bytes(data)
    return _open(p)


@pytest.mark.parametrize("indices", [[3], [0, 6, 2], [-1, 0], [4, 4, 1], list(range(7))[::-1]])
def test_read_indices_order_and_values(path, frames, indices):
    with _open(path) as r:
        got = r.read(indices=indices, squeeze=False)
    assert got.shape[0] == len(indices)
    np.testing.assert_array_equal(got.reshape(len(indices), *frames.shape[1:]), frames[indices])


def test_read_indices_bounded_path_and_out(path, frames):
    idx = [5, 1, 3]
    with _open(path) as r:
        bounded = r.read(indices=idx, max_pending_bytes=1)
        out = np.zeros((3, *frames.shape[1:]), np.uint16)
        filled = r.read(indices=idx, out=out)
    np.testing.assert_array_equal(bounded, frames[idx])
    assert filled.base is out or filled is out
    np.testing.assert_array_equal(out, frames[idx])


def test_read_indices_errors(path):
    with _open(path) as r:
        with pytest.raises(IndexError):
            r.read(indices=[0, 99])
        with pytest.raises(ValueError):
            r.read(indices=[])
        with pytest.raises(TypeError):
            r.read(indices=[1.5])


def test_read_indices_single_squeezes(path, frames):
    with _open(path) as r:
        np.testing.assert_array_equal(r.read(indices=[2]), frames[2])


# ----- attachments ---------------------------------------------------------

def _with_attachments(data: bytes, items) -> bytes:
    """Append ZISRAWATTACH segments and a ZISRAWATTDIR directory, and point the header at it."""
    import struct

    def segment(sid, payload):
        pad = (-len(payload)) % 32
        return struct.pack("<16sqq", sid.ljust(16, b"\x00"), len(payload) + pad, len(payload)) + payload + b"\x00" * pad

    out = bytearray(data)
    entries = []
    for name, ctype, blob in items:
        pos = len(out)
        entry = struct.pack("<2s10sqi16s8s80s", b"A1", b"", pos, 0, bytes(range(16)),
                            ctype.encode(), name.encode())
        out += segment(b"ZISRAWATTACH", struct.pack("<i12x", len(blob)) + entry + b"\x00" * 112 + blob)
        entries.append(entry)
    dir_pos = len(out)
    out += segment(b"ZISRAWATTDIR", struct.pack("<i252x", len(entries)) + b"".join(entries))
    out[104:112] = struct.pack("<q", dir_pos)       # attachment_directory_position in the file header
    return bytes(out)


@pytest.mark.parametrize("from_path", [True, False], ids=["path", "buffer"])
def test_attachments_listed_and_read_raw(tmp_path, frames, from_path):
    jpeg = b"\xff\xd8\xff\xe0" + bytes(range(256)) * 3
    data = _with_attachments(czi_bytes(frames, compression=6, hilo=True),
                             [("TimeStamps", "CZTIMS", b"\x10\x00\x00\x00" + b"\x01" * 12), ("Thumbnail", "JPG", jpeg)])
    src = tmp_path / "att.czi"
    src.write_bytes(data)
    with oc.get_codec("czi").open(str(src) if from_path else data) as r:
        names = [(a.name, a.content_file_type) for a in r.attachments()]
        assert names == [("TimeStamps", "CZTIMS"), ("Thumbnail", "JPG")]
        assert r.read_attachment("thumbnail") == jpeg              # case-insensitive, stored bytes unchanged
        assert r.read_attachment(r.attachments()[1]) == jpeg
        with pytest.raises(KeyError):
            r.read_attachment("label")
        np.testing.assert_array_equal(r.read(), frames)            # pixels unaffected


def test_no_attachment_directory_lists_none(path):
    with _open(path) as r:
        assert r.attachments() == []


@pytest.mark.parametrize("name", ["ome_axioscan_pyramid.czi", "idr0011_plate1_scene1.czi"])
def test_attachments_match_czifile(name):
    import os
    czifile = pytest.importorskip("czifile")
    p = os.path.join(os.path.dirname(__file__), "..", ".test_data", "czi", name)
    if not os.path.exists(p):
        pytest.skip("test data not present")
    with czifile.CziFile(p) as f:
        ref = {a.attachment_entry.name: a.data(raw=True) for a in f.attachments()}
    with oc.get_codec("czi").open(p) as r:
        assert {a.name: r.read_attachment(a) for a in r.attachments()} == ref


def test_lazy_directory_is_complete_under_concurrent_first_access(path, frames):
    for _ in range(20):
        with _open(path) as r, ThreadPoolExecutor(8) as ex:
            assert set(ex.map(lambda _: len(r.entries), range(32))) == {len(frames)}


def test_attachment_only_read_skips_the_directory(tmp_path, frames):
    data = _with_attachments(czi_bytes(frames), [("Thumbnail", "JPG", b"\xff\xd8jpeg")])
    p = tmp_path / "t.czi"
    p.write_bytes(data)
    with _open(p) as r:
        assert r.read_attachment("thumbnail") == b"\xff\xd8jpeg"
        assert r._entries is None                     # directory never read
        assert r.shape[0] == len(frames)              # and still there on demand


def test_closed_reader_says_so(path):
    r = _open(path)
    r.close()
    with pytest.raises(_czi_reader.CziError, match="closed"):
        r.entries


@pytest.mark.skipif(not _czi_reader._PAYLOAD_PREAD, reason="no positional reads on this platform")
def test_split_reads_match_whole_reads(path, frames, monkeypatch):
    monkeypatch.setattr(_czi_reader, "_SPLIT_PIECE", 64)     # split even these tiny payloads
    with _open(path) as r:
        np.testing.assert_array_equal(np.stack([r.read_tile(i) for i in range(len(r))]), frames)
        np.testing.assert_array_equal(r.read(indices=[3, 1]), frames[[3, 1]])
        np.testing.assert_array_equal(r.read(), frames)


# ----- payload index ------------------------------------------------------------

def test_payload_index_round_trip_reads_no_headers(path, frames, monkeypatch):
    with _open(path) as r:
        index = r.payload_index()
    reads = []
    real = _czi_reader.CziReader._parse_header
    monkeypatch.setattr(_czi_reader.CziReader, "_parse_header", lambda self: (reads.append(1), real(self))[1])
    with _czi_reader.CziReader(str(path), index=index) as r:
        np.testing.assert_array_equal(r.read(), frames)
        np.testing.assert_array_equal(r.read(indices=[5, 2]), frames[[5, 2]])
        np.testing.assert_array_equal(r.read_tile(3), frames[3])
        np.testing.assert_array_equal(r.read(max_pending_bytes=1), frames)
        assert reads == []                               # no file header, no directory
        assert r.shape == frames.shape and r.dtype == frames.dtype
    with _czi_reader.CziReader(buffer=path.read_bytes(), index=index) as r:   # any source
        np.testing.assert_array_equal(r.read(), frames)


def test_index_with_dtype_and_metadata_on_demand(path, frames):
    with _open(path) as r:
        index = r.payload_index()
        meta = r.metadata_bytes
    index = dict(index); index.pop("pixel_type"); index["dtype"] = "uint16"
    with _czi_reader.CziReader(str(path), index=index) as r:
        np.testing.assert_array_equal(r.read(), frames)
        assert r.metadata_bytes == meta                   # header parsed only now


def test_index_outside_the_file_is_refused(path):
    with _open(path) as r:
        index = r.payload_index()
    index["offsets"][0] = 10 ** 12
    with pytest.raises(_czi_reader.CziError):
        _czi_reader.CziReader(str(path), index=index)
