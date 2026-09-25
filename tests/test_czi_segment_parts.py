"""Sub-blocks emitted as header, payload and padding (priority 5).

The writer used to concatenate every sub-block into one string and then copy
it again to wrap it in a segment header. It now keeps a descriptor of parts.
What has to hold: the file bytes and directory offsets are unchanged, which
is checked against the test fixture's independent serializer; parts are
written correctly through a destination that only accepts short writes; a
failing sink aborts the writer cleanly; verification still catches corrupted
payloads (covered in test_czi_verification.py); complete files reopen
independently; and the borrowed-buffer path does not let a producer's later
mutation leak into a frame that was already written.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

import opencodecs as oc

pytestmark = pytest.mark.skipif(
    not oc.has_codec("czi"),
    reason="czi codec requires native zstd + bytetools extensions",
)

from _czi_fixture import czi_bytes  # noqa: E402

from opencodecs._czi_writer import (  # noqa: E402
    CziWriter, CziWriterError, _SubBlockSegment, _build_subblock,
)


def _frames(dtype, n=3, shape=(24, 40), seed=0):
    rs = np.random.RandomState(seed)
    if np.issubdtype(dtype, np.floating):
        return [rs.rand(*shape).astype(dtype) for _ in range(n)]
    info = np.iinfo(dtype)
    return [rs.randint(info.min, info.max, shape).astype(dtype) for _ in range(n)]


@pytest.mark.parametrize("compression,code", [("none", 0), ("zstd", 5), ("zstdhdr", 6)])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_file_bytes_match_the_fixture_serializer(tmp_path, compression, code, dtype):
    """Byte-for-byte against an independent implementation of the layout."""
    frames = _frames(dtype)
    path = tmp_path / f"parts_{compression}_{np.dtype(dtype).name}.czi"
    with CziWriter(path, compression=compression, hilo=True) as w:
        for f in frames:
            w.write(f)
    expected = czi_bytes(np.stack(frames), compression=code, hilo=True)
    assert path.read_bytes() == expected


def test_segment_parts_reassemble_to_the_old_single_string():
    frame = np.arange(16 * 24, dtype=np.uint16).reshape(16, 24)
    seg, entry = _build_subblock(
        frame, pixel_type=1, compression_code=6, hilo=True,
        file_position=1234, logical_shape=(16, 24))
    assert isinstance(seg, _SubBlockSegment)
    whole = seg.tobytes()
    assert len(whole) == seg.size
    assert len(whole) % 32 == 0
    assert whole[:14] == b"ZISRAWSUBBLOCK"
    # used == alloc == everything after the 16-byte sid, including the
    # pixel payload but not the trailing alignment.
    import struct
    alloc, used = struct.unpack_from("<qq", whole, 16)
    assert alloc == used == len(seg.header) - 32 + len(seg.payload)
    assert whole[len(seg.header):len(seg.header) + len(seg.payload)] == bytes(seg.payload)
    assert whole[len(seg.header) + len(seg.payload):] == bytes(seg.pad)


class _ShortWriter:
    """Accepts at most ``limit`` bytes per call, like a pipe under load."""

    def __init__(self, limit):
        self.limit = limit
        self.buffer = io.BytesIO()
        self.calls = 0

    def write(self, data):
        self.calls += 1
        view = memoryview(data)[: self.limit]
        return self.buffer.write(view)


@pytest.mark.parametrize("limit", [1, 7, 64, 1 << 20])
def test_parts_survive_short_writes(limit):
    frame = np.random.RandomState(3).randint(0, 65535, (20, 30)).astype(np.uint16)
    seg, _ = _build_subblock(
        frame, pixel_type=1, compression_code=0, hilo=False,
        file_position=0, logical_shape=(20, 30))
    dest = _ShortWriter(limit)
    seg.write_to(dest)
    assert dest.buffer.getvalue() == seg.tobytes()
    if limit < seg.size:
        assert dest.calls > 3


def test_sink_failure_aborts_the_writer(tmp_path):
    class Failing:
        def write(self, data):
            raise OSError("disk full")

        def flush(self):
            pass

        def close(self):
            pass

    path = tmp_path / "sink.czi"
    w = CziWriter(path, compression="zstdhdr")
    w.write(np.zeros((8, 8), dtype=np.uint16))
    w._file = Failing()
    with pytest.raises(OSError, match="disk full"):
        w.write(np.ones((8, 8), dtype=np.uint16))
    assert w._file is None
    with pytest.raises(CziWriterError, match="closed"):
        w.write(np.ones((8, 8), dtype=np.uint16))


@pytest.mark.parametrize("compression", ["none", "zstd", "zstdhdr"])
@pytest.mark.parametrize("verify", [False, True])
def test_written_files_reopen_independently(tmp_path, compression, verify):
    frames = _frames(np.uint16, n=4, shape=(33, 47), seed=5)
    path = tmp_path / f"reopen_{compression}_{verify}.czi"
    with CziWriter(path, compression=compression, verify=verify) as w:
        for f in frames:
            w.write(f)
    with oc.get_codec("czi").open(str(path)) as r:
        assert len(r.entries) == 4
        np.testing.assert_array_equal(np.squeeze(r.read()), np.stack(frames))
        positions = [e.file_position for e in r.entries]
        assert positions == sorted(positions)
        assert all(pos % 32 == 0 for pos in positions)


@pytest.mark.parametrize("compression", ["none", "zstdhdr"])
def test_borrowed_buffer_is_written_before_the_producer_mutates_it(tmp_path, compression):
    """Without verification the array's own buffer is the payload source;
    it must already be on disk when write() returns."""
    frame = np.zeros((16, 16), dtype=np.uint16)
    expected = []
    path = tmp_path / f"borrowed_{compression}.czi"
    with CziWriter(path, compression=compression) as w:
        for i in range(5):
            frame[:] = 1000 * (i + 1)
            expected.append(frame.copy())
            w.write(frame)
            frame.fill(65535)  # mutate immediately after write returns
    with oc.get_codec("czi").open(str(path)) as r:
        np.testing.assert_array_equal(np.squeeze(r.read()), np.stack(expected))


def test_non_contiguous_input_is_written_correctly(tmp_path):
    base = np.random.RandomState(8).randint(0, 65535, (40, 60)).astype(np.uint16)
    view = base[::2, ::3]  # strided view: needs a contiguous copy internally
    assert not view.flags.c_contiguous
    path = tmp_path / "strided.czi"
    with CziWriter(path, compression="zstdhdr") as w:
        w.write(view)
    with oc.get_codec("czi").open(str(path)) as r:
        np.testing.assert_array_equal(np.squeeze(r.read()), view)


def test_pending_verified_segment_uses_part_size_for_offsets(tmp_path):
    """The next segment's file position is computed from the pending
    descriptor's exact size, not from a materialized string."""
    frames = _frames(np.uint8, n=3, shape=(9, 13), seed=11)
    path = tmp_path / "offsets.czi"
    with CziWriter(path, compression="zstdhdr", verify=True) as w:
        for f in frames:
            w.write(f)
    plain = tmp_path / "offsets_plain.czi"
    with CziWriter(plain, compression="zstdhdr", verify=False) as w:
        for f in frames:
            w.write(f)
    assert path.read_bytes() == plain.read_bytes()
