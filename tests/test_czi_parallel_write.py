"""Bounded parallel CZI sub-block compression (priority 9).

``write_many`` compresses frames on workers and emits them in submission
order with positions assigned at emission. What has to hold: the file is
byte-identical to sequential ``write`` calls; variable-size frames keep a
valid ordered index; a producer that refills one buffer is snapshotted
before the source advances; ``copy_frames=False`` works for stable input;
a codec failure or a corrupted payload propagates and aborts the writer
with nothing unverified emitted; verification happens in the task, not in a
second pool; and the pyramid writer's level plan still runs in order.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc

pytestmark = pytest.mark.skipif(
    not oc.has_codec("czi"),
    reason="czi codec requires native zstd + bytetools extensions",
)

from opencodecs._czi_writer import (  # noqa: E402
    CziPyramidWriter, CziWriter, CziWriterError, _SubBlockSegment,
)
import opencodecs._czi_writer as implementation  # noqa: E402
from opencodecs.core.verification import LosslessVerificationError  # noqa: E402


def _frames(n, shape=(48, 64), dtype=np.uint16, seed=0):
    rs = np.random.RandomState(seed)
    return [rs.randint(0, 65535, shape).astype(dtype) for _ in range(n)]


@pytest.mark.parametrize("compression", ["none", "zstd", "zstdhdr"])
@pytest.mark.parametrize("verify", [False, True])
def test_parallel_file_is_byte_identical_to_sequential(tmp_path, compression, verify):
    frames = _frames(9)
    seq = tmp_path / f"seq_{compression}_{verify}.czi"
    par = tmp_path / f"par_{compression}_{verify}.czi"
    with CziWriter(seq, compression=compression, verify=verify) as w:
        for f in frames:
            w.write(f)
    with CziWriter(par, compression=compression, verify=verify) as w:
        w.write_many(frames, workers=4, max_pending_bytes=1 << 18)
    assert par.read_bytes() == seq.read_bytes()
    with oc.get_codec("czi").open(str(par)) as r:
        np.testing.assert_array_equal(np.squeeze(r.read()), np.stack(frames))


def test_variable_size_frames_keep_an_ordered_index(tmp_path):
    rs = np.random.RandomState(3)
    frames = [rs.randint(0, 65535, (int(rs.randint(1, 70)), int(rs.randint(1, 70)))).astype(np.uint16)
              for _ in range(12)]
    path = tmp_path / "variable.czi"
    with CziWriter(path, compression="zstdhdr") as w:
        w.write_many(frames, workers=4)
    with oc.get_codec("czi").open(str(path)) as r:
        assert len(r.entries) == 12
        positions = [e.file_position for e in r.entries]
        assert positions == sorted(positions) and all(p % 32 == 0 for p in positions)
        for i, f in enumerate(frames):
            np.testing.assert_array_equal(np.squeeze(r[i]), f)


def test_reused_producer_buffer_is_snapshotted_before_advancing(tmp_path):
    buffer = np.zeros((32, 32), dtype=np.uint16)
    expected = []

    def producer():
        for i in range(6):
            buffer[:] = 500 * (i + 1)
            expected.append(buffer.copy())
            yield buffer

    path = tmp_path / "reused.czi"
    with CziWriter(path, compression="zstdhdr") as w:
        w.write_many(producer(), workers=3)
    with oc.get_codec("czi").open(str(path)) as r:
        np.testing.assert_array_equal(np.squeeze(r.read()), np.stack(expected))


def test_copy_frames_false_with_stable_input(tmp_path):
    frames = _frames(5)
    path = tmp_path / "nocopy.czi"
    with CziWriter(path, compression="zstdhdr", verify=True) as w:
        w.write_many(frames, workers=2, copy_frames=False)
    with oc.get_codec("czi").open(str(path)) as r:
        np.testing.assert_array_equal(np.squeeze(r.read()), np.stack(frames))


def test_codec_failure_propagates_and_aborts(tmp_path, monkeypatch):
    build = implementation._build_subblock
    calls = []

    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise OSError("encoder stopped")
        return build(*args, **kwargs)

    monkeypatch.setattr(implementation, "_build_subblock", failing)
    w = CziWriter(tmp_path / "failed.czi", compression="zstdhdr")
    with pytest.raises(OSError, match="encoder stopped"):
        w.write_many(_frames(8), workers=2)
    assert w._file is None
    with pytest.raises(CziWriterError, match="closed"):
        w.write(_frames(1)[0])


def test_corrupted_payload_is_caught_in_task_and_nothing_after_is_emitted(tmp_path, monkeypatch):
    build = implementation._build_subblock
    built = []
    emitted = []

    def corrupt_fourth(*args, **kwargs):
        segment, entry = build(*args, **kwargs)
        built.append(1)
        if len(built) == 4:
            payload = bytearray(segment.payload)
            payload[3] ^= 1
            segment = segment.with_payload(bytes(payload))
        return segment, entry

    emit = CziWriter._emit_subblock

    def record(self, segment, entry):
        emitted.append(entry["file_position"])
        emit(self, segment, entry)

    monkeypatch.setattr(implementation, "_build_subblock", corrupt_fourth)
    monkeypatch.setattr(CziWriter, "_emit_subblock", record)
    w = CziWriter(tmp_path / "corrupt.czi", compression="zstdhdr", verify=True)
    with pytest.raises(LosslessVerificationError):
        # batch_bytes=1 pins one frame per task; with grouping, the whole
        # task containing the corrupt frame would be withheld instead.
        w.write_many(_frames(8), workers=1, batch_bytes=1)
    assert len(emitted) == 3          # frames before the corrupted one, in order
    assert emitted == sorted(emitted)
    assert w._file is None


def test_verification_runs_inside_the_worker_not_a_second_pool(tmp_path, monkeypatch):
    import threading

    verify = CziWriter._verify_subblock
    threads = set()

    def observed(original, segment, entry):
        threads.add(threading.current_thread().name)
        verify(original, segment, entry)

    monkeypatch.setattr(CziWriter, "_verify_subblock", staticmethod(observed))
    with CziWriter(tmp_path / "inworker.czi", compression="zstdhdr", verify=True) as w:
        w.write_many(_frames(6), workers=3)
    assert threads and all("czi-write" in name for name in threads)


@pytest.mark.parametrize("batch_bytes", [None, 1, 1 << 30])
def test_tiny_and_mixed_frames(tmp_path, batch_bytes):
    frames = [np.array([[7]], dtype=np.uint16)] + _frames(3, shape=(1, 5)) + _frames(2, shape=(200, 3))
    path = tmp_path / "tiny.czi"
    with CziWriter(path, compression="zstdhdr", verify=True) as w:
        w.write_many(frames, workers=4, batch_bytes=batch_bytes)
    with oc.get_codec("czi").open(str(path)) as r:
        for i, f in enumerate(frames):
            np.testing.assert_array_equal(np.squeeze(r[i]), np.squeeze(f))


def test_pyramid_writer_plans_levels_in_order(tmp_path):
    base = np.random.RandomState(5).randint(0, 65535, (64, 64)).astype(np.uint16)
    levels = [base, base[::2, ::2], base[::4, ::4]]
    seq, par = tmp_path / "pseq.czi", tmp_path / "ppar.czi"
    with CziPyramidWriter(seq, compression="zstdhdr") as w:
        w.write_pyramid(levels)
    with CziPyramidWriter(par, compression="zstdhdr") as w:
        w.write_many(levels, workers=3)
    assert par.read_bytes() == seq.read_bytes()
    with oc.get_codec("czi").open(str(par)) as r:
        assert [e.pyramid_type for e in r.entries] == [0, 2, 2]
        assert all(e.shape[:2] == (64, 64) for e in r.entries)


def test_write_many_after_write_drains_pending_verification(tmp_path):
    frames = _frames(6)
    path = tmp_path / "mixed.czi"
    with CziWriter(path, compression="zstdhdr", verify=True) as w:
        w.write(frames[0])
        w.write_many(frames[1:4], workers=2)
        w.write(frames[4])
        w.write_many(frames[5:], workers=2)
    with oc.get_codec("czi").open(str(path)) as r:
        np.testing.assert_array_equal(np.squeeze(r.read()), np.stack(frames))


def test_with_position_patches_only_the_directory_entry():
    frame = np.arange(4 * 6, dtype=np.uint16).reshape(4, 6)
    seg, _ = implementation._build_subblock(
        frame, pixel_type=1, compression_code=0, hilo=False,
        file_position=0, logical_shape=(4, 6))
    moved = seg.with_position(4096)
    assert isinstance(moved, _SubBlockSegment)
    a, b = seg.tobytes(), moved.tobytes()
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    assert diff and min(diff) >= _SubBlockSegment._POSITION_OFFSET
    assert max(diff) < _SubBlockSegment._POSITION_OFFSET + 8
    import struct
    assert struct.unpack_from("<q", b, _SubBlockSegment._POSITION_OFFSET)[0] == 4096
