"""Native byte shuffle and writer-owned scratch (priority 2).

The writer used to shuffle byte planes with a NumPy transpose plus tobytes().
That is now the native primitive writing into one reusable buffer. These tests
compare against the transpose spelled out here, so the reference is
independent of the shipped code rather than the same call in disguise, and
they pin the scratch invariants: a shorter frame after a longer one must not
inherit the tail, a pending verification must not see a recycled buffer, and
separate writers must not share one.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

import opencodecs as oc

pytestmark = pytest.mark.skipif(
    not oc.has_codec("czi"),
    reason="czi codec requires native zstd + bytetools extensions",
)

from _czi_fixture import czi_bytes  # noqa: E402,F401  (registers fixture path)


def reference_shuffle(natural: bytes, itemsize: int) -> bytes:
    """The NumPy transpose the native primitive replaced."""
    if itemsize == 1:
        return natural
    n = len(natural) // itemsize
    arr = np.frombuffer(natural, dtype=np.uint8).reshape(n, itemsize)
    return arr.T.tobytes()


@pytest.mark.parametrize("itemsize", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("n_elements", [0, 1, 2, 7, 1000])
def test_native_shuffle_matches_transpose(itemsize, n_elements):
    from opencodecs._czi_writer import _byteshuffle_encode

    payload = bytes(np.random.RandomState(itemsize * 100 + n_elements)
                    .randint(0, 256, itemsize * n_elements, dtype=np.uint8))
    got = _byteshuffle_encode(payload, itemsize)
    assert bytes(got) == reference_shuffle(payload, itemsize)


@pytest.mark.parametrize("itemsize", [2, 4, 8])
def test_shuffle_into_scratch_matches_allocating(itemsize):
    from opencodecs._czi_writer import _byteshuffle_encode
    from opencodecs.core.scratch import ScratchBuffer

    scratch = ScratchBuffer()
    payload = bytes(np.random.RandomState(7).randint(
        0, 256, itemsize * 500, dtype=np.uint8))
    into = _byteshuffle_encode(payload, itemsize,
                               out=scratch.bytes(len(payload)))
    assert bytes(into) == reference_shuffle(payload, itemsize)


def test_smaller_frame_after_larger_does_not_inherit_tail():
    """The scratch keeps its capacity; the view must stop at this frame."""
    from opencodecs._czi_writer import _byteshuffle_encode
    from opencodecs.core.scratch import ScratchBuffer

    scratch = ScratchBuffer()
    big = bytes(np.random.RandomState(1).randint(0, 256, 4096, dtype=np.uint8))
    small = bytes(np.random.RandomState(2).randint(0, 256, 64, dtype=np.uint8))

    _byteshuffle_encode(big, 2, out=scratch.bytes(len(big)))
    got = _byteshuffle_encode(small, 2, out=scratch.bytes(len(small)))
    assert len(got) == len(small)
    assert bytes(got) == reference_shuffle(small, 2)


def test_non_multiple_length_is_rejected():
    from opencodecs._czi_writer import _byteshuffle_encode

    with pytest.raises(ValueError):
        _byteshuffle_encode(b"abc", 2)


def test_zstdhdr_payload_identical_with_and_without_scratch():
    from opencodecs._czi_writer import _zstdhdr_encode
    from opencodecs.core.scratch import ScratchBuffer

    pixels = np.random.RandomState(3).randint(
        0, 65535, 4096, dtype=np.uint16).tobytes()
    plain = _zstdhdr_encode(pixels, 2, True)
    reused = _zstdhdr_encode(pixels, 2, True, scratch=ScratchBuffer())
    assert plain == reused


# ---------------------------------------------------------------------------
# Whole-file behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("verify", [False, True])
def test_written_file_round_trips(tmp_path, dtype, verify):
    from opencodecs._czi_writer import CziWriter

    rs = np.random.RandomState(4)
    if np.issubdtype(dtype, np.floating):
        frames = [rs.rand(24, 32).astype(dtype) for _ in range(3)]
    else:
        info = np.iinfo(dtype)
        frames = [rs.randint(info.min, info.max, (24, 32)).astype(dtype)
                  for _ in range(3)]

    path = tmp_path / f"shuffle_{np.dtype(dtype).name}_{verify}.czi"
    with CziWriter(path, compression="zstdhdr", hilo=True,
                   verify=verify) as w:
        for frame in frames:
            w.write(frame)

    with oc.get_codec("czi").open(str(path)) as r:
        out = np.squeeze(r.read())
    np.testing.assert_array_equal(out, np.stack(frames))


def test_float_payload_bits_survive(tmp_path):
    """Shuffling is bytewise, so NaN payloads and signed zero must persist."""
    from opencodecs._czi_writer import CziWriter

    frame = np.array([
        [np.nan, -0.0, 0.0, np.inf],
        [-np.inf, 1.5, -1.5, np.float32(5e-45)],
    ], dtype=np.float32)
    # A NaN with a non-default payload, which a float comparison would miss.
    frame[0, 0] = np.frombuffer(np.uint32(0x7FC0DEAD).tobytes(),
                                dtype=np.float32)[0]

    path = tmp_path / "float_bits.czi"
    with CziWriter(path, compression="zstdhdr", hilo=True) as w:
        w.write(frame)

    with oc.get_codec("czi").open(str(path)) as r:
        out = np.squeeze(r.read()).astype(np.float32)
    assert out.tobytes() == frame.tobytes()


def test_reused_producer_buffer_is_snapshotted(tmp_path):
    """Writing from one mutated buffer must store each frame as submitted."""
    from opencodecs._czi_writer import CziWriter

    scratch_frame = np.zeros((16, 16), dtype=np.uint16)
    expected = []
    path = tmp_path / "reused_buffer.czi"
    with CziWriter(path, compression="zstdhdr", hilo=True) as w:
        for i in range(4):
            scratch_frame[:] = np.uint16(1000 * (i + 1))
            expected.append(scratch_frame.copy())
            w.write(scratch_frame)

    with oc.get_codec("czi").open(str(path)) as r:
        out = np.squeeze(r.read())
    np.testing.assert_array_equal(out, np.stack(expected))


def test_concurrent_writers_do_not_share_scratch(tmp_path):
    from opencodecs._czi_writer import CziWriter

    def write_one(idx: int, results: dict) -> None:
        rs = np.random.RandomState(500 + idx)
        shape = (32, 32) if idx % 2 == 0 else (48, 24)
        frames = [rs.randint(0, 65535, shape).astype(np.uint16)
                  for _ in range(4)]
        path = tmp_path / f"concurrent_{idx}.czi"
        with CziWriter(path, compression="zstdhdr", hilo=True) as w:
            for frame in frames:
                w.write(frame)
        results[idx] = (path, np.stack(frames))

    results: dict[int, tuple] = {}
    threads = [threading.Thread(target=write_one, args=(i, results))
               for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 6
    for idx, (path, frames) in results.items():
        with oc.get_codec("czi").open(str(path)) as r:
            np.testing.assert_array_equal(np.squeeze(r.read()), frames)
