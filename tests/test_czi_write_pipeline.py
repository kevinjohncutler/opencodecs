"""CziWriter(background_encode=True) compressing large frames on a thread.

With the option on, compressed frames of 1 MiB or more are snapshotted and
compressed while the caller prepares the next frame; their positions are
assigned when written, as write_many does. It is off by default. The file must be byte-identical to compressing on the
calling thread, a caller refilling one buffer must not change what is
written, and a compression error must surface and abort the writer.
"""
from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc

pytestmark = pytest.mark.skipif(not oc.has_codec("czi"), reason="czi codec not built")

import opencodecs._czi_writer as cw  # noqa: E402
from opencodecs._czi_reader import CziReader  # noqa: E402
from opencodecs._czi_writer import CziPyramidWriter, CziWriter, CziWriterError  # noqa: E402


def _frames(sizes, seed=0):
    rs = np.random.RandomState(seed)
    return [rs.randint(0, 4000, s).astype(np.uint16) for s in sizes]


def _write(path, frames, *, pipelined, compression="zstdhdr"):
    with CziWriter(path, compression=compression, background_encode=pipelined) as w:
        for f in frames:
            w.write(f)
    return path.read_bytes()


@pytest.mark.parametrize("compression", ["zstdhdr", "zstd"])
def test_pipelined_file_is_byte_identical(tmp_path, compression):
    # Large and small frames interleaved: small ones stay on the calling
    # thread and must be written after the large frame before them.
    frames = _frames([(1024, 1024), (64, 64), (768, 1024), (1024, 1024), (32, 48)])
    a = _write(tmp_path / "a.czi", frames, pipelined=True, compression=compression)
    b = _write(tmp_path / "b.czi", frames, pipelined=False, compression=compression)
    assert a == b
    with CziReader(tmp_path / "a.czi") as r:
        for i, f in enumerate(frames):
            np.testing.assert_array_equal(np.squeeze(r[i]), f)


def test_caller_may_refill_its_buffer(tmp_path):
    frames = _frames([(1024, 1024)] * 4, seed=1)
    buffer = np.empty((1024, 1024), np.uint16)
    with CziWriter(tmp_path / "c.czi", compression="zstdhdr", background_encode=True) as w:
        for f in frames:
            buffer[...] = f
            w.write(buffer)
            buffer[...] = 0  # overwritten while the frame compresses
    with CziReader(tmp_path / "c.czi") as r:
        for i, f in enumerate(frames):
            np.testing.assert_array_equal(np.squeeze(r[i]), f)


def test_compression_error_surfaces_and_aborts(tmp_path, monkeypatch):
    frames = _frames([(1024, 1024)] * 3, seed=2)
    real = cw._build_subblock
    calls = []

    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("compressor failed")
        return real(*args, **kwargs)

    monkeypatch.setattr(cw, "_build_subblock", failing)
    w = CziWriter(tmp_path / "d.czi", compression="zstdhdr", background_encode=True)
    w.write(frames[0])
    w.write(frames[1])  # fails in the background
    with pytest.raises(RuntimeError, match="compressor failed"):
        w.write(frames[2])
    with pytest.raises(CziWriterError):
        w.write(frames[2])  # the writer is aborted
    assert w._encode_pool is None


def test_default_uncompressed_and_verified_writes_stay_on_the_calling_thread(tmp_path):
    frames = _frames([(1024, 1024)] * 2, seed=3)
    for kwargs in ({"compression": "none", "background_encode": True},
                   {"compression": "zstdhdr", "verify": True, "background_encode": True},
                   {"compression": "zstdhdr"}):  # off by default
        with CziWriter(tmp_path / "e.czi", **kwargs) as w:
            for f in frames:
                w.write(f)
                assert w._pending_encode is None


def test_pyramid_levels_and_write_many_after_pipelined_writes(tmp_path):
    base = _frames([(1024, 2048)], seed=4)[0]
    with CziPyramidWriter(tmp_path / "p.czi", compression="zstdhdr", background_encode=True) as w:
        w.write_pyramid([base, base[::2, ::2], base[::4, ::4]])
    with CziReader(tmp_path / "p.czi") as r:
        np.testing.assert_array_equal(np.squeeze(r[0]), base)
        np.testing.assert_array_equal(np.squeeze(r[1]), base[::2, ::2])
    frames = _frames([(1024, 1024), (1024, 1024), (64, 64)], seed=5)
    with CziWriter(tmp_path / "m.czi", compression="zstdhdr", background_encode=True) as w:
        w.write(frames[0])  # still compressing when write_many starts
        w.write_many(frames[1:])
    with CziReader(tmp_path / "m.czi") as r:
        for i, f in enumerate(frames):
            np.testing.assert_array_equal(np.squeeze(r[i]), f)
