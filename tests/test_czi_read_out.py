"""Caller-owned stack destinations for CziReader.read (priority 8).

``read(out=...)`` fills an array the caller owns, a ``numpy.memmap`` included,
through the priority 1 destination path, so no heap copy of the stack is
made. What has to hold: the same object comes back filled, both accepted
shapes work, both decode paths (pool and bounded) fill it, wrong storage is
refused with a clear message, and ``as_rgb`` is refused rather than silently
leaving the stored order different from the returned view.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc

pytestmark = pytest.mark.skipif(
    not oc.has_codec("czi"),
    reason="czi codec requires native zstd + bytetools extensions",
)

from _czi_fixture import czi_bytes  # noqa: E402


def _stack(n=6, shape=(20, 24), seed=0):
    rs = np.random.RandomState(seed)
    return np.stack([rs.randint(0, 65535, shape).astype(np.uint16) for _ in range(n)])


@pytest.fixture(scope="module")
def frames():
    return _stack()


@pytest.fixture(scope="module")
def data(frames):
    return czi_bytes(frames, compression=6, hilo=True)


def test_full_shape_destination_is_filled_and_returned(data, frames):
    with oc.get_codec("czi").open(data) as r:
        tile_shape = r.entries[0].stored_shape
        out = np.zeros((len(r.entries), *tile_shape), dtype=np.uint16)
        got = r.read(out=out, squeeze=False)
        assert got is out
        np.testing.assert_array_equal(np.squeeze(out), frames)


def test_squeezed_shape_destination_is_accepted(data, frames):
    with oc.get_codec("czi").open(data) as r:
        out = np.zeros(frames.shape, dtype=np.uint16)
        got = r.read(out=out)
        assert got.base is out or got is out
        np.testing.assert_array_equal(out, frames)


def test_memmap_destination(tmp_path, data, frames):
    path = tmp_path / "stack.dat"
    mm = np.memmap(path, dtype=np.uint16, mode="w+", shape=frames.shape)
    with oc.get_codec("czi").open(data) as r:
        got = r.read(out=mm)
        assert isinstance(got, np.memmap)
    mm.flush()
    del mm
    back = np.memmap(path, dtype=np.uint16, mode="r", shape=frames.shape)
    np.testing.assert_array_equal(np.asarray(back), frames)


@pytest.mark.parametrize("kw", [dict(n_workers=1), dict(n_workers=3),
                                dict(max_pending_bytes=1 << 12)])
def test_every_decode_path_fills_the_destination(data, frames, kw):
    with oc.get_codec("czi").open(data) as r:
        out = np.zeros(frames.shape, dtype=np.uint16)
        r.read(out=out, **kw)
        np.testing.assert_array_equal(out, frames)


def test_bad_destinations_are_refused(data, frames):
    with oc.get_codec("czi").open(data) as r:
        with pytest.raises(TypeError, match="ndarray"):
            r.read(out=bytearray(frames.nbytes))
        frozen = np.zeros(frames.shape, dtype=np.uint16)
        frozen.flags.writeable = False
        with pytest.raises(ValueError, match="writable"):
            r.read(out=frozen)
        with pytest.raises(ValueError, match="C-contiguous"):
            r.read(out=np.zeros((6, 20, 48), dtype=np.uint16)[:, :, ::2])
        with pytest.raises(ValueError, match="dtype"):
            r.read(out=np.zeros(frames.shape, dtype=np.uint8))
        with pytest.raises(ValueError, match="shape"):
            r.read(out=np.zeros((6, 24, 20), dtype=np.uint16))
        with pytest.raises(ValueError, match="shape"):
            r.read(out=np.zeros((5, 20, 24), dtype=np.uint16))


def test_as_rgb_is_refused_with_a_destination():
    frame = np.random.RandomState(1).randint(0, 256, (8, 8, 3)).astype(np.uint8)
    data = czi_bytes(frame, compression=6, hilo=True)
    with oc.get_codec("czi").open(data) as r:
        out = np.zeros((1, 8, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="as_rgb"):
            r.read(out=out, as_rgb=True)
        r.read(out=out, squeeze=False)
        np.testing.assert_array_equal(out[0], frame)
