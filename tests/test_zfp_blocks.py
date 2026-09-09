"""Fixed-rate zfp has addressable blocks, and we decoded whole streams.

In fixed-rate mode every 4x4x4 block occupies the same number of bits,
so block N begins at a computed bit offset and reaching it is a seek.
That is the whole reason fixed-rate mode exists, and it gives both
capabilities at once: one block without the others, and the block grid
split across threads.

Neither needs OpenMP. An earlier reading of the zfp documentation
concluded that intra-stream parallelism required ZFP_EXEC_OMP compiled
into the vendored build on three platforms; the block API plus a thread
per block range does it with the library as it already ships.

The variable-rate modes -- precision, accuracy, reversible -- pack
blocks at whatever size they need, so there is no offset to compute.
Those must be refused by name rather than silently decoding something
else, and must still decode correctly through the ordinary path.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc

from _perf import assert_faster, needs_cores

pytestmark = pytest.mark.skipif(
    not oc.has_codec("zfp"), reason="libzfp not built here")


@pytest.fixture(scope="module")
def codec():
    return oc.get_codec("zfp")


def _volume(shape, seed=0, dtype="f4"):
    return np.random.default_rng(seed).normal(0, 1, shape).astype(dtype)


# Shapes that are and are not multiples of 4 on each axis: the encoder
# pads edge blocks, and that padding decodes to real values that are
# not part of the array.
SHAPES = [(64, 64, 64), (32, 32, 32), (30, 17, 45), (4, 4, 4), (100, 3, 7),
          (5, 5, 5), (8, 12, 4)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_threaded_decode_is_bit_identical(codec, shape, dtype):
    a = _volume(shape, dtype=dtype)
    blob = codec.encode(a, mode="rate", rate=16)
    serial = codec.decode(blob, numthreads=1)
    for nt in (None, 2, 3, 8):
        got = codec.decode(blob, numthreads=nt)
        assert got.shape == serial.shape, (shape, dtype, nt)
        assert np.array_equal(got, serial), (shape, dtype, nt)


@pytest.mark.parametrize("shape", [(32, 32, 32), (30, 17, 45)])
def test_every_block_matches_the_full_decode(codec, shape):
    a = _volume(shape, seed=2)
    blob = codec.encode(a, mode="rate", rate=16)
    full = codec.decode(blob, numthreads=1)
    grid = codec.block_grid(blob)
    nbz, nby, nbx = grid["blocks"]
    # Pad the reference the way the encoder does, so edge blocks have
    # something to be compared against.
    padded = np.zeros((nbz * 4, nby * 4, nbx * 4), dtype=full.dtype)
    padded[:full.shape[0], :full.shape[1], :full.shape[2]] = full
    for kb in range(nbz):
        for jb in range(nby):
            for ib in range(nbx):
                idx = (kb * nby + jb) * nbx + ib
                blk = codec.decode_block(blob, idx)
                ref = padded[kb * 4:kb * 4 + 4,
                             jb * 4:jb * 4 + 4,
                             ib * 4:ib * 4 + 4]
                # Interior blocks must match exactly. Edge blocks carry
                # the encoder's padding, which our zeros do not model,
                # so only their in-range part is compared.
                zs = min(4, full.shape[0] - kb * 4)
                ys = min(4, full.shape[1] - jb * 4)
                xs = min(4, full.shape[2] - ib * 4)
                assert np.array_equal(blk[:zs, :ys, :xs],
                                      ref[:zs, :ys, :xs]), idx


def test_block_grid_reports_the_geometry(codec):
    a = _volume((30, 17, 45))
    blob = codec.encode(a, mode="rate", rate=16)
    g = codec.block_grid(blob)
    assert g["ndim"] == 3
    assert g["shape"] == (30, 17, 45)
    assert g["blocks"] == (8, 5, 12)          # ceil(dim / 4) per axis
    assert g["n_blocks"] == 8 * 5 * 12
    assert g["fixed_rate"] is True
    assert g["rate"] == 16.0


def test_negative_and_out_of_range_blocks(codec):
    a = _volume((16, 16, 16))
    blob = codec.encode(a, mode="rate", rate=16)
    n = codec.block_grid(blob)["n_blocks"]
    assert np.array_equal(codec.decode_block(blob, -1),
                          codec.decode_block(blob, n - 1))
    for bad in (n, n + 1, -n - 1):
        with pytest.raises(IndexError):
            codec.decode_block(blob, bad)


@pytest.mark.parametrize("mode,kw", [
    ("accuracy", {"accuracy": 1e-3}),
    ("precision", {"precision": 16}),
    ("reversible", {}),
])
def test_variable_rate_modes_are_refused_by_name(codec, mode, kw):
    """They have no computable block position, so decode_block must say
    so instead of decoding whatever is at a guessed offset."""
    a = _volume((16, 16, 16))
    blob = codec.encode(a, mode=mode, **kw)
    assert codec.block_grid(blob)["fixed_rate"] is False
    with pytest.raises(Exception, match="fixed-rate"):
        codec.decode_block(blob, 0)


@pytest.mark.parametrize("mode,kw", [
    ("accuracy", {"accuracy": 1e-3}),
    ("precision", {"precision": 16}),
    ("reversible", {}),
])
def test_variable_rate_modes_still_decode(codec, mode, kw):
    """The block path returning None must fall back, not fail."""
    a = _volume((16, 16, 16))
    blob = codec.encode(a, mode=mode, **kw)
    got = codec.decode(blob)
    assert got.shape == a.shape
    if mode == "reversible":
        assert np.array_equal(got, a)
    else:
        assert np.allclose(got, a, atol=1e-2)


def test_a_2d_stream_falls_back(codec):
    """Only 3-D is bound; 2-D must decode through the ordinary path."""
    a = _volume((32, 48))
    blob = codec.encode(a, mode="rate", rate=16)
    assert np.array_equal(codec.decode(blob), codec.decode(blob, numthreads=1))


@pytest.mark.perf
@needs_cores
def test_threading_helps_on_a_large_volume(codec):
    import time
    a = _volume((192, 192, 192), seed=7)
    blob = codec.encode(a, mode="rate", rate=8)

    def timed(nt):
        codec.decode(blob, numthreads=nt)
        best = 1e9
        for _ in range(3):
            t = time.perf_counter()
            codec.decode(blob, numthreads=nt)
            best = min(best, time.perf_counter() - t)
        return best

    assert_faster(timed(1), timed(None), "zfp block decode across threads")
