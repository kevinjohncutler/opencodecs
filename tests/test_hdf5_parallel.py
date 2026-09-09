"""HDF5 read() decompresses chunks in parallel.

The parallel path was already written and nothing called it, so every
read went through h5py one chunk at a time under libhdf5's process-wide
lock. That is the failure mode a capability flag is supposed to catch
and did: parallel_decode was honestly False for a reader that had the
code sitting in it.

The fast path declines on its own for datasets it cannot serve, so the
tests that matter are the ones where it must NOT be taken and must
still return the right bytes.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import opencodecs as oc

h5py = pytest.importorskip("h5py")


def _write(path, arr, *, chunks=None, compression=None, **kw):
    with h5py.File(path, "w") as f:
        opts = dict(kw)
        if chunks is not None:
            opts["chunks"] = chunks
        if compression is not None:
            opts["compression"] = compression
        f.create_dataset("img", data=arr, **opts)
    return path


@pytest.fixture(scope="module")
def arr():
    return np.random.default_rng(0).integers(
        0, 4000, (32, 256, 256)).astype("u2")


@pytest.mark.parametrize("kwargs", [
    dict(chunks=(1, 256, 256), compression="gzip"),
    dict(chunks=(1, 256, 256), compression="lzf"),
    dict(chunks=(4, 128, 128), compression="gzip"),
    dict(chunks=(1, 256, 256)),                     # chunked, uncompressed
    dict(),                                         # contiguous
    dict(chunks=(5, 100, 100), compression="gzip"),  # chunks do not divide
])
@pytest.mark.parametrize("numthreads", [None, 1, 8])
def test_read_matches_h5py(tmp_path, arr, kwargs, numthreads):
    p = _write(tmp_path / "d.h5", arr, **kwargs)
    with oc.get_codec("hdf5").open(str(p)) as r:
        got = r.read(numthreads=numthreads)
    assert got.dtype == arr.dtype
    assert np.array_equal(got, arr)


def test_ragged_edge_chunks_are_exact(tmp_path):
    """Chunk dims that do not divide the shape leave partial edge
    chunks, which the parallel path has an explicit fallback for."""
    a = np.random.default_rng(1).integers(0, 255, (13, 77, 51)).astype("u1")
    p = _write(tmp_path / "ragged.h5", a, chunks=(3, 16, 16),
               compression="gzip")
    with oc.get_codec("hdf5").open(str(p)) as r:
        assert np.array_equal(r.read(numthreads=8), a)


@pytest.mark.parametrize("dtype", ["u1", "i2", "u4", "f4", "f8"])
def test_dtypes_round_trip(tmp_path, dtype):
    a = (np.arange(8 * 64 * 64) % 200).astype(dtype).reshape(8, 64, 64)
    p = _write(tmp_path / f"d_{dtype}.h5", a, chunks=(1, 64, 64),
               compression="gzip")
    with oc.get_codec("hdf5").open(str(p)) as r:
        got = r.read(numthreads=8)
    assert got.dtype == np.dtype(dtype)
    assert np.array_equal(got, a)


def test_a_single_chunk_stays_serial(tmp_path, arr):
    """Below the threshold the pool costs more than it saves, so it
    must not be built. Correctness is the only thing asserted here;
    the timing claim lives in the slow test."""
    p = _write(tmp_path / "one.h5", arr, chunks=arr.shape, compression="gzip")
    with oc.get_codec("hdf5").open(str(p)) as r:
        assert np.array_equal(r.read(), arr)


@pytest.mark.slow
def test_parallel_read_is_faster(tmp_path):
    big = np.random.default_rng(2).integers(
        0, 4000, (64, 512, 512)).astype("u2")
    p = _write(tmp_path / "big.h5", big, chunks=(1, 512, 512),
               compression="gzip")
    with oc.get_codec("hdf5").open(str(p)) as r:
        r.read(numthreads=1)

        def timed(nt):
            best = 1e9
            for _ in range(3):
                t = time.perf_counter()
                r.read(numthreads=nt)
                best = min(best, time.perf_counter() - t)
            return best

        serial, auto = timed(1), timed(None)
    assert serial / auto > 1.5, f"{serial / auto:.2f}x with the pool"
