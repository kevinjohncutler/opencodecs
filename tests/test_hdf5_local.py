"""Local HDF5 tests — focus on the parallel-decompress fast path.

The interesting contract is that ``HdfReader.read_parallel`` produces
byte-equal output to h5py's standard ``ds[...]`` slicing for compressed
chunked datasets, while doing the decompression off the libhdf5 lock.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from opencodecs._hdf5_codec import HdfReader


@pytest.fixture
def gzip_hdf5(tmp_path):
    """Tiny chunked + gzip-compressed dataset (~16 MB)."""
    arr = np.random.default_rng(0).integers(
        0, 1000, size=(16, 256, 512), dtype=np.uint16,
    )
    p = tmp_path / "test.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset(
            "img", data=arr, chunks=(2, 128, 128),
            compression="gzip", compression_opts=4,
        )
    return p, arr


def test_read_parallel_full_byte_equal(gzip_hdf5):
    path, arr = gzip_hdf5
    r = HdfReader(str(path))
    try:
        out = r.read_parallel(n_workers=4)
    finally:
        r.close()
    np.testing.assert_array_equal(out, arr)


def test_read_parallel_slice_byte_equal(gzip_hdf5):
    path, arr = gzip_hdf5
    sl = np.s_[3:11, 50:200, 100:400]
    r = HdfReader(str(path))
    try:
        out = r.read_parallel(sl, n_workers=4)
    finally:
        r.close()
    np.testing.assert_array_equal(out, arr[sl])


def test_read_parallel_single_worker_falls_back(gzip_hdf5):
    """n_workers=1 should still return correct bytes (serial fallback)."""
    path, arr = gzip_hdf5
    r = HdfReader(str(path))
    try:
        out = r.read_parallel(n_workers=1)
    finally:
        r.close()
    np.testing.assert_array_equal(out, arr)


def test_read_parallel_uncompressed_falls_back(tmp_path):
    """For uncompressed datasets read_parallel should match h5py via the
    serial fallback path (no advantage but no regression)."""
    arr = np.random.default_rng(0).integers(
        0, 256, size=(8, 64, 64), dtype=np.uint8,
    )
    p = tmp_path / "uncompressed.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("img", data=arr, chunks=(2, 32, 32))
    r = HdfReader(str(p))
    try:
        out = r.read_parallel(n_workers=4)
    finally:
        r.close()
    np.testing.assert_array_equal(out, arr)


def test_read_parallel_int_index(gzip_hdf5):
    """A 1D int index along axis 0 should select a single frame correctly."""
    path, arr = gzip_hdf5
    r = HdfReader(str(path))
    try:
        out = r.read_parallel(np.s_[5], n_workers=4)
    finally:
        r.close()
    np.testing.assert_array_equal(out, arr[5:6])
