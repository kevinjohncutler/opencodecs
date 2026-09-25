"""Zarr regional reads: zstd chunks decoded straight into the output.

Arrays are written by zarr-python (the reference writer) and every region
read, on the direct zstd path and the decoded path, must equal NumPy
slicing of the source: N-d arrays, partial and edge chunks, missing chunks
(fill value), v2 and v3, one worker and several. Layouts the direct path
must not take (big-endian, other codecs) are checked to still read right.
"""
from __future__ import annotations

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
numcodecs = pytest.importorskip("numcodecs")

from opencodecs._omezarr import OmeZarrArray  # noqa: E402


def _write(tmp_path, data, chunks, *, fmt=2, compressor="zstd", fill=0, skip=None):
    path = tmp_path / f"a{fmt}_{compressor}.zarr"
    if fmt == 2:
        comp = {"zstd": numcodecs.Zstd(level=3), "blosc": numcodecs.Blosc("zstd", 3, 1),
                "zlib": numcodecs.Zlib(level=3)}[compressor]
        z = zarr.create_array(str(path), shape=data.shape, chunks=chunks, dtype=data.dtype,
                              compressors=comp, zarr_format=2, fill_value=fill)
    else:
        z = zarr.create_array(str(path), shape=data.shape, chunks=chunks, dtype=data.dtype,
                              compressors=zarr.codecs.ZstdCodec(level=3), zarr_format=3,
                              fill_value=fill)
    z[...] = data
    if skip is not None:
        # Delete one chunk so the reader must fill it.
        key = ".".join(map(str, skip)) if fmt == 2 else "c/" + "/".join(map(str, skip))
        (path / key).unlink()
    return OmeZarrArray(path)


def _regions(shape, rs, n=12):
    for _ in range(n):
        region = []
        for size in shape:
            a = int(rs.randint(0, size))
            b = int(rs.randint(a + 1, size + 1))
            region.append(slice(a, b))
        yield tuple(region)


@pytest.mark.parametrize("fmt", [2, 3])
@pytest.mark.parametrize("shape,chunks", [
    ((300, 410), (64, 96)),              # 2-D, edge chunks partial
    ((3, 5, 70, 90), (1, 2, 32, 40)),    # 4-D, leading axes chunked unevenly
    ((2, 130, 3), (1, 64, 2)),           # last axis tiny
])
@pytest.mark.parametrize("workers", [1, 4])
def test_zstd_direct_path_matches_numpy(tmp_path, fmt, shape, chunks, workers):
    rs = np.random.RandomState(sum(shape))
    data = rs.randint(0, 65535, shape).astype(np.uint16)
    arr = _write(tmp_path, data, chunks, fmt=fmt)
    assert arr._zstd_window_ok()
    np.testing.assert_array_equal(arr.read_region(tuple(slice(None) for _ in shape),
                                                  num_workers=workers), data)
    for region in _regions(shape, rs):
        np.testing.assert_array_equal(arr.read_region(region, num_workers=workers), data[region])


@pytest.mark.parametrize("workers", [1, 4])
def test_missing_chunk_reads_as_fill_value(tmp_path, workers):
    data = np.arange(200 * 300, dtype=np.uint16).reshape(200, 300)
    arr = _write(tmp_path, data, (64, 64), fill=7, skip=(1, 2))
    expected = data.copy()
    expected[64:128, 128:192] = 7
    np.testing.assert_array_equal(arr.read_region((slice(None), slice(None)), num_workers=workers),
                                  expected)
    np.testing.assert_array_equal(arr.read_region((slice(50, 150), slice(100, 250)),
                                                  num_workers=workers), expected[50:150, 100:250])


@pytest.mark.parametrize("compressor", ["blosc", "zlib"])
def test_other_codecs_keep_the_decoded_path(tmp_path, compressor):
    rs = np.random.RandomState(1)
    data = rs.randint(0, 65535, (150, 170)).astype(np.uint16)
    arr = _write(tmp_path, data, (64, 64), compressor=compressor)
    assert not arr._zstd_window_ok()
    for workers in (1, 4):
        np.testing.assert_array_equal(arr.read_region((slice(None), slice(None)),
                                                      num_workers=workers), data)


def test_big_endian_keeps_the_decoded_path(tmp_path):
    rs = np.random.RandomState(2)
    data = rs.randint(0, 65535, (100, 90)).astype(">u2")
    arr = _write(tmp_path, data, (32, 32))
    assert not arr._zstd_window_ok()
    np.testing.assert_array_equal(arr.read_region((slice(None), slice(None)), num_workers=4),
                                  data.astype("=u2"))
