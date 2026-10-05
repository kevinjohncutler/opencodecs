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
if not zarr.__version__.startswith("3"):
    # The fixtures are written with zarr 3's create_array (both formats).
    pytest.skip(f"needs zarr 3, have {zarr.__version__}", allow_module_level=True)

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


# ---------------------------------------------------------------------------
# The batched native path: local zstd chunks, sharded or not
# ---------------------------------------------------------------------------


def _write_sharded(tmp_path, data, chunks, shards, *, fill=0, drop_shard=None, blank=None):
    path = tmp_path / "sharded.zarr"
    z = zarr.create_array(str(path), shape=data.shape, chunks=chunks, shards=shards,
                          dtype=data.dtype, compressors=zarr.codecs.ZstdCodec(level=3),
                          zarr_format=3, fill_value=fill)
    written = data.copy()
    if blank is not None:
        # An all-fill inner chunk is left out of its shard (empty index entry).
        written[blank] = fill
    z[...] = written
    if drop_shard is not None:
        (path / ("c/" + "/".join(map(str, drop_shard)))).unlink()
    return OmeZarrArray(path), written


def _old_path(arr, monkeypatch):
    monkeypatch.setattr(type(arr), "_batch_ok", lambda self: False)


@pytest.mark.parametrize("workers", [None, 1, 3])
def test_sharded_batched_matches_numpy(tmp_path, workers):
    rs = np.random.RandomState(5)
    data = rs.randint(0, 65535, (150, 260)).astype(np.uint16)
    arr, written = _write_sharded(tmp_path, data, (16, 32), (64, 64),
                                  blank=(slice(16, 32), slice(64, 96)))
    assert arr._batch_ok() and not arr._zstd_window_ok()
    np.testing.assert_array_equal(arr.read_region((slice(None), slice(None)),
                                                  num_workers=workers), written)
    for region in _regions(data.shape, rs):
        np.testing.assert_array_equal(arr.read_region(region, num_workers=workers),
                                      written[region])


def test_sharded_missing_shard_reads_as_fill(tmp_path):
    data = np.arange(128 * 128, dtype=np.uint16).reshape(128, 128)
    arr, written = _write_sharded(tmp_path, data, (16, 16), (64, 64), fill=9,
                                  drop_shard=(1, 0))
    expected = written.copy()
    expected[64:128, 0:64] = 9
    for workers in (1, 4):
        np.testing.assert_array_equal(arr.read_region((slice(None), slice(None)),
                                                      num_workers=workers), expected)


def test_leading_singleton_chunks_take_the_vectorized_windows(tmp_path):
    # OME-Zarr's (t, c, z, y, x) with one plane per chunk.
    rs = np.random.RandomState(6)
    data = rs.randint(0, 65535, (2, 3, 4, 50, 70)).astype(np.uint16)
    arr = _write(tmp_path, data, (1, 1, 1, 16, 32), fmt=3)
    assert arr._batch_ok()
    for region in _regions(data.shape, rs, n=8):
        np.testing.assert_array_equal(arr.read_region(region, num_workers=3), data[region])


@pytest.mark.parametrize("sharded", [False, True])
def test_batched_corrupt_chunk_raises_the_old_error(tmp_path, monkeypatch, sharded):
    rs = np.random.RandomState(7)
    data = rs.randint(0, 65535, (128, 128)).astype(np.uint16)
    if sharded:
        arr, _ = _write_sharded(tmp_path, data, (32, 32), (128, 128))
        shard = tmp_path / "sharded.zarr" / "c" / "0" / "0"
        blob = bytearray(shard.read_bytes())
        blob[0:4] = b"\xff" * 4              # the first inner chunk's frame magic
        shard.write_bytes(bytes(blob))
    else:
        arr = _write(tmp_path, data, (32, 32), fmt=2)
        chunk = arr._root / "1.2"
        chunk.write_bytes(chunk.read_bytes()[:30])
    assert arr._batch_ok()
    with pytest.raises(Exception) as got:
        arr.read_region((slice(None), slice(None)), num_workers=4)
    fresh = OmeZarrArray(arr._root)
    _old_path(fresh, monkeypatch)
    with pytest.raises(Exception) as want:
        fresh.read_region((slice(None), slice(None)), num_workers=1)
    assert type(got.value) is type(want.value)
    assert str(got.value) == str(want.value)


def test_batched_matches_the_per_chunk_path(tmp_path, monkeypatch):
    rs = np.random.RandomState(8)
    data = rs.randint(0, 65535, (3, 70, 90)).astype(np.uint16)
    arr = _write(tmp_path, data, (2, 32, 40), fmt=2, fill=3, skip=(1, 1, 0))
    regions = list(_regions(data.shape, rs, n=10))
    got = [arr.read_region(r, num_workers=2) for r in regions]
    _old_path(arr, monkeypatch)
    for r, g in zip(regions, got):
        np.testing.assert_array_equal(g, arr.read_region(r, num_workers=1))


@pytest.mark.parametrize("sharded", [False, True])
def test_healthy_chunks_never_take_the_per_chunk_path(tmp_path, monkeypatch, sharded):
    rs = np.random.RandomState(9)
    data = rs.randint(0, 65535, (2, 96, 130)).astype(np.uint16)
    if sharded:
        arr, data = _write_sharded(tmp_path, data, (1, 16, 32), (1, 32, 64))
    else:
        arr = _write(tmp_path, data, (1, 16, 32), fmt=3)

    def refuse(*args, **kwargs):
        raise AssertionError("a healthy chunk was redone on the per-chunk path")

    monkeypatch.setattr(type(arr), "_place_zstd_chunk", refuse)
    monkeypatch.setattr(type(arr), "_load_chunk", refuse)
    for workers in (1, 3):
        np.testing.assert_array_equal(
            arr.read_region((slice(None), slice(None), slice(None)), num_workers=workers), data)
        np.testing.assert_array_equal(
            arr.read_region((slice(1, 2), slice(5, 90), slice(7, 101)), num_workers=workers),
            data[1:2, 5:90, 7:101])
