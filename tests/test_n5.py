"""N5 reader.

Synthetic datasets are built here per the specification, because they let
us cover compression variants, ranks and the sparse case exactly. The
real Janelia fixture covers what a self-built dataset cannot: it was
written by the toolchain N5 exists to serve, and it settles which of the
spec's two legal edge-block layouts real writers actually emit.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import struct

import numpy as np
import pytest

from opencodecs._n5 import N5Array, N5Error

REAL = (pathlib.Path(__file__).resolve().parent.parent / ".test_data" / "n5"
        / "jrc_hela-2.n5")
REAL_ARRAY = "em/fibsem-uint16/s4"


def write_n5(root: pathlib.Path, arr: np.ndarray, block_size, *,
             compression="raw", path="data", skip=()):
    """Write a minimal but spec-correct N5 dataset.

    Everything the format stores column-major is written column-major
    here, so a reader that forgets to reverse fails these tests.
    """
    d = root / path
    d.mkdir(parents=True, exist_ok=True)
    dtype_name = {"|u1": "uint8", "|i1": "int8", "<u2": "uint16",
                  "<i2": "int16", "<u4": "uint32", "<i4": "int32",
                  "<u8": "uint64", "<i8": "int64",
                  "<f4": "float32", "<f8": "float64"}[arr.dtype.str]
    (d / "attributes.json").write_text(json.dumps({
        "dimensions": list(reversed(arr.shape)),
        "blockSize": list(reversed(block_size)),
        "dataType": dtype_name,
        "compression": {"type": compression},
    }))
    be = arr.astype(">" + arr.dtype.str[1:])
    grid = tuple(-(-s // b) for s, b in zip(arr.shape, block_size))
    for idx in np.ndindex(*grid):
        if idx in skip:
            continue
        sel = tuple(slice(i * b, min((i + 1) * b, s))
                    for i, b, s in zip(idx, block_size, arr.shape))
        block = np.ascontiguousarray(be[sel])
        header = struct.pack(">HH", 0, arr.ndim)
        header += struct.pack(f">{arr.ndim}I", *reversed(block.shape))
        payload = block.tobytes()
        if compression == "gzip":
            payload = gzip.compress(payload)
        p = d.joinpath(*[str(i) for i in reversed(idx)])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(header + payload)
    return d


# --------------------------------------------------------------------
# synthetic
# --------------------------------------------------------------------

@pytest.mark.parametrize("compression", ["raw", "gzip"])
def test_roundtrip(tmp_path, compression):
    a = np.arange(4 * 6 * 8, dtype="<u2").reshape(4, 6, 8)
    write_n5(tmp_path, a, (2, 3, 4), compression=compression)
    z = N5Array(str(tmp_path), "data")
    assert z.shape == (4, 6, 8)
    assert z.chunks == (2, 3, 4)
    assert z.compression == compression
    assert np.array_equal(np.asarray(z.asarray(), dtype="<u2"), a)


def test_axis_order_is_reversed_from_disk(tmp_path):
    """The difference that makes N5 dangerous to treat as Zarr.

    dimensions and blockSize are column-major on disk. With distinct
    extents, a reader that takes them at face value returns a transposed
    volume of the correct rank, which no shape-only check catches.
    """
    a = np.arange(2 * 3 * 4, dtype="<u2").reshape(2, 3, 4)
    d = write_n5(tmp_path, a, (2, 3, 4))
    meta = json.loads((d / "attributes.json").read_text())
    assert meta["dimensions"] == [4, 3, 2]        # stored reversed
    z = N5Array(str(tmp_path), "data")
    assert z.shape == (2, 3, 4)                   # presented C-order
    assert np.array_equal(np.asarray(z.asarray(), dtype="<u2"), a)


def test_block_paths_are_nested_column_major(tmp_path):
    """C-order block (z, y, x) is stored at x/y/z."""
    a = np.arange(4 * 6 * 8, dtype="<u2").reshape(4, 6, 8)
    d = write_n5(tmp_path, a, (2, 3, 4))
    assert (d / "1" / "1" / "1").is_file()
    z = N5Array(str(tmp_path), "data")
    block = z.read_block((1, 1, 1))
    assert block is not None
    assert np.array_equal(np.asarray(block, dtype="<u2"), a[2:4, 3:6, 4:8])


def test_missing_block_reads_as_zeros(tmp_path):
    """Sparse datasets simply do not write empty regions."""
    a = np.ones((4, 4), dtype="<u2")
    write_n5(tmp_path, a, (2, 2), skip={(1, 1)})
    z = N5Array(str(tmp_path), "data")
    assert z.read_block((1, 1)) is None
    out = z.asarray()
    assert np.array_equal(out[2:, 2:], np.zeros((2, 2), dtype="<u2"))
    assert np.array_equal(np.asarray(out[:2, :2], dtype="<u2"), a[:2, :2])


def test_ragged_edge_blocks(tmp_path):
    """Blocks that do not divide the shape evenly."""
    a = np.arange(5 * 7, dtype="<u2").reshape(5, 7)
    write_n5(tmp_path, a, (2, 3))
    z = N5Array(str(tmp_path), "data")
    assert z.chunk_grid == (3, 3)
    assert np.array_equal(np.asarray(z.asarray(), dtype="<u2"), a)


@pytest.mark.parametrize("dtype", ["|u1", "<i2", "<u4", "<i8", "<f4", "<f8"])
def test_dtypes(tmp_path, dtype):
    a = (np.arange(24) % 17).astype(dtype).reshape(4, 6)
    write_n5(tmp_path, a, (2, 3))
    got = N5Array(str(tmp_path), "data").asarray()
    assert np.array_equal(np.asarray(got, dtype=dtype), a)


def test_group_without_dataset_metadata_raises(tmp_path):
    (tmp_path / "grp").mkdir()
    (tmp_path / "grp" / "attributes.json").write_text('{"n5": "2.0.0"}')
    with pytest.raises(N5Error, match="group, not a dataset"):
        N5Array(str(tmp_path), "grp")


def test_missing_metadata_raises(tmp_path):
    with pytest.raises(N5Error, match="no attributes.json"):
        N5Array(str(tmp_path), "nope")


def test_unsupported_dtype_raises(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "attributes.json").write_text(json.dumps({
        "dimensions": [2, 2], "blockSize": [2, 2],
        "dataType": "float16", "compression": {"type": "raw"}}))
    with pytest.raises(N5Error, match="unsupported dataType"):
        N5Array(str(tmp_path), "data")


def test_block_with_wrong_rank_raises(tmp_path):
    a = np.ones((2, 2), dtype="<u2")
    d = write_n5(tmp_path, a, (2, 2))
    # Rewrite the block claiming three dimensions.
    bad = struct.pack(">HH", 0, 3) + struct.pack(">3I", 2, 2, 1) + b"\x00" * 8
    (d / "0" / "0").write_bytes(bad)
    with pytest.raises(N5Error, match="declares 3 dimensions"):
        N5Array(str(tmp_path), "data").read_block((0, 0))


# --------------------------------------------------------------------
# real Janelia volume
# --------------------------------------------------------------------

pytestmark_real = pytest.mark.skipif(
    not (REAL / REAL_ARRAY / "attributes.json").is_file(),
    reason="run `python corpus/corpus.py fetch n5_janelia_hela`")


@pytestmark_real
def test_real_metadata():
    z = N5Array(str(REAL), REAL_ARRAY)
    assert z.attrs["dimensions"] == [750, 100, 398]     # column-major on disk
    assert z.shape == (398, 100, 750)                   # C-order presented
    assert z.chunks == (64, 64, 64)
    assert z.dtype == np.dtype(">u2")
    assert z.compression == "gzip"
    assert z.chunk_grid == (7, 2, 12)


@pytestmark_real
def test_real_blocks_decode():
    z = N5Array(str(REAL), REAL_ARRAY)
    b = z.read_block((0, 0, 0))
    assert b is not None
    assert b.shape == (64, 64, 64) and b.dtype == np.dtype(">u2")
    # Real FIB-SEM data: not constant, and within the uint16 range.
    assert int(b.min()) < int(b.max()) <= 65535


@pytestmark_real
def test_real_edge_block_is_padded_not_ragged():
    """This writer pads; the spec allows either, so readers must cope.

    Block 6 along the slowest axis covers rows 384..448 of a 398-row
    volume, so a ragged writer would store 14 planes. Janelia's stores a
    full 64, and the assembling code is what trims it.
    """
    z = N5Array(str(REAL), REAL_ARRAY)
    edge = z.read_block((6, 0, 0))
    assert edge is not None
    assert edge.shape == (64, 64, 64), (
        "expected a padded edge block; if upstream rewrote the volume "
        "ragged, this test documents the change rather than a bug")
    assert z.shape[0] - 6 * z.chunks[0] == 14


@pytestmark_real
def test_real_unfetched_blocks_read_as_zero():
    """Only four blocks are in the corpus, so the rest must be absent.

    That is the same code path a sparse dataset takes, exercised here
    without needing a sparse dataset.
    """
    z = N5Array(str(REAL), REAL_ARRAY)
    assert z.read_block((3, 1, 5)) is None
