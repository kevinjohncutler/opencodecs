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


# --------------------------------------------------------------------
# lz4: lz4-java's LZ4Block stream, which is what N5 writes
# --------------------------------------------------------------------

# One block file written by the N5 reference implementation itself (n5
# 2.5.1 N5FSWriter, Lz4Compression(64), lz4-java 1.8.0): a 10 x 8 uint16
# block, the N5 block header, then lz4-java's LZ4BlockOutputStream with
# 64-byte blocks. The first 64 bytes of data are zero, so that block is
# LZ4-compressed (method 0x20); the rest are pseudo-random, so lz4-java
# stored those two blocks raw (method 0x10); then the end block.
JAVA_LZ4_BLOCK_FILE = bytes.fromhex(
    "000000020000000a000000084c5a34426c6f636b200b000000400000000499fa031f"
    "000100275000000000004c5a34426c6f636b104000000040000000debb290b1c0d43"
    "f0e2b8d5f2535c8f7bcb3a6438d0af302bfd70fba92961d37b7a5b552ea92e5fe461"
    "8d5cd1077216e638c48adea7e58378dab341b94b0e7402f33008774c5a34426c6f63"
    "6b10200000002000000067ed690a0c99e786c36048a17c5b36ae2f3211aad75f4a8f"
    "2a020e3acc8f52dbda3602e54c5a34426c6f636b10000000000000000000000000")


def _java_lz4_values():
    """The samples the Java writer was given (its own generator)."""
    values, s = [0] * 32, 12345
    for _ in range(48):
        s = (s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        values.append(s >> 48)
    # N5 lists the fastest axis first: 10 columns, 8 rows.
    return np.array(values, dtype="<u2").reshape(8, 10)


def _write_lz4_dataset(root, payload, shape, block_size, dtype="uint16",
                       lz4_block_size=65536):
    d = root / "lz4"
    d.mkdir(parents=True, exist_ok=True)
    (d / "attributes.json").write_text(json.dumps({
        "dataType": dtype,
        "compression": {"type": "lz4", "blockSize": lz4_block_size},
        "blockSize": list(reversed(block_size)),
        "dimensions": list(reversed(shape)),
    }))
    (d / "0").mkdir(exist_ok=True)
    (d / "0" / "0").write_bytes(payload)


def test_lz4_reads_a_block_the_java_reference_wrote(tmp_path):
    _write_lz4_dataset(tmp_path, JAVA_LZ4_BLOCK_FILE, (8, 10), (8, 10),
                       lz4_block_size=64)
    z = N5Array(str(tmp_path), "lz4")
    assert z.compression == "lz4"
    got = np.asarray(z.asarray(), dtype="<u2")
    assert np.array_equal(got, _java_lz4_values())


def _lz4block_stream(data, block_size):
    """An LZ4BlockOutputStream stream, built from lz4-java's layout.

    Payloads come from imagecodecs' bare LZ4 block encoder and checksums
    from the xxhash package, so nothing here is opencodecs' own code.
    """
    ic = pytest.importorskip("imagecodecs")
    xxhash = pytest.importorskip("xxhash")
    level = max(0, block_size.bit_length() - 1 - 10)
    out = b""
    for start in range(0, len(data), block_size):
        chunk = data[start:start + block_size]
        check = xxhash.xxh32_intdigest(chunk, 0x9747B28C) & 0x0FFFFFFF
        packed = ic.lz4_encode(chunk)
        if len(packed) >= len(chunk):
            token, packed = 0x10 | level, chunk
        else:
            token = 0x20 | level
        out += (b"LZ4Block" + bytes([token])
                + struct.pack("<iii", len(packed), len(chunk), check) + packed)
    return out + b"LZ4Block" + bytes([0x10 | level]) + bytes(12)


def _n5_block(arr):
    header = struct.pack(">HH", 0, arr.ndim)
    header += struct.pack(f">{arr.ndim}I", *reversed(arr.shape))
    return header, arr.astype(">" + arr.dtype.str[1:]).tobytes()


def test_lz4_multi_block_stream(tmp_path):
    rng = np.random.default_rng(5)
    a = np.where(rng.random((96, 300)) < 0.7, 0,
                 rng.integers(0, 65535, (96, 300))).astype("<u2")
    header, body = _n5_block(a)
    _write_lz4_dataset(tmp_path, header + _lz4block_stream(body, 8192),
                       a.shape, a.shape, lz4_block_size=8192)
    got = N5Array(str(tmp_path), "lz4").asarray()
    assert np.array_equal(np.asarray(got, dtype="<u2"), a)


def test_lz4_checksum_and_truncation_are_errors(tmp_path):
    a = np.arange(40 * 50, dtype="<u2").reshape(40, 50)
    header, body = _n5_block(a)
    stream = bytearray(_lz4block_stream(body, 1024))
    stream[17] ^= 0x01                     # first block's checksum
    _write_lz4_dataset(tmp_path, header + bytes(stream), a.shape, a.shape)
    with pytest.raises(N5Error, match="checksum"):
        N5Array(str(tmp_path), "lz4").asarray()
    stream[17] ^= 0x01
    _write_lz4_dataset(tmp_path, header + bytes(stream[:-21]), a.shape,
                       a.shape)            # no end block
    with pytest.raises(N5Error, match="prematurely"):
        N5Array(str(tmp_path), "lz4").asarray()


def test_lz4_frame_payload_still_reads(tmp_path):
    ic = pytest.importorskip("imagecodecs")
    a = np.arange(30 * 20, dtype="<u2").reshape(30, 20)
    header, body = _n5_block(a)
    _write_lz4_dataset(tmp_path, header + ic.lz4f_encode(body), a.shape,
                       a.shape)
    got = N5Array(str(tmp_path), "lz4").asarray()
    assert np.array_equal(np.asarray(got, dtype="<u2"), a)
