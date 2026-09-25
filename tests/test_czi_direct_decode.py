"""Direct-destination CZI decoding: parity, ownership, checks, threads.

The direct-decode change removed the
tile-sized intermediates from CZI decode: unshuffled payloads now decompress
straight into the destination, and shuffled payloads decompress into
per-thread scratch and unshuffle into the destination.

These tests pin what that removal could plausibly break. Identical pixels for
every compression variant. Standalone tiles that still own their memory after
the reader closes, since a result aliasing reusable scratch would decay
silently. Destinations that are validated rather than trusted. Malformed
payloads that raise instead of returning a truncated tile. And concurrent
decodes that cannot overwrite each other's scratch.
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


def _frame(dtype, shape, seed):
    rs = np.random.RandomState(seed)
    if np.issubdtype(dtype, np.floating):
        return rs.rand(*shape).astype(dtype)
    info = np.iinfo(dtype)
    return rs.randint(info.min, info.max, shape).astype(dtype)


# (compression, hilo, dtype, shape): every decode branch the reader has,
# plus a color type, a singleton axis and a 1-byte shuffle (identity).
VARIANTS = [
    pytest.param(0, False, np.uint8, (16, 24), id="raw-u1"),
    pytest.param(5, False, np.uint8, (32, 32), id="zstd0-u1"),
    pytest.param(6, False, np.uint8, (16, 16), id="zstdhdr-noshuffle-u1"),
    pytest.param(6, True, np.uint8, (16, 16), id="zstdhdr-shuffle-u1"),
    pytest.param(6, True, np.uint16, (24, 32), id="zstdhdr-shuffle-u2"),
    pytest.param(6, True, np.float32, (16, 16), id="zstdhdr-shuffle-f4"),
    pytest.param(6, True, np.uint8, (8, 12, 3), id="zstdhdr-shuffle-bgr24"),
    pytest.param(6, True, np.uint16, (1, 8), id="zstdhdr-shuffle-singleton"),
]


@pytest.mark.parametrize("compression,hilo,dtype,shape", VARIANTS)
def test_decode_parity_for_every_variant(compression, hilo, dtype, shape):
    """Both the destination path and the allocating path yield the pixels."""
    arr = _frame(dtype, shape, seed=11)
    data = czi_bytes(arr, compression=compression, hilo=hilo)
    with oc.get_codec("czi").open(data) as r:
        np.testing.assert_array_equal(np.squeeze(r.read()), np.squeeze(arr))
        # r[0] takes dest=None, which allocates rather than borrowing scratch.
        np.testing.assert_array_equal(np.squeeze(r[0]), np.squeeze(arr))


@pytest.mark.parametrize("compression,hilo,dtype,shape", VARIANTS)
def test_standalone_tile_owns_memory_after_close(compression, hilo, dtype, shape):
    """A tile must survive its reader: scratch would be reused by then."""
    arr = _frame(dtype, shape, seed=12)
    data = czi_bytes(arr, compression=compression, hilo=hilo)
    with oc.get_codec("czi").open(data) as r:
        tile = r[0]
    assert tile.base is None or tile.flags.owndata or tile.base.flags.owndata
    assert tile.flags.writeable
    np.testing.assert_array_equal(np.squeeze(tile), np.squeeze(arr))
    # Decoding more tiles cycles the scratch; the earlier result must not move.
    keep = np.squeeze(tile).copy()
    for seed in range(4):
        other = _frame(dtype, shape, seed=20 + seed)
        with oc.get_codec("czi").open(
                czi_bytes(other, compression=compression, hilo=hilo)) as r2:
            r2.read()
    np.testing.assert_array_equal(np.squeeze(tile), keep)


def test_read_into_destination_returns_that_destination():
    from opencodecs._czi_reader import CziReader

    arr = _frame(np.uint16, (12, 20), seed=13)
    data = czi_bytes(arr, compression=6, hilo=True)
    with oc.get_codec("czi").open(data) as r:
        entry = r.entries[0]
        view, _ = r._pixel_data_view(entry)
        dest = np.zeros(entry.stored_shape, dtype=entry.dtype)
        got = CziReader._decode_payload(
            view, entry.pixel_type, entry.compression, entry.stored_shape,
            dest=dest)
    assert got is dest
    np.testing.assert_array_equal(np.squeeze(dest), arr)


# ---------------------------------------------------------------------------
# Destination validation
# ---------------------------------------------------------------------------


def test_destination_validation_rejects_unusable_storage():
    from opencodecs._czi_reader import _payload_destination

    dtype = np.dtype(np.uint16)
    shape = (4, 5)
    n = 20

    with pytest.raises(TypeError, match="ndarray"):
        _payload_destination(bytearray(40), dtype, shape, n)

    frozen = np.zeros(shape, dtype=dtype)
    frozen.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        _payload_destination(frozen, dtype, shape, n)

    strided = np.zeros((4, 10), dtype=dtype)[:, ::2]
    assert not strided.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        _payload_destination(strided, dtype, shape, n)

    with pytest.raises(ValueError, match="dtype"):
        _payload_destination(np.zeros(shape, dtype=np.uint8), dtype, shape, n)

    with pytest.raises(ValueError, match="elements"):
        _payload_destination(np.zeros((4, 4), dtype=dtype), dtype, shape, n)


def test_destination_byte_view_covers_exactly_one_tile():
    from opencodecs._czi_reader import _payload_destination

    dest = np.zeros((3, 4), dtype=np.uint16)
    got, as_bytes = _payload_destination(dest, np.dtype(np.uint16), (3, 4), 12)
    assert got is dest
    assert as_bytes.nbytes == dest.nbytes
    as_bytes[:] = 0xAB
    assert np.all(dest == 0xABAB)


# ---------------------------------------------------------------------------
# Malformed payloads
# ---------------------------------------------------------------------------


def _payload_of(data):
    """Return (reader-independent) payload view, entry and its reader."""
    reader = oc.get_codec("czi").open(data)
    entry = reader.entries[0]
    view, _ = reader._pixel_data_view(entry)
    return reader, entry, view


def test_payload_decoding_short_raises_czi_error():
    """Claiming more pixels than the payload holds must not pad silently."""
    from opencodecs._czi_reader import CziError, CziReader

    arr = _frame(np.uint16, (8, 8), seed=14)
    reader, entry, view = _payload_of(czi_bytes(arr, compression=6, hilo=True))
    try:
        bigger = (entry.stored_shape[0], entry.stored_shape[1] * 2,
                  *entry.stored_shape[2:])
        with pytest.raises(CziError, match="decoded to|does not decode"):
            CziReader._decode_payload(
                view, entry.pixel_type, entry.compression, bigger)
    finally:
        reader.close()


def test_payload_decoding_long_raises_instead_of_truncating():
    """A tile smaller than the frame used to yield unshuffled garbage."""
    from opencodecs._czi_reader import CziError, CziReader

    arr = _frame(np.uint16, (8, 8), seed=15)
    reader, entry, view = _payload_of(czi_bytes(arr, compression=6, hilo=True))
    try:
        smaller = (entry.stored_shape[0] // 2, *entry.stored_shape[1:])
        with pytest.raises(CziError, match="does not decode|decoded to"):
            CziReader._decode_payload(
                view, entry.pixel_type, entry.compression, smaller)
    finally:
        reader.close()


# ---------------------------------------------------------------------------
# Concurrency: per-thread scratch
# ---------------------------------------------------------------------------


def test_parallel_read_matches_serial_iteration():
    frames = np.stack([_frame(np.uint16, (32, 40), seed=100 + i)
                       for i in range(12)])
    data = czi_bytes(frames, compression=6, hilo=True)
    with oc.get_codec("czi").open(data) as r:
        serial = np.stack([np.squeeze(t) for t in r.iter_tiles()])
    with oc.get_codec("czi").open(data) as r:
        parallel = np.squeeze(r.read(n_workers=8))
    np.testing.assert_array_equal(serial, frames)
    np.testing.assert_array_equal(parallel, frames)


def test_parallel_read_bounded_path_matches():
    """The map_bounded path decodes into the same destinations."""
    frames = np.stack([_frame(np.uint16, (24, 24), seed=200 + i)
                       for i in range(8)])
    data = czi_bytes(frames, compression=6, hilo=True)
    with oc.get_codec("czi").open(data) as r:
        bounded = np.squeeze(r.read(max_pending_bytes=1 << 16))
    np.testing.assert_array_equal(bounded, frames)


def test_concurrent_tile_decodes_do_not_share_scratch():
    """Different sizes on many threads: one shared buffer would corrupt."""
    from concurrent.futures import ThreadPoolExecutor

    small = np.stack([_frame(np.uint16, (16, 16), seed=300 + i)
                      for i in range(6)])
    large = np.stack([_frame(np.uint16, (64, 48), seed=400 + i)
                      for i in range(6)])
    small_data = czi_bytes(small, compression=6, hilo=True)
    large_data = czi_bytes(large, compression=6, hilo=True)

    with oc.get_codec("czi").open(small_data) as rs, \
            oc.get_codec("czi").open(large_data) as rl:
        def job(spec):
            reader, idx = spec
            return idx, np.squeeze(reader.read_tile(idx))

        specs = [(rs, i) for i in range(6)] * 4 + [(rl, i) for i in range(6)] * 4
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(job, specs))

        for (reader, idx), (got_idx, tile) in zip(specs, results):
            assert got_idx == idx
            expected = small[idx] if reader is rs else large[idx]
            np.testing.assert_array_equal(tile, expected)
