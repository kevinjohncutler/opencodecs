"""Bounded iteration and direct slice assembly (priority 3).

Slice access now allocates its stack once and decodes each tile into a row,
instead of holding a list of tiles and the stack together. Iteration gained an
opt-in bounded prefetch path alongside the serial default.

The risks worth pinning: a preallocated stack must still reproduce every slice
form, including the errors the old np.stack raised; the bounded path must
yield in directory order and hand back arrays that stay valid as iteration
continues (per-thread scratch is reused underneath); abandoning the iterator
must join its workers and leave the reader usable; and a failing decode must
surface to the caller rather than hang.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc

pytestmark = pytest.mark.skipif(
    not oc.has_codec("czi"),
    reason="czi codec requires native zstd + bytetools extensions",
)

from _czi_fixture import czi_bytes, pyramid_czi_bytes  # noqa: E402


def _stack(n=8, shape=(24, 32), seed=0):
    rs = np.random.RandomState(seed)
    return np.stack([rs.randint(0, 65535, shape).astype(np.uint16)
                     for _ in range(n)])


@pytest.fixture(scope="module")
def frames():
    return _stack()


@pytest.fixture(scope="module")
def data(frames):
    return czi_bytes(frames, compression=6, hilo=True)


# ---------------------------------------------------------------------------
# Slice assembly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sl", [
    slice(None), slice(1, 5), slice(None, None, 2), slice(-3, None),
    slice(0, 1), slice(None, None, 3), slice(7, 0, -2),
])
def test_slice_matches_tile_by_tile(data, frames, sl):
    with oc.get_codec("czi").open(data) as r:
        got = r[sl]
        expected = np.stack([np.squeeze(r[i])
                             for i in range(*sl.indices(len(r)))])
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(got, frames[sl])


def test_empty_slice_still_raises(data):
    with oc.get_codec("czi").open(data) as r:
        with pytest.raises(ValueError):
            r[5:1]


def test_mixed_tile_shapes_still_raise_value_error():
    """A pyramid's levels differ in stored shape; stacking them must fail."""
    base = np.random.RandomState(9).randint(
        0, 65535, (64, 64)).astype(np.uint16)
    data = pyramid_czi_bytes([base, base[::2, ::2], base[::4, ::4]],
                             compression=6, hilo=True)
    with oc.get_codec("czi").open(data) as r:
        assert len({e.stored_shape for e in r.entries}) > 1
        with pytest.raises(ValueError):
            r[:]


def test_singleton_tiles_stack_like_before():
    """Every axis of size one squeezes away; the stack is still per tile."""
    frames = np.arange(5, dtype=np.uint16).reshape(5, 1, 1)
    data = czi_bytes(frames, compression=6, hilo=True)
    with oc.get_codec("czi").open(data) as r:
        got = r[:]
        expected = np.stack([np.squeeze(r[i]) for i in range(len(r))])
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(got.reshape(-1), frames.reshape(-1))


# ---------------------------------------------------------------------------
# Bounded iteration
# ---------------------------------------------------------------------------


def test_bounded_iteration_matches_serial_order(data, frames):
    with oc.get_codec("czi").open(data) as r:
        serial = [np.squeeze(t) for t in r.iter_tiles()]
    with oc.get_codec("czi").open(data) as r:
        bounded = list(r.iter_tiles(n_workers=4, max_pending_bytes=1 << 15))
    assert len(bounded) == len(serial) == len(frames)
    for i, (a, b) in enumerate(zip(serial, bounded)):
        np.testing.assert_array_equal(a, b, err_msg=f"tile {i} out of order")
    np.testing.assert_array_equal(np.stack(bounded), frames)


def test_yielded_tiles_stay_valid_while_iterating(data, frames):
    """Scratch is reused per worker; a yielded array must not follow it."""
    with oc.get_codec("czi").open(data) as r:
        held = list(r.iter_tiles(n_workers=4, max_pending_bytes=1 << 15))
    # Every array was retained across many later decodes before this compare.
    np.testing.assert_array_equal(np.stack(held), frames)


def test_bounded_iteration_respects_a_small_byte_budget(data, frames):
    """A budget below one tile must still complete, running items alone."""
    with oc.get_codec("czi").open(data) as r:
        tiles = list(r.iter_tiles(n_workers=4, max_pending_bytes=1))
    np.testing.assert_array_equal(np.stack(tiles), frames)


def test_early_close_joins_and_leaves_reader_usable(data, frames):
    from contextlib import closing

    with oc.get_codec("czi").open(data) as r:
        with closing(r.iter_tiles(n_workers=4,
                                  max_pending_bytes=1 << 15)) as it:
            first = next(it)
            second = next(it)
        np.testing.assert_array_equal(first, frames[0])
        np.testing.assert_array_equal(second, frames[1])
        # The reader is borrowed by the iterator, not consumed by it.
        np.testing.assert_array_equal(np.squeeze(r[3]), frames[3])
        np.testing.assert_array_equal(np.squeeze(r.read()), frames)


def test_decode_failure_propagates_from_bounded_iteration(data):
    with oc.get_codec("czi").open(data) as r:
        original = r._decode_one

        def explode(entry, **kw):
            if entry is r.entries[3]:
                raise RuntimeError("boom")
            return original(entry, **kw)

        r._decode_one = explode
        with pytest.raises(RuntimeError, match="boom"):
            list(r.iter_tiles(n_workers=4, max_pending_bytes=1 << 15))


def test_as_rgb_applies_in_both_iteration_paths():
    frame = np.random.RandomState(12).randint(
        0, 256, (8, 12, 3)).astype(np.uint8)
    data = czi_bytes(frame, compression=6, hilo=True)
    with oc.get_codec("czi").open(data) as r:
        serial = list(r.iter_tiles(as_rgb=True))
        bounded = list(r.iter_tiles(as_rgb=True, n_workers=2))
    np.testing.assert_array_equal(serial[0], frame[..., ::-1])
    np.testing.assert_array_equal(bounded[0], frame[..., ::-1])
