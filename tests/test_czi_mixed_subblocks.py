"""A CZI whose sub-blocks are not all the same size, and how ``read`` says so.

``read`` stacks every sub-block into one rectangular array, which only exists
when they all hold the same number of pixels. A pyramid CZI interleaves
down-scaled sub-blocks with the full-resolution ones, and a mosaic can clip
its edge tiles, so both break that assumption. Sizing the stack from the
first sub-block and letting the decode destination check fail reported an
element count the caller never chose -- on a real AxioScan slide,
``destination holds 3080192 elements, expected 1048576`` -- which named
neither the file's shape mix nor the API that does work. These pin the
up-front refusal and the fact that a uniform file is untouched by it.

Also here: the whole-stack read hands work out in byte-sized tasks rather
than one contiguous run per worker, so the parallel result has to stay
identical to the serial one whatever the sub-block size.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from _czi_fixture import mosaic_czi_bytes, pyramid_czi_bytes  # noqa: E402

from opencodecs._czi_reader import CziError, CziReader  # noqa: E402


def _tile(h, w, seed):
    return np.random.RandomState(seed).randint(0, 4096, (h, w)).astype(np.uint16)


def test_mixed_tile_sizes_are_not_uniform():
    data = mosaic_czi_bytes([(_tile(64, 64, 1), (0, 0)),
                             (_tile(64, 48, 2), (0, 64))])
    with CziReader(buffer=data) as r:
        assert r.is_uniform is False


def test_uniform_tiles_are_uniform_and_read():
    data = mosaic_czi_bytes([(_tile(64, 64, 1), (0, 0)),
                             (_tile(64, 64, 2), (0, 64))])
    with CziReader(buffer=data) as r:
        assert r.is_uniform is True
        assert r.read().shape == (2, 64, 64)


def test_read_refuses_a_mixed_file_and_names_the_way_out():
    data = mosaic_czi_bytes([(_tile(64, 64, 1), (0, 0)),
                             (_tile(64, 48, 2), (0, 64))])
    with CziReader(buffer=data) as r:
        with pytest.raises(CziError) as exc:
            r.read()
    message = str(exc.value)
    # The shapes it could not stack, and every API that does work.
    assert "(64, 64)" in message and "(64, 48)" in message
    assert "read_region" in message
    assert "entries_at_level" in message
    assert "read_tile" in message
    # Not the element-count error from deep inside the codec.
    assert "elements" not in message


def test_read_refuses_a_pyramid_and_says_it_is_pyramidal():
    base = _tile(64, 64, 3)
    data = pyramid_czi_bytes([base, base[::2, ::2], base[::4, ::4]])
    with CziReader(buffer=data) as r:
        assert r.is_pyramidal is True
        assert r.is_uniform is False
        with pytest.raises(CziError, match="pyramidal"):
            r.read()


def test_mixed_file_still_reads_one_tile_at_a_time():
    """The refusal is about the stack, not about the sub-blocks."""
    tiles = [_tile(64, 64, 4), _tile(64, 48, 5)]
    data = mosaic_czi_bytes([(tiles[0], (0, 0)), (tiles[1], (0, 64))])
    with CziReader(buffer=data) as r:
        for i, expected in enumerate(tiles):
            assert np.array_equal(r.read_tile(i), expected)


@pytest.mark.parametrize("n_tiles,size", [
    (40, 16),    # many sub-blocks well under one task's byte budget
    (3, 512),    # each sub-block its own task
])
def test_parallel_read_matches_serial(n_tiles, size):
    tiles = [_tile(size, size, i) for i in range(n_tiles)]
    data = mosaic_czi_bytes(
        [(t, (0, i * size)) for i, t in enumerate(tiles)], compression=6,
        hilo=True)
    with CziReader(buffer=data) as r:
        serial = r.read(n_workers=1)
        parallel = r.read()
        assert np.array_equal(serial, parallel)
        assert np.array_equal(serial, np.stack(tiles, axis=0))
        # An explicit count is a budget, not an instruction to spend it.
        assert np.array_equal(serial, r.read(n_workers=64))
