"""BCn decodes in bands: across threads, and a band on its own.

Block compression exists so that every 4x4 block is independently
decodable at a fixed offset. Neither half of that was being used: a
surface decoded in one serial pass, and reaching 64 rows of a 4096-row
texture decoded every block in it.

Both fall out of the same primitive. A band's blocks are a contiguous
slice of the input and decode as a surface of that height, so there is
no offset arithmetic to get wrong -- which is what these tests pin, by
comparing every band against the same rows of a full serial decode.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc

pytestmark = pytest.mark.skipif(
    not oc.has_codec("bcn"), reason="bcn backend not built here")

BLOCK_BYTES = {"bc1": 8, "bc2": 16, "bc3": 16, "bc7": 16}
FORMATS = sorted(BLOCK_BYTES)


def _blocks(fmt, width, height, seed=0):
    n = (width // 4) * (height // 4) * BLOCK_BYTES[fmt]
    return np.random.default_rng(seed).integers(
        0, 255, n, dtype=np.uint8).tobytes()


@pytest.fixture(scope="module")
def codec():
    return oc.get_codec("bcn")


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("numthreads", [None, 1, 2, 3, 8])
def test_threaded_decode_is_bit_identical(codec, fmt, numthreads):
    """Including 3, which does not divide the block-row count evenly."""
    w, h = 256, 512
    data = _blocks(fmt, w, h)
    ref = codec.decode(data, format=fmt, width=w, height=h, numthreads=1)
    got = codec.decode(data, format=fmt, width=w, height=h,
                       numthreads=numthreads)
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("band", [(0, 4), (0, 64), (64, 128), (448, 512),
                                  (4, 8), (0, 512)])
def test_a_band_matches_those_rows_of_the_whole(codec, fmt, band):
    w, h = 128, 512
    data = _blocks(fmt, w, h, seed=2)
    full = codec.decode(data, format=fmt, width=w, height=h, numthreads=1)
    y0, y1 = band
    got = codec.decode_rows(data, format=fmt, width=w, height=h,
                            y0=y0, y1=y1)
    assert got.shape == (y1 - y0, w, 4)
    assert np.array_equal(got, full[y0:y1])


@pytest.mark.parametrize("fmt", FORMATS)
def test_bands_tile_the_whole_surface(codec, fmt):
    """Concatenating every band must reconstruct the image exactly.

    Stronger than checking one band: an off-by-one in the input slice
    would still let a single band match while making the bands overlap
    or leave a gap.
    """
    w, h = 64, 256
    data = _blocks(fmt, w, h, seed=5)
    full = codec.decode(data, format=fmt, width=w, height=h, numthreads=1)
    stacked = np.concatenate([
        codec.decode_rows(data, format=fmt, width=w, height=h,
                          y0=y, y1=y + 32)
        for y in range(0, h, 32)])
    assert np.array_equal(stacked, full)


@pytest.mark.parametrize("fmt", FORMATS)
def test_rows_off_the_block_grid_are_refused(codec, fmt):
    w, h = 64, 128
    data = _blocks(fmt, w, h)
    for y0, y1 in ((1, 8), (0, 6), (2, 10)):
        with pytest.raises(ValueError, match="multiples of 4"):
            codec.decode_rows(data, format=fmt, width=w, height=h,
                              y0=y0, y1=y1)


@pytest.mark.parametrize("fmt", FORMATS)
def test_out_of_range_bands_are_refused(codec, fmt):
    w, h = 64, 128
    data = _blocks(fmt, w, h)
    for y0, y1 in ((0, 132), (128, 132), (64, 64), (32, 16), (-4, 8)):
        with pytest.raises(ValueError):
            codec.decode_rows(data, format=fmt, width=w, height=h,
                              y0=y0, y1=y1)


@pytest.mark.parametrize("fmt", ["bc4", "bc5", "bc6h"])
def test_non_rgba_formats_say_so(codec, fmt):
    with pytest.raises(ValueError, match="not an RGBA BC format"):
        codec.decode_rows(b"\x00" * 4096, format=fmt, width=64, height=64,
                          y0=0, y1=8)


def test_out_argument_still_honored(codec):
    w, h = 64, 128
    data = _blocks("bc7", w, h)
    out = np.empty((h, w, 4), dtype=np.uint8)
    got = codec.decode(data, format="bc7", width=w, height=h, out=out,
                       numthreads=4)
    assert got is out
    assert np.array_equal(out, codec.decode(data, format="bc7", width=w,
                                            height=h, numthreads=1))


@pytest.mark.slow
def test_threading_helps_on_a_large_surface(codec):
    import time
    w = h = 2048
    data = _blocks("bc7", w, h, seed=9)

    def timed(nt):
        codec.decode(data, format="bc7", width=w, height=h, numthreads=nt)
        best = 1e9
        for _ in range(3):
            t = time.perf_counter()
            codec.decode(data, format="bc7", width=w, height=h, numthreads=nt)
            best = min(best, time.perf_counter() - t)
        return best

    assert timed(1) / timed(None) > 1.5
