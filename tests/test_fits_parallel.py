"""Compressed-FITS tiles decode across threads, for every ZCMPTYPE.

A tile-compressed FITS HDU is a BINTABLE with one row per tile, each an
independent codestream, and the whole table and heap are read in one
block before any tile is touched. So there is no I/O to serialize and
each tile writes a rectangle no other tile writes to.

One of the five decoders was not safe to run that way. cfitsio's
fits_hdecompress keeps its bit-reader position in three file-scope
variables, so two overlapping HCOMPRESS_1 tiles corrupted each other
and every tile after the first came back as status=414, "bad format
code". Serial decode never touched it. That is why every algorithm is
tested threaded here, not just the one that seemed most likely to
break.

astropy writes the fixtures, so the expected values come from the
reference implementation rather than from this decoder's own output.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import opencodecs as oc

fits = pytest.importorskip("astropy.io.fits")

ALGORITHMS = ["RICE_1", "GZIP_1", "GZIP_2", "PLIO_1", "HCOMPRESS_1"]


def _write(path, arr, algo, tile):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits.CompImageHDU(data=arr, compression_type=algo,
                          tile_shape=tile).writeto(path, overwrite=True)
    return path


def _payload(algo, shape=(512, 512), seed=0):
    a = np.random.default_rng(seed).integers(0, 3000, shape).astype("i2")
    # PLIO_1 is a mask coder: cfitsio restricts it to small positive
    # values, so a full-range int16 image is not a legal input.
    return (a % 100).astype("i2") if algo == "PLIO_1" else a


@pytest.mark.parametrize("algo", ALGORITHMS)
@pytest.mark.parametrize("numthreads", [None, 1, 2, 8])
def test_threaded_tiles_match_astropy(tmp_path, algo, numthreads):
    a = _payload(algo)
    p = _write(tmp_path / f"{algo}.fits", a, algo, (64, 64))
    expected = fits.getdata(str(p))
    with oc.get_codec("fits").open(str(p)) as r:
        got = r.read(numthreads=numthreads)
    assert np.array_equal(got, expected), f"{algo} at numthreads={numthreads}"


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_partial_edge_tiles(tmp_path, algo):
    """Tile dims that do not divide the image leave partial tiles at
    the right and bottom edges, which are placed by different
    arithmetic than the full ones."""
    a = _payload(algo, shape=(301, 173), seed=3)
    p = _write(tmp_path / f"edge_{algo}.fits", a, algo, (64, 64))
    with oc.get_codec("fits").open(str(p)) as r:
        assert np.array_equal(r.read(numthreads=8), fits.getdata(str(p)))


def test_hcompress_many_tiles_on_many_threads(tmp_path):
    """The specific shape that exposed the shared bit-reader.

    Enough tiles that several are genuinely in flight at once. With the
    cfitsio statics shared this raised HcompError on the second tile;
    it has to be run more than once, because a race that happens to
    interleave harmlessly one time is exactly what this guards.
    """
    a = _payload("HCOMPRESS_1", shape=(1024, 1024), seed=7)
    p = _write(tmp_path / "hc_many.fits", a, "HCOMPRESS_1", (64, 64))
    expected = fits.getdata(str(p))
    for _ in range(5):
        with oc.get_codec("fits").open(str(p)) as r:
            assert np.array_equal(r.read(numthreads=16), expected)


def test_float_quantized_tiles(tmp_path):
    """Quantized floats carry per-tile ZSCALE / ZZERO, read out of the
    BINTABLE row rather than out of the tile, so the threaded path has
    to index those per tile too."""
    a = np.random.default_rng(11).normal(100, 10, (256, 256)).astype("f4")
    p = _write(tmp_path / "quant.fits", a, "RICE_1", (64, 64))
    expected = fits.getdata(str(p))
    with oc.get_codec("fits").open(str(p)) as r:
        got = r.read(numthreads=8)
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("algo", ["RICE_1", "HCOMPRESS_1"])
def test_threading_helps(tmp_path, algo):
    import time
    a = _payload(algo, shape=(2048, 2048), seed=5)
    p = _write(tmp_path / f"big_{algo}.fits", a, algo, (256, 256))
    codec = oc.get_codec("fits")

    def timed(nt):
        best = 1e9
        for _ in range(3):
            t = time.perf_counter()
            with codec.open(str(p)) as r:
                r.read(numthreads=nt)
            best = min(best, time.perf_counter() - t)
        return best

    with codec.open(str(p)) as r:
        r.read(numthreads=1)
    assert timed(1) / timed(8) > 1.5, f"{algo} did not speed up"
