"""Threaded tile decode for a single TIFF.

TIFF tiles and strips are independent, so one image can decode across
cores. Two things have to hold and neither is obvious:

  * every thread count produces byte-identical pixels, including on
    the layouts that need the awkward general path (bilevel, planar=2,
    predictors, partial edge tiles);
  * turning threads on does not make anything slower. It did, until
    uncompressed segments were excluded: there is no decode work to
    divide there, so the pool was pure overhead.

The speedup itself is measured in _resolve_tiff_workers' docstring
rather than asserted here -- a wall-clock assertion in CI is a flaky
test waiting to happen.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc
from opencodecs._tiff_codec import _resolve_tiff_workers


def _img(h=600, w=800, seed=0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 128 + 90 * np.sin(yy / 21) * np.cos(xx / 17)
    noise = np.random.default_rng(seed).normal(0, 10, (h, w))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


# Tile sizes chosen so the image does NOT divide evenly: partial edge
# tiles are where a threaded placement bug would show up first.
# packbits is decode-only in this writer, so it cannot be round-tripped
# here; its decode path is the same general per-segment path these
# cover.
@pytest.mark.parametrize("compression", ["zstd", "deflate", "lzw", "none"])
@pytest.mark.parametrize("tile", [(128, 128), (256, 256)])
def test_threaded_decode_matches_serial(tmp_path, compression, tile):
    img = _img()
    p = tmp_path / f"t_{compression}_{tile[0]}.tif"
    with oc.TiffWriter(str(p)) as w:
        w.write_page(img, tile=tile, compression=compression)

    with oc.get_codec("tiff").open(str(p), numthreads=1) as r:
        serial = r.read()
    assert np.array_equal(serial, img)

    for nt in (2, 4, 8, None):
        with oc.get_codec("tiff").open(str(p), numthreads=nt) as r:
            got = r.read()
        assert np.array_equal(got, serial), (
            f"{compression} tile={tile} numthreads={nt} differs from serial")


def test_threaded_decode_matches_serial_for_strips(tmp_path):
    """Strips take the same general path but with tiles_x == 1."""
    img = _img(500, 300)
    p = tmp_path / "strips.tif"
    with oc.TiffWriter(str(p)) as w:
        w.write_page(img, compression="deflate")
    with oc.get_codec("tiff").open(str(p), numthreads=1) as r:
        serial = r.read()
    with oc.get_codec("tiff").open(str(p), numthreads=8) as r:
        assert np.array_equal(r.read(), serial)


def test_threaded_multipage_reads_every_page(tmp_path):
    pages = [_img(300, 400, seed=i) for i in range(4)]
    p = tmp_path / "multi.tif"
    with oc.TiffWriter(str(p)) as w:
        for a in pages:
            w.write_page(a, tile=(128, 128), compression="zstd")
    with oc.get_codec("tiff").open(str(p), numthreads=8) as r:
        stack = r.read()
    assert stack.shape == (4, 300, 400)
    for i, a in enumerate(pages):
        assert np.array_equal(stack[i], a), f"page {i}"


def test_uncompressed_declines_threads_by_default():
    """The regression that made this gate necessary.

    Uncompressed segments have no decode to divide, and threading them
    measured 3x SLOWER than serial. `None` means "decide", and deciding
    correctly here means declining.
    """
    assert _resolve_tiff_workers(None, 144, has_decode_work=False) == 1
    assert _resolve_tiff_workers(None, 144, has_decode_work=True) > 1


def test_explicit_thread_count_is_a_budget_not_a_mandate():
    """A thread count means "use up to N", and N can be too many.

    Pinning to 1 is how you get a reproducible serial run. Pinning
    high is honored wherever it can help -- but not on uncompressed
    segments, where spending the budget measured 2x SLOWER than not.
    Quietly doing what was asked at half the speed is a worse answer
    than declining.
    """
    assert _resolve_tiff_workers(1, 500) == 1
    assert _resolve_tiff_workers(0, 500) == 1
    assert _resolve_tiff_workers(4, 500) == 4
    assert _resolve_tiff_workers(4, 500, has_decode_work=False) == 1
    assert _resolve_tiff_workers(None, 500, has_decode_work=False) == 1
    # Never more workers than there is work.
    assert _resolve_tiff_workers(64, 3) == 3


def test_too_few_segments_stays_serial():
    assert _resolve_tiff_workers(None, 1) == 1
    assert _resolve_tiff_workers(None, 3) == 1


def test_codec_declares_parallel_decode():
    """The flag is public API -- README tells callers to read it."""
    assert oc.get_codec("tiff").parallel_decode is True


def test_numthreads_reaches_every_read_path(tmp_path):
    """read(), iter_frames() and [idx] all decode segments.

    They each call asarray() separately, so a numthreads that only
    reached one of them would leave the others silently serial -- and
    nothing about the returned pixels would say so.
    """
    pages = [_img(300, 400, seed=i) for i in range(3)]
    p = tmp_path / "paths.tif"
    with oc.TiffWriter(str(p)) as w:
        for a in pages:
            w.write_page(a, tile=(128, 128), compression="deflate")

    codec = oc.get_codec("tiff")
    with codec.open(str(p), numthreads=8) as r:
        assert r._numthreads == 8
        assert np.array_equal(r[1], pages[1])
        for i, frame in enumerate(r.iter_frames()):
            assert np.array_equal(frame, pages[i]), f"iter_frames page {i}"
        assert np.array_equal(r.read()[2], pages[2])


def test_page_asarray_takes_numthreads_directly(tmp_path):
    img = _img(400, 500)
    p = tmp_path / "one.tif"
    with oc.TiffWriter(str(p)) as w:
        w.write_page(img, tile=(128, 128), compression="lzw")
    with oc.get_codec("tiff").open(str(p)) as r:
        page = r.page(0)
        serial = page.asarray(numthreads=1)
        assert np.array_equal(serial, img)
        assert np.array_equal(page.asarray(numthreads=6), serial)
        assert np.array_equal(page.asarray(), serial)


def test_codec_decode_accepts_numthreads(tmp_path):
    img = _img(300, 300)
    p = tmp_path / "dec.tif"
    with oc.TiffWriter(str(p)) as w:
        w.write_page(img, tile=(128, 128), compression="zstd")
    codec = oc.get_codec("tiff")
    assert np.array_equal(codec.decode(str(p), numthreads=4), img)
    assert np.array_equal(codec.decode(str(p), numthreads=1), img)


def test_a_worker_exception_is_not_swallowed(tmp_path, monkeypatch):
    """ex.map returns a lazy iterator.

    If the results are never consumed, a decode failure in a worker
    disappears and the caller gets a half-filled array with no error.
    """
    img = _img(400, 400)
    p = tmp_path / "boom.tif"
    with oc.TiffWriter(str(p)) as w:
        w.write_page(img, tile=(64, 64), compression="deflate")
    with oc.get_codec("tiff").open(str(p), numthreads=4) as r:
        page = r.page(0)
        monkeypatch.setattr(page, "_decode_segment",
                            lambda raw: (_ for _ in ()).throw(
                                RuntimeError("segment exploded")))
        with pytest.raises(RuntimeError, match="segment exploded"):
            page.asarray(numthreads=4)
