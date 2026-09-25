"""Optional decoded-tile cache (priority 7).

The cache spends memory to skip decompression on repeated or nearby regions.
What has to hold: it is off unless asked for; results with it are identical
to results without it; cached tiles are read-only so no caller can corrupt
what another will be served; the budget bounds retained bytes with LRU
eviction and refuses items larger than itself; concurrent misses for one
key decode once while the others wait; a failed claimant releases its key
so waiters proceed; and explicit invalidation empties it.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import opencodecs as oc
from opencodecs.core.tile_cache import DecodedTileCache

pytestmark = pytest.mark.skipif(
    not oc.has_codec("czi"),
    reason="czi codec requires native zstd + bytetools extensions",
)

from _czi_fixture import mosaic_czi_bytes  # noqa: E402

from opencodecs._czi_reader import CziPyramidReader  # noqa: E402


# ---------------------------------------------------------------- unit ---


def test_lru_eviction_by_bytes_and_oversized_skip():
    cache = DecodedTileCache(max_bytes=3 * 100)
    for k in range(3):
        assert cache.claim(k) == ("miss", None)
        cache.store(k, np.zeros(100, dtype=np.uint8))
    assert len(cache) == 3 and cache.bytes == 300
    cache.claim(0)                       # touch 0: 1 is now least recent
    cache.claim(3); cache.store(3, np.zeros(100, dtype=np.uint8))
    assert len(cache) == 3 and cache.evictions == 1
    assert cache.claim(1) == ("miss", None)    # evicted
    cache.release(1)
    assert cache.claim(0)[0] == "hit"          # kept
    cache.claim(9); cache.store(9, np.zeros(1000, dtype=np.uint8))
    assert cache.skipped_oversized == 1 and cache.claim(9) == ("miss", None)
    cache.release(9)


def test_cached_arrays_are_read_only():
    cache = DecodedTileCache(1 << 20)
    cache.claim("a")
    arr = cache.store("a", np.arange(10))
    with pytest.raises(ValueError):
        arr[0] = 99
    status, again = cache.claim("a")
    assert status == "hit" and again is arr and not again.flags.writeable


def test_concurrent_misses_decode_once_and_waiters_get_the_result():
    cache = DecodedTileCache(1 << 20)
    decoded = []
    started = threading.Barrier(6)

    def worker(n):
        started.wait()
        status, arr = cache.claim("tile")
        if status == "miss":
            decoded.append(n)
            arr = cache.store("tile", np.full(4, 7))
        return arr

    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(worker, range(6)))
    assert len(decoded) == 1
    assert all(np.array_equal(r, [7, 7, 7, 7]) for r in results)
    assert cache.hits + cache.misses == 6 and cache.misses == 1


def test_failed_claimant_releases_key_for_waiters():
    cache = DecodedTileCache(1 << 20)
    assert cache.claim("k") == ("miss", None)
    got = []

    def waiter():
        got.append(cache.claim("k"))

    t = threading.Thread(target=waiter)
    t.start()
    cache.release("k")            # the first decoder failed
    t.join(5)
    assert got == [("miss", None)]  # waiter now owns the claim
    cache.store("k", np.zeros(2))
    assert cache.claim("k")[0] == "hit"


def test_clear_and_stats():
    cache = DecodedTileCache(1000)
    cache.claim(1); cache.store(1, np.zeros(10, dtype=np.uint8))
    assert cache.stats()["entries"] == 1
    cache.clear()
    assert cache.stats()["entries"] == 0 and cache.bytes == 0


def test_budget_must_be_positive():
    with pytest.raises(ValueError):
        DecodedTileCache(0)


# --------------------------------------------------------- integration ---


def _tiles(n=16, side=32, seed=0):
    return [(np.random.RandomState(seed + i).randint(0, 65535, (side, side)).astype(np.uint16),
             ((i // 4) * side, (i % 4) * side)) for i in range(n)]


def _canvas(tiles):
    h = max(sy + t.shape[0] for t, (sy, sx) in tiles)
    w = max(sx + t.shape[1] for t, (sy, sx) in tiles)
    c = np.zeros((h, w), dtype=np.uint16)
    for t, (sy, sx) in tiles:
        c[sy:sy + t.shape[0], sx:sx + t.shape[1]] = t
    return c


def test_cache_is_off_by_default_and_on_when_asked():
    data = mosaic_czi_bytes(_tiles(), compression=6, hilo=True)
    plain = CziPyramidReader(oc.get_codec("czi").open(data))
    cached = CziPyramidReader(oc.get_codec("czi").open(data), decoded_cache_bytes=1 << 20)
    try:
        assert plain.tile_cache is None and cached.tile_cache is not None
        plain.read_region(0, y=(0, 40), x=(0, 40))
        assert "cache_hits" not in plain.region_stats
    finally:
        plain.close(); cached.close()


def test_repeated_regions_hit_and_match_uncached_pixels():
    tiles = _tiles()
    canvas = _canvas(tiles)
    data = mosaic_czi_bytes(tiles, compression=6, hilo=True)
    p = CziPyramidReader(oc.get_codec("czi").open(data), decoded_cache_bytes=1 << 20)
    try:
        boxes = [(0, 40, 0, 40), (10, 70, 20, 90), (0, 40, 0, 40), (100, 128, 100, 128)]
        for y0, y1, x0, x1 in boxes:
            got = p.read_region(0, y=(y0, y1), x=(x0, x1))
            np.testing.assert_array_equal(got, canvas[y0:y1, x0:x1])
        st = p.region_stats
        assert p.tile_cache.hits > 0
        # Third box repeats the first: served entirely from the cache.
        p.read_region(0, y=(0, 40), x=(0, 40))
        st = p.region_stats
        assert st["cache_hits"] == 4 and st["cache_misses"] == 0 and st["tiles_decoded"] == 0
        # Batched path shares the cache too.
        outs = p.read_regions(0, [((0, 40), (0, 40)), ((10, 70), (20, 90))])
        np.testing.assert_array_equal(outs[0], canvas[0:40, 0:40])
        np.testing.assert_array_equal(outs[1], canvas[10:70, 20:90])
    finally:
        p.close()


def test_budget_smaller_than_the_region_still_returns_correct_pixels():
    tiles = _tiles()
    canvas = _canvas(tiles)
    data = mosaic_czi_bytes(tiles, compression=6, hilo=True)
    p = CziPyramidReader(oc.get_codec("czi").open(data), decoded_cache_bytes=3 * 32 * 32 * 2)
    try:
        for _ in range(3):
            got = p.read_region(0)
            np.testing.assert_array_equal(got, canvas)
        assert p.tile_cache.bytes <= p.tile_cache.max_bytes
        assert p.tile_cache.evictions > 0
    finally:
        p.close()


def test_invalidate_cache_empties_it_and_decodes_again():
    tiles = _tiles()
    data = mosaic_czi_bytes(tiles, compression=6, hilo=True)
    p = CziPyramidReader(oc.get_codec("czi").open(data), decoded_cache_bytes=1 << 20)
    try:
        p.read_region(0, y=(0, 32), x=(0, 32))
        assert len(p.tile_cache) == 1
        p.invalidate_cache()
        assert len(p.tile_cache) == 0
        p.read_region(0, y=(0, 32), x=(0, 32))
        assert p.region_stats["cache_misses"] == 1
    finally:
        p.close()


def test_concurrent_readers_share_the_cache_and_agree():
    tiles = _tiles()
    canvas = _canvas(tiles)
    data = mosaic_czi_bytes(tiles, compression=6, hilo=True)
    p = CziPyramidReader(oc.get_codec("czi").open(data), decoded_cache_bytes=1 << 20)
    try:
        boxes = [(0, 64, 0, 64), (32, 96, 32, 96), (0, 128, 0, 128), (64, 128, 0, 64)] * 4

        def job(box):
            y0, y1, x0, x1 = box
            return box, p.read_region(0, y=(y0, y1), x=(x0, x1))

        with ThreadPoolExecutor(8) as ex:
            for (y0, y1, x0, x1), out in ex.map(job, boxes):
                np.testing.assert_array_equal(out, canvas[y0:y1, x0:x1])
        assert p.tile_cache.stores == 16
    finally:
        p.close()


def test_decode_failure_releases_claims():
    tiles = _tiles()
    data = mosaic_czi_bytes(tiles, compression=6, hilo=True)
    r = oc.get_codec("czi").open(data)
    p = CziPyramidReader(r, decoded_cache_bytes=1 << 20)
    try:
        original = r._decode_one

        def boom(entry, **kw):
            raise RuntimeError("boom")

        r._decode_one = boom
        with pytest.raises(RuntimeError, match="boom"):
            p.read_region(0, y=(0, 64), x=(0, 64))
        r._decode_one = original
        # Claims were released: a fresh read decodes and succeeds.
        out = p.read_region(0, y=(0, 64), x=(0, 64))
        assert out.shape == (64, 64) and p.region_stats["cache_misses"] == 4
    finally:
        p.close()
