"""HTTPDataSource read-ahead + covering-cache regression tests.

Two related features:

* **Speculative read-ahead**. When ``read_at(offset, n)`` misses cache
  and ``n`` is small, the implementation issues a larger Range
  request (up to ``readahead_window`` bytes). Subsequent small reads
  in the same neighborhood hit the cache instead of forcing fresh
  RTTs.

* **Covering-cache lookup**. When ``read_at(off, n)`` misses the exact
  ``(off, n)`` cache key, the LRU is scanned for any blob whose
  ``[c_off, c_off + c_len)`` covers the requested range; if found,
  slice it out and cache the exact view too.

The tests use the existing range-supporting HTTP server in
``_range_http_server.py`` and count requests on the server side —
the only reliable way to verify the prefetch path is actually
saving round-trips.
"""

from __future__ import annotations

import pathlib
import sys
from contextlib import contextmanager

import pytest

# The range-http-server helper is a non-test module; import it directly
# from the tests directory.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _range_http_server import range_http_server  # noqa: E402

from opencodecs._tiff_http import HTTPDataSource


@contextmanager
def _served_file(tmp_path, content: bytes, name="probe.bin"):
    """Drop ``content`` into ``tmp_path``, serve via the test HTTP server,
    yield ``(url, tracker)``."""
    p = tmp_path / name
    p.write_bytes(content)
    with range_http_server(tmp_path) as (base, tracker):
        yield f"{base}/{name}", tracker


# ---------- speculative read-ahead ----------


def test_readahead_serves_subsequent_small_reads_from_cache(tmp_path):
    """A 2 KB read followed by 5 nearby 1 KB reads should hit the
    server exactly once for the data fetch — read-ahead caches the
    surrounding window."""
    payload = bytes(range(256)) * 256  # 64 KB
    with _served_file(tmp_path, payload) as (url, tracker):
        ds = HTTPDataSource(
            url, prefetch_bytes=0,    # disable head prefetch for clarity
            readahead_threshold=8 * 1024,
            readahead_window=32 * 1024,
        )
        # Trigger size discovery first (one HEAD-style request).
        baseline = tracker.requests

        # First read: 2 KB at offset 0. Should fetch 32 KB.
        b0 = ds.read_at(0, 2048)
        assert b0 == payload[:2048]
        # Subsequent small reads within the 32 KB window should be free.
        b1 = ds.read_at(2048, 1024)
        b2 = ds.read_at(3072, 1024)
        b3 = ds.read_at(4096, 4096)
        b4 = ds.read_at(8192, 1024)
        assert b1 == payload[2048:3072]
        assert b2 == payload[3072:4096]
        assert b3 == payload[4096:8192]
        assert b4 == payload[8192:9216]

        # One data request for the 32 KB window after the size probe.
        # ``baseline`` accounts for the request the size discovery made
        # (it may have already happened on construction). The total
        # request count after the four small reads must be at most
        # baseline + 1 (the read-ahead fetch).
        assert tracker.requests <= baseline + 1, (
            f"read-ahead leaked: served {tracker.requests - baseline} "
            f"requests for 5 reads inside one 32 KB window"
        )
        ds.close()


def test_readahead_disabled_falls_back_to_per_call_requests(tmp_path):
    """With ``readahead_window=0`` each small read should round-trip
    to the server — verifies the new code path is opt-out friendly."""
    payload = bytes(range(256)) * 256
    with _served_file(tmp_path, payload) as (url, tracker):
        ds = HTTPDataSource(
            url, prefetch_bytes=0, readahead_window=0,
        )
        baseline = tracker.requests

        ds.read_at(0, 2048)
        ds.read_at(2048, 1024)
        ds.read_at(3072, 1024)

        # Three reads should hit the server three times.
        assert tracker.requests - baseline >= 3
        ds.close()


def test_readahead_doesnt_extend_large_reads(tmp_path):
    """Tile-sized reads (>= threshold) should NOT be extended — we'd
    waste bandwidth on data the caller probably won't touch."""
    payload = b"\x00" * (200 * 1024)
    with _served_file(tmp_path, payload) as (url, tracker):
        ds = HTTPDataSource(
            url, prefetch_bytes=0,
            readahead_threshold=8 * 1024,
            readahead_window=64 * 1024,
        )
        ds.read_at(0, 100 * 1024)
        # The big read should have fetched exactly its requested
        # length, NOT been bumped up to the readahead window.
        assert tracker.bytes_served <= 100 * 1024 + 1024  # +1 KB headroom
        ds.close()


# ---------- covering-cache lookup ----------


def test_covering_cache_serves_inner_range(tmp_path):
    """A big read followed by a small read inside the same range should
    hit cache via the covering lookup."""
    payload = bytes(range(256)) * 256
    with _served_file(tmp_path, payload) as (url, tracker):
        ds = HTTPDataSource(
            url, prefetch_bytes=0, readahead_window=0,
        )
        baseline = tracker.requests
        # Big read at offset 1000.
        big = ds.read_at(1000, 20 * 1024)
        assert big == payload[1000:1000 + 20 * 1024]
        big_reqs = tracker.requests - baseline

        # Inner read with a different offset and length should NOT
        # round-trip — the covering lookup slices the cached blob.
        inner = ds.read_at(5000, 4096)
        assert inner == payload[5000:9096]
        assert tracker.requests - baseline == big_reqs, (
            "covering-cache miss: inner read forced an extra request"
        )
        ds.close()


def test_covering_cache_skips_partial_overlaps(tmp_path):
    """If a cached blob partially overlaps the requested range but
    doesn't fully cover it, we MUST fetch from the server (no silent
    truncation)."""
    payload = bytes(range(256)) * 256
    with _served_file(tmp_path, payload) as (url, tracker):
        ds = HTTPDataSource(
            url, prefetch_bytes=0, readahead_window=0,
        )
        # Cache bytes [1000, 5000).
        ds.read_at(1000, 4000)
        before = tracker.requests
        # Request [4000, 6000) — extends past the cached range.
        spanning = ds.read_at(4000, 2000)
        assert spanning == payload[4000:6000]
        assert tracker.requests > before, (
            "covering cache mis-served a partial overlap"
        )
        ds.close()


# ---------- read_many hits the same paths ----------


def test_read_many_uses_covering_cache(tmp_path):
    """A batch read after a single big read should be free."""
    payload = bytes(range(256)) * 256
    with _served_file(tmp_path, payload) as (url, tracker):
        ds = HTTPDataSource(
            url, prefetch_bytes=0, readahead_window=0,
        )
        ds.read_at(0, 40 * 1024)
        before = tracker.requests
        out = ds.read_many([(1024, 512), (8192, 1024), (16384, 2048)])
        assert out == [
            payload[1024:1536], payload[8192:9216], payload[16384:18432]
        ]
        assert tracker.requests == before, (
            "read_many forced extra requests despite covering cache"
        )
        ds.close()
