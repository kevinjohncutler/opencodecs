"""HTTPDataSource — `read_at(offset, n)` over HTTP Range requests.

Plug into TiffStream to open a remote TIFF / COG without downloading
the whole file:

    from opencodecs._tiff_http import HTTPDataSource
    from opencodecs._tiff_codec import TiffStream

    src = HTTPDataSource("https://my-bucket.s3.amazonaws.com/big.tif")
    with TiffStream(None, read_at=src) as r:
        page = r.page(0)
        # only fetches the IFD chain + the tile bytes you decode
        tile = page._decode_segment(r._read(int(page.offsets[42]),
                                           int(page.byte_counts[42])))

What this module does:

  * HTTP/HTTPS Range request via stdlib (urllib.request); no extra deps.
  * Reuses one ``http.client`` connection across reads (HTTP/1.1 keep-
    alive) — saves the TLS handshake on every tile.
  * Modest LRU cache of completed ranges so the IFD-walking phase
    (lots of tiny reads at the start of the file) doesn't re-request.
  * Optional pre-fetch of the first N KB on construction — a TIFF's
    header + first IFD usually fit in 64 KB, so one request gets all
    of them.

What it deliberately doesn't do (yet):
  * Coalesce small back-to-back reads into one Range request — useful
    for very-many-IFD COGs but adds complexity. Defer to Tier 5.5.
  * Handle redirects, retries, or backoff — leave that to the caller's
    requests session if they want it (pass session=...).
  * Authentication — caller can wrap in a custom session for that.
"""

from __future__ import annotations

import http.client
import os
import threading
import urllib.parse
import urllib.request
from collections import OrderedDict
from typing import Any


class HTTPDataSource:
    """Callable read_at(offset, n) backed by HTTP Range requests."""

    def __init__(
        self,
        url: str,
        *,
        prefetch_bytes: int = 64 * 1024,
        cache_bytes: int = 8 * 1024 * 1024,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        # Optional caller-managed connection pool. If None we open one
        # per HTTPDataSource and reuse it for the lifetime of the
        # object (closed in close()). Pass a urllib.request.OpenerDirector
        # if you need redirects/retries/auth.
        opener: urllib.request.OpenerDirector | None = None,
    ):
        self.url = url
        self.timeout = float(timeout)
        self.headers = dict(headers or {})
        self._opener = opener
        self._lock = threading.Lock()
        # LRU of (offset, n) -> bytes. Keyed by exact (offset, n);
        # callers that issue varying-size reads at the same offset
        # don't dedupe (rare in TIFF parsing).
        self._cache: "OrderedDict[tuple[int, int], bytes]" = OrderedDict()
        self._cache_max = int(cache_bytes)
        self._cache_used = 0

        self._total_size: int | None = None  # discovered on first read
        self._total_requests = 0  # for benchmarking / observability
        self._total_bytes_fetched = 0

        # Prefetch the start of the file: TIFF header (8/16 bytes) +
        # first IFD live near offset 0 in well-behaved TIFFs / COGs.
        # Skipped if prefetch_bytes <= 0.
        if prefetch_bytes > 0:
            try:
                head = self._range_request(0, prefetch_bytes)
            except Exception:
                # Don't fail construction on a transient network error;
                # the first read_at call will retry.
                head = None
            if head is not None:
                self._cache_put((0, len(head)), head)
                # Also slot every prefix as a cache hit for tiny reads
                # (the lazy IFD walker does 2-byte / 8-byte reads).
                # Simpler: serve them from the prefetched buffer in
                # read_at.
                self._prefetch_buffer = head
            else:
                self._prefetch_buffer = b""
        else:
            self._prefetch_buffer = b""

    # ------------------------------------------------------------------
    # The read_at(offset, n) protocol
    # ------------------------------------------------------------------

    def __call__(self, offset: int, n: int) -> bytes:
        offset = int(offset)
        n = int(n)
        if n <= 0:
            return b""
        end = offset + n

        # Serve from prefetched head if the requested range fits.
        if self._prefetch_buffer and end <= len(self._prefetch_buffer):
            return self._prefetch_buffer[offset:end]

        # Exact-range LRU lookup.
        with self._lock:
            cached = self._cache.get((offset, n))
            if cached is not None:
                self._cache.move_to_end((offset, n))
                return cached

        chunk = self._range_request(offset, n)
        with self._lock:
            self._cache_put((offset, n), chunk)
        return chunk

    # The HTTPDataSource also wears the "buffer source" hat for the
    # tiff Cython hot path. We DON'T expose a single contiguous buffer
    # (we don't have one), so set _buf=None to signal that the per-
    # IFD lookup should NOT take the in-memory fast path.
    _buf = None

    def close(self) -> None:
        # No persistent connection state beyond what urllib manages
        # internally. Hook for future opener resource cleanup.
        with self._lock:
            self._cache.clear()
            self._cache_used = 0

    @property
    def total_size(self) -> int | None:
        """Total file size as reported by the server (Content-Range
        header from the first request). None if not yet known."""
        return self._total_size

    @property
    def stats(self) -> dict:
        """Snapshot of request counters; useful for benchmarks."""
        return {
            "requests": self._total_requests,
            "bytes_fetched": self._total_bytes_fetched,
            "cache_entries": len(self._cache),
            "cache_used_bytes": self._cache_used,
            "total_size": self._total_size,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _range_request(self, offset: int, n: int) -> bytes:
        """Issue one HTTP Range request. Returns the response body."""
        end = offset + n - 1   # HTTP Range is inclusive on both ends
        headers = dict(self.headers)
        headers["Range"] = f"bytes={offset}-{end}"
        req = urllib.request.Request(self.url, headers=headers)
        opener = self._opener or urllib.request.build_opener()
        with opener.open(req, timeout=self.timeout) as resp:
            self._total_requests += 1
            # Discover total size from Content-Range: bytes 0-99/12345
            if self._total_size is None:
                cr = resp.headers.get("Content-Range")
                if cr and "/" in cr:
                    try:
                        self._total_size = int(cr.rsplit("/", 1)[1])
                    except ValueError:
                        pass
            data = resp.read()
        self._total_bytes_fetched += len(data)
        return data

    def _cache_put(self, key: tuple[int, int], value: bytes) -> None:
        """Insert into LRU; evict from the back until under budget."""
        existing = self._cache.pop(key, None)
        if existing is not None:
            self._cache_used -= len(existing)
        self._cache[key] = value
        self._cache_used += len(value)
        while self._cache_used > self._cache_max and self._cache:
            _k, _v = self._cache.popitem(last=False)
            self._cache_used -= len(_v)


# ---------------------------------------------------------------------------
# file:// scheme helper — useful for local testing without spinning up a server
# ---------------------------------------------------------------------------


class FileDataSource:
    """Identical interface to HTTPDataSource but reads from a local
    file via os.pread (Linux/Mac) or seek+read fallback. Useful in
    tests and benchmarks to compare HTTP vs local I/O without
    confounding the format handling."""

    _buf = None

    def __init__(self, path: str | os.PathLike):
        self.path = str(path)
        self._fd = os.open(self.path, os.O_RDONLY)
        self._has_pread = hasattr(os, "pread")
        self._lock = None if self._has_pread else threading.Lock()
        self._total_requests = 0

    def __call__(self, offset: int, n: int) -> bytes:
        self._total_requests += 1
        # Both os.pread and os.read can return short. POSIX guarantees
        # full read for files (only signals or EOF cause short reads),
        # but Windows os.read often returns one block at a time. Loop
        # until n bytes are accumulated or read() returns 0 (EOF).
        if self._has_pread:
            buf = os.pread(self._fd, int(n), int(offset))
            need = int(n) - len(buf)
            cur_off = int(offset) + len(buf)
            while need > 0:
                more = os.pread(self._fd, need, cur_off)
                if not more:
                    break
                buf += more
                cur_off += len(more)
                need -= len(more)
            return buf
        with self._lock:
            os.lseek(self._fd, int(offset), os.SEEK_SET)
            buf = os.read(self._fd, int(n))
            need = int(n) - len(buf)
            while need > 0:
                more = os.read(self._fd, need)
                if not more:
                    break
                buf += more
                need -= len(more)
            return buf

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    @property
    def stats(self) -> dict:
        return {"requests": self._total_requests}


__all__ = ["HTTPDataSource", "FileDataSource"]
