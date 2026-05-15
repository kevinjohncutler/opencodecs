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

import concurrent.futures
import http.client
import os
import threading
import urllib.parse
import urllib.request
from collections import OrderedDict
from typing import Any, Sequence

from .core.io import DataSource, Range, coalesce_ranges


class HTTPDataSource(DataSource):
    """Random-access bytes source backed by HTTP Range requests.

    Subclasses :class:`DataSource`, so ``ds(offset, n)`` keeps working
    as before *and* you can call ``ds.read_many(ranges)`` to fan out
    a batch of requests in parallel (thread pool, automatic
    coalescing of adjacent ranges via
    :func:`opencodecs.core.io.coalesce_ranges`).
    """

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
        # read_many parallel fetch knobs. Tuned for typical S3/GCS:
        # 8 concurrent Range requests saturates one TCP connection per
        # core on a 1 Gbit link without ringing the server.
        max_workers: int = 8,
        coalesce_gap: int = 16 * 1024,
        coalesce_max: int = 4 * 1024 * 1024,
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

        self._max_workers = int(max_workers)
        self._coalesce_gap = int(coalesce_gap)
        self._coalesce_max = int(coalesce_max)
        # Lazily-created thread pool for read_many. We don't pay the
        # 8-thread startup cost on single-read workflows.
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None

        # Persistent http.client connection pool — keyed by thread id.
        # urllib.request opens a fresh TCP connection per call, which
        # not only adds a handshake per read (~1ms loopback, 50+ms WAN)
        # but also stresses the kernel's TIME_WAIT recycling on rapid
        # sequential reads (e.g. h5py walking a B-tree). One persistent
        # HTTP/1.1 keep-alive connection per worker thread eliminates
        # both. Initialized lazily in _range_request.
        parsed = urllib.parse.urlsplit(self.url)
        self._scheme = parsed.scheme
        self._netloc = parsed.netloc
        self._path_q = (
            parsed.path + ("?" + parsed.query if parsed.query else "")
        ) or "/"
        self._conn_local = threading.local()

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

    def read_at(self, offset: int, n: int) -> bytes:
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

    def read_many(self, ranges: Sequence[Range]) -> list[bytes]:
        """Fetch many ranges in parallel; results returned in input order.

        Behavior:
          1. Serve from prefetch buffer / LRU where possible (no network).
          2. Coalesce remaining nearby ranges into bigger fetches.
          3. Issue the coalesced fetches concurrently on the thread pool.
          4. Slice the merged responses to fill each requested range.
          5. Cache merged blobs so subsequent ``read_at`` calls hit.

        Empty input returns an empty list.
        """
        n_in = len(ranges)
        if n_in == 0:
            return []

        out: list[bytes | None] = [None] * n_in
        # Step 1: cache / prefetch lookups.
        miss_idx: list[int] = []
        miss_ranges: list[Range] = []
        with self._lock:
            for i, (off, length) in enumerate(ranges):
                off = int(off); length = int(length)
                if length <= 0:
                    out[i] = b""
                    continue
                end = off + length
                if (self._prefetch_buffer
                        and end <= len(self._prefetch_buffer)):
                    out[i] = self._prefetch_buffer[off:end]
                    continue
                hit = self._cache.get((off, length))
                if hit is not None:
                    self._cache.move_to_end((off, length))
                    out[i] = hit
                    continue
                miss_idx.append(i)
                miss_ranges.append((off, length))

        if not miss_ranges:
            return [b if b is not None else b"" for b in out]

        # Step 2: coalesce.
        merged, splits = coalesce_ranges(
            miss_ranges,
            max_gap=self._coalesce_gap,
            max_combined=self._coalesce_max,
        )

        # Step 3: parallel fetch of merged ranges.
        if len(merged) == 1 or self._max_workers <= 1:
            fetched = [self._range_request(o, L) for o, L in merged]
        else:
            if self._pool is None:
                self._pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(self._max_workers, len(merged)),
                    thread_name_prefix="opencodecs-http",
                )
            futures = [
                self._pool.submit(self._range_request, o, L)
                for o, L in merged
            ]
            fetched = [f.result() for f in futures]

        # Step 4 + 5: split, fill, cache.
        with self._lock:
            for (m_off, m_len), data in zip(merged, fetched):
                # Cache the merged blob too — future ``read_at(m_off, m_len)``
                # will hit. Also stash sub-slices keyed by their requested
                # (offset, length) so the next single-read hits as well.
                self._cache_put((m_off, m_len), data)
            for i_local, splits_for_orig in enumerate(splits):
                orig_idx = miss_idx[i_local]
                # With non-overlapping inputs each splits[i] has length 1;
                # if a caller passes overlapping ranges we use the first.
                m_idx, s_start, s_end = splits_for_orig[0]
                blob = fetched[m_idx]
                piece = blob[s_start:s_end]
                out[orig_idx] = piece
                # Cache the exact-range view so single read_at(o, n)
                # hits without slicing again.
                self._cache_put(
                    (miss_ranges[i_local][0], miss_ranges[i_local][1]),
                    piece,
                )

        return [b if b is not None else b"" for b in out]

    # The HTTPDataSource also wears the "buffer source" hat for the
    # tiff Cython hot path. We DON'T expose a single contiguous buffer
    # (we don't have one), so set _buf=None to signal that the per-
    # IFD lookup should NOT take the in-memory fast path.
    _buf = None

    def close(self) -> None:
        # Drop the LRU cache, shut down the read_many pool, and close
        # any persistent http.client connection on the main thread.
        # Worker-thread connections are closed by the threading.local
        # finalizer when their thread terminates (which the pool's
        # shutdown takes care of).
        with self._lock:
            self._cache.clear()
            self._cache_used = 0
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        conn = getattr(self._conn_local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._conn_local.conn = None

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

    def _get_conn(self) -> http.client.HTTPConnection:
        """One persistent HTTPConnection per worker thread."""
        conn = getattr(self._conn_local, "conn", None)
        if conn is None:
            if self._scheme == "https":
                conn = http.client.HTTPSConnection(
                    self._netloc, timeout=self.timeout
                )
            else:
                conn = http.client.HTTPConnection(
                    self._netloc, timeout=self.timeout
                )
            self._conn_local.conn = conn
        return conn

    def _range_request(self, offset: int, n: int) -> bytes:
        """Issue one HTTP Range request over a persistent connection."""
        # Caller path 1: a user-supplied opener (auth / redirects /
        # retries). Use it as-is — we don't try to plumb keep-alive
        # through their custom stack.
        if self._opener is not None:
            return self._range_request_via_opener(offset, n)

        end = offset + n - 1
        headers = dict(self.headers)
        headers["Range"] = f"bytes={offset}-{end}"
        headers.setdefault("Connection", "keep-alive")
        headers.setdefault("Accept-Encoding", "identity")

        # http.client connections aren't safe across threads — that's
        # why we keep one per thread via threading.local. One retry on
        # ConnectionError handles the case where a long-idle connection
        # was reaped by the server between our requests.
        last_err: Exception | None = None
        for attempt in (0, 1):
            conn = self._get_conn()
            try:
                conn.request("GET", self._path_q, headers=headers)
                resp = conn.getresponse()
                if resp.status not in (200, 206):
                    err = http.client.HTTPException(
                        f"unexpected status {resp.status} for {self._path_q}"
                    )
                    resp.read()  # drain so the conn can be reused
                    raise err
                if self._total_size is None:
                    cr = resp.getheader("Content-Range")
                    if cr and "/" in cr:
                        try:
                            self._total_size = int(cr.rsplit("/", 1)[1])
                        except ValueError:
                            pass
                    elif resp.status == 200:
                        # Server returned the WHOLE file (didn't honor
                        # Range, or didn't advertise it). The
                        # Content-Length header is the file size in
                        # that case — we may as well learn it.
                        cl = resp.getheader("Content-Length")
                        if cl is not None:
                            try:
                                self._total_size = int(cl)
                            except ValueError:
                                pass
                data = resp.read()
                self._total_requests += 1
                self._total_bytes_fetched += len(data)
                # If the server signals it will close the connection
                # (HTTP/1.0 default, or explicit "Connection: close"),
                # drop our cached conn so the next request opens a
                # fresh one. Otherwise we'd send to a half-closed
                # socket and hit the timeout.
                if resp.will_close:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    self._conn_local.conn = None
                return data
            except (http.client.HTTPException, ConnectionError, OSError) as e:
                last_err = e
                # Stale connection — drop it and retry once.
                try:
                    conn.close()
                except Exception:
                    pass
                self._conn_local.conn = None
                if attempt == 1:
                    raise
        # Unreachable, but keeps type-checker happy.
        raise last_err  # type: ignore[misc]

    def _range_request_via_opener(self, offset: int, n: int) -> bytes:
        """Fallback path when the caller supplied their own urllib
        opener (typically for auth / retries / redirects). No keep-
        alive — urllib closes after each request."""
        end = offset + n - 1
        headers = dict(self.headers)
        headers["Range"] = f"bytes={offset}-{end}"
        req = urllib.request.Request(self.url, headers=headers)
        with self._opener.open(req, timeout=self.timeout) as resp:
            self._total_requests += 1
            if self._total_size is None:
                cr = resp.headers.get("Content-Range")
                if cr and "/" in cr:
                    try:
                        self._total_size = int(cr.rsplit("/", 1)[1])
                    except ValueError:
                        pass
                elif resp.status == 200:
                    cl = resp.headers.get("Content-Length")
                    if cl is not None:
                        try:
                            self._total_size = int(cl)
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


class FileDataSource(DataSource):
    """Same protocol as HTTPDataSource but reads from a local file via
    ``os.pread`` (POSIX) or seek+read fallback (Windows). Useful in
    tests and benchmarks to compare HTTP vs local I/O without
    confounding the format handling.

    ``read_many`` defaults to a serial loop (``max_workers=1``) because
    on a single SSD pread is ~10 µs and thread-pool dispatch
    overhead (~10s of µs) dominates. Pass ``max_workers=4+`` for
    NFS / network shares where each pread takes ms."""

    _buf = None

    def __init__(self, path: str | os.PathLike, *, max_workers: int = 1):
        self.path = str(path)
        # Windows os.open() defaults to TEXT mode: a 0x1A byte (Ctrl-Z)
        # in binary data triggers a soft-EOF mid-file, and CR/LF gets
        # translated. OR in O_BINARY when available (Windows only).
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        self._fd = os.open(self.path, flags)
        self._has_pread = hasattr(os, "pread")
        self._lock = None if self._has_pread else threading.Lock()
        self._total_requests = 0
        self._max_workers = int(max_workers)
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None
        try:
            self.size = os.fstat(self._fd).st_size
        except OSError:  # pragma: no cover - shouldn't happen on open FD
            self.size = None

    def read_at(self, offset: int, n: int) -> bytes:
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

    def read_many(self, ranges: Sequence[Range]) -> list[bytes]:
        """Parallel ``pread`` fan-out on POSIX (where it's thread-safe).

        On Windows we serialize through the lock anyway, so we don't
        spin up a pool — plain serial is the same speed.
        """
        if not ranges:
            return []
        if not self._has_pread or self._max_workers <= 1 or len(ranges) == 1:
            return [self.read_at(o, n) for o, n in ranges]
        if self._pool is None:
            self._pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self._max_workers, len(ranges)),
                thread_name_prefix="opencodecs-pread",
            )
        futures = [
            self._pool.submit(self.read_at, o, n) for o, n in ranges
        ]
        return [f.result() for f in futures]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    @property
    def stats(self) -> dict:
        return {"requests": self._total_requests}


def http_fetch_all(
    url: str,
    *,
    timeout: float = 60.0,
    headers: dict[str, str] | None = None,
) -> bytes:
    """Download a URL fully into bytes.

    Convenience helper for readers whose underlying codec wants the
    whole stream (libjxl, libpng, libtiff-no-range-mode). Falls back
    to a single GET; no Range. Equivalent to HTTPDataSource(url, prefetch=full)
    but without the LRU machinery.

    Use HTTPDataSource for readers that do scattered slice access
    (TIFF tile-by-tile, NDTiff frame-by-frame); use http_fetch_all
    for readers that just want the file as bytes (JXL, single-image
    PNG, full-volume reads).
    """
    req = urllib.request.Request(url, headers=dict(headers or {}))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


__all__ = ["HTTPDataSource", "FileDataSource", "http_fetch_all"]
