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

Adjacent ranges are coalesced by read_many. Small sequential misses can
trigger adaptive read-ahead; cached ranges use an interval index. Responses
must honor Range: a whole-file HTTP 200 response is rejected before its body
is consumed. Pass opener= for caller-managed redirects or authentication.
"""

from __future__ import annotations

import concurrent.futures
import http.client
import os
import re
import urllib.error
import threading
import urllib.parse
import urllib.request
from collections import OrderedDict
from typing import Any, Sequence

from .core.io import O_BINARY, pread_all
from .core.io import DataSource, Range, coalesce_ranges
from .core._range_index import RangeIndex


class RangeResponseError(ValueError):
    """A server response cannot satisfy the requested byte range safely."""


def _source_pool_reads(source, ranges, read, name):
    """Share one source pool and join this call's work before propagating errors."""
    futures = []
    try:
        with source._pool_lock:
            if source._pool_closed:
                raise ValueError("data source is closed")
            if source._pool is None:
                source._pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=source._max_workers, thread_name_prefix=name)
            for offset, length in ranges:
                futures.append(source._pool.submit(read, offset, length))
        return [future.result() for future in futures]
    except BaseException:
        for future in futures:
            future.cancel()
        for future in futures:
            try:
                future.exception()
            except concurrent.futures.CancelledError:
                pass
        raise


def _close_source_pool(source):
    with source._pool_lock:
        source._pool_closed = True
        pool, source._pool = source._pool, None
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)


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
        # Speculative read-ahead. When a ``read_at(off, n)`` call
        # misses cache and ``n`` is small (a FITS header block, a TIFF
        # IFD, an HDF5 B-tree node), fetch ``readahead_window`` bytes
        # instead so subsequent small reads in the same region hit
        # the covering cache without a fresh HTTP round-trip.
        #
        # OFF by default because read-ahead trades bytes for RTTs: a
        # workload that opens a file then makes one small read pays
        # ``readahead_window`` bytes of wasted bandwidth even though
        # it never needed the extra data. Opt in when you know the
        # access pattern is sequential-ish (FITS HDU walk, h5py
        # B-tree traversal) by passing ``readahead_window=65536`` (or
        # whatever size matches your locality).
        readahead_threshold: int = 16 * 1024,
        readahead_window: int = 0,
        # Adaptive read-ahead. ``readahead_window`` above is the fixed
        # always-on window (default 0 = off); the adaptive path watches
        # the access pattern and auto-enables a ``adaptive_window``-byte
        # read-ahead after seeing N=``adaptive_streak_threshold``
        # consecutive small cache misses where each falls within
        # ``adaptive_locality`` of the previous one (the kind of pattern
        # FITS HDU walks, h5py B-tree traversal, and TIFF IFD chains
        # produce). Set ``adaptive_window=0`` to disable.
        adaptive_window: int = 64 * 1024,
        adaptive_streak_threshold: int = 3,
        adaptive_locality: int = 64 * 1024,
        # Sequential-access mode (libvips-style streaming). When set to
        # "sequential", read_at uses a single rolling buffer rather than
        # the LRU cache — assumes the caller reads the file ~once in
        # forward order (raster-scan over a huge COG, FITS HDU walk,
        # NDTiff frame stream). Memory stays bounded to
        # ``sequential_chunk_bytes`` regardless of file size; backward
        # seeks still work but invalidate the buffer and issue a fresh
        # Range request. Default "random" keeps the LRU + adaptive
        # read-ahead path for the bulk of callers (TIFF IFD walks,
        # pyramid level switches, FITS table scans).
        access: str = "random",
        sequential_chunk_bytes: int = 4 * 1024 * 1024,
    ):
        self.url = url
        self.timeout = float(timeout)
        self.headers = dict(headers or {})
        self._opener = opener
        self._lock = threading.Lock()
        # Least recently used (LRU) payloads, with an exact-key fast path
        # and a separate interval index for covering-range lookups.
        self._cache: "OrderedDict[tuple[int, int], bytes]" = OrderedDict()
        self._range_index = RangeIndex()
        self._cache_max = int(cache_bytes)
        self._cache_used = 0

        self._total_size: int | None = None  # discovered on first read
        self._total_requests = 0  # for benchmarking / observability
        self._total_bytes_fetched = 0

        self._max_workers = int(max_workers)
        self._coalesce_gap = int(coalesce_gap)
        self._coalesce_max = int(coalesce_max)
        self._readahead_threshold = int(readahead_threshold)
        self._readahead_window = int(readahead_window)
        # Adaptive read-ahead bookkeeping. The streak counter advances
        # whenever a small cache miss is "near" the previous one; once
        # it crosses the threshold, ``_should_adaptive_readahead``
        # returns True and the next miss-path read gets bumped to
        # ``_adaptive_window`` bytes.
        self._adaptive_window = int(adaptive_window)
        self._adaptive_streak_threshold = int(adaptive_streak_threshold)
        self._adaptive_locality = int(adaptive_locality)
        self._adaptive_last_end: int | None = None
        self._adaptive_streak = 0

        if access not in ("random", "sequential"):
            raise ValueError(
                f"access must be 'random' or 'sequential', got {access!r}"
            )
        self._access = access
        self._sequential_chunk_bytes = int(sequential_chunk_bytes)
        # Single rolling buffer that covers [_seq_buf_off, _seq_buf_off + len).
        # Sequential mode keeps this in lieu of the LRU.
        self._seq_buf: bytes | None = None
        self._seq_buf_off: int = 0
        # Bookkeeping for sequential observability — analogous to
        # _total_requests/_total_bytes_fetched for the LRU path.
        self._sequential_refills = 0
        self._sequential_backward_seeks = 0
        # Lazily-created thread pool for read_many. We don't pay the
        # 8-thread startup cost on single-read workflows.
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()
        self._pool_closed = False

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
            except RangeResponseError:
                raise
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
        if offset < 0 or n < 0:
            raise ValueError("negative byte range")
        if n == 0:
            return b""
        end = offset + n

        # Serve from prefetched head if the requested range fits.
        if self._prefetch_buffer and end <= len(self._prefetch_buffer):
            return self._prefetch_buffer[offset:end]

        if self._access == "sequential":
            return self._read_at_sequential(offset, n)

        # Exact-range LRU lookup.
        with self._lock:
            cached = self._cache.get((offset, n))
            if cached is not None:
                self._cache.move_to_end((offset, n))
                return cached
            # Covering-range lookup: an earlier read-ahead may have
            # fetched a bigger blob that includes this range.
            covering = self._covering_lookup(offset, n)
            if covering is not None:
                return covering

        # Speculative read-ahead. Small misses get extended to a
        # bigger fetch so the next adjacent small read hits the
        # covering cache without a round-trip. Two trigger paths:
        #
        #   1. Always-on ``readahead_window`` — user opted in by
        #      passing the kwarg at construction (off by default).
        #
        #   2. Adaptive: the streak counter (advanced inside
        #      ``_observe_miss`` below) crossed its threshold,
        #      meaning we've seen N consecutive nearby small misses.
        #      The next miss-path read gets the bigger fetch.
        #
        # Tile/chunk-sized requests (>= threshold) are never extended
        # — over-fetching wastes bandwidth on data the caller probably
        # won't touch.
        with self._lock:
            fetch_n = n
            is_small = n <= self._readahead_threshold
            explicit = self._readahead_window > 0 and self._readahead_window > n
            adaptive = (
                self._adaptive_window > 0
                and self._adaptive_streak >= self._adaptive_streak_threshold
                and self._adaptive_window > n
                and self._adaptive_last_end is not None
                and abs(offset - self._adaptive_last_end) <= self._adaptive_locality
            )
            if is_small and (explicit or adaptive):
                fetch_n = max(
                    self._readahead_window if explicit else 0,
                    self._adaptive_window if adaptive else 0,
                )
                # Don't fetch past EOF when the file size is known.
                if self._total_size is not None:
                    fetch_n = min(fetch_n, max(0, self._total_size - offset))
                if fetch_n < n:
                    fetch_n = n

            # Streak bookkeeping for the adaptive path. Done BEFORE the
            # actual fetch so the *current* miss counts toward the streak
            # for the next call. We only count small reads; large reads
            # are tiles/chunks and signal a different access pattern.
            if is_small:
                self._observe_miss(offset, n)
            else:
                self._adaptive_streak = 0
                self._adaptive_last_end = None

        chunk = self._range_request(offset, fetch_n)
        with self._lock:
            self._cache_put((offset, len(chunk)), chunk)
            if len(chunk) > n:
                # Slice + cache the exact requested view so future
                # ``read_at(offset, n)`` calls hit the exact-key path.
                exact = chunk[:n]
                self._cache_put((offset, n), exact)
                return exact
        return chunk

    def _read_at_sequential(self, offset: int, n: int) -> bytes:
        """Serve ``[offset, offset+n)`` from a single rolling buffer.

        Sequential mode replaces the LRU + adaptive-readahead machinery
        with one ``sequential_chunk_bytes``-sized buffer that slides
        forward as the caller reads. Memory is bounded to one chunk
        regardless of file size.

        Cases:

        * Hit: ``[offset, offset+n)`` lies within the current buffer
          (``[_seq_buf_off, _seq_buf_off+len(_seq_buf))``). Slice and
          return. No network call.
        * Miss (forward): the request starts at or past the current
          buffer end, or starts before the buffer (backward seek). Fetch
          a fresh ``max(sequential_chunk_bytes, n)`` bytes starting at
          ``offset`` and replace the buffer. Bytes behind ``offset`` are
          dropped — sequential mode trades random-access locality for
          bounded memory.
        """
        end = offset + n
        buf = self._seq_buf
        if buf is not None:
            bo = self._seq_buf_off
            be = bo + len(buf)
            if bo <= offset and end <= be:
                return buf[offset - bo : offset - bo + n]
            if offset < bo:
                # Backward seek — count for observability so users can
                # spot a workload that's actually random and would
                # benefit from access="random".
                self._sequential_backward_seeks += 1

        chunk_n = max(self._sequential_chunk_bytes, n)
        if self._total_size is not None:
            chunk_n = min(chunk_n, max(0, self._total_size - offset))
            if chunk_n < n:
                # Caller asked for past-EOF; let the range request
                # surface the underlying error rather than silently
                # truncating.
                chunk_n = n
        chunk = self._range_request(offset, chunk_n)
        self._seq_buf = chunk
        self._seq_buf_off = offset
        self._sequential_refills += 1
        return chunk[:n] if len(chunk) >= n else chunk

    def _observe_miss(self, offset: int, n: int) -> None:
        """Update the adaptive-read-ahead streak counter.

        Called from inside the cache-miss path of ``read_at`` after we
        know the request isn't a tile / chunk read (``n`` is small).
        Advances the streak when the new miss is "near" the previous
        one (offset within ``adaptive_locality`` of the prior read's
        end); resets to 1 otherwise. Once the streak crosses
        ``adaptive_streak_threshold``, the *next* small miss gets the
        bigger fetch.
        """
        if self._adaptive_window <= 0:
            return
        last_end = self._adaptive_last_end
        if (last_end is not None
                and abs(offset - last_end) <= self._adaptive_locality):
            self._adaptive_streak += 1
        else:
            self._adaptive_streak = 1
        self._adaptive_last_end = offset + n

    def _covering_lookup(self, offset: int, n: int) -> bytes | None:
        """Return bytes for ``[offset, offset+n)`` from any cached blob
        that fully covers the range, else None. Caller holds ``self._lock``.

        Length buckets avoid scanning unrelated small entries. The cache
        remains the authority for payload ownership and eviction order.
        """
        key = self._range_index.covering(offset, n)
        if key is None:
            return None
        blob = self._cache[key]
        lo = offset - key[0]
        view = blob[lo:lo + n]
        self._cache.move_to_end(key)
        self._cache_put((offset, n), view)
        return view

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
                if off < 0 or length < 0:
                    raise ValueError("negative byte range")
                if length == 0:
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
                # Same covering-cache hit path read_at uses — a prior
                # read-ahead may have already pulled this range in.
                covering = self._covering_lookup(off, length)
                if covering is not None:
                    out[i] = covering
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
            fetched = _source_pool_reads(
                self, merged, self._range_request, "opencodecs-http")

        # Step 4 + 5: split, fill, cache.
        with self._lock:
            for (m_off, m_len), data in zip(merged, fetched):
                # Cache the merged blob too — future ``read_at(m_off, m_len)``
                # will hit. Also stash sub-slices keyed by their requested
                # (offset, length) so the next single-read hits as well.
                self._cache_put((m_off, len(data)), data)
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
                    (miss_ranges[i_local][0], len(piece)),
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
            self._range_index.clear()
            self._prefetch_buffer = b""
            self._seq_buf = None
            self._cache_used = 0
        _close_source_pool(self)
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
            "access": self._access,
            "sequential_refills": self._sequential_refills,
            "sequential_backward_seeks": self._sequential_backward_seeks,
            "sequential_buf_bytes": (
                len(self._seq_buf) if self._seq_buf is not None else 0
            ),
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
                data = self._read_range_response(resp, offset, n)
                # If the server signals it will close the connection
                # (HTTP/1.0 default, or explicit "Connection: close"),
                # drop our cached conn so the next request opens a
                # fresh one. Otherwise we'd send to a half-closed
                # socket and hit the timeout.
                if resp.status == 416 or resp.will_close:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    self._conn_local.conn = None
                return data
            except RangeResponseError:
                resp.close()
                conn.close()
                self._conn_local.conn = None
                raise
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
        headers.setdefault("Accept-Encoding", "identity")
        req = urllib.request.Request(self.url, headers=headers)
        try:
            resp = self._opener.open(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            if exc.code != 416:
                raise
            resp = exc
        with resp:
            return self._read_range_response(resp, offset, n)

    def _read_range_response(self, resp, offset: int, n: int) -> bytes:
        """Validate both transports before bytes enter an offset cache.

        Reject whole-file responses without consuming their potentially
        huge bodies. A range source must not silently become a download.
        """
        with self._lock:
            self._total_requests += 1
        cr = resp.headers.get("Content-Range", "")
        if resp.status == 416:
            match = re.fullmatch(r"bytes \*/(\d+)", cr)
            if match:
                self._total_size = int(match[1])
                if offset < self._total_size:
                    raise RangeResponseError("server rejected a satisfiable byte range")
            # Error bodies need not be bounded by the requested range.
            resp.close()
            return b""
        if resp.status != 206:
            raise RangeResponseError(
                f"expected HTTP 206 for byte range, got {resp.status}; "
                "the server must support Range requests")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", cr)
        if match is None:
            raise RangeResponseError("missing or invalid Content-Range")
        start, end = int(match[1]), int(match[2])
        total = None if match[3] == "*" else int(match[3])
        expected_end = offset + n - 1
        if total is not None:
            expected_end = min(expected_end, total - 1)
        if start != offset or end < start or end != expected_end:
            raise RangeResponseError(f"Content-Range does not match request: {cr}")
        length = end - start + 1
        content_length = resp.headers.get("Content-Length")
        if content_length is not None and (
                re.fullmatch(r"[0-9]+", content_length.strip()) is None or
                int(content_length) != length):
            raise RangeResponseError("Content-Length does not match Content-Range")
        if resp.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise RangeResponseError("encoded range response is not byte-addressable")
        data = resp.read(length + 1)
        with self._lock:
            self._total_bytes_fetched += len(data)
        if len(data) != length:
            raise RangeResponseError("truncated or oversized range response")
        if total is not None:
            self._total_size = total
        return data

    def _cache_put(self, key: tuple[int, int], value: bytes) -> None:
        """Insert into LRU; evict from the back until under budget."""
        if not value:
            return
        existing = self._cache.pop(key, None)
        if existing is not None:
            self._cache_used -= len(existing)
        self._cache[key] = value
        self._range_index.add(key)
        self._cache_used += len(value)
        while self._cache_used > self._cache_max and self._cache:
            _k, _v = self._cache.popitem(last=False)
            self._cache_used -= len(_v)
            self._range_index.discard(_k)


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
        flags = os.O_RDONLY | O_BINARY
        self._fd = os.open(self.path, flags)
        self._has_pread = hasattr(os, "pread")
        self._lock = None if self._has_pread else threading.Lock()
        self._total_requests = 0
        self._max_workers = int(max_workers)
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()
        self._pool_closed = False
        try:
            self.size = os.fstat(self._fd).st_size
        except OSError:  # pragma: no cover - shouldn't happen on open FD
            self.size = None

    def read_at(self, offset: int, n: int) -> bytes:
        self._total_requests += 1
        return pread_all(self._fd, n, offset, self._lock)

    def read_many(self, ranges: Sequence[Range]) -> list[bytes]:
        """Parallel ``pread`` fan-out on POSIX (where it's thread-safe).

        On Windows we serialize through the lock anyway, so we don't
        spin up a pool — plain serial is the same speed.
        """
        if not ranges:
            return []
        if not self._has_pread or self._max_workers <= 1 or len(ranges) == 1:
            return [self.read_at(o, n) for o, n in ranges]
        return _source_pool_reads(self, ranges, self.read_at, "opencodecs-pread")

    def close(self) -> None:
        _close_source_pool(self)
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
