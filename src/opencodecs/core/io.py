"""I/O primitives shared across codecs.

Two related substrates, each tuned to a different access pattern:

* ``BackgroundChunkReader`` — sequential, bg-thread chunk producer with
  a bounded queue. Used by streaming codecs (JXL incremental decode,
  some TIFF compressed strips).
* ``DataSource`` — random-access, "give me bytes at offset O of length N"
  abstraction with batched ``read_many`` for parallel fetch. Used by
  container readers (TIFF tiles, CZI sub-blocks, HDF5 chunks, DICOMweb
  frames). The concrete subclasses (FileDataSource / HTTPDataSource)
  live in ``opencodecs._tiff_http`` for historical reasons; this module
  defines the ABC + the cross-cutting helpers (``coalesce_ranges``).
"""

from __future__ import annotations

import os
import queue
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Sequence


_DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024   # 4 MiB
_DEFAULT_PREFETCH_CHUNKS = 4              # up to 16 MiB outstanding


class BackgroundChunkReader:
    """Read a file (or file-like) in chunks on a background thread.

    The bg thread fills a bounded ``queue.Queue`` with ``bytes`` chunks of
    ``chunk_size`` each. The consumer pops via ``next(reader)`` /
    ``reader.get()``. When the file is exhausted, ``get()`` returns
    ``None`` (sentinel) and subsequent calls keep returning ``None``.

    The queue's ``maxsize=prefetch`` provides backpressure: if the
    consumer is slower than the I/O, the bg thread blocks on ``put()``
    instead of buffering unbounded ahead.

    Real I/O+codec overlap depends on the consumer running its codec
    work inside a ``with nogil:`` block (or otherwise releasing the GIL
    during compute). When the consumer is in nogil, the bg thread can
    acquire the GIL and call ``file.read()``, which releases the GIL
    again during the actual syscall. So both happen at once on different
    cores.

    Parameters
    ----------
    src : str | Path | file-like
        Path to open for reading, or an existing file-like with .read().
    chunk_size : int
        Bytes per chunk (default 4 MiB). Larger chunks amortize syscall
        overhead but reduce overlap granularity.
    prefetch : int
        Maximum chunks queued ahead (default 4). Bigger = more overlap
        room but more peak memory.

    Attributes
    ----------
    file_size : int | None
        Total file size if known (for path inputs and seekable files),
        else None.

    Usage
    -----
    >>> with BackgroundChunkReader("foo.jxl") as r:
    ...     while True:
    ...         chunk = r.get()
    ...         if chunk is None: break
    ...         feed_to_codec(chunk)
    """

    def __init__(
        self,
        src,
        *,
        chunk_size: int = _DEFAULT_CHUNK_BYTES,
        prefetch: int = _DEFAULT_PREFETCH_CHUNKS,
    ):
        if chunk_size < 1024:
            raise ValueError(f"chunk_size too small: {chunk_size}")
        if prefetch < 1:
            raise ValueError(f"prefetch must be >= 1: {prefetch}")

        self._chunk_size = int(chunk_size)
        self._prefetch = int(prefetch)
        self._owns_file = False
        self._closed = False

        if isinstance(src, (str, os.PathLike)):
            self._file = open(src, "rb")
            self._owns_file = True
            try:
                self.file_size = os.fstat(self._file.fileno()).st_size
            except OSError:  # pragma: no cover - fstat on a freshly-opened path always succeeds
                self.file_size = None
        else:
            if not hasattr(src, "read"):
                raise TypeError(
                    "BackgroundChunkReader needs a path or file-like, "
                    f"got {type(src).__name__}"
                )
            self._file = src
            self.file_size = self._try_size(src)

        self._queue: queue.Queue = queue.Queue(maxsize=self._prefetch)
        self._stop = threading.Event()
        self._eof_seen = False
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="opencodecs-chunk-reader",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _try_size(src) -> int | None:
        """Best-effort file size from a file-like."""
        # Prefer fstat on the fd if available
        try:
            fd = src.fileno()
        except (AttributeError, OSError):
            fd = None
        if fd is not None:
            try:
                return os.fstat(fd).st_size
            except OSError:  # pragma: no cover - fstat on a live fd is reliable
                pass
        # Fall back to seek-trickery if seekable
        try:
            cur = src.tell()
            src.seek(0, 2)  # SEEK_END
            sz = src.tell()
            src.seek(cur)
            return sz
        except (AttributeError, OSError):
            return None

    def _reader_loop(self):
        try:
            while not self._stop.is_set():
                chunk = self._file.read(self._chunk_size)
                if not chunk:
                    self._queue.put(None)  # EOF sentinel
                    return
                # `put` blocks if queue is full → natural backpressure
                self._queue.put(chunk)
        except Exception as e:  # noqa: BLE001
            # Propagate to the consumer instead of silently dying
            self._queue.put(e)

    def get(self) -> bytes | None:
        """Pop the next chunk, or None at EOF. Re-raises bg-thread errors."""
        if self._eof_seen:
            return None
        item = self._queue.get()
        if item is None:
            self._eof_seen = True
            return None
        if isinstance(item, BaseException):
            self._eof_seen = True
            raise item
        return item

    def __iter__(self) -> Iterator[bytes]:
        while True:
            chunk = self.get()
            if chunk is None:
                return
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        # Drain a couple of items so the bg thread isn't blocked on
        # queue.put() and can notice the stop event.
        try:
            for _ in range(self._prefetch + 1):
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=1.0)
        if self._owns_file and self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None


# ---------------------------------------------------------------------------
# Random-access DataSource ABC
# ---------------------------------------------------------------------------


Range = tuple[int, int]  # (offset, length)


class DataSource(ABC):
    """Random-access byte source with batched parallel fetch.

    Concrete subclasses live in ``opencodecs._tiff_http``
    (``FileDataSource``, ``HTTPDataSource``); both predate this ABC and
    keep their ``__call__(offset, n)`` shorthand for back-compat. The
    ABC formalizes:

    * ``read_at(offset, n) -> bytes`` — the primitive every consumer
      uses.
    * ``read_many(ranges) -> list[bytes]`` — fan out a known batch in
      parallel. Default impl just loops ``read_at``; overrides exist
      where parallel fetch is a win (HTTPDataSource).
    * ``size`` — total bytes, ``None`` if not yet discovered.
    * ``close()`` — release any held resources (sockets / FDs).

    Range coalescing lives in ``coalesce_ranges``; subclasses that
    benefit from it call it inside ``read_many``.
    """

    size: int | None = None

    @abstractmethod
    def read_at(self, offset: int, n: int) -> bytes:
        """Fetch ``n`` bytes starting at ``offset``."""

    def __call__(self, offset: int, n: int) -> bytes:
        """Callable shorthand — keeps the existing read_at-callable API."""
        return self.read_at(offset, n)

    def read_many(self, ranges: Sequence[Range]) -> list[bytes]:
        """Fetch many ranges, return results in input order.

        Default impl: serial loop. Subclasses with a faster batched
        path (parallel HTTP, parallel pread on a fast filesystem)
        override.
        """
        return [self.read_at(o, n) for o, n in ranges]

    def close(self) -> None:  # pragma: no cover - default no-op
        pass

    def __enter__(self) -> "DataSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def coalesce_ranges(
    ranges: Sequence[Range],
    *,
    max_gap: int = 16 * 1024,
    max_combined: int = 4 * 1024 * 1024,
) -> tuple[list[Range], list[list[tuple[int, int, int]]]]:
    """Merge nearby ranges into bigger fetches.

    A single TIFF tile / HDF5 chunk read is usually a few KB; if the
    next requested range starts a few KB later we want one HTTP Range
    request, not two. We sort the input by offset, then greedily
    merge while:

    * the gap from the previous range's end to the next range's start
      is ``<= max_gap`` bytes, AND
    * the resulting combined range is ``<= max_combined`` bytes.

    Returns
    -------
    merged
        ``[(offset, length), ...]`` — the actual fetches the caller
        should issue.
    splits
        For each input range, ``splits[i]`` is a list of
        ``(merged_index, slice_start, slice_end)`` describing how to
        reconstruct the original bytes by slicing the merged result.
        With non-overlapping inputs each list has length 1, but
        overlapping inputs split fine.

    Both lists are in the **original input order** so callers can
    reuse their existing per-range indexing.
    """
    if not ranges:
        return [], []

    n = len(ranges)
    # Sort by (offset, length); remember original index so we can map back.
    indexed = sorted(
        ((int(o), int(L), i) for i, (o, L) in enumerate(ranges)),
        key=lambda t: (t[0], -t[1]),
    )
    merged: list[Range] = []
    # splits[i] is populated as merged ranges close.
    splits: list[list[tuple[int, int, int]]] = [[] for _ in range(n)]

    cur_off, cur_end = -1, -1
    cur_idx = -1  # index into `merged`
    for off, length, orig in indexed:
        end = off + length
        if cur_off < 0:
            cur_off, cur_end, cur_idx = off, end, len(merged)
            merged.append((cur_off, cur_end - cur_off))
        elif (
            off - cur_end <= max_gap
            and (max(end, cur_end) - cur_off) <= max_combined
        ):
            if end > cur_end:
                cur_end = end
                merged[cur_idx] = (cur_off, cur_end - cur_off)
        else:
            cur_off, cur_end = off, end
            cur_idx = len(merged)
            merged.append((cur_off, cur_end - cur_off))
        splits[orig].append((cur_idx, off - cur_off, end - cur_off))
    return merged, splits


__all__ = [
    "BackgroundChunkReader",
    "DataSource",
    "Range",
    "coalesce_ranges",
]
