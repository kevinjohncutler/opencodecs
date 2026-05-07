"""I/O primitives shared across codecs.

The big one: ``BackgroundChunkReader`` — a bg-thread file-chunk producer
with a bounded queue. The bg thread holds the GIL only while doing
Python work (queue.put, etc.); during the actual ``file.read()`` syscall
the GIL is released, so the consumer thread (running libjxl /
TIFFReadStrip / numpy frombuffer / etc. inside ``with nogil:``) gets
real wall-clock overlap between I/O and codec work.

For chunked formats (JXL incremental SetInput, TIFF tile pyramids, NPY
row ranges, zarr chunks), this is the substrate the codec wrapper
plugs into.
"""

from __future__ import annotations

import os
import queue
import threading
from pathlib import Path
from typing import Iterator


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


__all__ = ["BackgroundChunkReader"]
