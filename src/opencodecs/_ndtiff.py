"""NDTiffDataset — native reader for Micro-Manager / Pycro-Manager NDTiff folders.

A complete NDTiff acquisition lives in a single directory containing:

  * ``NDTiff.index`` — a flat binary side-index mapping
    ``frozenset({axis_name: value, ...})`` to ``(file, pixel_offset,
    w, h, pixel_type, pixel_compression, metadata_offset,
    metadata_length, metadata_compression)``.
  * One or more ``*_NDTiffStack[_N].tif`` files, each capped at the
    4 GB TIFF limit. Frames are concatenated with a fixed pixel-byte
    region followed by per-image JSON metadata.

This reader treats the side-index as the source of truth: we never
walk the IFDs inside the .tif files. The index already records the
exact byte offset of each frame's pixel data, so a single ``os.pread``
gets the bytes — no TIFF parsing per frame.

Two parallelism axes:

  * The index parse runs in a Cython nogil loop (5-20× faster than
    ndstorage's pure-Python loop).
  * Multi-frame reads run on a persistent ThreadPoolExecutor; each
    worker uses ``os.pread`` so reads issue concurrently at the OS
    level without sharing an fd.

Both paths reuse opencodecs's existing parallel-pread harness from
``_czi_reader.py`` (same module-level pool sizing).

Compatible with the read_at(offset, n) callable contract — pass an
``HTTPDataSource`` or any custom data source instead of a directory
to open an NDTiff over HTTP / S3 / mmap. The index file still has to
be locally readable (it's small, KB-MB range) but the .tif file bytes
can come from anywhere.
"""

from __future__ import annotations

import json
import mmap
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from .core.codec import Reader
from .core._optional_backend import import_or_stubs

(
    _parse_ndtiff_index, PIXEL_TYPE_EIGHT_BIT, PIXEL_TYPE_SIXTEEN_BIT,
    PIXEL_TYPE_EIGHT_BIT_RGB, PIXEL_TYPE_TEN_BIT, PIXEL_TYPE_TWELVE_BIT,
    PIXEL_TYPE_FOURTEEN_BIT, PIXEL_TYPE_ELEVEN_BIT,
    _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._ndtiff",
    "parse_ndtiff_index",
    "PIXEL_TYPE_EIGHT_BIT", "PIXEL_TYPE_SIXTEEN_BIT",
    "PIXEL_TYPE_EIGHT_BIT_RGB", "PIXEL_TYPE_TEN_BIT",
    "PIXEL_TYPE_TWELVE_BIT", "PIXEL_TYPE_FOURTEEN_BIT",
    "PIXEL_TYPE_ELEVEN_BIT",
)


# Storage dtype (uint8 vs uint16) keyed by NDTiff pixel-type code. Bit
# depths below 16 (10/11/12/14) are still stored in uint16 words.
_DTYPE_FOR_PIXEL_TYPE: dict[int, tuple[np.dtype, int]] = {}


def _build_dtype_table() -> None:
    """Build the pixel-type → (dtype, bit_depth) table once at import.

    The stubs from import_or_stubs return callables when the backend
    is unbuilt; guard against calling them like constants.
    """
    if _DTYPE_FOR_PIXEL_TYPE:
        return
    try:
        _DTYPE_FOR_PIXEL_TYPE.update({
            int(PIXEL_TYPE_EIGHT_BIT):     (np.dtype(np.uint8),  8),
            int(PIXEL_TYPE_SIXTEEN_BIT):   (np.dtype(np.uint16), 16),
            int(PIXEL_TYPE_EIGHT_BIT_RGB): (np.dtype(np.uint8),  8),  # 3 samples
            int(PIXEL_TYPE_TEN_BIT):       (np.dtype(np.uint16), 10),
            int(PIXEL_TYPE_TWELVE_BIT):    (np.dtype(np.uint16), 12),
            int(PIXEL_TYPE_FOURTEEN_BIT):  (np.dtype(np.uint16), 14),
            int(PIXEL_TYPE_ELEVEN_BIT):    (np.dtype(np.uint16), 11),
        })
    except TypeError:
        # Backend stubs — leave the table empty; reads will raise on first use.
        pass


# Module-level persistent thread pool — same pattern as CZI reader.
# Promoting to a shared helper is a follow-up; keeping the duplication
# minimal (just 8 lines) for now.
_DEFAULT_POOL_SIZE = max(2 * (os.cpu_count() or 4), 8)
_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


def _get_pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = ThreadPoolExecutor(
                    max_workers=_DEFAULT_POOL_SIZE,
                    thread_name_prefix="opencodecs-ndtiff",
                )
    return _POOL


# ---------------------------------------------------------------------------
# Index entry — one record per frame
# ---------------------------------------------------------------------------


class NDTiffIndexEntry:
    """One frame's index record. Lightweight; we make millions of these."""

    __slots__ = (
        "axes", "filename", "pixel_offset", "image_width", "image_height",
        "pixel_type", "pixel_compression",
        "metadata_offset", "metadata_length", "metadata_compression",
    )

    def __init__(
        self,
        axes: dict[str, Any],
        filename: str,
        pixel_offset: int,
        image_width: int,
        image_height: int,
        pixel_type: int,
        pixel_compression: int,
        metadata_offset: int,
        metadata_length: int,
        metadata_compression: int,
    ):
        self.axes = axes
        self.filename = filename
        self.pixel_offset = pixel_offset
        self.image_width = image_width
        self.image_height = image_height
        self.pixel_type = pixel_type
        self.pixel_compression = pixel_compression
        self.metadata_offset = metadata_offset
        self.metadata_length = metadata_length
        self.metadata_compression = metadata_compression

    @property
    def dtype(self) -> np.dtype:
        _build_dtype_table()
        return _DTYPE_FOR_PIXEL_TYPE[self.pixel_type][0]

    @property
    def bit_depth(self) -> int:
        _build_dtype_table()
        return _DTYPE_FOR_PIXEL_TYPE[self.pixel_type][1]

    @property
    def samples_per_pixel(self) -> int:
        return 3 if self.pixel_type == PIXEL_TYPE_EIGHT_BIT_RGB else 1

    @property
    def shape(self) -> tuple[int, ...]:
        if self.samples_per_pixel == 1:
            return (self.image_height, self.image_width)
        return (self.image_height, self.image_width, self.samples_per_pixel)

    @property
    def pixel_nbytes(self) -> int:
        return (self.image_width * self.image_height
                * self.samples_per_pixel * self.dtype.itemsize)

    def __repr__(self) -> str:
        return (f"<NDTiffIndexEntry axes={self.axes} {self.filename}"
                f" @{self.pixel_offset} {self.shape} {self.dtype.name}>")


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class NDTiffDataset(Reader):
    """Read-only access to an NDTiff acquisition directory.

    Construct with a path to the directory containing ``NDTiff.index``
    plus one or more ``*_NDTiffStack[_N].tif`` files.

    Random access by axes coordinates::

        ds = NDTiffDataset("/path/to/Acq_647nm_0/Cam-1")
        img = ds.read_frame(z=42, c=0)      # one frame, one pread
        imgs = ds.read_many(keys=[...])      # parallel pread + decode

    Or use the Reader protocol for sequential iteration::

        with NDTiffDataset(dir) as ds:
            for frame in ds.iter_frames():
                ...
    """

    is_chunked = True

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        # For HTTP / cloud sources: a factory that takes a filename
        # (e.g. "NDTiffStack_2.tif") and returns a read_at(offset, n)
        # callable for that file. Default opens a local fd + os.pread.
        data_source_factory: Callable[[str], Callable[[int, int], bytes]] | None = None,
        # Pre-fetched index bytes; if provided, ``path`` may be None
        # (useful for HTTP/cloud — the caller downloads the small
        # NDTiff.index file once and passes the bytes here).
        index_bytes: bytes | None = None,
    ):
        if path is not None:
            self._path = Path(path)
            if not self._path.is_dir():
                raise FileNotFoundError(
                    f"NDTiff directory not found: {self._path}")
        else:
            self._path = None

        if index_bytes is not None:
            idx_bytes = index_bytes
        elif self._path is not None:
            index_path = self._path / "NDTiff.index"
            if not index_path.is_file():
                raise FileNotFoundError(
                    f"NDTiff.index missing in {self._path}; "
                    "not an NDTiff folder?"
                )
            idx_bytes = index_path.read_bytes()
        else:
            raise ValueError(
                "NDTiffDataset: either pass a directory path or "
                "index_bytes + data_source_factory"
            )

        # Cython index parse — single nogil walk over the buffer.
        records = _parse_ndtiff_index(idx_bytes)

        # Intern axes JSON: identical blobs across the same acquisition
        # are common (same axis set, different values). One json.loads
        # per *unique* blob saves time on big indices.
        json_cache: dict[bytes, dict] = {}

        self.entries: list[NDTiffIndexEntry] = []
        self._by_axes: dict[frozenset, int] = {}

        for rec in records:
            axes_blob, filename, p_off, w, h, pt, pc, m_off, m_len, mc = rec
            axes = json_cache.get(axes_blob)
            if axes is None:
                axes = json.loads(axes_blob.decode("utf-8"))
                # Normalize to dict; tolerate edge cases where the JSON
                # is a list (rare older variants).
                if isinstance(axes, list):
                    axes = dict(axes)
                json_cache[axes_blob] = axes
            entry = NDTiffIndexEntry(
                axes=axes,
                filename=filename,
                pixel_offset=p_off,
                image_width=w,
                image_height=h,
                pixel_type=pt,
                pixel_compression=pc,
                metadata_offset=m_off,
                metadata_length=m_len,
                metadata_compression=mc,
            )
            self.entries.append(entry)
            key = frozenset(axes.items())
            self._by_axes[key] = len(self.entries) - 1

        # Per-file read_at(offset, n) callables. Opened lazily on first
        # frame read for that file. Default factory opens a local fd
        # and wraps os.pread; pass a custom factory for HTTP / S3 /
        # mmap (e.g. ``lambda fn: HTTPDataSource(base_url + fn)``).
        self._source_cache: dict[str, Callable[[int, int], bytes]] = {}
        self._source_lock = threading.Lock()
        self._source_factory = (
            data_source_factory
            if data_source_factory is not None
            else self._default_source_factory
        )
        # fds opened by the default factory; close() releases them.
        self._owned_fds: list[int] = []

        # Populate Reader contract attrs from frame 0.
        if not self.entries:
            raise ValueError(f"NDTiff index in {self._path} is empty")
        first = self.entries[0]
        self.dtype = first.dtype
        self.shape = first.shape
        self.n_frames = len(self.entries)

    @classmethod
    def from_http(
        cls,
        base_url: str,
        *,
        index_url: str | None = None,
        prefetch_bytes: int = 64 * 1024,
        cache_bytes_per_file: int = 8 * 1024 * 1024,
    ) -> "NDTiffDataset":
        """Open a remote NDTiff acquisition over HTTP Range requests.

        ``base_url`` is the folder URL (must end with ``/`` or have an
        implicit one). Each ``NDTiffStack[_N].tif`` file referenced by
        the index will be fetched lazily via HTTPDataSource.

        ``index_url`` defaults to ``base_url + "NDTiff.index"``. The
        index file is small (KB-MB) so we download it in full once.

        Example::

            ds = NDTiffDataset.from_http(
                "https://lab.example.org/Acq_647nm/Cam-1/")
            frame = ds.read_frame(z=42)   # → one Range request
        """
        import urllib.request

        from ._tiff_http import HTTPDataSource

        if not base_url.endswith("/"):
            base_url = base_url + "/"
        if index_url is None:
            index_url = base_url + "NDTiff.index"

        with urllib.request.urlopen(index_url) as resp:
            idx_bytes = resp.read()

        def factory(filename: str) -> Callable[[int, int], bytes]:
            return HTTPDataSource(
                base_url + filename,
                prefetch_bytes=prefetch_bytes,
                cache_bytes=cache_bytes_per_file,
            )

        return cls(
            path=None,
            data_source_factory=factory,
            index_bytes=idx_bytes,
        )

    # ----- Per-file data source management -----

    def _default_source_factory(
        self, filename: str,
    ) -> Callable[[int, int], bytes]:
        """Default: open the file via os.pread. Caller must close the
        fd in close()."""
        if self._path is None:
            raise RuntimeError(
                "NDTiffDataset has no path; pass a custom "
                "data_source_factory for non-local data sources"
            )
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        fd = os.open(str(self._path / filename), flags)
        # Stash fd on the closure so close() can release it.
        self._owned_fds.append(fd)
        return _make_pread_callable(fd)

    def _source_for(self, filename: str) -> Callable[[int, int], bytes]:
        src = self._source_cache.get(filename)
        if src is not None:
            return src
        with self._source_lock:
            src = self._source_cache.get(filename)
            if src is None:
                src = self._source_factory(filename)
                self._source_cache[filename] = src
        return src

    def close(self) -> None:
        # Close all factory-allocated callables. For HTTP/custom
        # sources, the caller's source may expose a .close() method.
        with self._source_lock:
            for src in self._source_cache.values():
                close = getattr(src, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:  # pragma: no cover - defensive
                        pass
            self._source_cache.clear()
        # Close any fds we opened for the default factory.
        for fd in self._owned_fds:
            try:
                os.close(fd)
            except OSError:  # pragma: no cover - already-closed defense
                pass
        self._owned_fds.clear()

    # ----- Random access -----

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx) -> np.ndarray:
        """Random access by ordinal index (insertion order) OR by an axes
        dict-like::

            ds[0]            # first frame in index order
            ds[{"z": 42}]    # by axes coordinates
        """
        if isinstance(idx, dict):
            return self.read_frame(**idx)
        if isinstance(idx, frozenset):
            entry = self.entries[self._by_axes[idx]]
            return self._read_entry(entry)
        if isinstance(idx, int):
            if idx < 0:
                idx += len(self.entries)
            return self._read_entry(self.entries[idx])
        raise TypeError(f"NDTiffDataset[{idx!r}]: int / dict / frozenset only")

    def read_frame(self, **axes) -> np.ndarray:
        """Read one frame by axes coordinates. Single pread."""
        key = frozenset(axes.items())
        idx = self._by_axes.get(key)
        if idx is None:
            raise KeyError(
                f"no frame at axes {axes}; available keys near: "
                f"{list(self._by_axes.keys())[:3]!r} ..."
            )
        return self._read_entry(self.entries[idx])

    def has_frame(self, **axes) -> bool:
        return frozenset(axes.items()) in self._by_axes

    @property
    def axes_names(self) -> set[str]:
        """Union of axis names across all frames in the index."""
        s: set[str] = set()
        for e in self.entries:
            s.update(e.axes.keys())
        return s

    def axis_values(self, axis: str) -> list:
        """Sorted unique values seen for ``axis``."""
        seen = set()
        for e in self.entries:
            if axis in e.axes:
                seen.add(e.axes[axis])
        return sorted(seen)

    # ----- Reader protocol -----

    def iter_frames(self) -> Iterator[np.ndarray]:
        for e in self.entries:
            yield self._read_entry(e)

    def __enter__(self) -> "NDTiffDataset":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    # ----- Parallel multi-frame read -----

    def read_many(
        self,
        keys: list[dict] | list[int] | None = None,
        *,
        n_workers: int | None = None,
    ) -> np.ndarray:
        """Read N frames concurrently. Returns a stacked ndarray.

        ``keys`` may be a list of axes dicts, a list of integer indices,
        or None (= read every frame). All frames must share the same
        shape and dtype; if they don't, raise.
        """
        if keys is None:
            indices = range(len(self.entries))
        else:
            indices = [self._key_to_index(k) for k in keys]
        if not indices:
            return np.empty((0,), dtype=self.dtype)

        first = self.entries[indices[0]]
        ref_shape = first.shape
        ref_dtype = first.dtype

        out = np.empty((len(indices), *ref_shape), dtype=ref_dtype)

        def _worker(slot: int) -> None:
            entry = self.entries[indices[slot]]
            if entry.shape != ref_shape or entry.dtype != ref_dtype:
                raise ValueError(
                    f"read_many: frame {slot} shape={entry.shape}/"
                    f"dtype={entry.dtype} differs from reference "
                    f"({ref_shape}/{ref_dtype})"
                )
            out[slot] = self._read_entry(entry)

        if n_workers == 1 or len(indices) == 1:
            for slot in range(len(indices)):
                _worker(slot)
        elif n_workers is None:
            pool = _get_pool()
            list(pool.map(_worker, range(len(indices))))
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                list(ex.map(_worker, range(len(indices))))
        return out

    def iter_frames_parallel(
        self,
        keys: list[dict] | list[int] | None = None,
        *,
        prefetch: int = 16,
    ) -> Iterator[np.ndarray]:
        """Yield frames in submitted order with parallel read-ahead.

        Bounded memory: at most ``prefetch`` frames in flight at once,
        so this is safe for arbitrarily large acquisitions (no risk of
        allocating a 100 GB output array up front like ``read_many``).

        Match-for-match replacement for ndstorage's iteration pattern
        with the added bonus that I/O issues in parallel.
        """
        if keys is None:
            indices = list(range(len(self.entries)))
        else:
            indices = [self._key_to_index(k) for k in keys]
        if not indices:
            return

        pool = _get_pool()
        # Pipeline: submit up to ``prefetch`` reads in advance; as each
        # completes, yield it and submit the next one.
        in_flight: list = []
        next_to_submit = 0

        def _submit_more():
            nonlocal next_to_submit
            while len(in_flight) < prefetch and next_to_submit < len(indices):
                idx = indices[next_to_submit]
                fut = pool.submit(self._read_entry, self.entries[idx])
                in_flight.append(fut)
                next_to_submit += 1

        _submit_more()
        while in_flight:
            fut = in_flight.pop(0)
            yield fut.result()
            _submit_more()

    # ----- Internals -----

    def _key_to_index(self, key) -> int:
        if isinstance(key, int):
            return key if key >= 0 else key + len(self.entries)
        if isinstance(key, dict):
            return self._by_axes[frozenset(key.items())]
        if isinstance(key, frozenset):
            return self._by_axes[key]
        raise TypeError(f"unsupported key type {type(key).__name__}")

    def _read_entry(self, entry: NDTiffIndexEntry) -> np.ndarray:
        read_at = self._source_for(entry.filename)
        # For compressed frames, the index records the compressed
        # byte count separately — but NDTiff's official spec doesn't
        # publish a "compressed_byte_count" field. ndstorage solves
        # this by reading the TIFF IFD's StripByteCounts via a
        # separate scan; we instead use the trailing-record trick:
        # the next record's pixel_offset (in the same file) tells us
        # where this frame ends. For uncompressed frames the math is
        # trivial (pixel_nbytes); we use that fast path when
        # compression == NONE.
        if entry.pixel_compression == 0:
            nbytes = entry.pixel_nbytes
            raw = read_at(entry.pixel_offset, nbytes)
            if len(raw) < nbytes:
                raise EOFError(
                    f"NDTiff: short read ({len(raw)}/{nbytes} bytes) "
                    f"for frame {entry.axes} in {entry.filename}"
                )
            arr = np.frombuffer(
                raw, dtype=entry.dtype,
                count=nbytes // entry.dtype.itemsize,
            )
            return arr.reshape(entry.shape)

        # Compressed: look up the size of the compressed payload from
        # the index records (entries in the same file are stored in
        # write order with monotonically increasing pixel_offset).
        comp_size = self._compressed_nbytes_for(entry)
        comp_bytes = read_at(entry.pixel_offset, comp_size)
        if len(comp_bytes) < comp_size:
            raise EOFError(
                f"NDTiff: short read ({len(comp_bytes)}/{comp_size}) "
                f"for compressed frame {entry.axes} in {entry.filename}"
            )
        from .core.segment_compression import decode_segment
        raw = decode_segment(bytes(comp_bytes), entry.pixel_compression)
        # raw might be bytes (for deflate/zstd/lzw/packbits) or ndarray
        # (for jpeg/jxl/lerc/jpeg2k/webp). For byte-stream codecs we
        # reshape; for image-codecs we trust the codec's shape.
        if isinstance(raw, np.ndarray):
            return raw.reshape(entry.shape)
        nbytes = entry.pixel_nbytes
        if len(raw) < nbytes:
            raise EOFError(
                f"NDTiff decompress short: {len(raw)}/{nbytes} bytes "
                f"for frame {entry.axes}"
            )
        arr = np.frombuffer(
            raw, dtype=entry.dtype,
            count=nbytes // entry.dtype.itemsize,
        )
        return arr.reshape(entry.shape)

    def _compressed_nbytes_for(self, entry: "NDTiffIndexEntry") -> int:
        """Compute the on-disk byte size of a compressed frame.

        NDTiff's index records pixel_offset + metadata_offset for each
        frame; the compressed payload runs from pixel_offset up to
        metadata_offset (metadata is written immediately after the
        pixel bytes). For the last frame in a file we fall back to
        the file size.
        """
        # metadata_offset is set per-record by the writer; equals
        # pixel_offset + compressed_payload_size.
        return int(entry.metadata_offset) - int(entry.pixel_offset)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pread_callable(fd: int) -> Callable[[int, int], bytes]:
    """Wrap an open fd as a read_at(offset, n) -> bytes callable.

    Uses os.pread on POSIX (parallel-safe — no shared seek state).
    Windows falls back to a locked seek+read since pread is unavailable.
    Both paths loop on short reads.
    """
    has_pread = hasattr(os, "pread")
    if has_pread:
        def read_at(offset: int, n: int) -> bytes:
            buf = os.pread(fd, int(n), int(offset))
            need = int(n) - len(buf)
            cur = int(offset) + len(buf)
            while need > 0:
                more = os.pread(fd, need, cur)
                if not more:
                    break  # EOF — caller handles short return
                buf += more
                cur += len(more)
                need -= len(more)
            return buf
        return read_at

    # Windows: serialize seek+read with a lock since the fd has shared
    # state. Slower than pread under contention, but correctness first;
    # users wanting parallel I/O on Windows should use HTTPDataSource.
    lock = threading.Lock()

    def read_at(offset: int, n: int) -> bytes:
        with lock:
            os.lseek(fd, int(offset), os.SEEK_SET)
            buf = os.read(fd, int(n))
            need = int(n) - len(buf)
            while need > 0:
                more = os.read(fd, need)
                if not more:
                    break
                buf += more
                need -= len(more)
            return buf
    return read_at


__all__ = ["NDTiffDataset", "NDTiffIndexEntry"]
