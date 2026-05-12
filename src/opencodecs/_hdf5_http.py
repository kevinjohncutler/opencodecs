"""Remote HDF5 reader — open an HDF5 file straight from an HTTP(S) URL.

h5py 3.x can open from any Python file-like object via
``h5py.File(fobj, 'r')``. We give it a file-like that delegates byte
range reads to our existing ``HTTPDataSource`` (HTTP/1.1 keep-alive
client with Range support and an LRU cache, see ``_tiff_http.py``).

Result: chunked + compressed HDF5 datasets stored in S3 / cloud
buckets stream on demand — only the chunks you read get fetched.

Usage
=====

.. code-block:: python

    from opencodecs._hdf5_http import open_remote_hdf5

    with open_remote_hdf5("https://...amazonaws.com/file.h5") as f:
        arr = f["dataset/path"][0:1024]
        # only the chunks covering the slice were fetched

This is the same pattern as kerchunk / xarray + fsspec, but without
the runtime dependency on either — we get range-request streaming
from our stdlib-based ``HTTPDataSource``.

Caveats
=======

* h5py's file-like driver is single-threaded — concurrent reads to
  the same handle aren't safe. Open separate handles for parallel
  workers.
* The HDF5 superblock + B-tree chunk index are queried with many
  small reads near the start of the file; the ``HTTPDataSource``
  prefetch (64 KB by default) absorbs those.
* For HDF5 files that grow at the *end* (e.g. partial writes still in
  progress on the server), pass ``prefetch_bytes=0`` to disable the
  prefetch and force every read through HTTP Range.
"""

from __future__ import annotations

from typing import Any

from ._tiff_http import HTTPDataSource
from .core.io import DataSource

# Track DataSource per open h5py.File so prefetch_hdf5_chunks(dataset,
# sel) can find it without the user passing source= explicitly.
# Keyed by file.id (h5py's FileID is hashable and equality-stable
# across dataset.file lookups, even though the wrapper object isn't).
_SOURCE_REGISTRY: dict[Any, DataSource] = {}


class _HTTPFileLike:
    """File-like wrapper exposing ``HTTPDataSource`` as a seekable
    stream. The h5py driver only calls ``read``, ``readinto``,
    ``seek``, ``tell``, and ``close``."""

    def __init__(self, src: HTTPDataSource):
        self._src = src
        self._pos = 0

    # ------------------------------------------------------------------

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            # h5py occasionally asks for "all remaining"; we discover
            # the total size lazily via the source.
            total = self._src._total_size
            if total is None:
                # Trigger discovery by issuing a tiny range read at
                # offset 0 (cached) — HTTPDataSource fills in
                # _total_size from the Content-Range header.
                self._src(0, 1)
                total = self._src._total_size or 0
            n = max(0, int(total) - self._pos)
        data = self._src(self._pos, n)
        self._pos += len(data)
        return data

    def readinto(self, buf) -> int:
        n = len(buf)
        chunk = self.read(n)
        buf[: len(chunk)] = chunk
        return len(chunk)

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = int(offset)
        elif whence == 1:
            self._pos += int(offset)
        elif whence == 2:
            total = self._src._total_size
            if total is None:
                self._src(0, 1)
                total = self._src._total_size or 0
            self._pos = int(total) + int(offset)
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if self._pos < 0:
            self._pos = 0
        return self._pos

    def tell(self) -> int:
        return self._pos

    def close(self) -> None:
        # HTTPDataSource has its own close that flushes the keep-alive
        # connection.
        close = getattr(self._src, "close", None)
        if callable(close):
            close()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False


def open_remote_hdf5(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    prefetch_bytes: int = 64 * 1024,
    cache_bytes: int = 8 * 1024 * 1024,
    max_workers: int = 8,
):
    """Open a remote HDF5 file. Returns an ``h5py.File`` handle.

    The returned handle has one extra attribute, ``opencodecs_source``,
    which is the underlying :class:`HTTPDataSource`. Pass it to
    :func:`prefetch_hdf5_chunks` before reading a large slice to fan
    out chunk fetches in parallel — h5py's single-threaded driver
    then reads them out of the LRU cache instead of round-tripping.

    The caller closes via ``f.close()`` or a ``with`` statement;
    closing flushes the keep-alive HTTP connection and shuts down
    the prefetch thread pool.

    Parameters mirror :class:`opencodecs._tiff_http.HTTPDataSource`
    so the same prefetch / cache tuning works for HDF5 just like for
    cloud-optimized GeoTIFF.
    """
    try:
        import h5py
    except ImportError as e:  # pragma: no cover - h5py-missing branch
        raise ImportError(
            "h5py is required for remote HDF5 support: pip install h5py"
        ) from e

    src = HTTPDataSource(
        url,
        headers=headers,
        timeout=timeout,
        prefetch_bytes=prefetch_bytes,
        cache_bytes=cache_bytes,
        max_workers=max_workers,
    )
    fobj = _HTTPFileLike(src)
    h = h5py.File(fobj, mode="r")
    _SOURCE_REGISTRY[h.id] = src
    # Wrap close so the registry entry + the HTTPDataSource get cleaned
    # up automatically. h5py.File.close() is idempotent so the chained
    # close is safe.
    _orig_close = h.close

    def _wrapped_close():
        _SOURCE_REGISTRY.pop(h.id, None)
        try:
            src.close()
        finally:
            _orig_close()

    h.close = _wrapped_close
    return h


def prefetch_hdf5_chunks(
    dataset: Any,
    sel: Any,
    source: DataSource | None = None,
) -> int:
    """Pre-fetch all HDF5 chunks covered by ``sel`` in parallel.

    Walks ``dataset.iter_chunks(sel)`` to enumerate which chunks the
    selection touches, queries ``dataset.id.get_chunk_info_by_coord``
    for each chunk's ``(byte_offset, size)``, then issues a single
    parallel batch via ``source.read_many``. After this returns,
    a subsequent ``dataset[sel]`` reads each chunk out of the LRU
    cache the first time h5py asks for it — no network round-trip
    in the hot loop.

    Parameters
    ----------
    dataset
        The h5py dataset, typically obtained from
        ``open_remote_hdf5(url)['path/to/array']``.
    sel
        Any selection that ``iter_chunks`` accepts (``np.s_[:1024,
        :1024]``, ``slice(0, 1000)``, ``Ellipsis``, etc).
    source
        The DataSource backing the file. If omitted, we look up
        ``dataset.file.opencodecs_source`` (set by
        :func:`open_remote_hdf5`).

    Returns
    -------
    int
        Number of chunks prefetched.
    """
    if source is None:
        source = _SOURCE_REGISTRY.get(dataset.file.id)
        if source is None:
            raise ValueError(
                "prefetch_hdf5_chunks: no DataSource registered for "
                "this file; pass source= or open via open_remote_hdf5()"
            )
    ranges: list[tuple[int, int]] = []
    for chunk_slice in dataset.iter_chunks(sel):
        coord = tuple(int(s.start) for s in chunk_slice)
        info = dataset.id.get_chunk_info_by_coord(coord)
        # byte_offset is unset (huge value) for never-written chunks
        # that fall back to the fill value; those don't need fetching.
        if info.byte_offset >= (1 << 63):  # HADDR_UNDEF
            continue
        if info.size <= 0:
            continue
        ranges.append((int(info.byte_offset), int(info.size)))
    if ranges:
        source.read_many(ranges)
    return len(ranges)
