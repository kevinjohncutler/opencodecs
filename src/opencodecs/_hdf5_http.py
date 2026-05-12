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
):
    """Open a remote HDF5 file. Returns an ``h5py.File`` handle.

    The caller closes via ``f.close()`` or a ``with`` statement;
    closing flushes the underlying keep-alive HTTP connection.

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
    )
    fobj = _HTTPFileLike(src)
    return h5py.File(fobj, mode="r")
