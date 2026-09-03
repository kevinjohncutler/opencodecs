"""Shared input/output helpers for codec adapters.

Every codec adapter accepts the same union of source/dest types:

  * ``bytes`` / ``bytearray`` / ``memoryview``       — buffer protocol
  * ``mmap.mmap``                                    — buffer protocol
  * ``numpy.ndarray``                                — raw bytes via .tobytes()
  * file-like objects with ``.read()`` / ``.write()``
  * ``str`` / ``pathlib.Path``                       — disk path

Centralising this here means one place to fix bugs (like the missing
ndarray case the comprehensive edge-case tests turned up).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def read_src(src: Any) -> bytes:
    """Coerce *src* to bytes for codec input.

    Accepts the buffer protocol (bytes / bytearray / memoryview / mmap),
    numpy arrays (uses .tobytes()), file-like objects with ``.read()``,
    and strings / paths (treated as disk files).
    """
    if isinstance(src, np.ndarray):
        # Caller is responsible for remembering shape + dtype if they
        # want the original back.
        return src.tobytes()
    if isinstance(src, (bytes, bytearray, memoryview)):
        return bytes(src)
    if hasattr(src, "read"):
        return src.read()
    return Path(src).read_bytes()


def write_dest(data: bytes, dest: Any) -> bytes | None:
    """Write *data* to *dest*, or return it if *dest* is None.

    Accepts file-like objects with ``.write()`` and strings / paths.
    """
    if dest is None:
        return data
    if hasattr(dest, "write"):
        dest.write(data)
        return None
    Path(dest).write_bytes(data)
    return None


__all__ = ["read_src", "write_dest"]

def open_read_at(src: Any):
    """Return ``(read_at, close)`` for random access to *src*.

    ``read_at(offset, n) -> bytes`` is the contract the TIFF, FITS and
    MRC readers already share, and it is what lets a reader open a file
    it never downloads: a path becomes a seek, an http(s) URL becomes a
    range request, bytes become a slice.

    Factored here because three readers had grown their own copy of it.
    A format whose data sits at a computable offset should not have to
    reimplement the plumbing to reach it.
    """
    import os

    if callable(src) and not isinstance(
            src, (str, os.PathLike, bytes, bytearray, memoryview)):
        return src, lambda: None

    if isinstance(src, str) and src.startswith(("http://", "https://")):
        from .._tiff_http import HTTPDataSource
        ds = HTTPDataSource(src)
        # Learn the length up front with a one-byte read, then clamp.
        # Readers open by asking for a generous header chunk, and a
        # range that runs past the end of a small file comes back empty
        # rather than short, which reads as "not a valid file" three
        # layers up. The clamp costs one tiny request at open.
        try:
            ds.read_at(0, 1)
        except Exception:                                # noqa: BLE001
            pass
        total = getattr(ds, "total_size", None)

        def read_at(offset: int, n: int, _ds=ds, _total=total) -> bytes:
            if _total is not None:
                n = min(n, max(0, _total - offset))
                if n <= 0:
                    return b""
            return _ds.read_at(offset, n)

        closer = getattr(ds, "close", None)
        return read_at, (closer if callable(closer) else (lambda: None))

    if isinstance(src, (str, os.PathLike)):
        fh = open(src, "rb")

        def read_at(off: int, n: int, _f=fh) -> bytes:
            _f.seek(off)
            return _f.read(n)
        return read_at, fh.close

    if isinstance(src, (bytes, bytearray, memoryview)):
        buf = bytes(src)

        def read_at(off: int, n: int, _b=buf) -> bytes:
            return _b[off:off + n]
        return read_at, lambda: None

    if hasattr(src, "read") and hasattr(src, "seek"):
        def read_at(off: int, n: int, _f=src) -> bytes:
            _f.seek(off)
            return _f.read(n)
        return read_at, lambda: None

    raise TypeError(
        f"unsupported src type {type(src).__name__}; pass a path, an "
        f"http(s) URL, bytes, a seekable file-like, or a read_at callable")
