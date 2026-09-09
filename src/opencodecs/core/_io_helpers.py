"""Shared input/output helpers for codec adapters.

Every codec adapter accepts the same union of source/dest types:

  * ``bytes`` / ``bytearray`` / ``memoryview``       — buffer protocol
  * ``mmap.mmap``                                    — buffer protocol
  * ``numpy.ndarray``                                — raw bytes via .tobytes()
  * file-like objects with ``.read()`` / ``.write()``
  * ``str`` / ``pathlib.Path``                       — disk path

Centralizing this here means one place to fix bugs (like the missing
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
    strings / paths (treated as disk files), and http(s) URLs.

    A URL is fetched in FULL rather than by range, and that is the
    honest thing for the codecs that reach storage through here: they
    want a complete codestream, so there is nothing to skip. Coercing
    through a range-reading data source instead would move the same
    bytes while putting an `http` tick in the capability manifest
    against formats that cannot use it -- the same reasoning, and the
    same helper, as `_pyramid_bytes`. Formats that CAN read at offsets
    take a data source rather than bytes and never come through here.

    Before this, a URL fell through to ``Path(src).read_bytes()`` and
    failed as a missing file, so about thirty codecs could not open one
    at all.
    """
    if isinstance(src, np.ndarray):
        # Caller is responsible for remembering shape + dtype if they
        # want the original back.
        return src.tobytes()
    if isinstance(src, (bytes, bytearray, memoryview)):
        return bytes(src)
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        from .._tiff_http import http_fetch_all
        return http_fetch_all(src)
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
