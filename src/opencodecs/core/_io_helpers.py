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

    A thin adapter over :func:`opencodecs.core.io.coerce_data_source`,
    which is where source coercion lives. This stayed a separate
    function only because it predates that one, and for a while the two
    grew apart: this accepted buffers, URLs and callables while the
    other took a path or a DataSource, so which sources a reader
    supported came down to which helper it happened to call. Two sets
    of rules for one question drift, and these did.

    Prefer ``coerce_data_source`` in new code: it also reports the size
    and exposes ``read_many`` for batched parallel fetch. This form
    remains for the readers already written against a plain callable,
    which is all a reader needs when it only ever calls ``read_at``.

    Deliberately NOT memory-mapped, which was tried and measured. The
    readers behind this helper (MRC, NRRD, DICOM, FITS) fetch their
    bulk data in one large read, and mapping only removes per-read
    syscall overhead, of which one read has none. Alternating A/B on a
    134 MB volume: MRC 1.00x, NRRD 0.97x, against a control reader that
    does not use this helper at all and moved 1.05x, which is the noise
    floor. Mapping earns its keep in the TIFF reader instead, where a
    page is hundreds of separate tile reads.
    """
    from .io import coerce_data_source

    ds, owns, _size = coerce_data_source(src)
    return ds.read_at, (ds.close if owns else (lambda: None))
