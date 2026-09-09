"""FitsCodec — Codec adapter for the native FITS reader.

FITS is the canonical astronomy file format (HST, JWST, Vera Rubin,
every sky survey). This codec exposes ``read(src)`` and the
``open(src)`` reader contract for FITS files; encode is not
supported (FITS is a container format, not a compression codec —
the closest equivalent is the rcomp / RICE_1 codec which we ship
separately).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src


class FitsCodec(Codec):
    """FITS astronomical image reader."""

    name = "fits"
    aliases = ("fts", "fit")
    file_extensions = (".fits", ".fts", ".fit")

    has_native = True
    has_delegate = False
    can_encode = False
    can_decode = True
    multi_frame = True
    # An HDU is a seek: the headers are walked, but only the requested
    # HDU's data is read. Measured on a 24-HDU 12.7 MB file, reading
    # the last one moves 5.2% of it over HTTP -- one frame plus the
    # headers in front of it.
    chunked = True
    streaming_decode = True
    # A compressed-image HDU is a BINTABLE with one row per tile, and
    # tiles are independent codestreams. The whole table and heap are
    # read in one block first, so there is no I/O to serialize.
    # Measured on 4096x4096 with 256x256 tiles: HCOMPRESS_1 603.3 ms
    # to 86.7 ms, RICE_1 106.5 to 19.2, GZIP_1 86.5 to 20.6, PLIO_1
    # 46.2 to 18.5, all bit-identical to astropy.
    parallel_decode = True

    # FITS holds any of these via BITPIX; the reader returns the dtype
    # advertised by the primary HDU's BITPIX card. None of the values
    # below is enforced — they're advisory for the registry.
    supported_dtypes = (
        np.uint8, np.int16, np.int32, np.int64, np.float32, np.float64,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        # First card must be ``SIMPLE  = `` (primary) or ``XTENSION=``
        # (extension HDU loaded as a standalone subset).
        return (
            head.startswith(b"SIMPLE  = ")
            or head.startswith(b"XTENSION= ")
        )

    def open(self, src: Any, *, numthreads: int | None = None):
        # ._fits, not ._fits_reader: the latter has never existed, so
        # this raised ModuleNotFoundError for every caller of
        # oc.open(..., format="fits") and for decode(), which goes
        # through it. Nothing noticed because no test called either.
        from ._fits import FitsStream
        return FitsStream(_read_src_or_path(src), numthreads=numthreads)

    def decode(self, src: Any, **opts) -> np.ndarray:
        """Read the primary (or first data-bearing) HDU as an ndarray."""
        with self.open(src, numthreads=opts.pop("numthreads", None)) as r:
            return r.read()


def _read_src_or_path(src: Any) -> Any:
    """Preserve path / file-like / DataSource for FitsStream's own
    open path. Falls back to ``read_src`` (bytes) for objects that
    only support the buffer protocol."""
    import os
    if isinstance(src, (str, os.PathLike, bytes, bytearray, memoryview)):
        return src
    if callable(src) or (hasattr(src, "read") and hasattr(src, "seek")):
        return src
    return _read_src(src)


__all__ = ["FitsCodec"]
