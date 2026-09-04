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
    streaming_decode = True
    parallel_decode = False

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

    def open(self, src: Any):
        # ._fits, not ._fits_reader: the latter has never existed, so
        # this raised ModuleNotFoundError for every caller of
        # oc.open(..., format="fits") and for decode(), which goes
        # through it. Nothing noticed because no test called either.
        from ._fits import FitsStream
        return FitsStream(_read_src_or_path(src))

    def decode(self, src: Any, **opts) -> np.ndarray:
        """Read the primary (or first data-bearing) HDU as an ndarray."""
        with self.open(src) as r:
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
