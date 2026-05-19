"""NoneCodec — identity passthrough.

Used as the "no compression" entry in filter chains (zarr, ome-tiff,
hdf5) where a codec slot is required but no transformation should be
applied. Mirrors ``imagecodecs.none_encode`` / ``none_decode``.

Returns ``bytes(data)`` for both directions so the surrounding
machinery (which expects bytes-out for compressors) sees a consistent
type — ndarray inputs get a single ``tobytes()`` copy on encode.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest


class NoneCodec(Codec):
    """Identity (no-op) codec."""

    name = "none"
    aliases = ("identity", "raw")
    file_extensions = ()

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.uint8,)
    supports_color = False

    def signature(self, head: bytes) -> bool:
        # No way to recognise an unframed byte stream; treat everything
        # as not-ours so the dispatcher never auto-routes here.
        return False

    def encode(self, data: Any, *, dest=None, **opts) -> bytes | None:
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        elif not isinstance(data, (bytes, bytearray, memoryview)):
            data = bytes(data)
        return _write_dest(bytes(data), dest)

    def decode(self, src: Any, **opts) -> bytes:
        return bytes(_read_src(src))


__all__ = ["NoneCodec"]
