"""BrotliCodec — Codec adapter wrapping the native _brotli extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

_brotli_encode, _brotli_decode, _brotli_check_signature, _HAVE_BACKEND = import_or_stubs(
    "opencodecs.codecs._brotli",
    "encode", "decode", "check_signature",
)


class BrotliCodec(Codec):
    """Native brotli codec.

    Bytes-in / bytes-out only. Brotli streams have no fixed magic header,
    so signature-based dispatch always returns False; use ``format='brotli'``
    or ``.br`` extension for routing.
    """

    name = "brotli"
    file_extensions = (".br",)

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
        return _brotli_check_signature(head)

    def encode(self, data: Any, *, dest=None, level: int | None = None,
               **opts) -> bytes | None:
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        compressed = _brotli_encode(data, level=level)
        return _write_dest(compressed, dest)

    def decode(self, src: Any, **opts) -> bytes:
        return _brotli_decode(_read_src(src))



__all__ = ["BrotliCodec"]
