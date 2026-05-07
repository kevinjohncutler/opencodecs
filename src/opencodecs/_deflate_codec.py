"""DeflateCodec — Codec adapter wrapping the native _deflate extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

_zlib_encode, _zlib_decode, _zlib_check_signature, _HAVE_BACKEND = import_or_stubs(
    "opencodecs.codecs._deflate",
    "encode", "decode", "check_signature",
)


class DeflateCodec(Codec):
    """Native zlib / deflate codec (matches imagecodecs.zlib_encode/decode)."""

    name = "deflate"
    file_extensions = (".zlib",)
    aliases = ("zlib",)

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
        return _zlib_check_signature(head)

    def encode(self, data: Any, *, dest=None, level: int | None = None,
               **opts) -> bytes | None:
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        compressed = _zlib_encode(data, level=level)
        return _write_dest(compressed, dest)

    def decode(self, src: Any, **opts) -> bytes:
        return _zlib_decode(_read_src(src))



__all__ = ["DeflateCodec"]
