"""Jpeg2kCodec — Codec adapter wrapping the native _jpeg2k extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

_jp2_encode, _jp2_decode, _jp2_check_signature, _HAVE_BACKEND = import_or_stubs(
    "opencodecs.codecs._jpeg2k",
    "encode", "decode", "check_signature",
)


class Jpeg2kCodec(Codec):
    """Native JPEG-2000 codec via OpenJPEG."""

    name = "jpeg2k"
    file_extensions = (".jp2", ".j2k", ".jpx", ".jpc")
    aliases = ("j2k", "jp2")

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.uint8, np.uint16)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        return _jp2_check_signature(head)

    def encode(self, data: Any, *, dest=None, level: int | None = None,
               lossless: bool = False, codec: str = "jp2",
               numthreads: int | None = None,
               **opts) -> bytes | None:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        encoded = _jp2_encode(
            data, level=level, lossless=lossless, codec=codec,
            numthreads=numthreads,
        )
        return _write_dest(encoded, dest)

    def decode(self, src: Any, *, numthreads: int | None = None,
               **opts) -> np.ndarray:
        return _jp2_decode(_read_src(src), numthreads=numthreads)



__all__ = ["Jpeg2kCodec"]
