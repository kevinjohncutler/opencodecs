"""WebpCodec — Codec adapter wrapping the native _webp extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

_webp_encode, _webp_decode, _webp_check_signature, _HAVE_BACKEND = import_or_stubs(
    "opencodecs.codecs._webp",
    "encode", "decode", "check_signature",
)


class WebpCodec(Codec):
    """Native WebP codec via libwebp."""

    name = "webp"
    file_extensions = (".webp",)

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.uint8,)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        return _webp_check_signature(head)

    def encode(self, data: Any, *, dest=None, level: int | None = None,
               lossless: bool = False, **opts) -> bytes | None:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        encoded = _webp_encode(data, level=level, lossless=lossless)
        return _write_dest(encoded, dest)

    def decode(self, src: Any, **opts) -> np.ndarray:
        return _webp_decode(_read_src(src))



__all__ = ["WebpCodec"]
