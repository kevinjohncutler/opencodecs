"""PngCodec — Codec adapter wrapping the native _png extension (libspng)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

_png_encode, _png_decode, _png_check_signature, _HAVE_BACKEND = import_or_stubs(
    "opencodecs.codecs._png",
    "encode", "decode", "check_signature",
)


class PngCodec(Codec):
    """Native PNG codec backed by libspng.

    Decode preserves PNG color type and bit depth (8 or 16), with 1/2/4-bit
    images upscaled to 8-bit and indexed palettes expanded to RGBA.
    """

    name = "png"
    file_extensions = (".png",)

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
        return _png_check_signature(head)

    def encode(self, data: Any, *, dest=None, level: int | None = None,
               **opts) -> bytes | None:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        encoded = _png_encode(data, level=level)
        return _write_dest(encoded, dest)

    def decode(self, src: Any, **opts) -> np.ndarray:
        return _png_decode(_read_src(src))



__all__ = ["PngCodec"]
