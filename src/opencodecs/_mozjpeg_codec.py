"""MozJpegCodec — JPEG encoder via MozJPEG (smaller files than libjpeg-turbo).

MozJPEG is Mozilla's libjpeg-turbo fork that adds progressive encoding
with trellis-quantization optimization. Files are 10-15% smaller than
libjpeg-turbo at the same quality level, fully decodable by any
standard JPEG decoder. Encode is slower (~2x); decode is identical.

This codec is **encode-focused**: ``decode`` works but is no faster
than the regular ``jpeg`` codec (same underlying libjpeg-turbo decoder
in MozJPEG). Pair MozJPEG with the standard JPEG decoder on the read
side for typical archive/web pipelines.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _moz_encode, _moz_decode, _moz_check_signature, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._mozjpeg",
    "encode", "decode", "check_signature",
)


class MozJpegCodec(Codec):
    """MozJPEG — smaller-JPEG encoder via Mozilla's libjpeg-turbo fork."""

    name = "mozjpeg"
    file_extensions = (".jpg", ".jpeg", ".mjpg")

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
        return _moz_check_signature(head)

    def encode(self, data: Any, *, dest=None,
               level: int | None = None,
               subsampling: object = None,
               progressive: bool = True,
               **opts) -> bytes | None:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        out = _moz_encode(
            data, level=level, subsampling=subsampling,
            progressive=progressive,
        )
        return _write_dest(out, dest)

    def decode(self, src: Any, **opts) -> np.ndarray:
        return _moz_decode(_read_src(src))


__all__ = ["MozJpegCodec"]
