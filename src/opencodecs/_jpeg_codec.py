"""JpegCodec — Codec adapter wrapping the native _jpeg extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec
from .core.buffers import array_output
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _jpeg_encode, _jpeg_decode, _jpeg_check_signature,
    _jpeg_read_icc, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._jpeg",
    "encode", "decode", "check_signature", "read_icc_profile",
)


class JpegCodec(Codec):
    """Native JPEG codec via libjpeg-turbo (TurboJPEG API v3)."""

    name = "jpeg"
    file_extensions = (".jpg", ".jpeg")
    aliases = ("jpg",)

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
        return _jpeg_check_signature(head)

    def encode(
        self,
        data: Any,
        *,
        dest=None,
        level: int | None = None,
        iccprofile: bytes | None = None,
        **opts,
    ) -> bytes | None:
        """Encode an ndarray as JPEG.

        ``iccprofile`` embeds an ICC color profile in an APP2 marker.
        """
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        encoded = _jpeg_encode(data, level=level, iccprofile=iccprofile)
        return _write_dest(encoded, dest)

    def decode(self, src: Any, *, out=None, scale=None,
               scale_num=None, scale_denom=None, **opts) -> np.ndarray:
        return _jpeg_decode(
            _read_src(src), out=out if out is None else array_output(out),
            scale=scale, scale_num=scale_num, scale_denom=scale_denom)

    def decoder(self):
        """Create an explicitly owned reusable decode handle."""
        from .codecs._jpeg import DecoderContext
        from .core.buffers import ImageDecoderContext
        return ImageDecoderContext(DecoderContext())

    def read_icc_profile(self, src: Any) -> bytes | None:
        """Return the embedded ICC profile bytes, or ``None`` if absent."""
        return _jpeg_read_icc(_read_src(src))



__all__ = ["JpegCodec"]
