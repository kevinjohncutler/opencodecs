"""AvifCodec — Codec adapter wrapping the native _avif extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _avif_encode, _avif_decode, _avif_check_signature,
    _avif_read_icc, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._avif",
    "encode", "decode", "check_signature", "read_icc_profile",
)


class AvifCodec(Codec):
    """Native AVIF codec via libavif."""

    name = "avif"
    file_extensions = (".avif",)

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
        return _avif_check_signature(head)

    def encode(self, data: Any, *, dest=None, level: int | None = None,
               lossless: bool = False, speed: int = 6,
               color=None, bit_depth: int | None = None,
               numthreads: int | None = None,
               iccprofile: bytes | None = None,
               **opts) -> bytes | None:
        """Encode an array as AVIF.

        ``iccprofile`` embeds an ICC color profile.
        """
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        encoded = _avif_encode(
            data, level=level, lossless=lossless, speed=speed,
            color=color, bit_depth=bit_depth, numthreads=numthreads,
            iccprofile=iccprofile,
        )
        return _write_dest(encoded, dest)

    def decode(self, src: Any, *, numthreads: int | None = None,
               out=None, **opts) -> np.ndarray:
        return _avif_decode(_read_src(src), numthreads=numthreads, out=out)

    def read_icc_profile(self, src: Any) -> bytes | None:
        """Return the embedded ICC profile bytes, or ``None`` if absent."""
        return _avif_read_icc(_read_src(src))



__all__ = ["AvifCodec"]
