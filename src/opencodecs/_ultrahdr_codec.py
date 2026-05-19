"""UltraHdrCodec — Codec adapter for the native ``_ultrahdr`` extension.

Ultra HDR (ISO 21496) is a backwards-compatible HDR image format —
an SDR JPEG with a small embedded gainmap JPEG. SDR viewers see a
normal JPEG; HDR viewers (Android 14+, iOS 18+) multiply by the
gainmap to recover the HDR image.

Default I/O surface: encode takes ``(H, W, 4) float16`` linear-light
RGBA in BT.2100 primaries; decode returns the same. ``dtype=np.uint8``
opt-outs to the SDR-only tonemapped base JPEG.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs


(
    _uhdr_encode, _uhdr_decode, _uhdr_check_signature, _HAVE_BACKEND
) = import_or_stubs(
    "opencodecs.codecs._ultrahdr",
    "encode", "decode", "check_signature",
)


class UltraHdrCodec(Codec):
    """Ultra HDR (gainmap JPEG) image codec."""

    name = "ultrahdr"
    aliases = ("uhdr",)
    file_extensions = (".jpg", ".jpeg")  # gainmap JPEGs use the JPEG extension

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.float16, np.uint8, np.uint16)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        # A bare JPEG magic byte match is too loose to claim Ultra HDR
        # ownership, so the dispatcher routes JPEG by default — callers
        # opt in to ultrahdr explicitly via ``read(..., codec="ultrahdr")``.
        return False

    def encode(self, data: Any, *, dest=None,
               level: int | None = None,
               scale: int | None = None,
               fast: bool = False,
               **opts) -> bytes | None:
        arr = data if isinstance(data, np.ndarray) else np.asarray(data)
        out = _uhdr_encode(arr, level=level, scale=scale, fast=fast)
        return _write_dest(out, dest)

    def decode(self, src: Any, *, dtype=None, boost=None,
               **opts) -> np.ndarray:
        return _uhdr_decode(_read_src(src), dtype=dtype, boost=boost)


__all__ = ["UltraHdrCodec"]
