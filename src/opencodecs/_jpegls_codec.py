"""JpegLsCodec — JPEG-LS via CharLS.

JPEG-LS (ISO/IEC 14495) is a predictive lossless / near-lossless image
codec used heavily in DICOM medical imaging. The opencodecs name
``"jpegls"`` matches imagecodecs's naming for compatibility.

Modes::

    near_lossless=0   # mathematically lossless (default)
    near_lossless=N   # bounded error: each sample within N of source
                      # (smaller files, more error; N=1..9 typical)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _jpegls_encode, _jpegls_decode, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._charls",
    "encode", "decode",
)


class JpegLsCodec(Codec):
    """JPEG-LS via CharLS — predictive lossless / near-lossless."""

    name = "jpegls"
    aliases = ("jls", "charls")
    file_extensions = (".jls",)

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
        # JPEG-LS bitstream starts with SOI (0xFFD8) followed by SOF55
        # (0xFFF7). Some implementations include extra app markers
        # between SOI and SOF; we accept either layout by matching the
        # SOI prefix and looking for the SOF55 marker in the first 64
        # bytes (avoids false-positives on regular JPEG).
        if len(head) < 4 or head[0] != 0xFF or head[1] != 0xD8:
            return False
        # Look for 0xFFF7 in the first 64 bytes after SOI.
        scan = bytes(head[:64])
        return b"\xff\xf7" in scan

    def encode(self, data: Any, *, dest=None, near_lossless: int = 0,
               **opts) -> bytes | None:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        out = _jpegls_encode(data, near_lossless=int(near_lossless))
        return _write_dest(out, dest)

    def decode(self, src: Any, **opts) -> np.ndarray:
        return _jpegls_decode(_read_src(src))


__all__ = ["JpegLsCodec"]
