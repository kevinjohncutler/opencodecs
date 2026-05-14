"""Htj2kCodec — HTJ2K (JPEG-2000 Part 15) via OpenJPH.

HTJ2K is the high-throughput JPEG-2000 codestream defined in ISO/IEC
15444-15. It targets 10-30× faster encode/decode than classic JPEG-2000
at near-identical compression ratios, while staying within the
JPEG-2000 ecosystem (same wavelet basis, same image model).

Used in DICOM medical imaging (transfer syntax 1.2.840.10008.1.2.4.201)
and increasingly in the broadcast / cinema pipeline.

Modes::

    level=None     # reversible — mathematically lossless (default)
    level=0.1      # irreversible (lossy) — smaller files, more loss
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _htj2k_encode, _htj2k_decode, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._openjph",
    "encode", "decode",
)


class Htj2kCodec(Codec):
    """HTJ2K (JPEG-2000 Part-15) via OpenJPH."""

    name = "htj2k"
    aliases = ("openjph", "jph", "j2c")
    file_extensions = (".j2c", ".jph")

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.uint8, np.uint16, np.int8, np.int16)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        # HTJ2K raw codestream starts with SOC marker 0xFF4F + SIZ
        # marker 0xFF51. JP2-wrapped HTJ2K starts with the JP2
        # signature box (\x00\x00\x00\x0Cjp2 \r\n\x87\n) — same as
        # classic JPEG-2000. Match either.
        if len(head) < 4:
            return False
        if head[0] == 0xFF and head[1] == 0x4F and \
           head[2] == 0xFF and head[3] == 0x51:
            return True
        return len(head) >= 12 and bytes(head[:12]) == (
            b"\x00\x00\x00\x0Cjp2 \r\n\x87\n"
        )

    def encode(self, data: Any, *, dest=None,
               level: float | None = None,
               num_decomp: int = 5,
               **opts) -> bytes | None:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        out = _htj2k_encode(data, level=level, num_decomp=int(num_decomp))
        return _write_dest(out, dest)

    def decode(self, src: Any, **opts) -> np.ndarray:
        return _htj2k_decode(_read_src(src))


__all__ = ["Htj2kCodec"]
