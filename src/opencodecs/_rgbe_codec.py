"""RgbeCodec — Codec adapter for the native ``_rgbe`` extension.

Radiance HDR (``.hdr``) is the canonical interchange format for high-
dynamic-range floating-point imagery. The file is a text-headed
container holding an RLE-compressed RGBE pixel stream — each pixel is
three 8-bit mantissas sharing one 8-bit exponent, giving ~127 stops of
range in 32 bits per pixel.

Encoded bytes are the full file (header + RLE pixels); decoded output
is an ``(H, W, 3)`` float32 array.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core.buffers import array_output
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs


_rgbe_encode, _rgbe_decode, _rgbe_check_signature, _HAVE_BACKEND = import_or_stubs(
    "opencodecs.codecs._rgbe",
    "encode", "decode", "check_signature",
)


class RgbeCodec(Codec):
    """Radiance HDR (RGBE) image codec."""

    name = "rgbe"
    aliases = ("hdr", "radiance")
    file_extensions = (".hdr",)

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.float32,)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        return _rgbe_check_signature(head)

    def encode(self, data: Any, *, header=None, rle=None, dest=None,
               **opts) -> bytes | None:
        """Encode float RGB. ``header`` (default True) and ``rle``
        (default: same as ``header``) follow imagecodecs
        ``rgbe_encode``, except that ``rle=False`` with a header writes
        flat scanlines, where imagecodecs writes RLE regardless; see
        :func:`opencodecs.codecs._rgbe.encode`."""
        arr = data if isinstance(data, np.ndarray) else np.asarray(data)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        out = _rgbe_encode(arr, header=header, rle=rle)
        return _write_dest(out, dest)

    def decode(self, src: Any, *, header=None, rle=None, out=None,
               **opts) -> np.ndarray:
        """Decode to ``(H, W, 3)`` float32, honoring the orientation the
        resolution line states. ``header=False`` reads a bare pixel
        stream into ``out``, as imagecodecs ``rgbe_decode`` does."""
        data = _read_src(src)
        return _rgbe_decode(data, header=header, rle=rle,
                            out=out if out is None else array_output(out))


__all__ = ["RgbeCodec"]
