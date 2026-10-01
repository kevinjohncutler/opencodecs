"""MozJpegCodec — JPEG encoder via MozJPEG (smaller files than libjpeg-turbo).

MozJPEG is Mozilla's libjpeg-turbo fork that adds progressive encoding
with trellis-quantization optimization. Files are 10-15% smaller than
libjpeg-turbo at the same quality level, fully decodable by any
standard JPEG decoder. Encode is slower (~2x); decode is identical.

This codec is **encode-focused**: ``decode`` works but is no faster
than the regular ``jpeg`` codec (same underlying libjpeg-turbo decoder
in MozJPEG), and streams MozJPEG cannot decode (12-bit, lossless) are
handed to the ``jpeg`` codec. Pair MozJPEG with the standard JPEG
decoder on the read side for typical archive/web pipelines.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core.buffers import array_output
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
               colorspace: int | str | None = None,
               outcolorspace: int | str | None = None,
               subsampling: str | tuple[int, int] | None = None,
               optimize: bool | None = None,
               smoothing: int | None = None,
               notrellis: bool | None = None,
               quanttable: int | None = None,
               progressive: bool | None = True) -> bytes | None:
        """Encode an ndarray as JPEG via MozJPEG.

        The parameters are those of ``imagecodecs.mozjpeg_encode`` (see
        :func:`opencodecs.codecs._mozjpeg.encode`). An option this
        codec does not know is a ``TypeError``, and one MozJPEG's API
        cannot honor raises, rather than either being dropped.
        """
        out = _moz_encode(
            data, level=level, colorspace=colorspace,
            outcolorspace=outcolorspace, subsampling=subsampling,
            optimize=optimize, smoothing=smoothing, notrellis=notrellis,
            quanttable=quanttable, progressive=progressive,
        )
        return _write_dest(out, dest)

    def decoder(self):
        """Create an explicitly owned reusable decode handle."""
        from .codecs._mozjpeg import DecoderContext
        from .core.buffers import ImageDecoderContext
        return ImageDecoderContext(DecoderContext())

    def decode(self, src: Any, *, out=None, tables=None, header=None,
               colorspace=None, outcolorspace=None, fancyupsampling=None,
               shape=None, bitspersample=None, scale=None,
               scale_num=None, scale_denom=None) -> np.ndarray:
        """Decode a JPEG stream; see :func:`opencodecs.codecs._mozjpeg.decode`.

        12-bit and lossless streams, which MozJPEG cannot decode, are
        decoded by the ``jpeg`` codec.
        """
        return _moz_decode(
            _read_src(src), out=out if out is None else array_output(out),
            tables=tables, header=header, colorspace=colorspace,
            outcolorspace=outcolorspace, fancyupsampling=fancyupsampling,
            shape=shape, bitspersample=bitspersample,
            scale=scale, scale_num=scale_num, scale_denom=scale_denom)


__all__ = ["MozJpegCodec"]
