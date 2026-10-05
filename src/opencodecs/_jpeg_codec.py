"""JpegCodec — Codec adapter wrapping the native _jpeg extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec
from .core.buffers import array_output
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs
from .backends import select as _select_backend

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

    supported_dtypes = (np.uint8, np.uint16)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        return _jpeg_check_signature(head)

    def encode(
        self,
        data: Any,
        *,
        dest=None,
        level: int | None = None,
        colorspace: int | str | None = None,
        outcolorspace: int | str | None = None,
        subsampling: str | tuple[int, int] | None = None,
        optimize: bool | None = None,
        smoothing: int | None = None,
        lossless: bool | None = None,
        predictor: int | None = None,
        bitspersample: int | None = None,
        validate: bool | None = None,
        iccprofile: bytes | None = None,
        backend: str | None = None,
    ) -> bytes | None:
        """Encode an ndarray as JPEG.

        The parameters are those of ``imagecodecs.jpeg8_encode`` (see
        :func:`opencodecs.codecs._jpeg.encode`), plus ``iccprofile``,
        which embeds an ICC color profile in APP2 markers. Every one is
        passed to the encoder; an option this codec does not know is a
        ``TypeError`` rather than something silently dropped.
        ``validate`` has no effect, in imagecodecs as here; it is taken
        for compatibility.

        ``backend="nvimgcodec"`` encodes on an NVIDIA GPU (opt-in; see
        :mod:`opencodecs.backends`): 8-bit gray or RGB, lossy, with
        ``level``, ``subsampling`` and ``optimize``; other options raise.
        """
        hardware = _select_backend(backend, self.name, "encode")
        if hardware is not None:
            encoded = hardware.encode_jpeg(
                data, level=level, colorspace=colorspace,
                outcolorspace=outcolorspace, subsampling=subsampling,
                optimize=optimize, smoothing=smoothing, lossless=lossless,
                predictor=predictor, bitspersample=bitspersample,
                validate=validate, iccprofile=iccprofile)
            return _write_dest(encoded, dest)
        encoded = _jpeg_encode(
            data, level=level, colorspace=colorspace,
            outcolorspace=outcolorspace, subsampling=subsampling,
            optimize=optimize, smoothing=smoothing, lossless=lossless,
            predictor=predictor, bitspersample=bitspersample,
            validate=validate, iccprofile=iccprofile)
        return _write_dest(encoded, dest)

    def decode(self, src: Any, *, out=None, tables=None, header=None,
               colorspace=None, outcolorspace=None, fancyupsampling=None,
               shape=None, bitspersample=None, scale=None,
               scale_num=None, scale_denom=None,
               backend: str | None = None) -> np.ndarray:
        """Decode a JPEG stream; see :func:`opencodecs.codecs._jpeg.decode`.

        The parameters are those of ``imagecodecs.jpeg_decode`` plus the
        DCT-domain ``scale``.

        ``backend="nvimgcodec"`` decodes 8-bit gray or color lossy JPEG
        on an NVIDIA GPU (opt-in; see :mod:`opencodecs.backends`).
        Pixels differ from libjpeg-turbo's by a few levels (IDCT and
        upsampling rounding). Only ``out=`` is taken with it; it may be a
        CuPy array or one from :func:`opencodecs.backends.pinned_empty`.
        """
        hardware = _select_backend(backend, self.name, "decode")
        if hardware is not None:
            given = dict(tables=tables, header=header, colorspace=colorspace,
                         outcolorspace=outcolorspace,
                         fancyupsampling=fancyupsampling, shape=shape,
                         bitspersample=bitspersample, scale=scale,
                         scale_num=scale_num, scale_denom=scale_denom)
            bad = sorted(k for k, v in given.items() if v is not None)
            if bad:
                raise ValueError(
                    f"jpeg decode: backend='nvimgcodec' does not take "
                    f"{', '.join(bad)}; use backend=None for them")
            return hardware.decode(self.name, _read_src(src), out=out)
        return _jpeg_decode(
            _read_src(src), out=out if out is None else array_output(out),
            tables=tables, header=header, colorspace=colorspace,
            outcolorspace=outcolorspace, fancyupsampling=fancyupsampling,
            shape=shape, bitspersample=bitspersample,
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
