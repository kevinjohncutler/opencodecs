"""PngCodec — Codec adapter wrapping the native _png extension (libspng)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec
from .core.buffers import array_output
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

# Stands in for "filter_choice not passed", so the native default
# applies and the imagecodecs-style filter= alias can be used instead.
_UNSET = object()

(
    _png_encode, _png_decode, _png_check_signature,
    _png_read_icc, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._png",
    "encode", "decode", "check_signature", "read_icc_profile",
)


def _no_encode_out(name, out):
    """Accept imagecodecs' ``out=None``; refuse a buffer loudly."""
    if out is not None:
        raise TypeError(
            f"{name} encode: out= is not supported; the encoded bytes are "
            f"returned, or written to dest=")


class PngCodec(Codec):
    """Native PNG codec backed by libspng.

    Decode keeps the PNG color type and returns uint8 or uint16 samples:
    gray is ``(H, W)`` with 1/2/4-bit samples scaled to 8 bits, gray+alpha
    ``(H, W, 2)``, RGB ``(H, W, 3)``, RGBA ``(H, W, 4)``, and an indexed
    image is ``(H, W, 3)`` with the palette applied. A tRNS chunk adds the
    alpha channel it defines, so gray becomes ``(H, W, 2)`` and RGB or
    indexed ``(H, W, 4)``. This is the same mapping imagecodecs (libpng
    with expand and tRNS-to-alpha) returns.
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

    def encode(
        self,
        data: Any,
        *,
        dest=None,
        level: int | None = None,
        filter_choice: Any = _UNSET,
        filter: Any = None,
        strategy: int | None = None,
        iccprofile: bytes | None = None,
        iccprofile_name: str = "ICC profile",
        numthreads: int | None = None,
        out: Any = None,
        **opts,
    ) -> bytes | None:
        """Encode an ndarray as PNG.

        ``level`` is the deflate level 0-9. ``filter_choice`` picks the
        row filters libspng tries (``"fast"`` by default, or ``"all"``,
        ``"off"``, ``"none"``, ``"sub"``, ``"up"``, ``"avg"``,
        ``"paeth"``, or a bitmask); ``filter`` is the imagecodecs name
        for the same setting and takes its ``PNG.FILTER`` values or
        names. ``strategy`` is the zlib strategy 0-4 or its
        ``PNG.STRATEGY`` name, as in imagecodecs.

        ``iccprofile`` embeds an ICC color profile in an ``iCCP``
        chunk. The PNG renderer will use it as the document's
        canonical color space description. ``iccprofile_name`` is the
        free-text identifier (truncated to 79 ASCII chars).

        uint16 input of either byte order is stored by value.
        ``numthreads``, which the package's other encoders take and
        ``imagecodecs.png_encode`` does not define, is accepted and has
        nothing to do, since PNG encoding is single-threaded. ``out``,
        imagecodecs' output buffer, may be ``None``; anything else raises
        ``TypeError``, since the encoded bytes are returned or written to
        ``dest``. Options this codec does not know raise ``TypeError``
        rather than being dropped.
        """
        if opts:
            raise TypeError(
                f"png encode: unexpected option(s) {sorted(opts)}")
        _no_encode_out("png", out)
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        options = {} if filter_choice is _UNSET else {"filter_choice": filter_choice}
        encoded = _png_encode(
            data, level=level, filter=filter, strategy=strategy,
            iccprofile=iccprofile,
            iccprofile_name=iccprofile_name,
            **options,
        )
        return _write_dest(encoded, dest)

    def decode(self, src: Any, *, out=None,
               numthreads: int | None = None) -> np.ndarray:
        """Decode a PNG; see the class docstring for the output layout.

        ``out`` is the only option ``imagecodecs.png_decode`` defines.
        ``numthreads`` is accepted, as by ``encode``, and has nothing to
        do, since PNG decoding is single-threaded. Any other
        option raises ``TypeError`` rather than being dropped.
        """
        return _png_decode(_read_src(src), out=out if out is None else array_output(out))

    def decode_rows(self, src):
        """Yield owned complete rows or explicitly located Adam7 pass updates."""
        from .core.rows import decode_png_rows
        return decode_png_rows(src)

    def encode_rows(self, rows, *, shape, dtype, dest=None,
                    numthreads: int | None = None, **options):
        """Encode ordinary PNG from sequential rows and a declared image shape.

        ``numthreads`` is accepted, as by ``encode``, and has nothing to
        do, since PNG encoding is single-threaded.
        """
        from .core.rows import encode_png_rows
        return encode_png_rows(rows, shape=shape, dtype=dtype, dest=dest, **options)

    def read_icc_profile(self, src: Any) -> bytes | None:
        """Return the embedded ICC profile bytes, or ``None`` if absent.

        Only reads PNG header chunks — fast even for large files.
        """
        return _png_read_icc(_read_src(src))



__all__ = ["PngCodec"]
