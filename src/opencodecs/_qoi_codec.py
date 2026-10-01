"""QoiCodec — Codec adapter wrapping the native _qoi extension.

QOI is a trivial-to-implement lossless image format, vendored from
phoboslab/qoi as a single header. Encode/decode are bytes-in/bytes-out
only — no streaming, no multi-frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core.buffers import array_output
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs
from ._png_codec import _no_encode_out

_qoi_encode, _qoi_decode, _qoi_check_signature, _HAVE_BACKEND = import_or_stubs(
    "opencodecs.codecs._qoi",
    "encode", "decode", "check_signature",
)


class QoiCodec(Codec):
    """Native QOI codec (Quite OK Image Format)."""

    name = "qoi"
    file_extensions = (".qoi",)

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
        return _qoi_check_signature(head)

    def encode(self, arr: np.ndarray, *, dest=None, srgb: bool = True,
               numthreads: int | None = None, out: Any = None,
               **opts) -> bytes | None:
        """Encode a (H, W, 3) or (H, W, 4) uint8 array as QOI.

        ``srgb`` sets the informative header colorspace byte: True (the
        default, for RGB and RGBA alike, as the QOI specification and its
        reference encoder do) writes 0, "sRGB with linear alpha"; False
        writes 1, "all channels linear". imagecodecs writes 1 for RGBA,
        so RGBA output differs from imagecodecs in header byte 13 only;
        the pixel data is identical. ``numthreads``, which the package's
        other encoders take and ``imagecodecs.qoi_encode`` does not
        define, is accepted and has nothing to do, since QOI encoding is
        single-threaded. ``out``, imagecodecs' output buffer, may be
        ``None``; anything else raises ``TypeError``, since the encoded
        bytes are returned or written to ``dest``. Unknown options raise
        ``TypeError``.
        """
        if opts:
            raise TypeError(
                f"qoi encode: unexpected option(s) {sorted(opts)}")
        _no_encode_out("qoi", out)
        data = _qoi_encode(arr, srgb=bool(srgb))
        return _write_dest(data, dest)

    def decode(self, src: Any, *, out=None,
               numthreads: int | None = None) -> np.ndarray:
        """Decode a QOI image to ``(H, W, 3)`` or ``(H, W, 4)`` uint8.

        ``out`` is the only option ``imagecodecs.qoi_decode`` defines.
        ``numthreads`` is accepted, as by ``encode``, and has nothing to
        do, since QOI decoding is single-threaded. Any other
        option raises ``TypeError`` rather than being dropped.
        """
        if out is None:
            return _qoi_decode(_read_src(src))
        return _qoi_decode(_read_src(src), out=array_output(out))



__all__ = ["QoiCodec"]
