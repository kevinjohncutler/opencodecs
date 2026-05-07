"""QoiCodec — Codec adapter wrapping the native _qoi extension.

QOI is a trivial-to-implement lossless image format, vendored from
phoboslab/qoi as a single header. Encode/decode are bytes-in/bytes-out
only — no streaming, no multi-frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

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
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return _qoi_check_signature(head)

    def encode(self, arr: np.ndarray, *, dest=None, **opts) -> bytes | None:
        srgb = bool(opts.get("srgb", True))
        data = _qoi_encode(arr, srgb=srgb)
        return _write_dest(data, dest)

    def decode(self, src: Any, **opts) -> np.ndarray:
        return _qoi_decode(_read_src(src))



__all__ = ["QoiCodec"]
