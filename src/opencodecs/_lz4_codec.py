"""Lz4Codec — Codec adapter wrapping the native _lz4 extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

_lz4_encode, _lz4_decode, _lz4_check_signature, _HAVE_BACKEND = import_or_stubs(
    "opencodecs.codecs._lz4",
    "encode", "decode", "check_signature",
)


class Lz4Codec(Codec):
    """Native LZ4 codec (frame format, .lz4 files).

    Bytes-in / bytes-out only — LZ4 is a generic compressor, not an
    image codec. Uses the LZ4 frame format (magic 0x184D2204), which is
    self-describing and matches imagecodecs's ``lz4f_encode``/``lz4f_decode``.
    """

    name = "lz4"
    file_extensions = (".lz4",)

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
        return _lz4_check_signature(head)

    def encode(self, data: Any, *, dest=None, level: int | None = None,
               **opts) -> bytes | None:
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        compressed = _lz4_encode(data, level=level)
        return _write_dest(compressed, dest)

    def decode(self, src: Any, **opts) -> bytes:
        return _lz4_decode(_read_src(src))



__all__ = ["Lz4Codec"]
