"""Blosc2Codec — Codec adapter wrapping the native _blosc2 extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

_blosc2_encode, _blosc2_decode, _blosc2_check_signature, _HAVE_BACKEND = import_or_stubs(
    "opencodecs.codecs._blosc2",
    "encode", "decode", "check_signature",
)


class Blosc2Codec(Codec):
    """Native blosc2 meta-compressor (c-blosc2)."""

    name = "blosc2"
    file_extensions = (".b2",)

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
        return _blosc2_check_signature(head)

    def encode(self, data: Any, *, dest=None, level: int | None = None,
               compressor: str | None = None,
               typesize: int | None = None,
               shuffle: bool | None = None,
               **opts) -> bytes | None:
        if isinstance(data, np.ndarray):  # pragma: no cover - blosc2 is byte-oriented; ndarray-aware encode unused in tests
            if typesize is None:
                typesize = data.dtype.itemsize
            data = data.tobytes()
        compressed = _blosc2_encode(
            data, level=level, compressor=compressor,
            typesize=typesize, shuffle=shuffle,
        )
        return _write_dest(compressed, dest)

    def decode(self, src: Any, **opts) -> bytes:
        return _blosc2_decode(_read_src(src))



__all__ = ["Blosc2Codec"]
