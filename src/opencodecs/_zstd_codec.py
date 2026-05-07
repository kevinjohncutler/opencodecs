"""ZstdCodec — Codec adapter wrapping the native _zstd extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._optional_backend import import_or_stubs
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest

_zstd_encode, _zstd_decode, _zstd_check_signature, _HAVE_BACKEND = import_or_stubs(
    "opencodecs.codecs._zstd",
    "encode", "decode", "check_signature",
)


class ZstdCodec(Codec):
    """Native zstd codec (Facebook's Zstandard).

    Bytes-in / bytes-out only — zstd is a generic compressor, not an
    image codec. Useful as a chunk compressor in zarr or as a transport
    compressor for any opencodecs format.
    """

    name = "zstd"
    file_extensions = (".zst",)

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
        return _zstd_check_signature(head)

    def encode(self, data: Any, *, dest=None, level: int | None = None,
               **opts) -> bytes | None:
        # Accept ndarrays too — flatten via tobytes(). For arrays the
        # caller is responsible for remembering shape/dtype.
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        compressed = _zstd_encode(data, level=level)
        return _write_dest(compressed, dest)

    def decode(self, src: Any, **opts) -> bytes:
        return _zstd_decode(_read_src(src))


__all__ = ["ZstdCodec"]
