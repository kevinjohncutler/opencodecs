"""SnappyCodec — Google's Snappy block compression via libsnappy.

Snappy compresses at ~500 MB/s and decompresses at ~1 GB/s with ~2x
compression ratios. Used heavily in Parquet, Hadoop, Bigtable
pipelines where speed dominates ratio. Raw block format — no framing,
no checksums.

This codec wraps libsnappy's C API directly via Cython. Performance
is bounded by libsnappy itself (already SIMD-tuned upstream); the
wrapper overhead is at parity with imagecodecs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _snappy_encode, _snappy_decode, _snappy_check_signature, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._snappy",
    "encode", "decode", "check_signature",
)


class SnappyCodec(Codec):
    """Native Snappy — Google's fast block compressor."""

    name = "snappy"
    file_extensions = (".sz",)

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
        return _snappy_check_signature(head)

    def encode(self, data: Any, *, dest=None, **opts) -> bytes | None:
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        return _write_dest(_snappy_encode(data), dest)

    def decode(self, src: Any, **opts) -> bytes:
        return _snappy_decode(_read_src(src))


__all__ = ["SnappyCodec"]
