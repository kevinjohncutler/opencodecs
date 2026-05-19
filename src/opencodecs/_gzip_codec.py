"""GzipCodec — gzip-format compression via the stdlib gzip module.

The gzip format wraps a raw deflate stream with a 10-byte header (RFC
1952: magic, compression method, flags, mtime, XFL, OS) plus an 8-byte
trailer (CRC32 + original size). It is the canonical interchange format
for ``.gz`` archives and the HTTP ``Content-Encoding: gzip`` transport.

The compressor is the same deflate engine used by :class:`DeflateCodec`,
so encode/decode throughput is governed by the underlying zlib (or
zlib-ng-compat, when linked). The stdlib ``gzip`` module is a thin
wrapper over zlib — overhead vs raw deflate is dominated by the
wrapper bytes, not Python.
"""

from __future__ import annotations

import gzip
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest


class GzipCodec(Codec):
    """gzip via the stdlib ``gzip`` module."""

    name = "gzip"
    file_extensions = (".gz", ".gzip")

    has_native = True   # stdlib gzip always present
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.uint8,)
    supports_color = False

    def signature(self, head: bytes) -> bool:
        # RFC 1952: gzip stream starts with 0x1F 0x8B.
        return len(head) >= 2 and head[0] == 0x1F and head[1] == 0x8B

    def encode(self, data: Any, *, dest=None,
               level: int | None = None,
               **opts) -> bytes | None:
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        clevel = 6 if level is None else int(level)
        if clevel < 0:
            clevel = 0
        if clevel > 9:
            clevel = 9
        out = gzip.compress(data, compresslevel=clevel, mtime=0)
        return _write_dest(out, dest)

    def decode(self, src: Any, **opts) -> bytes:
        return gzip.decompress(_read_src(src))


__all__ = ["GzipCodec"]
