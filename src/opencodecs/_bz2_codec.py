"""Bz2Codec — bzip2 via Python's stdlib bz2 module.

bzip2 is an older block-sorting compressor (Burrows-Wheeler + RLE +
Huffman). It typically achieves a slightly higher ratio than gzip but
is meaningfully slower than both zstd and lzma. Mainly relevant for
reading legacy ``.bz2`` archives — zstd or lzma is preferred for new
writes.

The stdlib ``bz2`` module wraps libbz2 and releases the GIL during the
compression call.
"""

from __future__ import annotations

import bz2
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest


class Bz2Codec(Codec):
    """bzip2 via the stdlib ``bz2`` module."""

    name = "bz2"
    aliases = ("bzip2",)
    file_extensions = (".bz2",)

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
        # bzip2 file magic: 'BZh' followed by an ASCII digit (block size).
        return (
            len(head) >= 4
            and head[0] == 0x42 and head[1] == 0x5A and head[2] == 0x68
            and 0x31 <= head[3] <= 0x39
        )

    def encode(self, data: Any, *, dest=None,
               level: int | None = None,
               **opts) -> bytes | None:
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        clevel = 9 if level is None else int(level)
        # bz2 accepts compresslevel 1..9.
        if clevel < 1:
            clevel = 1
        if clevel > 9:
            clevel = 9
        out = bz2.compress(data, compresslevel=clevel)
        return _write_dest(out, dest)

    def decode(self, src: Any, **opts) -> bytes:
        return bz2.decompress(_read_src(src))


__all__ = ["Bz2Codec"]
