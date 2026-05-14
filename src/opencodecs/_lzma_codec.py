"""LzmaCodec — LZMA/XZ via Python's stdlib lzma module.

LZMA achieves higher compression ratios than zstd / deflate at the cost
of being 5-10x slower to encode. Common file format on Linux (.xz), in
package archives (.tar.xz), and as an HDF5 dataset filter.

The stdlib ``lzma`` module is a C wrapper around liblzma (xz-utils) and
releases the GIL during the compression call, so this codec gets
parallelism for free when used from threaded code.

Defaults to ``preset=6`` (matches ``xz``'s default — a sensible
speed/ratio balance). Encoded blobs are valid XZ stream format.
"""

from __future__ import annotations

import lzma
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest


class LzmaCodec(Codec):
    """LZMA / XZ via the stdlib ``lzma`` module."""

    name = "lzma"
    aliases = ("xz",)
    file_extensions = (".xz", ".lzma")

    has_native = True   # stdlib lzma always present
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.uint8,)
    supports_color = False

    def signature(self, head: bytes) -> bool:
        # XZ stream header: \xFD 7z X Z \x00 (6 bytes).
        return len(head) >= 6 and bytes(head[:6]) == b"\xfd7zXZ\x00"

    def encode(self, data: Any, *, dest=None,
               level: int | None = None,
               **opts) -> bytes | None:
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        preset = 6 if level is None else int(level)
        # Clamp to lzma's accepted range 0..9 (with extreme flag for 0-9e).
        if preset < 0:
            preset = 0
        if preset > 9:
            preset = 9
        out = lzma.compress(data, preset=preset)
        return _write_dest(out, dest)

    def decode(self, src: Any, **opts) -> bytes:
        return lzma.decompress(_read_src(src))


__all__ = ["LzmaCodec"]
