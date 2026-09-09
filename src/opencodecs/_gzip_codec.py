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
import zlib
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
        """Inflate a gzip stream, allocating the output once.

        ``gzip.decompress`` discovers the output size by growing a
        buffer and joining the pieces, which peaks at over twice the
        result: 148.8 MB to produce 67 MB. A gzip member ends with
        ISIZE, the uncompressed length mod 2**32, so the size is known
        up front and zlib can be handed it as the initial buffer --
        67.2 MB and 4.5 ms against 10.5 ms, for the same bytes.

        Note which zlib entry point: the module-level ``decompress``
        takes ``bufsize`` as an initial allocation, while
        ``decompressobj().decompress`` takes ``max_length``, which caps
        what is returned and does nothing about the growth. Using that
        one instead still peaked at 134 MB.
        """
        data = _read_src(src)
        out = self._decode_single_member(data)
        # Concatenated members are legal gzip and zlib stops after the
        # first, so anything not provably single-member goes to the
        # stdlib, which handles them.
        return gzip.decompress(data) if out is None else out

    @staticmethod
    def _decode_single_member(data) -> bytes | None:
        """The pre-sized inflate, or None if it cannot be trusted.

        ISIZE describes the LAST member, so on a concatenated stream it
        can agree with the first member's length and the short read
        looks correct -- two 198-byte members did exactly that. The
        witness that rules it out is the member's own trailer, CRC32
        followed by length: if those eight bytes end the file and occur
        nowhere else in it, the member that ``out`` came from is the
        only one, because a second member would have to sit after that
        trailer. A chance match costs the slow path and nothing else.
        """
        if len(data) < 4:
            return None
        data = bytes(data)
        isize = int.from_bytes(data[-4:], "little")
        if not 0 < isize < (1 << 32) - 1:
            # 0 is an empty member, and a 4 GB output has wrapped ISIZE
            # and would under-allocate. Neither is worth a special case.
            return None
        try:
            out = zlib.decompress(data, 31, isize)
        except zlib.error:
            return None      # malformed: let gzip report it
        if len(out) != isize:
            return None
        trailer = zlib.crc32(out).to_bytes(4, "little") + data[-4:]
        if not data.endswith(trailer) or data.count(trailer) != 1:
            return None
        return out


__all__ = ["GzipCodec"]
