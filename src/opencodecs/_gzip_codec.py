"""GzipCodec: gzip-format compression (RFC 1952).

The gzip format wraps a raw deflate stream with a 10-byte header (RFC
1952: magic, compression method, flags, mtime, XFL, OS) plus an 8-byte
trailer (CRC32 + original size). It is the canonical interchange format
for ``.gz`` archives and the HTTP ``Content-Encoding: gzip`` transport.

Encode uses the same deflate engine as :class:`DeflateCodec`
(libdeflate when linked, as imagecodecs does) and writes a fixed header:
MTIME 0, OS 255 ("unknown"), XFL from the level. The bytes are then the
same on every platform and Python version, and match imagecodecs's
``gzip_encode``. Encoding through the stdlib ``gzip`` module instead let
zlib write its own OS byte, 19 on macOS or 3 on Linux under Python 3.12
and 255 under 3.13, so identical input gave different files. A build
without the ``_deflate`` extension encodes with the stdlib ``zlib``
module and writes the same header, as the extension's own zlib fallback
does, so gzip encode works on every build.

Decode uses libdeflate when it is linked (``_deflate.gzip_decode``),
which reads every member of a multi-member file as the stdlib does and
is 1.7x faster than the stdlib's zlib on the same data. Input libdeflate
rejects goes to the stdlib ``gzip`` module, so a corrupt file raises the
same error it always has. Builds without libdeflate decode with the
stdlib.
"""

from __future__ import annotations

import gzip
import zlib
from typing import Any

import numpy as np

from .core.codec import Codec
from .core.buffers import decoded_output, encoded_output
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

_native_gzip_encode, _HAVE_BACKEND = import_or_stubs(
    "opencodecs.codecs._deflate", "gzip_encode",
)
_native_gzip_decode, _native_backend, _HAVE_DECODE = import_or_stubs(
    "opencodecs.codecs._deflate", "gzip_decode", "backend",
)
#: libdeflate is linked, so decode goes through ``_deflate.gzip_decode``.
_LIBDEFLATE_DECODE = _HAVE_DECODE and _native_backend() == "libdeflate"


def _zlib_gzip_encode(data, level=None) -> bytes:
    """One gzip member from the stdlib ``zlib`` module, header fixed.

    For builds without the ``_deflate`` extension. The header is the one
    ``_deflate.gzip_encode`` writes (MTIME 0, XFL from the level, OS
    255); the level is clamped to zlib's 0 to 9, 6 by default.
    """
    lvl = 6 if level is None else min(max(int(level), 0), 9)
    view = memoryview(data).cast("B")
    compressor = zlib.compressobj(lvl, zlib.DEFLATED, -15)
    body = compressor.compress(view) + compressor.flush()
    xfl = 4 if lvl < 2 else (2 if lvl >= 8 else 0)
    header = bytes((0x1F, 0x8B, 8, 0, 0, 0, 0, 0, xfl, 255))
    trailer = ((zlib.crc32(view) & 0xFFFFFFFF).to_bytes(4, "little")
               + (len(view) & 0xFFFFFFFF).to_bytes(4, "little"))
    return header + body + trailer


def gzip_encode(data, level=None) -> bytes:
    """Encode bytes-like ``data`` as one gzip member with a fixed header.

    ``_deflate.gzip_encode`` when the extension is built, otherwise the
    stdlib ``zlib`` with the same header.
    """
    if _HAVE_BACKEND:
        return _native_gzip_encode(data, level=level)
    return _zlib_gzip_encode(data, level)


class GzipCodec(Codec):
    """gzip: the native deflate engine both ways, the stdlib as fallback."""

    name = "gzip"
    file_extensions = (".gz", ".gzip")

    has_native = True   # both directions fall back to the stdlib zlib
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
               out=None) -> bytes | None:
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        # Level 6 by default, clamped to 0..12 (libdeflate's range; the
        # zlib fallbacks top out at 9), as imagecodecs.gzip_encode does.
        return encoded_output(
            gzip_encode(data, level=level), out, self.name, dest)

    def decode(self, src: Any, *, out=None) -> bytes | memoryview:
        """Inflate a gzip stream, allocating the output once.

        ``out`` follows imagecodecs.gzip_decode's contract (an int
        capacity or a writable buffer); the inflated bytes are copied
        into a given buffer, and a result that does not fit raises.

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
        if _LIBDEFLATE_DECODE:
            # Every member, sized from the trailer (see gzip_decode).
            # Whatever libdeflate refuses goes to the stdlib below, so
            # malformed input raises exactly what it raised before.
            try:
                decoded = _native_gzip_decode(data)
            except RuntimeError:
                decoded = None
        else:
            decoded = self._decode_single_member(data)
        # Concatenated members are legal gzip and zlib stops after the
        # first, so anything not provably single-member goes to the
        # stdlib, which handles them.
        if decoded is None:
            decoded = gzip.decompress(data)
        return decoded_output(decoded, out, self.name)

    @staticmethod
    def _decode_single_member(data) -> bytes | None:
        """The pre-sized stdlib inflate, or None if it cannot be trusted.

        The decode path of builds without libdeflate.

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


__all__ = ["GzipCodec", "gzip_encode"]
