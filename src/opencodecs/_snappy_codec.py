"""SnappyCodec — Google's Snappy block compression via libsnappy.

Snappy compresses at ~500 MB/s and decompresses at ~1 GB/s with ~2x
compression ratios. Used heavily in Parquet, Hadoop, Bigtable
pipelines where speed dominates ratio. Raw block format — no framing,
no checksums.

This codec wraps libsnappy's C API directly via Cython. Performance
is bounded by libsnappy itself (already SIMD-tuned upstream); the
wrapper overhead is at parity with imagecodecs.

Two codecs live here, because Snappy has two formats. ``snappy`` is the
raw block, byte-identical to imagecodecs.snappy_encode / snappy_decode.
``snappy_framed`` is Snappy's framing format (Google's
framing_format.txt), a stream of checksummed chunks behind the stream
identifier ``ff 06 00 00 sNaPpY``. The framing format is the one that
names ``.sz`` as its file extension, so ``.sz`` files belong to
``snappy_framed``; routing them to the raw block codec made every real
``.sz`` file fail to read.

opencodecs 0.4.0 and earlier wrote ``.sz`` files as raw Snappy blocks.
Those still read: ``snappy_framed`` decodes input that does not start
with the stream identifier as a raw block, and raises if it is not a
valid one either. A raw block cannot be mistaken for a framed stream
unless its first ten bytes happen to spell the stream identifier.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core.buffers import decoded_output, encoded_output, native_decoded
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _snappy_encode, _snappy_decode, _snappy_check_signature, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._snappy",
    "encode", "decode", "check_signature",
)
(
    _framed_encode, _framed_decode, _framed_check_signature, _HAVE_FRAMED,
) = import_or_stubs(
    "opencodecs.codecs._snappy",
    "framed_encode", "framed_decode", "framed_check_signature",
)


class SnappyCodec(Codec):
    """Native Snappy, Google's fast block compressor (raw block format)."""

    name = "snappy"
    # No extension: the raw block format defines none, and ".sz" is the
    # framing format's (see SnappyFramedCodec).
    file_extensions = ()

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

    def encode(self, data: Any, *, dest=None, out=None) -> bytes | None:
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        return encoded_output(_snappy_encode(data), out, self.name, dest)

    def decode(self, src: Any, *, out=None) -> bytes | memoryview:
        return native_decoded(_snappy_decode, _read_src(src), out)


class SnappyFramedCodec(Codec):
    """Snappy framing format, the ``.sz`` stream format."""

    name = "snappy_framed"
    aliases = ("snappy-framed", "sz")
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
        return _framed_check_signature(head)

    def encode(self, data: Any, *, dest=None, out=None) -> bytes | None:
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        return encoded_output(_framed_encode(data), out, self.name, dest)

    def decode(self, src: Any, *, out=None,
               verify: bool = True) -> bytes | memoryview:
        """Decode a framing-format stream, or a legacy raw block.

        Data without the stream identifier is what opencodecs 0.4.0 and
        earlier wrote to ``.sz``, a raw Snappy block, and is decoded as
        one; if it is not a valid raw block either, this raises rather
        than guessing. ``verify=False`` skips the framed chunks' CRC-32C
        checks (raw blocks carry no checksum).
        """
        data = _read_src(src)
        if _framed_check_signature(data):
            decoded = _framed_decode(data, verify=verify)
        else:
            try:
                decoded = _snappy_decode(data)
            except RuntimeError as exc:
                raise type(exc)(
                    "snappy_framed decode: the data does not start with "
                    "the framing format's stream identifier, and it is "
                    f"not a raw Snappy block either ({exc})") from exc
        return decoded_output(decoded, out, self.name)


__all__ = ["SnappyCodec", "SnappyFramedCodec"]
