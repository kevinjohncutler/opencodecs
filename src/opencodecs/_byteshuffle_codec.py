"""ByteshuffleCodec — element-byte-plane shuffle for compression preprocessors.

For multi-byte element arrays (uint16 / uint32 / float32 / ...) the
high-byte and low-byte streams typically have very different
entropy: high bytes are often constant or slowly-varying, low bytes
look random. A bytes-in/bytes-out compressor (zstd, lz4, deflate)
sees one interleaved stream of all bytes and matches both streams
together — losing the redundancy in the high-byte plane.

Byteshuffle rearranges memory so all the high bytes come first, then
all the low bytes (etc. for >2 byte types). The result is a byte
stream the compressor can squeeze ~1.5-3× harder for typical
scientific arrays. Bitshuffle (a finer-grained sibling — see
``BitshuffleCodec``) often beats it on noisy data; byteshuffle is
cheaper to encode/decode and frequently wins on smooth data.

Composes with any byte-level compressor:

    byteshuffled = oc.get_codec("byteshuffle").encode(arr.tobytes(), itemsize=2)
    compressed = oc.get_codec("zstd").encode(byteshuffled)

The underlying nogil loops live in
``opencodecs.codecs._bytetools.byteshuffle_{encode,decode}``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .codecs._bytetools import (
    byteshuffle_encode as _bs_encode,
    byteshuffle_decode as _bs_decode,
)


class ByteshuffleCodec(Codec):
    """Element-byte-plane shuffle (compression preprocessor)."""

    name = "byteshuffle"
    aliases = ()
    file_extensions = ()

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (
        np.uint8, np.int8, np.uint16, np.int16,
        np.uint32, np.int32, np.uint64, np.int64,
        np.float16, np.float32, np.float64,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        # Byteshuffle is a filter, not a container — no magic bytes.
        return False

    def encode(
        self,
        data: Any,
        *,
        dest=None,
        itemsize: int | None = None,
        **opts,
    ) -> bytes | None:
        """Shuffle bytes.

        ``data`` may be a numpy array (itemsize inferred from dtype) or
        a bytes-like; for bytes-like input, ``itemsize`` is required.
        """
        if isinstance(data, np.ndarray):
            if itemsize is None:
                itemsize = data.dtype.itemsize
            n_elements = data.size
            buf = np.ascontiguousarray(data).tobytes()
        else:
            if itemsize is None:
                raise ValueError(
                    "byteshuffle encode: itemsize= is required for "
                    "non-ndarray input")
            buf = bytes(data) if not isinstance(data, (bytes, bytearray)) else data
            if len(buf) % itemsize != 0:
                raise ValueError(
                    f"byteshuffle encode: data length {len(buf)} is not "
                    f"a multiple of itemsize {itemsize}")
            n_elements = len(buf) // itemsize
        out = _bs_encode(buf, int(itemsize), int(n_elements))
        return _write_dest(out, dest)

    def decode(
        self,
        src: Any,
        *,
        itemsize: int,
        n_elements: int | None = None,
        out=None,
        **opts,
    ) -> bytes:
        """Reverse a byteshuffle.

        ``itemsize`` is required (the codec can't infer it from the
        byteshuffled bytes alone). ``n_elements`` defaults to
        ``len(src) // itemsize``.
        """
        buf = _read_src(src)
        if n_elements is None:
            if len(buf) % itemsize != 0:
                raise ValueError(
                    f"byteshuffle decode: data length {len(buf)} is not "
                    f"a multiple of itemsize {itemsize}")
            n_elements = len(buf) // itemsize
        return _bs_decode(buf, int(itemsize), int(n_elements), out=out)


__all__ = ["ByteshuffleCodec"]
