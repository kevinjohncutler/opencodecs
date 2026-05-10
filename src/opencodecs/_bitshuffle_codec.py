"""BitshuffleCodec — Codec adapter wrapping the native _bitshuffle extension.

Bitshuffle is a *filter* (bit-level transpose), not a stand-alone compressor.
For an N-element array of M-byte items, the output collects bit k from every
element into one contiguous run, repeated for k=0..M*8-1. Output size equals
input size, but the bit-correlated output is far more friendly to LZ77/zstd
than the raw bytes.

Usage::

    import numpy as np
    import opencodecs as oc

    arr = np.arange(10000, dtype=np.uint16)
    shuffled = oc.write(None, arr.tobytes(), format="bitshuffle", itemsize=2)
    raw      = oc.read(shuffled,                  format="bitshuffle", itemsize=2)
    assert raw == arr.tobytes()

Pair with zstd / lz4 / blosc2 (where available) for a complete pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _bs_encode, _bs_decode, _bs_check_signature, _bs_default_blocksize,
    _bs_version, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._bitshuffle",
    "encode", "decode", "check_signature", "default_blocksize", "version",
)


class BitshuffleCodec(Codec):
    """Bitshuffle bit-level transpose filter (vendored, native)."""

    name = "bitshuffle"
    aliases = ("bshuf",)
    file_extensions = ()  # filter, not a container

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (
        np.uint8, np.int8,
        np.uint16, np.int16,
        np.uint32, np.int32, np.float32,
        np.uint64, np.int64, np.float64,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return False  # filter: no magic

    def encode(self, data: Any, *, dest=None,
               itemsize: int | None = None,
               blocksize: int = 0,
               **opts) -> bytes | None:
        if isinstance(data, np.ndarray):
            if itemsize is None:
                itemsize = int(data.dtype.itemsize)
            buf = np.ascontiguousarray(data).tobytes()
        else:
            buf = bytes(data) if not isinstance(data, (bytes, bytearray)) else data
            if itemsize is None:
                itemsize = 1
        out = _bs_encode(buf, itemsize=int(itemsize), blocksize=int(blocksize))
        return _write_dest(out, dest)

    def decode(self, src: Any, *,
               itemsize: int = 1,
               blocksize: int = 0,
               **opts) -> bytes:
        return _bs_decode(_read_src(src), itemsize=int(itemsize), blocksize=int(blocksize))


__all__ = ["BitshuffleCodec"]
