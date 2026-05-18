"""RcompCodec — Rice compression for FITS astronomy data.

Rice coding (a special case of Golomb-Rice) is a lightweight entropy
coder optimised for streams of signed integers concentrated near
zero. It's the canonical compression for FITS BINTABLE columns and
the ``RICE_1`` tile-compression algorithm in compressed FITS images.

Implementation: thin wrapper around the Cython ``_rcomp`` extension,
which binds cfitsio's vendored ``ricecomp.c``. The previous
pure-Python implementation ran ~1000x slower than imagecodecs —
profile showed ~60 k Python-level bit-write calls per 4 k-element
array — so it's been replaced. Old in-memory blobs from the pure-
Python era will NOT decode here (different payload format); rcomp
is an in-process compressor, not a long-term storage format, so the
break is acceptable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

_rcomp_encode, _rcomp_decode, _rcomp_check_signature, _HAVE_BACKEND = import_or_stubs(
    "opencodecs.codecs._rcomp", "encode", "decode", "check_signature",
)




class RcompCodec(Codec):
    """Rice compression (Golomb-Rice) for FITS-style integer streams."""

    name = "rcomp"
    aliases = ("rice", "rice1", "ricecomp")
    file_extensions = ()

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    # cfitsio's ricecomp.c supports int8 / int16 / int32 (i.e. 1/2/4
    # byte signed integers); unsigned inputs of the same itemsize are
    # passed through bit-for-bit. int64 is not supported by the cfitsio
    # backend — callers with 8-byte ints should down-convert.
    supported_dtypes = (
        np.int8, np.uint8, np.int16, np.uint16,
        np.int32, np.uint32,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return False  # opaque header, no magic

    def encode(self, data: Any, *, dest=None, blocksize: int = 32,
               **opts) -> bytes | None:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        out = _rcomp_encode(data, blocksize=blocksize)
        return _write_dest(out, dest)

    def decode(self, src: Any, *, dtype=None, shape=None, out=None,
               **opts) -> np.ndarray:
        buf = _read_src(src)
        arr = _rcomp_decode(buf)
        # cfitsio's rdecomp_* writes UNSIGNED output (uint8/16/32). For
        # signed inputs the bit pattern matches; we just need to
        # reinterpret. Use ``view`` so values stay numerically correct
        # under two's-complement (``astype`` would clip negatives).
        if dtype is not None:
            target = np.dtype(dtype)
            if target.itemsize == arr.dtype.itemsize:
                arr = arr.view(target)
            else:
                arr = arr.astype(target, copy=False)
        if shape is not None:
            arr = arr.reshape(shape)
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError(
                    f"rcomp decode: out= must be an ndarray, "
                    f"got {type(out).__name__}")
            if out.shape != arr.shape or out.dtype != arr.dtype:
                raise ValueError("rcomp decode: out= shape/dtype mismatch")
            np.copyto(out, arr)
            return out
        return arr


__all__ = ["RcompCodec"]
