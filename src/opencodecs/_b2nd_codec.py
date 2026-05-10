"""B2ndCodec — Codec adapter wrapping the native _b2nd extension.

Blosc2 NDim ("b2nd") is c-blosc2's multidimensional layer. Each cframe
is a self-contained byte buffer that round-trips an ndarray with full
shape and dtype — no out-of-band metadata required.

This is the natural shape-aware companion to the existing flat blosc2
codec. Use cases:

* Persisting numerical arrays as cframes inside HDF5 / Zarr / a tar
* Network-transfer of multidim arrays (the cframe is a complete record)
* Scientific time-series where each frame is a chunk

Example::

    import numpy as np
    import opencodecs as oc

    arr = np.random.rand(64, 128, 128).astype(np.float32)
    blob = oc.write(None, arr, format="b2nd", compressor="zstd", shuffle="bit")
    back = oc.read(blob, format="b2nd")
    assert back.shape == arr.shape and back.dtype == arr.dtype
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _b2nd_encode, _b2nd_decode, _b2nd_inspect, _b2nd_check_signature,
    _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._b2nd",
    "encode", "decode", "inspect", "check_signature",
)


class B2ndCodec(Codec):
    """Native blosc2 NDim — ndarray ↔ self-describing cframe."""

    name = "b2nd"
    aliases = ("blosc2nd", "blosc2-nd")
    file_extensions = (".b2nd",)

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (
        np.uint8, np.int8,
        np.uint16, np.int16, np.float16,
        np.uint32, np.int32, np.float32,
        np.uint64, np.int64, np.float64,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return _b2nd_check_signature(head)

    def encode(self, data: Any, *, dest=None,
               level: int = 5,
               compressor: str | None = "zstd",
               shuffle: Any = "bit",
               **opts) -> bytes | None:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        out = _b2nd_encode(
            data, level=int(level), compressor=compressor, shuffle=shuffle,
        )
        return _write_dest(out, dest)

    def decode(self, src: Any, **opts) -> np.ndarray:
        return _b2nd_decode(_read_src(src))

    def inspect(self, src: Any) -> dict:
        """Return {ndim, shape, dtype, itemsize} without decompressing."""
        return _b2nd_inspect(_read_src(src))


__all__ = ["B2ndCodec"]
