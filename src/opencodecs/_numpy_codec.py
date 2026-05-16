"""NumpyCodec — round-trip an ndarray through the ``.npy`` byte format.

The ``.npy`` format is a thin, self-describing wrapper around raw
array bytes: a magic number, version, and a Python-literal header
naming dtype/shape/fortran-order, followed by the array data. It's
the only ndarray serialisation format that is both numpy-native and
specified well enough to share across libraries.

Useful as:

* A trivial fallback "compressor" in pipelines that demand a codec
  interface (e.g. zarr's codec chain) but want raw passthrough.
* A self-describing wire format when you can't ship dtype/shape out
  of band.
* A reference for what "no compression" looks like in a benchmark.

Mirrors imagecodecs's ``numpy_encode`` / ``numpy_decode``.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest


class NumpyCodec(Codec):
    """``.npy``-format passthrough codec."""

    name = "numpy"
    aliases = ("npy",)
    file_extensions = (".npy",)

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    # Every numpy dtype is supported (it's literal raw bytes plus a
    # dtype header). We list a representative set for the registry's
    # surface; numpy.save itself accepts any np.dtype.
    supported_dtypes = (
        np.uint8, np.int8, np.uint16, np.int16,
        np.uint32, np.int32, np.uint64, np.int64,
        np.float16, np.float32, np.float64,
        np.complex64, np.complex128,
    )
    supports_color = True

    def signature(self, head: bytes) -> bool:
        # ``.npy`` files start with the 6-byte magic ``\x93NUMPY``
        # followed by a 2-byte version (major, minor).
        return len(head) >= 8 and head[:6] == b"\x93NUMPY"

    def encode(self, data: Any, *, dest=None, **opts) -> bytes | None:
        arr = np.ascontiguousarray(data)
        buf = io.BytesIO()
        # ``allow_pickle=False`` matches modern numpy's default and
        # guarantees the output is a pure-binary header+data dump
        # (no pickle protocol bytes for object dtypes).
        np.save(buf, arr, allow_pickle=False)
        return _write_dest(buf.getvalue(), dest)

    def decode(self, src: Any, *, out=None, **opts) -> np.ndarray:
        # numpy.load returns a fresh ndarray; if the caller wants
        # the result in their preallocated buffer, copy into it.
        # True zero-alloc isn't possible here because np.load owns
        # the buffer it creates, but out= still saves the second
        # allocation a caller would do.
        arr = np.load(io.BytesIO(_read_src(src)), allow_pickle=False)
        if out is None:
            return arr
        if not isinstance(out, np.ndarray):
            raise TypeError(
                f"numpy decode: out= must be an ndarray, "
                f"got {type(out).__name__}")
        if out.shape != arr.shape:
            raise ValueError(
                f"numpy decode: out= shape {out.shape} does not match "
                f"decoded {arr.shape}")
        if out.dtype != arr.dtype:
            raise ValueError(
                f"numpy decode: out= dtype {out.dtype} does not match "
                f"decoded {arr.dtype}")
        if not out.flags["C_CONTIGUOUS"]:
            raise ValueError("numpy decode: out= must be C-contiguous")
        np.copyto(out, arr)
        return out


__all__ = ["NumpyCodec"]
