"""ZfpCodec — Codec adapter wrapping the native _zfp extension.

ZFP is the standard for *fast* lossy compression of 1D-4D float / int
arrays in HPC. Self-describing blob (full header) carries shape, dtype,
and mode metadata.

Modes (pick one)::

    mode='reversible'                 # lossless
    mode='rate', rate=4               # 4 bits per value (predictable size)
    mode='precision', precision=12    # 12 bits of mantissa (predictable accuracy)
    mode='accuracy', accuracy=1e-3    # absolute error <= 1e-3 (predictable error)

Example::

    arr = np.random.rand(64, 128, 128).astype(np.float32)
    blob = oc.write(None, arr, format="zfp", mode="accuracy", accuracy=1e-4)
    back = oc.read(blob, format="zfp")
    assert np.abs(arr - back).max() <= 1e-4
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _zfp_encode, _zfp_decode, _zfp_check_signature, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._zfp",
    "encode", "decode", "check_signature",
)


class ZfpCodec(Codec):
    """Native ZFP codec — fast lossy compression for 1D-4D arrays."""

    name = "zfp"
    file_extensions = (".zfp",)

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.int32, np.int64, np.float32, np.float64)
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return _zfp_check_signature(head)

    def encode(self, data: Any, *, dest=None,
               mode: str = "reversible",
               rate=None, precision=None, accuracy=None,
               **opts) -> bytes | None:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        out = _zfp_encode(
            data, mode=mode,
            rate=rate, precision=precision, accuracy=accuracy,
        )
        return _write_dest(out, dest)

    def decode(self, src: Any, **opts) -> np.ndarray:
        return _zfp_decode(_read_src(src))


__all__ = ["ZfpCodec"]
