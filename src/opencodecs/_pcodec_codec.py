"""PcodecCodec — Codec adapter wrapping the native _pcodec extension.

pcodec (https://github.com/mwlon/pcodec) is a 2024+ lossless numerical
compressor that beats zstd on dense numerical arrays by ~1.5-3× without
filtering. It's a drop-in replacement for zstd-on-floats in scientific
pipelines.

Example::

    arr = np.random.rand(100000).astype(np.float32)
    blob = oc.write(None, arr, format="pcodec", level=8)
    back = oc.read(blob, format="pcodec")
    np.testing.assert_array_equal(arr, back)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _pco_encode, _pco_decode, _pco_check_signature, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._pcodec",
    "encode", "decode", "check_signature",
)


class PcodecCodec(Codec):
    """Native pcodec — modern lossless numerical compressor."""

    name = "pcodec"
    aliases = ("pco",)
    file_extensions = (".pco",)

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
        return _pco_check_signature(head)

    def encode(self, data: Any, *, dest=None,
               level: int = 8,
               max_page_n: int = 0,
               **opts) -> bytes | None:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        out = _pco_encode(data, level=int(level), max_page_n=int(max_page_n))
        return _write_dest(out, dest)

    def decode(self, src: Any, **opts) -> np.ndarray:
        return _pco_decode(_read_src(src))


__all__ = ["PcodecCodec"]
