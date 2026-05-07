"""Native JPEG XL codec — wraps the JxlReader / JxlWriter cdef classes
in the unified Codec / Reader interface.

Sits at the package root (not under codecs/) because the codecs/__init__
loader has to run first to load the _jxl extension via the off-NAS cache;
importing from codecs/_jxl_codec.py would create a circular dep.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .core.codec import Codec, Reader
from .core._optional_backend import import_or_stubs

(_JxlReader, _JxlWriter, _jxl_encode, _jxl_decode, _jxl_check_signature,
 _HAVE_BACKEND) = import_or_stubs(
    "opencodecs.codecs._jxl",
    "JxlReader", "JxlWriter", "encode", "decode", "check_signature",
)


class JpegXLReader(Reader):
    """Reader adapter wrapping the cdef JxlReader."""

    is_chunked = True  # multi-frame JXLs support frame-by-frame iteration

    def __init__(self, src: Any, **opts):
        self._inner = _JxlReader(src, **opts)
        self.shape = self._inner.frame_shape
        self.dtype = self._inner.dtype
        self.color = self._inner.color
        self.icc_profile = None  # lazy on the inner reader
        self.n_frames = self._inner.n_frames

    @property
    def basic_info(self) -> dict:
        return self._inner.basic_info

    @property
    def is_animation(self) -> bool:
        return self._inner.is_animation

    def iter_frames(self) -> Iterator[np.ndarray]:
        return self._inner.iter_frames()

    def read(self) -> np.ndarray:
        return self._inner.read()

    def close(self) -> None:
        self._inner.close()


class JpegXLCodec(Codec):
    """Native JPEG XL codec (libjxl 0.11).

    Streaming reader, multi-frame animation support, P3 + HDR (PQ/HLG)
    color via ColorSpec, optional bg-thread streaming for very large
    files (off by default — see JxlReader docs).
    """

    name = "jxl"
    aliases = ("jpegxl", "jpeg-xl")
    file_extensions = (".jxl",)

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = True
    chunked = True
    streaming_decode = True
    parallel_decode = False  # per-frame parallel via parallel.read_files; v0.2: jxli-box random access

    supported_dtypes = (np.uint8, np.uint16, np.float16, np.float32)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        return _jxl_check_signature(head)

    def encode(self, arr: np.ndarray, *, dest=None, **opts) -> bytes | None:
        return _jxl_encode(arr, dest=dest, **opts)

    def decode(self, src: Any, **opts) -> np.ndarray:
        return _jxl_decode(src, **opts)

    def open(self, src: Any, **opts) -> JpegXLReader:
        return JpegXLReader(src, **opts)


__all__ = ["JpegXLCodec", "JpegXLReader"]
