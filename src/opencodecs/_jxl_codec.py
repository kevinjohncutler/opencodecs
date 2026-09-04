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

from .core.codec import Codec, Reader, Writer
from .core._optional_backend import import_or_stubs

(_JxlReader, _JxlWriter, _jxl_encode, _jxl_decode, _jxl_check_signature,
 _HAVE_BACKEND) = import_or_stubs(
    "opencodecs.codecs._jxl",
    "JxlReader", "JxlWriter", "encode", "decode", "check_signature",
)


if _HAVE_BACKEND:
    # Same story as GifWriter: an extension type cannot inherit a
    # Python ABC, and this one already has write_frame + close, so
    # register it as a virtual subclass. Registering the name imported
    # above rather than re-importing, so the module keeps one binding
    # for it.
    Writer.register(_JxlWriter)


class _JxlStreamWriter(Writer):
    """Holds one frame back so ``close()`` can mark the last one.

    libjxl needs the final frame of an animation flagged when it is
    submitted -- ``is_last=True`` sets a bit in that frame's header, and
    ``JxlEncoderCloseInput`` at close time is not a substitute. A caller
    driving the Writer contract only has ``write_frame(arr)`` and cannot
    know which frame is last until there are no more.

    Without this the failure is quiet in the worst way: the stream is
    exactly the same length either way, so nothing looks wrong until
    something tries to decode it and reports a truncated file. Deferring
    by one frame costs one frame of memory and makes the contract call
    produce a valid animation.
    """

    def __init__(self, dest: Any = None, **opts):
        self._inner = _JxlWriter(dest, **opts)
        self._pending: np.ndarray | None = None
        self._closed = False
        self._result: bytes | None = None

    def write_frame(self, arr: np.ndarray, **opts) -> None:
        if self._closed:
            raise RuntimeError("jxl: writer is closed")
        if self._pending is not None:
            self._inner.write_frame(self._pending, **opts)
        self._pending = np.asarray(arr)

    def close(self) -> bytes | None:
        if self._closed:
            return self._result
        self._closed = True
        if self._pending is not None:
            self._inner.write_frame(self._pending, is_last=True)
            self._pending = None
        self._result = self._inner.close()
        return self._result

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class JpegXLReader(Reader):
    """Reader adapter wrapping the cdef JxlReader."""

    is_chunked = True  # multi-frame JXLs support frame-by-frame iteration

    def __init__(self, src: Any, **opts):
        self._inner = _JxlReader(src, **opts)
        self.shape = self._inner.frame_shape
        # np.dtype, not the scalar type the inner reader hands back:
        # Reader annotates dtype as np.dtype and every other format
        # returns one, so `r.dtype.kind` should not depend on which
        # codec produced the reader. The two compare equal, so callers
        # testing `r.dtype == np.uint8` are unaffected.
        self.dtype = np.dtype(self._inner.dtype)
        self.color = self._inner.color
        self.icc_profile = None  # lazy on the inner reader
        # The inner reader answers None for everything, because a JXL
        # codestream does not carry a frame count -- frames are
        # discovered as they decode. That is honest for an animation and
        # needlessly unhelpful for a still, where the answer is one and
        # costs nothing. None therefore still means "unknown", just no
        # longer for the overwhelmingly common case.
        inner_n = self._inner.n_frames
        if inner_n is None and not self._inner.is_animation:
            inner_n = 1
        self.n_frames = inner_n

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

    def writer(self, dest: Any = None, **opts):
        """A real streaming JXL encoder. ``dest=None`` returns bytes."""
        return _JxlStreamWriter(dest, **opts)

    def open(self, src: Any, **opts) -> JpegXLReader:
        return JpegXLReader(src, **opts)


__all__ = ["JpegXLCodec", "JpegXLReader"]
