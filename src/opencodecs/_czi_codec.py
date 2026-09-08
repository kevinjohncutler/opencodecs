"""CziCodec — unified Codec API for the native CZI reader.

The actual reader lives in ``opencodecs._czi_reader.CziReader``. This
module wraps it as a ``Codec`` so it shows up in the global registry
and works through ``opencodecs.read('foo.czi')`` and ``opencodecs.open``.

Encode is intentionally not implemented — writing CZI requires Zen-side
metadata (XML stage info, scene tree, attachments) that we don't
synthesise. Most workflows only ever read CZI; encode would land later
once we can reproduce the metadata schema faithfully.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec, Reader
from ._czi_reader import CziReader


class CziCodec(Codec):
    """Native CZI reader (Zeiss ZISRAW container).

    Supports compression types ``0`` (uncompressed) and ``6`` (ZSTDHDR /
    Zstd1) — the only ones present in the lab's archive (verified across
    23 sampled files spanning 2022-2024). JPEG-XR sub-blocks (rare in
    modern Zen) raise ``NotImplementedError``.
    """

    name = "czi"
    file_extensions = (".czi",)

    has_native = True
    has_delegate = False
    can_encode = False        # not implemented — CZI write requires Zen metadata
    can_decode = True
    multi_frame = True
    chunked = True
    streaming_decode = True
    parallel_decode = True

    supported_dtypes = (np.uint8, np.uint16, np.float32)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        # ZISRAWFILE segment header begins with the literal magic; first
        # 10 bytes are unique enough to disambiguate from other ISO BMFF /
        # TIFF / JXL containers.
        return len(head) >= 10 and head[:10] == b"ZISRAWFILE"

    def decode(self, src: Any, **opts) -> np.ndarray:
        """Decode an entire CZI file to a stacked ndarray."""
        with self._reader(src) as r:
            return r.read(**opts)

    def writer(self, dest: Any = None, **opts):
        """A real streaming CZI writer: one sub-block per frame."""
        from ._czi_writer import CziWriter
        if dest is None:
            raise ValueError("czi: writer() needs a destination path")
        return CziWriter(dest, **opts)

    def open(self, src: Any, **opts) -> Reader:
        return self._reader(src)

    @staticmethod
    def _reader(src: Any) -> "CziReader":
        """Open `src` without putting it on disk if it is already bytes.

        CziReader takes either a path, which it maps, or a buffer, and
        it has taken both for as long as it has existed. This wrote
        every non-path source to a temp file anyway -- a 43 MB CZI
        already in memory went to disk to be read back -- on the
        reasoning, recorded in a docstring, that "CZI parsing wants a
        seekable mmap-able fd". Its own reader says otherwise.
        """
        if isinstance(src, (str, Path)):
            return CziReader(str(src))
        if isinstance(src, (bytes, bytearray, memoryview)):
            return CziReader(buffer=src)
        if hasattr(src, "read") and hasattr(src, "seek"):
            src.seek(0)
            return CziReader(buffer=src.read())
        raise TypeError(f"unsupported CZI source: {type(src).__name__}")



__all__ = ["CziCodec"]
