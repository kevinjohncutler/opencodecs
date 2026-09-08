"""Blosc2Codec — Codec adapter wrapping the native _blosc2 extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

_blosc2_encode, _blosc2_decode, _blosc2_check_signature, _HAVE_BACKEND = import_or_stubs(
    "opencodecs.codecs._blosc2",
    "encode", "decode", "check_signature",
)


class Blosc2Codec(Codec):
    """Native blosc2 meta-compressor (c-blosc2)."""

    name = "blosc2"
    file_extensions = (".b2",)

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    # A blosc2 chunk is a series of independently compressed blocks, so
    # a range of items touches only the blocks it overlaps:
    # decode_partial() pulls 1/64th of a 64 MB chunk 64x cheaper than
    # expanding the whole thing. That is what `chunked` means here.
    chunked = True
    # blosc2_decompress_ctx spreads those blocks across threads --
    # 5.9x on a 64 MB chunk. Opt-in via numthreads= because it is
    # slower than serial on small chunks; see _blosc2.decode.
    parallel_decode = True

    supported_dtypes = (np.uint8,)
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return _blosc2_check_signature(head)

    def encode(self, data: Any, *, dest=None, level: int | None = None,
               compressor: str | None = None,
               typesize: int | None = None,
               shuffle: bool | None = None,
               **opts) -> bytes | None:
        if isinstance(data, np.ndarray):  # pragma: no cover - blosc2 is byte-oriented; ndarray-aware encode unused in tests
            if typesize is None:
                typesize = data.dtype.itemsize
            data = data.tobytes()
        compressed = _blosc2_encode(
            data, level=level, compressor=compressor,
            typesize=typesize, shuffle=shuffle,
        )
        return _write_dest(compressed, dest)

    def decode(self, src: Any, *, numthreads: int | None = None,
               **opts) -> bytes:
        return _blosc2_decode(_read_src(src), numthreads=numthreads)

    def decode_partial(self, src: Any, start: int, nitems: int, *,
                       typesize: int | None = None,
                       numthreads: int | None = None) -> bytes:
        """Decompress items ``[start, start + nitems)`` and nothing else.

        Random access into a compressed buffer, without expanding the
        rest of it. ``start`` and ``nitems`` count ITEMS, of the width
        the chunk was written with -- blosc2's shuffle filters are
        defined over items, so the width is fixed at write time and
        reads have to use it. Passing ``typesize`` asserts what you
        believe that width to be and raises if the chunk disagrees; it
        cannot change the addressing.
        """
        from .codecs._blosc2 import decode_partial as _partial
        return _partial(_read_src(src), start, nitems,
                        typesize=typesize, numthreads=numthreads)



__all__ = ["Blosc2Codec"]
