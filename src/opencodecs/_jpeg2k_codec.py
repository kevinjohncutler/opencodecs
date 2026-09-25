"""Jpeg2kCodec — Codec adapter wrapping the native _jpeg2k extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec
from .core.buffers import array_output, SeekableDestination
from .core._write_helpers import binary_destination, write_all
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs
from .core.pipeline import native_workers
from .core.native_source import NativeSource

_jp2_encode, _jp2_decode, _jp2_check_signature, _HAVE_BACKEND = import_or_stubs(
    "opencodecs.codecs._jpeg2k",
    "encode", "decode", "check_signature",
)


class Jpeg2kCodec(Codec):
    """Native JPEG-2000 codec via OpenJPEG."""

    name = "jpeg2k"
    file_extensions = (".jp2", ".j2k", ".jpx", ".jpc")
    aliases = ("j2k", "jp2")

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    # decode_region() and decode_tile() take a window or a tile out of
    # a codestream without expanding the rest, which is most of why
    # JPEG 2000 exists for large imagery. Measured on 2048x2048: a
    # 256x256 window costs 1.9 ms against 25 ms for the whole image,
    # and one tile of a 16-tile file is 13x cheaper than all of them.
    chunked = True
    # opj_codec_set_threads() in _jpeg2k.pyx decode(); measured 6.6x
    # on 2048x2048 going from one thread to eight.
    parallel_decode = True

    supported_dtypes = (np.uint8, np.uint16)
    supports_color = True

    def decode_region(self, src: Any, y0: int, y1: int, x0: int, x1: int,
                      *, reduce: int = 0,
                      numthreads: int | None = None) -> np.ndarray:
        """Decode the window ``[y0:y1, x0:x1]`` and nothing else.

        Coordinates are on the full-resolution image and stay that way
        when ``reduce`` is used, so a caller picks a window once and
        changes only the zoom.
        """
        from .codecs._jpeg2k import decode_region as _region
        if isinstance(src, (bytes, bytearray, memoryview)):
            return _region(src, y0, y1, x0, x1, reduce=reduce,
                           numthreads=native_workers(numthreads))
        with NativeSource(src) as source:
            return _region(source, y0, y1, x0, x1, reduce=reduce,
                           numthreads=native_workers(numthreads))

    def decode_tile(self, src: Any, tile_index: int, *,
                    numthreads: int | None = None) -> np.ndarray:
        """Decode one tile of a tiled codestream, in raster order."""
        from .codecs._jpeg2k import decode_tile as _tile
        if isinstance(src, (bytes, bytearray, memoryview)):
            return _tile(src, tile_index, numthreads=native_workers(numthreads))
        with NativeSource(src) as source:
            return _tile(source, tile_index, numthreads=native_workers(numthreads))

    def signature(self, head: bytes) -> bool:
        return _jp2_check_signature(head)

    def encode(self, data: Any, *, dest=None, level: int | None = None,
               lossless: bool = True, codec: str = "jp2",
               numthreads: int | None = None,
               **opts) -> bytes | None:
        # ``lossless=True`` by default to match
        # ``imagecodecs.jpeg2k_encode`` — see
        # docs/codec_api_conventions.md "Default settings: Pareto-better
        # than the reference, no cheating."
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        if dest is not None:
            with binary_destination(dest) as stream:
                if hasattr(stream, "seek") and hasattr(stream, "tell") and (
                    not hasattr(stream, "seekable") or stream.seekable()
                ):
                    return _jp2_encode(data, level=level, lossless=lossless, codec=codec,
                                       numthreads=native_workers(numthreads),
                                       destination=SeekableDestination(stream))
                encoded = _jp2_encode(data, level=level, lossless=lossless, codec=codec,
                                       numthreads=native_workers(numthreads))
                write_all(stream, encoded)
                return None
        encoded = _jp2_encode(
            data, level=level, lossless=lossless, codec=codec,
            numthreads=native_workers(numthreads),
        )
        return _write_dest(encoded, dest)

    def decode(self, src: Any, *, numthreads: int | None = None,
               out=None, reduce: int = 0, **opts) -> np.ndarray:
        if out is None:
            return _jp2_decode(_read_src(src), numthreads=native_workers(numthreads),
                               reduce=reduce)
        return _jp2_decode(_read_src(src), numthreads=native_workers(numthreads),
                           out=array_output(out), reduce=reduce)



__all__ = ["Jpeg2kCodec"]
