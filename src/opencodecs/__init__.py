"""opencodecs — streaming, network-aware image codecs for scientific imaging.

Top-level API:

  opencodecs.read(src, *, format=None, **opts) -> ndarray
  opencodecs.write(dest, arr, *, format=None, **opts) -> bytes | None
  opencodecs.open(src, *, format=None, **opts) -> Reader
  opencodecs.list_codecs() -> [{name, native, encode, decode, ...}]
  opencodecs.has_codec(name) -> bool

Format auto-detection: by file extension when the input is a path, by
magic bytes when it's bytes/file-like. Override with format="png".
"""

from __future__ import annotations

import os
from typing import Any

from .core.codec import (
    Codec,
    Reader,
    Writer,
    register_codec,
    get_codec,
    list_codecs,
    has_codec,
    codec_for_path,
    codec_for_bytes,
    _resolve_codec,
)
from .core.color import ColorSpec, parse_color
from .core.errors import OpenCodecsError

# Importing the codec subpackage triggers each format's
# register_codec(...) at module-init time, populating the registry.
from . import codecs as _codecs_pkg  # noqa: F401

# Direct back-compat surface. Both modules below handle a missing libjxl
# backend internally: they import cleanly and only raise (with a
# helpful message) when a function is actually called. So
# ``import opencodecs`` succeeds on platforms without libjxl built.
from . import jxl, parallel
from .jxl import (
    JxlReader,
    JxlWriter,
    encode as jxl_encode,
    decode as jxl_decode,
    iter_frames as jxl_iter_frames,
    open as jxl_open,
)
from ._tiff_writer import TiffWriter, imwrite as tiff_imwrite
from ._omezarr import OmeZarrArray, OmeZarrPyramidDataset
from ._omezarr_writer import write_zarr_array, write_omezarr_pyramid
from ._czi_reader import CziPyramidReader
from ._czi_writer import CziWriter, CziPyramidWriter


def read(src: Any, *, format: str | None = None, **opts):
    """Decode `src` to an ndarray. Codec auto-detected from path/bytes."""
    return _resolve_codec(src, format=format).decode(src, **opts)


def write(
    dest: Any,
    arr,
    *,
    format: str | None = None,
    **opts,
):
    """Encode `arr` to `dest`. Codec auto-detected from dest path or
    `format=`.

    `dest` may be a path, file-like, or None (in-memory: returns bytes).
    """
    if format is None:
        if isinstance(dest, (str, os.PathLike)):
            codec = codec_for_path(dest)
        else:
            raise ValueError(
                "write() needs format=... when dest isn't a path"
            )
    else:
        codec = get_codec(format)
    return codec.encode(arr, dest=dest, **opts)


def open(  # noqa: A001
    src: Any,
    *,
    format: str | None = None,
    **opts,
) -> Reader:
    """Open `src` for streaming / random-access reading."""
    return _resolve_codec(src, format=format).open(src, **opts)


__all__ = [
    # Top-level unified API
    "read", "write", "open",
    "list_codecs", "has_codec", "get_codec",
    # Core types (subclassable)
    "Codec", "Reader", "Writer", "register_codec",
    # Color
    "ColorSpec", "parse_color",
    # Errors
    "OpenCodecsError",
    # Submodules + native JXL surface (back-compat)
    "jxl", "parallel",
    "JxlReader", "JxlWriter",
    "jxl_encode", "jxl_decode", "jxl_iter_frames", "jxl_open",
    "TiffWriter", "tiff_imwrite",
    "OmeZarrArray", "OmeZarrPyramidDataset",
    "write_zarr_array", "write_omezarr_pyramid",
    "CziPyramidReader",
    "CziWriter", "CziPyramidWriter",
]

__version__ = "0.2.0.dev0"
