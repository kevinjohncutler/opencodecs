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
from ._fits import FitsStream, FitsHDU, imread as fits_imread
from ._rgbe import encode as rgbe_encode, decode as rgbe_decode, \
    imread as rgbe_imread, imwrite as rgbe_imwrite
from ._czi_reader import CziPyramidReader
from ._czi_writer import CziWriter, CziPyramidWriter
from ._tiff_pyramid import TiffPyramidReader
from ._tiff_http import HTTPDataSource, FileDataSource
from .core.pyramid import PyramidReader, PyramidLevel


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


def open_pyramid(
    src: Any,
    *,
    format: str | None = None,
    **opts,
) -> PyramidReader:
    """Open `src` as a multi-resolution pyramid reader.

    Dispatches by format/extension/URL:

    * TIFF (.tif/.tiff/.btf, or http(s) URL ending in those) →
      :class:`TiffPyramidReader`. For HTTP sources, range-requests
      fetch only the COG tiles overlapping each :meth:`read_region`.
    * Zarr (.zarr/ome.zarr) → :class:`OmeZarrPyramidDataset`
    * CZI (.czi) → :class:`CziPyramidReader`
    * NDTiff directory → :class:`NDTiffPyramidReader` (when available)

    Examples
    --------
    Remote COG over HTTP::

        with oc.open_pyramid("https://bucket/big.tif") as p:
            overview = p.read_region(p.best_level_for(max_pixels_y=1024))

    Local OME-TIFF or COG::

        with oc.open_pyramid("scan.ome.tif") as p:
            level = p.best_level_for(max_pixels_y=2048)
            tile  = p.read_region(level, y=(0, 1024), x=(0, 1024))
    """
    fmt = (format or "").lower()
    is_url = isinstance(src, str) and src.startswith(("http://", "https://"))
    # Path → extension heuristic.
    path_lower = ""
    if isinstance(src, (str, os.PathLike)):
        path_lower = str(src).lower()
    if not fmt:
        if is_url or any(path_lower.endswith(ext) for ext in
                         (".tif", ".tiff", ".btf", ".ome.tif", ".ome.tiff")):
            fmt = "tiff"
        elif path_lower.endswith((".zarr", ".ome.zarr")):
            fmt = "omezarr"
        elif path_lower.endswith(".czi"):
            fmt = "czi"
    if fmt in ("tiff", "tif", "btf", "bigtiff", "cog", "ome-tiff"):
        if is_url:
            # Build an HTTPDataSource and feed it through read_at.
            ds = HTTPDataSource(src, **opts.pop("http_opts", {}))
            return TiffPyramidReader(ds, read_at=ds.read_at, **opts)
        return TiffPyramidReader(src, **opts)
    if fmt in ("omezarr", "ome-zarr", "zarr"):
        return OmeZarrPyramidDataset(src, **opts)
    if fmt == "czi":
        return CziPyramidReader(src, **opts)
    raise ValueError(
        f"open_pyramid: cannot determine format for src={src!r}; "
        f"pass format='tiff'|'omezarr'|'czi'"
    )


__all__ = [
    # Top-level unified API
    "read", "write", "open", "open_pyramid",
    "list_codecs", "has_codec", "get_codec",
    # Core types (subclassable)
    "Codec", "Reader", "Writer", "register_codec",
    "PyramidReader", "PyramidLevel",
    # Color
    "ColorSpec", "parse_color",
    # Errors
    "OpenCodecsError",
    # Submodules + native JXL surface (back-compat)
    "jxl", "parallel",
    "JxlReader", "JxlWriter",
    "jxl_encode", "jxl_decode", "jxl_iter_frames", "jxl_open",
    "TiffWriter", "tiff_imwrite",
    "TiffPyramidReader",
    "HTTPDataSource", "FileDataSource",
    "OmeZarrArray", "OmeZarrPyramidDataset",
    "write_zarr_array", "write_omezarr_pyramid",
    "FitsStream", "FitsHDU", "fits_imread",
    "rgbe_encode", "rgbe_decode", "rgbe_imread", "rgbe_imwrite",
    "CziPyramidReader",
    "CziWriter", "CziPyramidWriter",
]

__version__ = "0.2.0.dev0"
