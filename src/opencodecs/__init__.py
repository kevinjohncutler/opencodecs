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
import pathlib
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
from ._n5 import N5Array, N5Error
from ._imaris import ImarisReader, ImarisError
from ._jpeg_pyramid import JpegPyramidReader
from ._mozjpeg_pyramid import MozjpegPyramidReader
from ._jpeg2k_pyramid import Jpeg2kPyramidReader
from ._htj2k_pyramid import Htj2kPyramidReader
from ._dicom import DicomFile, DicomError
from ._dicom_codec import DicomCodec
from ._nrrd import NrrdFile, NrrdError
from ._dm import DmFile, DmError
from ._emd import EmdFile, EmdError
from ._mrc_writer import encode_mrc, write_mrc
from ._nifti_writer import encode_nifti, write_nifti
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


def writer(
    dest: Any = None,
    *,
    format: str | None = None,
    **opts,
) -> Writer:
    """Open `dest` for streaming / multi-frame writing.

    The write-side counterpart of :func:`open`. Dispatch matches
    :func:`write` rather than :func:`open`, because there are no bytes
    to sniff yet: the codec comes from `dest`'s extension, or from
    `format=`.

    `dest` may be a path, a file-like, or None for the formats that
    return bytes from `close()`::

        with oc.writer("stack.tif") as w:
            for plane in volume:
                w.write_frame(plane)

    Codecs with a real streaming writer hand one back; the rest get a
    writer that buffers frames and encodes on close, so the interface
    is the same either way.
    """
    if format is None:
        if isinstance(dest, (str, os.PathLike)):
            codec = codec_for_path(dest)
        else:
            raise ValueError(
                "writer() needs format=... when dest isn't a path"
            )
    else:
        codec = get_codec(format)
    return codec.writer(dest, **opts)


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
    * Imaris (.ims) → :class:`ImarisReader`
    * JPEG 2000 (.jp2/.j2k/.jpx/.jpc) → :class:`Jpeg2kPyramidReader`
    * HTJ2K (.jph/.j2c) → :class:`Htj2kPyramidReader`
    * JPEG (.jpg/.jpeg) → :class:`JpegPyramidReader`
    * VSI / ETS (``.vsi`` or ``.ets``) → :class:`VsiPyramidReader`. A
      genuine tiled pyramid: ``read_region`` decodes only the tiles the
      box covers. A ``.vsi`` with several stacks holds several images
      rather than more levels, so it needs ``stack=``.
    * DICOM whole-slide (a directory or list of instances, or ``.dcm``)
      → :class:`DicomWsiPyramid`. Unlike the others this is a SERIES of
      files: the levels are separate instances sharing a
      SeriesInstanceUID, and nothing inside any one of them says which
      level it is. A directory has to be asked for with
      ``format="dicom"``, since no extension implies one.

    The last three are a different kind of pyramid: those formats store
    one image and decode a smaller one out of it, rather than storing
    several resolutions. So they need no pyramid on disk, but they also
    cannot fetch only the tiles a region covers -- the saving is
    resolution, not bytes read. See :mod:`opencodecs._scaled_pyramid`
    for what each one actually costs.

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
                         (".tif", ".tiff", ".btf", ".ome.tif", ".ome.tiff",
                          # TIFF-derived container formats from imaging
                          # ecosystems we know about. Treating them as
                          # TIFF works because they're TIFF + private tags.
                          ".svs", ".ndpi", ".scn", ".qptiff")):
            fmt = "tiff"
        elif path_lower.endswith((".zarr", ".ome.zarr")):
            fmt = "omezarr"
        elif path_lower.endswith(".czi"):
            fmt = "czi"
        elif path_lower.endswith(".ims"):
            fmt = "imaris"
        elif path_lower.endswith((".jp2", ".j2k", ".jpx", ".jpc")):
            fmt = "jpeg2k"
        elif path_lower.endswith((".jph", ".j2c")):
            fmt = "htj2k"
        elif path_lower.endswith((".jpg", ".jpeg")):
            fmt = "jpeg"
        elif path_lower.endswith(".dcm"):
            fmt = "dicom"
        elif path_lower.endswith((".vsi", ".ets")):
            fmt = "vsi"
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
    if fmt in ("imaris", "ims"):
        return ImarisReader(src, **opts)
    if fmt in ("vsi", "ets", "cellsens"):
        # A .vsi is an index; the pyramid lives in a sibling .ets. A
        # file with more than one stack holds more than one IMAGE, not
        # more levels, so those need stack= rather than a silent pick.
        from ._vsi_pyramid import VsiPyramidReader
        return VsiPyramidReader(src, **opts)
    if fmt in ("dicom", "dcm", "wsi"):
        # A whole-slide DICOM pyramid is a SERIES of files, so this one
        # takes a directory or a list rather than a path to "the file".
        # No extension can imply that, which is why a directory has to
        # be asked for by format=.
        from ._dicom_pyramid import DicomWsiPyramid
        return DicomWsiPyramid(src, **opts)
    # Single-codestream pyramids. These formats store one image and
    # decode a smaller one out of it, so the reader is handed bytes
    # rather than a seekable source -- there is nothing to seek to.
    if fmt in ("jpeg2k", "jp2", "j2k", "jpx", "jpc"):
        return Jpeg2kPyramidReader(_pyramid_bytes(src), **opts)
    if fmt in ("htj2k", "jph", "j2c"):
        return Htj2kPyramidReader(_pyramid_bytes(src), **opts)
    if fmt in ("jpeg", "jpg"):
        return JpegPyramidReader(_pyramid_bytes(src), **opts)
    if fmt == "mozjpeg":
        return MozjpegPyramidReader(_pyramid_bytes(src), **opts)
    raise ValueError(
        f"open_pyramid: cannot determine format for src={src!r}; pass "
        f"format='tiff'|'omezarr'|'czi'|'imaris'|'jpeg'|'jpeg2k'|'htj2k'"
        f"|'dicom'|'vsi'"
    )


def _pyramid_bytes(src: Any) -> bytes:
    """Read a whole-codestream source into bytes.

    JPEG, JPEG 2000 and HTJ2K decoders here all want the complete
    codestream, so an http(s) URL is fetched in full rather than by
    range: pretending otherwise would put a `http` tick against formats
    that cannot use it. The reduced-resolution decode still pays off,
    because the saving is CPU and memory rather than bytes moved.
    """
    if isinstance(src, (bytes, bytearray, memoryview)):
        return bytes(src)
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        from ._tiff_http import http_fetch_all
        return http_fetch_all(src)
    if isinstance(src, (str, os.PathLike)):
        # `open` is this module's own reader factory, so go through
        # pathlib rather than shadowing our way into a bug.
        return pathlib.Path(src).read_bytes()
    if hasattr(src, "read"):
        return src.read()
    raise TypeError(
        f"open_pyramid: cannot read a codestream from {type(src).__name__}")


__all__ = [
    # Top-level unified API
    "read", "write", "open", "open_pyramid",
    # N5 (Janelia chunked arrays)
    "N5Array", "N5Error",
    # Imaris (.ims)
    "ImarisReader", "ImarisError",
    # DICOM files
    "DicomFile", "DicomError", "DicomCodec",
    # NRRD
    "NrrdFile", "NrrdError",
    # Gatan Digital Micrograph
    "DmFile", "DmError",
    # EMD (Berkeley / Velox)
    "EmdFile", "EmdError",
    # Volume writers
    "encode_mrc", "write_mrc",
    "encode_nifti", "write_nifti",
    "list_codecs", "has_codec", "get_codec", "writer",
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
    "JpegPyramidReader", "MozjpegPyramidReader",
    "Jpeg2kPyramidReader", "Htj2kPyramidReader",
    "HTTPDataSource", "FileDataSource",
    "OmeZarrArray", "OmeZarrPyramidDataset",
    "write_zarr_array", "write_omezarr_pyramid",
    "FitsStream", "FitsHDU", "fits_imread",
    "rgbe_encode", "rgbe_decode", "rgbe_imread", "rgbe_imwrite",
    "CziPyramidReader",
    "CziWriter", "CziPyramidWriter",
]

__version__ = "0.1.13"
