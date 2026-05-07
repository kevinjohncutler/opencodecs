"""Native codec registrations — no runtime delegation.

Each codec lives in its own ``.pyx`` Cython extension under
``opencodecs/codecs/``. The extension loader in ``__init__.py`` builds
its module before this file runs, so by the time we register here,
``_jxl``, ``_qoi``, etc. are in ``sys.modules`` (or absent if not built).

Adding a codec:

  1. Drop ``_<name>.pyx`` (and ``<name>.pxd`` for C declarations) into
     this directory. Use one of the existing codecs as a template.
  2. Add ``"_<name>"`` to ``_EXTENSIONS`` in ``__init__.py``.
  3. Wire the build into ``setup.py`` (Extension entry).
  4. Add a ``register_codec(MyCodec())`` call below.
  5. Add tests parity-checking against imagecodecs in
     ``tests/test_<name>.py``.

Each Codec subclass owns its file-extension list, signature check,
encode/decode, and (for multi-frame formats) Reader/Writer adapters.
"""

from __future__ import annotations

import sys

from ..core.codec import Codec, register_codec


# ---------------------------------------------------------------------------
# JPEG XL — native (libjxl 0.11.2, vendored)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._jxl" in sys.modules:
    # Defined at the package root to avoid circular dep with the loader.
    from .._jxl_codec import JpegXLCodec
    register_codec(JpegXLCodec())


# ---------------------------------------------------------------------------
# QOI — native (vendored qoi.h, single-header BSD)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._qoi" in sys.modules:
    from .._qoi_codec import QoiCodec
    register_codec(QoiCodec())


# ---------------------------------------------------------------------------
# zstd — native (system libzstd)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._zstd" in sys.modules:
    from .._zstd_codec import ZstdCodec
    register_codec(ZstdCodec())


# ---------------------------------------------------------------------------
# LZ4 — native (system liblz4, frame format)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._lz4" in sys.modules:
    from .._lz4_codec import Lz4Codec
    register_codec(Lz4Codec())


# ---------------------------------------------------------------------------
# brotli — native (system libbrotli; also a libjxl transitive dep)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._brotli" in sys.modules:
    from .._brotli_codec import BrotliCodec
    register_codec(BrotliCodec())


# ---------------------------------------------------------------------------
# blosc2 — native (system c-blosc2)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._blosc2" in sys.modules:
    from .._blosc2_codec import Blosc2Codec
    register_codec(Blosc2Codec())


# ---------------------------------------------------------------------------
# deflate / zlib — native (system zlib)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._deflate" in sys.modules:
    from .._deflate_codec import DeflateCodec
    register_codec(DeflateCodec())


# ---------------------------------------------------------------------------
# JPEG — native (libjpeg-turbo, TurboJPEG v3 API)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._jpeg" in sys.modules:
    from .._jpeg_codec import JpegCodec
    register_codec(JpegCodec())


# ---------------------------------------------------------------------------
# WebP — native (libwebp)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._webp" in sys.modules:
    from .._webp_codec import WebpCodec
    register_codec(WebpCodec())


# ---------------------------------------------------------------------------
# JPEG-2000 — native (OpenJPEG)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._jpeg2k" in sys.modules:
    from .._jpeg2k_codec import Jpeg2kCodec
    register_codec(Jpeg2kCodec())


# ---------------------------------------------------------------------------
# AVIF — native (libavif; depends on system aom/dav1d/svt-av1)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._avif" in sys.modules:
    from .._avif_codec import AvifCodec
    register_codec(AvifCodec())


# ---------------------------------------------------------------------------
# HEIF / HEIC — native (libheif; depends on libde265 / x265)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._heif" in sys.modules:
    from .._heif_codec import HeifCodec
    register_codec(HeifCodec())


# ---------------------------------------------------------------------------
# HDF5 — container reader (h5py wrapper, optional)
# ---------------------------------------------------------------------------

try:
    from .._hdf5_codec import HdfCodec as _HdfCodec
    if _HdfCodec.has_native:  # only register when h5py is importable
        register_codec(_HdfCodec())
except ImportError:  # pragma: no cover - h5py-missing branch
    pass


# ---------------------------------------------------------------------------
# CZI — native (Zeiss ZISRAW; mmap + parallel zstd via opencodecs._zstd)
# ---------------------------------------------------------------------------

# CziReader uses our native zstd codec for decompression — register only
# when that extension is available.
if "opencodecs.codecs._zstd" in sys.modules:
    from .._czi_codec import CziCodec
    register_codec(CziCodec())


# ---------------------------------------------------------------------------
# BMP — native (pure Python + numpy, no external library)
# ---------------------------------------------------------------------------

from .._bmp_codec import BmpCodec
register_codec(BmpCodec())


# ---------------------------------------------------------------------------
# PNG — native (libspng, system)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._png" in sys.modules:
    from .._png_codec import PngCodec
    register_codec(PngCodec())


# ---------------------------------------------------------------------------
# Native codec roadmap
# ---------------------------------------------------------------------------
#
# These have native ``_<name>.pyx`` extensions in flight. Each lands as
# a separate engineering task; the build pattern matches JpegXL's
# vendor/ + setup.py Extension + Codec subclass.
#
# Compression-only (byte → byte, no shape):
#   _zstd     libzstd            system [DONE]
#   _lz4      liblz4             system, frame format [DONE]
#   _brotli   libbrotli          shared with libjxl deps
#   _blosc2   libblosc2          system or vendored
#
# Lossless still images:
#   _qoi      vendored qoi.h     single-header, zero deps [DONE]
#   _bmp      pure Python+numpy  zero deps [DONE]
#   _png      libspng            system, byte-parity tested [DONE]
#   _lerc     LERC2              vendored
#
# Lossy still images:
#   _jpeg     libjpeg-turbo      system or vendored
#   _webp    libwebp            system or vendored
#   _avif     libavif            heavy: libaom or dav1d
#   _heif     libheif            heavy: libde265 or libx265
#   _jpeg2k   openjpeg           system or vendored
#   _jpegxr   jxrlib             vendored
#   _jpegls   charls             vendored
#
# Containers (multi-frame / chunked, parallel-friendly):
#   _tiff     libtiff or own     own parser is preferable for chunk-parallel
#                                tile decode via BackgroundChunkReader
#   _czi      Zeiss CZI parser   port from hiprpy.io.czi (parallel pread)
#   _hdf5     h5py wrapper       Reader interface adapting h5py.Dataset
#   _zarr_v3  BytesBytesCodec    native zarr v3 codec wrapping JxlCodec
#
# Each can be developed independently. The unified Codec interface
# means user code never changes.
