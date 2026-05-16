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
# Blosc2 NDim (b2nd) — multidimensional layer of c-blosc2
# ---------------------------------------------------------------------------

if "opencodecs.codecs._b2nd" in sys.modules:
    from .._b2nd_codec import B2ndCodec
    register_codec(B2ndCodec())


# ---------------------------------------------------------------------------
# AEC — CCSDS 121.0-B-2 adaptive entropy coding (libaec, system)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._aec" in sys.modules:
    from .._aec_codec import AecCodec
    register_codec(AecCodec())


# ---------------------------------------------------------------------------
# LERC — Esri Limited Error Raster Compression (system liblerc)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._lerc" in sys.modules:
    from .._lerc_codec import LercCodec
    register_codec(LercCodec())


# ---------------------------------------------------------------------------
# ZFP — lossy 1D-4D float / int array compression (system libzfp)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._zfp" in sys.modules:
    from .._zfp_codec import ZfpCodec
    register_codec(ZfpCodec())


# ---------------------------------------------------------------------------
# SZ3 — error-bounded lossy scientific compressor (system SZ3c)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._sz3" in sys.modules:
    from .._sz3_codec import Sz3Codec
    register_codec(Sz3Codec())


# ---------------------------------------------------------------------------
# SPERR — wavelet-based error-bounded lossy compressor (libSPERR)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._sperr" in sys.modules:
    from .._sperr_codec import SperrCodec
    register_codec(SperrCodec())


# ---------------------------------------------------------------------------
# Brunsli — lossless JPEG transcoder (~20% smaller)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._brunsli" in sys.modules:
    from .._brunsli_codec import BrunsliCodec
    register_codec(BrunsliCodec())


# ---------------------------------------------------------------------------
# pcodec — modern (2024+) lossless numerical compressor (Rust cdylib)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._pcodec" in sys.modules:
    from .._pcodec_codec import PcodecCodec
    register_codec(PcodecCodec())


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
# JPEG-LS — native (CharLS); used heavily in DICOM
# ---------------------------------------------------------------------------

if "opencodecs.codecs._charls" in sys.modules:
    from .._jpegls_codec import JpegLsCodec
    register_codec(JpegLsCodec())


# ---------------------------------------------------------------------------
# MozJPEG — smaller-JPEG encoder (Mozilla libjpeg-turbo fork)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._mozjpeg" in sys.modules:
    from .._mozjpeg_codec import MozJpegCodec
    register_codec(MozJpegCodec())


# ---------------------------------------------------------------------------
# HTJ2K — JPEG-2000 Part 15 high-throughput codestream (OpenJPH)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._openjph" in sys.modules:
    from .._htj2k_codec import Htj2kCodec
    register_codec(Htj2kCodec())


# ---------------------------------------------------------------------------
# LZMA / XZ — stdlib (always available)
# ---------------------------------------------------------------------------

from .._lzma_codec import LzmaCodec
register_codec(LzmaCodec())


# ---------------------------------------------------------------------------
# bzip2 — stdlib (always available)
# ---------------------------------------------------------------------------

from .._bz2_codec import Bz2Codec
register_codec(Bz2Codec())


# ---------------------------------------------------------------------------
# numpy — .npy-format passthrough (stdlib + numpy.save/load)
# ---------------------------------------------------------------------------

from .._numpy_codec import NumpyCodec
register_codec(NumpyCodec())


# ---------------------------------------------------------------------------
# byteshuffle — element-byte-plane shuffle preprocessor
# ---------------------------------------------------------------------------

if "opencodecs.codecs._bytetools" in sys.modules:
    from .._byteshuffle_codec import ByteshuffleCodec
    register_codec(ByteshuffleCodec())


# ---------------------------------------------------------------------------
# delta / xor / floatpred — composable byte-level predictors
# ---------------------------------------------------------------------------

from .._predictor_codec import DeltaCodec, XorCodec, FloatpredCodec
register_codec(DeltaCodec())
register_codec(XorCodec())
register_codec(FloatpredCodec())


# ---------------------------------------------------------------------------
# quantize / packints — lossy / bit-width filters
# ---------------------------------------------------------------------------

from .._quantize_codec import QuantizeCodec
register_codec(QuantizeCodec())

from .._packints_codec import PackintsCodec
register_codec(PackintsCodec())


# ---------------------------------------------------------------------------
# bcn — BC1-7 GPU texture decoder (dispatcher around _bcdec)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._bcdec" in sys.modules:
    from .._bcn_codec import BcnCodec
    register_codec(BcnCodec())


# ---------------------------------------------------------------------------
# rcomp — Rice compression (FITS RICE_1)
# ---------------------------------------------------------------------------

from .._rcomp_codec import RcompCodec
register_codec(RcompCodec())


# ---------------------------------------------------------------------------
# dicomrle — DICOM RLE Image Compression (PS 3.5 §G)
# ---------------------------------------------------------------------------

from .._dicomrle_codec import DicomRleCodec
register_codec(DicomRleCodec())


# ---------------------------------------------------------------------------
# GIF — giflib (libgif)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._gif" in sys.modules:
    from .._gif_codec import GifCodec
    register_codec(GifCodec())


# ---------------------------------------------------------------------------
# Snappy — fast block compression (libsnappy, Google)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._snappy" in sys.modules:
    from .._snappy_codec import SnappyCodec
    register_codec(SnappyCodec())


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
# ND2 — delegate (Nikon NIS-Elements; wraps the `nd2` Python package)
# ---------------------------------------------------------------------------

try:
    from .._nd2_codec import Nd2Codec as _Nd2Codec
    # Always register — has_native=True (raw ND2 works without the
    # nd2 package); has_delegate handles legacy + compressed variants.
    register_codec(_Nd2Codec())
except ImportError:  # pragma: no cover - import-time error branch
    pass


# ---------------------------------------------------------------------------
# LIF — delegate (Leica LAS-X; wraps the `readlif` package)
# ---------------------------------------------------------------------------

try:
    from .._lif_codec import LifCodec as _LifCodec
    # Always register — native parser handles typical LIFs without
    # readlif; readlif is the fallback for files with LAS-X frame-
    # order overrides.
    register_codec(_LifCodec())
except ImportError:  # pragma: no cover - import-time error branch
    pass


# ---------------------------------------------------------------------------
# OIB / OIF — delegate (Olympus FluoView; wraps the `oiffile` package)
# ---------------------------------------------------------------------------

try:
    from .._oib_codec import OibCodec as _OibCodec
    # Always register — native OLE2+TIFF reader handles OIB without
    # oiffile; oiffile is the fallback for OIF directory variants.
    register_codec(_OibCodec())
except ImportError:  # pragma: no cover - import-time error branch
    pass


# ---------------------------------------------------------------------------
# VSI — Olympus CellSens virtual slide. Top-level container is TIFF,
# full-res tiles live in a sibling .ets file we don't decode yet.
# ---------------------------------------------------------------------------

try:
    from .._vsi_codec import VsiCodec as _VsiCodec
    register_codec(_VsiCodec())
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# OIR — Olympus FluoView newer format. Format-detection stub only;
# decode raises NotImplementedError pointing at bioformats.
# ---------------------------------------------------------------------------

try:
    from .._oir_codec import OirCodec as _OirCodec
    register_codec(_OirCodec())
except ImportError:  # pragma: no cover
    pass


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
# Bitshuffle — native (vendored 3rdparty/bitshuffle)
# ---------------------------------------------------------------------------

if "opencodecs.codecs._bitshuffle" in sys.modules:
    from .._bitshuffle_codec import BitshuffleCodec
    register_codec(BitshuffleCodec())


# ---------------------------------------------------------------------------
# TIFF — native reader (no libtiff dep). See _tiff_codec.py for design notes.
# ---------------------------------------------------------------------------

if "opencodecs.codecs._tiff" in sys.modules:
    from .._tiff_codec import TiffCodec
    register_codec(TiffCodec())


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
