"""Repoint tifffile's compression dispatch at opencodecs's native codecs.

tifffile uses ``imagecodecs.zstd_decode`` / ``imagecodecs.deflate_decode`` /
etc. via a single module-level reference. We replace that reference with a
shim object that forwards most calls to imagecodecs but overrides specific
codecs with our native implementations.

Usage::

    import tifffile
    import opencodecs.tifffile_patch as patch
    patch.install()                       # idempotent

    arr = tifffile.imread('big.tif')      # now uses opencodecs's zstd path

Or scoped::

    with patch.patched():
        arr = tifffile.imread('big.tif')

The wrapper signatures match what tifffile calls:

    zstd_decode(data, out=int_or_buffer) -> bytes
    deflate_decode(data, out=int_or_buffer) -> bytes
    zstd_encode(data, level=int) -> bytes
    deflate_encode(data, level=int) -> bytes

The ``out`` argument is honored as a size hint (we allocate to fit) so the
buffer-sized variant returns bytes of length ``out``. tifffile only inspects
the returned object's length; pre-allocating into the caller's buffer
isn't required for correctness.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any


# ---------------------------------------------------------------------------
# Adapter functions: match imagecodecs's tifffile-facing signature
# ---------------------------------------------------------------------------


def _bytes_out(data: Any, out: Any) -> bytes:
    """Coerce data + out arg to a bytes return matching imagecodecs."""
    if isinstance(out, (bytes, bytearray, memoryview)):
        return bytes(data[:len(out)])
    return bytes(data)  # pragma: no cover - tifffile always passes bytes-like out


def zstd_decode(data, out=None, **kw):
    from .codecs._zstd import decode as _decode
    decoded = _decode(bytes(data) if not isinstance(data, (bytes, bytearray)) else data)
    return _bytes_out(decoded, out) if out is not None and not isinstance(out, int) else decoded


def zstd_encode(data, level=None, out=None, **kw):
    from .codecs._zstd import encode as _encode
    if not isinstance(data, (bytes, bytearray)):
        try:
            data = bytes(data)
        except Exception:
            data = data.tobytes()
    return _encode(data, level=level)


def deflate_decode(data, out=None, **kw):
    from .codecs._deflate import decode as _decode
    return _decode(bytes(data) if not isinstance(data, (bytes, bytearray)) else data)


def deflate_encode(data, level=None, out=None, **kw):
    from .codecs._deflate import encode as _encode
    if not isinstance(data, (bytes, bytearray)):
        try:
            data = bytes(data)
        except Exception:
            data = data.tobytes()
    return _encode(data, level=level)


def zlib_decode(data, out=None, **kw):
    return deflate_decode(data, out=out, **kw)


def zlib_encode(data, level=None, out=None, **kw):
    return deflate_encode(data, level=level, out=out, **kw)


def lz4_decode(data, out=None, **kw):
    from .codecs._lz4 import decode as _decode
    return _decode(bytes(data) if not isinstance(data, (bytes, bytearray)) else data)


def lz4_encode(data, level=None, out=None, **kw):
    from .codecs._lz4 import encode as _encode
    if not isinstance(data, (bytes, bytearray)):
        try:
            data = bytes(data)
        except Exception:
            data = data.tobytes()
    return _encode(data, level=level)


def _no_encode_out(name, out):
    """Encoders here return new bytes; refuse an output buffer loudly."""
    if out is not None:
        raise TypeError(f"{name}: out= is not supported, the encoded "
                        f"bytes are returned")


def png_decode(data, out=None):
    """``imagecodecs.png_decode(data, *, out=None)``."""
    from .codecs._png import decode as _decode
    data = bytes(data) if not isinstance(data, (bytes, bytearray)) else data
    if out is None or isinstance(out, int):
        return _decode(data)        # an int is only a size hint
    return _decode(data, out=out)


def png_encode(data, level=None, strategy=None, filter=None, out=None, **kw):
    """``imagecodecs.png_encode``: ``level``, ``strategy`` and ``filter``
    are forwarded with imagecodecs' meanings. Options opencodecs does
    not implement raise ``TypeError`` rather than being dropped."""
    from .codecs._png import encode as _encode
    _no_encode_out("png_encode", out)
    return _encode(data, level=level, strategy=strategy, filter=filter, **kw)


def webp_decode(data, index=None, hasalpha=None, out=None):
    """``imagecodecs.webp_decode(data, index=None, *, hasalpha=None,
    out=None)``. tifffile passes ``hasalpha=True`` for four-sample
    images, whose all-opaque tiles libwebp stores without alpha."""
    from ._webp_codec import decode_webp
    data = bytes(data) if not isinstance(data, (bytes, bytearray)) else data
    if isinstance(out, int):
        out = None                  # an int is only a size hint
    return decode_webp(data, index=index, hasalpha=hasalpha, out=out)


def webp_encode(data, level=None, lossless=None, method=None,
                numthreads=None, out=None):
    """``imagecodecs.webp_encode``: lossless by default, as imagecodecs
    and therefore plain tifffile are; ``level``, ``lossless``, ``method``
    and ``numthreads`` keep imagecodecs' meanings."""
    from .codecs._webp import encode as _encode
    _no_encode_out("webp_encode", out)
    return _encode(data, level=level, lossless=lossless, method=method,
                   numthreads=numthreads)


def _jpeg_decode(name, data, out, kw):
    # tifffile passes tables (JPEGTables), header, colorspace and
    # outcolorspace (from PhotometricInterpretation), shape and
    # bitspersample; the native decoder takes all of them, with
    # imagecodecs' meanings, and raises on any it does not know.
    from .codecs._jpeg import JpegError, decode as _decode
    stream = bytes(data) if not isinstance(data, (bytes, bytearray)) \
        else data
    try:
        return _decode(stream, **kw)
    except (JpegError, NotImplementedError):
        fallback = getattr(_original, name, None)
        if fallback is None or not _turbojpeg_cannot_decode(stream, kw):
            raise
    # A JPEG with a component count TurboJPEG has no colorspace for
    # (2, as in a DNG-style lossless tile with two samples per pixel),
    # or a lossless JPEG the decoder raises for, such as one asked for
    # a color conversion TurboJPEG's lossless mode does not make (a
    # tile of a lossless TIFF whose PhotometricInterpretation is YCbCr,
    # read as RGB), is valid T.81 that imagecodecs reads (the YCbCr
    # tile as its stored samples), so the patch hands it the call
    # rather than fail a file that reads without the patch.
    return fallback(data, out=out, **kw)


def _turbojpeg_cannot_decode(stream, kw) -> bool:
    from .codecs._jpeg_common import frame_header, splice_stream
    try:
        fh = frame_header(splice_stream(stream, kw.get("tables"),
                                        kw.get("header")))
    except ValueError:
        return False
    return fh is not None and (fh.components not in (1, 3, 4)
                               or fh.lossless)


def jpeg_decode(data, out=None, **kw):
    return _jpeg_decode("jpeg_decode", data, out, kw)


def jpeg8_decode(data, out=None, **kw):
    return _jpeg_decode("jpeg8_decode", data, out, kw)


def jpeg_encode(data, level=None, out=None, **kw):
    # colorspace, outcolorspace, subsampling, bitspersample, lossless...
    # decide what tifffile records in the TIFF tags, so they must reach
    # the encoder rather than be dropped.
    from .codecs._jpeg import encode as _encode
    if kw.get("subsampling") is not None or kw.get("lossless"):
        kw = _tifffile_jpeg_args(kw)
    return _encode(data, level=level, **kw)


def _tifffile_jpeg_args(kw):
    # tifffile's subsampling is the YCbCrSubSampling tag (TIFF 6.0
    # section 21), chroma subsampling, and for every contiguous RGB JPEG
    # it writes it passes subsampling, (2, 2) by default, with
    # colorspace "RGB" and outcolorspace "YCBCR", also when
    # compressionargs ask for something else. imagecodecs applies
    # neither where the JPEG it writes has no chroma: an RGB JPEG, and a
    # lossless one, which libjpeg-turbo stores unconverted and 1x1. The
    # jpeg codec would honor or refuse them, so the adapter leaves them
    # out there and writes the bytes imagecodecs writes.
    from .codecs._jpeg_common import colorspace_name
    kw = dict(kw)
    out_cs = colorspace_name(kw.get("outcolorspace"), "outcolorspace")
    if kw.get("lossless"):
        kw["subsampling"] = None
        if out_cs == "ycbcr" and colorspace_name(
                kw.get("colorspace"), "colorspace") != "ycbcr":
            kw["outcolorspace"] = None
    elif out_cs not in (None, "ycbcr"):
        kw["subsampling"] = None
    return kw


def jpegxl_decode(data, out=None, **kw):
    import opencodecs as oc
    return oc.read(bytes(data) if not isinstance(data, (bytes, bytearray)) else data, format="jxl")


def jpegxl_encode(data, level=None, distance=None, effort=None, lossless=None,
                   out=None, **kw):
    import opencodecs as oc
    kwargs = {}
    if effort is not None: kwargs["effort"] = effort
    if distance is not None: kwargs["distance"] = distance
    if lossless is not None: kwargs["lossless"] = lossless
    return oc.write(None, data, format="jxl", **kwargs)


def jpeg2k_decode(data, out=None, **kw):
    from .codecs._jpeg2k import decode as _decode
    return _decode(bytes(data) if not isinstance(data, (bytes, bytearray)) else data)


def jpeg2k_encode(data, level=None, lossless=None, out=None, **kw):
    from .codecs._jpeg2k import encode as _encode
    return _encode(data, level=level, lossless=bool(lossless))


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------


_OVERRIDES = {
    # bytes-in / bytes-out compressors
    "zstd_decode": zstd_decode,
    "zstd_encode": zstd_encode,
    "deflate_decode": deflate_decode,
    "deflate_encode": deflate_encode,
    "zlib_decode": zlib_decode,
    "zlib_encode": zlib_encode,
    "lz4_decode": lz4_decode,
    "lz4_encode": lz4_encode,
    # image codecs
    "png_decode": png_decode,
    "png_encode": png_encode,
    "webp_decode": webp_decode,
    "webp_encode": webp_encode,
    "jpeg_decode": jpeg_decode,
    "jpeg_encode": jpeg_encode,
    "jpeg8_decode": jpeg8_decode,
    "jpeg8_encode": jpeg_encode,
    "jpegxl_decode": jpegxl_decode,
    "jpegxl_encode": jpegxl_encode,
    "jpeg2k_decode": jpeg2k_decode,
    "jpeg2k_encode": jpeg2k_encode,
}


_installed: bool = False
_original: Any = None


def _patch_module(module: Any) -> None:
    """Replace ``imagecodecs`` reference inside the given tifffile module
    with a SimpleNamespace that forwards most attributes but overrides ours.
    """
    global _original
    if _original is None:
        _original = module.imagecodecs

    fwd = SimpleNamespace()
    # Forward every public attribute from the real imagecodecs object.
    for name in dir(_original):
        if not name.startswith("_"):
            setattr(fwd, name, getattr(_original, name))
    # Override with our adapters.
    for name, func in _OVERRIDES.items():
        setattr(fwd, name, func)
    module.imagecodecs = fwd


def install() -> None:
    """Install opencodecs as tifffile's codec backend (idempotent)."""
    global _installed
    if _installed:
        return
    import tifffile.tifffile as _tt
    _patch_module(_tt)
    # tifffile builds CompressionCodec on first access; clear the cached
    # property so it re-resolves through the new imagecodecs reference.
    try:
        _tt.TIFF.__dict__.pop("COMPRESSORS", None)
        _tt.TIFF.__dict__.pop("DECOMPRESSORS", None)
    except Exception:  # pragma: no cover - dict.pop on a class dict is safe
        pass
    _installed = True


def uninstall() -> None:
    """Restore the original tifffile codec backend."""
    global _installed, _original
    if not _installed or _original is None:
        return
    import tifffile.tifffile as _tt
    _tt.imagecodecs = _original
    try:
        _tt.TIFF.__dict__.pop("COMPRESSORS", None)
        _tt.TIFF.__dict__.pop("DECOMPRESSORS", None)
    except Exception:  # pragma: no cover - dict.pop on a class dict is safe
        pass
    _installed = False


@contextlib.contextmanager
def patched():
    """Context manager: install on enter, uninstall on exit."""
    install()
    try:
        yield
    finally:
        uninstall()


__all__ = ["install", "uninstall", "patched"]
