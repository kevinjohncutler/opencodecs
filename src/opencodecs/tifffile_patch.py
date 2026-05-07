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


def png_decode(data, out=None, **kw):
    from .codecs._png import decode as _decode
    return _decode(bytes(data) if not isinstance(data, (bytes, bytearray)) else data)


def png_encode(data, level=None, out=None, **kw):
    from .codecs._png import encode as _encode
    return _encode(data, level=level)


def webp_decode(data, hasalpha=None, out=None, **kw):
    from .codecs._webp import decode as _decode
    return _decode(bytes(data) if not isinstance(data, (bytes, bytearray)) else data)


def webp_encode(data, level=None, lossless=False, out=None, **kw):
    from .codecs._webp import encode as _encode
    return _encode(data, level=level, lossless=lossless)


def jpeg_decode(data, out=None, **kw):
    from .codecs._jpeg import decode as _decode
    return _decode(bytes(data) if not isinstance(data, (bytes, bytearray)) else data)


def jpeg_encode(data, level=None, out=None, **kw):
    from .codecs._jpeg import encode as _encode
    return _encode(data, level=level)


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
    "jpeg8_decode": jpeg_decode,
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
