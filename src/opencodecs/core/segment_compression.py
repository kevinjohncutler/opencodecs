"""Shared per-segment compression dispatcher for container codecs.

Container formats (TIFF tiles/strips, NDTiff frames, CZI sub-blocks,
Zarr chunks, future formats) all face the same problem: pick a codec
by name or numeric tag, compress one segment of bytes, write it out;
on read, look up the same tag and decompress.

This module is a single source of truth for that mapping. It lets
every container reader/writer in opencodecs call:

    from opencodecs.core.segment_compression import (
        encode_segment, decode_segment, codec_name_to_tiff_code,
    )

and avoid re-implementing the dispatch table per format. The
underlying compressors are opencodecs's already-built native codecs
(``opencodecs.codecs._deflate``, ``._zstd``, etc.); we never link
to a new C library here.

The default tag values follow the TIFF 6 + community-assigned
compression codes (e.g. 8 = deflate, 50000 = zstd) so any container
that re-uses TIFF's namespace gets a free mapping.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Codec name → TIFF/community compression code mapping
# ---------------------------------------------------------------------------
#
# Sourced from opencodecs/codecs/_tiff.pyx CMP_* constants. Kept here
# duplicated so this module can be imported without dragging in the
# native TIFF extension on platforms where it wasn't built.

NONE          = 1
LZW           = 5
JPEG          = 7
DEFLATE       = 8
PACKBITS      = 32773
LZMA          = 34925
ZSTD          = 50000
WEBP          = 50001
JXL           = 50002
JPEG2000      = 34712
LERC          = 34887
LERC_LEGACY   = 33003
ADOBE_DEFLATE = 32946


# Public name → numeric code. Aliases ("zlib" → deflate) live here.
_NAME_TO_CODE: dict[str, int] = {
    "none":     NONE,
    "raw":      NONE,
    "deflate":  DEFLATE,
    "zlib":     DEFLATE,            # alias — same wire format
    "adeflate": ADOBE_DEFLATE,
    "lzw":      LZW,
    "packbits": PACKBITS,
    "jpeg":     JPEG,
    "jpeg2000": JPEG2000,
    "lerc":     LERC,
    "zstd":     ZSTD,
    "jxl":      JXL,
    "jpegxl":   JXL,
    "webp":     WEBP,
}


def codec_name_to_code(name: str | int) -> int:
    """Resolve a friendly name (``"zstd"``) or a numeric tag to the
    canonical TIFF compression code.
    """
    if isinstance(name, int):
        return name
    s = name.lower().strip()
    if s in _NAME_TO_CODE:
        return _NAME_TO_CODE[s]
    raise ValueError(
        f"unknown segment-compression codec {name!r}; "
        f"expected one of {sorted(_NAME_TO_CODE.keys())}"
    )


def codec_code_to_name(code: int) -> str:
    """Friendly name for a numeric code. Falls back to ``code=<n>``."""
    for k, v in _NAME_TO_CODE.items():
        if v == code:
            return k
    return f"code={code}"


# ---------------------------------------------------------------------------
# Encode / decode dispatchers
# ---------------------------------------------------------------------------


_DECODER_MOD: dict[int, str] = {
    DEFLATE:       "opencodecs.codecs._deflate",
    ADOBE_DEFLATE: "opencodecs.codecs._deflate",
    ZSTD:          "opencodecs.codecs._zstd",
    JPEG:          "opencodecs.codecs._jpeg",
    JPEG2000:      "opencodecs.codecs._jpeg2k",
    JXL:           "opencodecs.codecs._jxl",
    WEBP:          "opencodecs.codecs._webp",
    LERC:          "opencodecs.codecs._lerc",
    LERC_LEGACY:   "opencodecs.codecs._lerc",
    # LZW + PackBits live inside the TIFF codec (vendored decoders).
    LZW:           "opencodecs.codecs._tiff",
    PACKBITS:      "opencodecs.codecs._tiff",
}


# Some codecs have asymmetric encode/decode attribute names
# (e.g. _tiff exports ``lzw_decode`` but no ``lzw_encode`` — LZW
# encode is rare in our use cases and not implemented yet).
_DECODE_FN: dict[int, str] = {
    LZW:       "lzw_decode",
    PACKBITS:  "packbits_decode",
}
_ENCODE_FN: dict[int, str] = {
    # LZW + PackBits encode not implemented in opencodecs.codecs._tiff
    # yet — encoders that emit these would have to vendor one.
    DEFLATE:   "encode",
    ADOBE_DEFLATE: "encode",
    ZSTD:      "encode",
    JXL:       "encode",
    JPEG:      "encode",
    JPEG2000:  "encode",
    WEBP:      "encode",
    LERC:      "encode",
    LERC_LEGACY: "encode",
}


_FN_CACHE: dict[tuple[int, str], Callable] = {}


def _lookup_fn(code: int, side: str) -> Callable:
    """Resolve the encode/decode callable for one compression code."""
    key = (code, side)
    fn = _FN_CACHE.get(key)
    if fn is not None:
        return fn
    modname = _DECODER_MOD.get(code)
    if modname is None:
        raise NotImplementedError(
            f"segment_compression: no opencodecs codec for "
            f"compression code {code} ({codec_code_to_name(code)})"
        )
    if side == "decode":
        attr = _DECODE_FN.get(code, "decode")
    else:
        attr = _ENCODE_FN.get(code)
        if attr is None:
            raise NotImplementedError(
                f"segment_compression: encode not implemented for "
                f"{codec_code_to_name(code)} (code {code}); use a "
                f"different codec or read-only path"
            )
    try:
        mod = import_module(modname)
    except ImportError as exc:
        raise NotImplementedError(
            f"segment_compression: {codec_code_to_name(code)} backend "
            f"({modname}) not built on this platform: {exc}"
        ) from exc
    fn = getattr(mod, attr)
    _FN_CACHE[key] = fn
    return fn


def encode_segment(data, codec: str | int, *, level: int | None = None,
                   **codec_kwargs) -> bytes:
    """Compress ``data`` (bytes-like) using ``codec``.

    ``codec`` is either a name (``"zstd"``) or a numeric TIFF code.
    ``level`` is passed through to the underlying codec when the codec
    accepts a ``level`` kwarg (deflate, zstd, jxl, ...). Codec-specific
    extras can be passed via ``codec_kwargs``.
    """
    code = codec_name_to_code(codec)
    if code == NONE:
        return bytes(data) if not isinstance(data, bytes) else data
    fn = _lookup_fn(code, "encode")
    kw: dict[str, Any] = dict(codec_kwargs)
    if level is not None and "level" not in kw:
        kw["level"] = level
    return fn(data, **kw)


def decode_segment(data, codec: str | int, **codec_kwargs) -> bytes:
    """Decompress ``data`` using ``codec``.

    Returns ``bytes`` for byte-stream codecs (deflate, zstd, lzw,
    packbits). For image-format codecs (jpeg, jxl, lerc, jpeg2k, webp)
    the underlying codec returns an ndarray; this dispatcher passes
    that through transparently (use ``decode_segment_array`` if you
    want a strict ndarray return type).
    """
    code = codec_name_to_code(codec)
    if code == NONE:
        return bytes(data) if not isinstance(data, bytes) else data
    fn = _lookup_fn(code, "decode")
    return fn(data, **codec_kwargs)


__all__ = [
    "NONE", "LZW", "JPEG", "DEFLATE", "PACKBITS", "LZMA",
    "ZSTD", "WEBP", "JXL", "JPEG2000", "LERC", "LERC_LEGACY",
    "ADOBE_DEFLATE",
    "codec_name_to_code", "codec_code_to_name",
    "encode_segment", "decode_segment",
]
