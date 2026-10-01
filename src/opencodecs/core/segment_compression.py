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
# (e.g. _tiff exports ``lzw_decode`` / ``lzw_encode`` and
# ``packbits_decode``; PackBits encode is still not implemented).
_DECODE_FN: dict[int, str] = {
    LZW:       "lzw_decode",
    PACKBITS:  "packbits_decode",
}
_ENCODE_FN: dict[int, str] = {
    # PackBits encode is the last asymmetric pair — rarely seen in
    # the wild and not worth vendoring; encoders that need it can
    # always fall back to none/deflate.
    LZW:       "lzw_encode",
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
    if side == "encode_buffer" and code == ZSTD:
        attr = "encode_buffer"
    elif side == "decode":
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
    if code == JPEG and side != "decode":
        fn = _tiff_jpeg_encoder(fn, mod.JpegError)
    _FN_CACHE[key] = fn
    return fn


def _tiff_jpeg_encoder(encode: Callable, error: type) -> Callable:
    """Wrap the JPEG encoder for the TIFF writers' segments.

    The writers record BitsPerSample from the array's dtype and
    PhotometricInterpretation from its sample count, so only an 8-bit
    grayscale or RGB JPEG matches those tags. The jpeg codec also writes
    uint16 arrays (as 12-bit JPEG, which a reader rejects under
    BitsPerSample 16) and four samples (as CMYK, which a reader rejects
    under PhotometricInterpretation RGB), so those raise here, as they
    did before the codec wrote them.
    """
    import numpy as np

    def encode_tiff_segment(data, **kw):
        a = data if isinstance(data, np.ndarray) else np.asarray(data)
        if a.dtype != np.uint8 or not (
                a.ndim == 2 or (a.ndim == 3 and a.shape[2] in (1, 3))):
            raise error(
                f"TIFF JPEG compression: the TIFF writers store 8-bit "
                f"grayscale or RGB JPEG, not {a.dtype} samples shaped "
                f"{a.shape}")
        return encode(data, **kw)
    return encode_tiff_segment


def segment_input_kind(codec: str | int) -> str:
    """Required native input representation, independent of quality policy."""
    return "image" if codec_name_to_code(codec) in (
        JPEG, JPEG2000, JXL, WEBP, LERC, LERC_LEGACY) else "bytes"


def prepare_segment_input(data, codec: str | int, *, copy=False):
    """Preserve shape for image codecs and provide byte views to byte codecs.

    A writer advancing a producer with reusable buffers must request a copy
    before advancing. Stable borrowed arrays can stay zero-copy when contiguous.
    Compression quality/defaults remain the caller's responsibility.
    """
    import numpy as np
    if segment_input_kind(codec) == "image":
        return (np.array(data, copy=True, order="C") if copy
                else np.ascontiguousarray(data))
    if isinstance(data, np.ndarray):
        view = memoryview(np.ascontiguousarray(data)).cast("B")
        return view.tobytes() if copy else view
    if copy:
        return memoryview(data).tobytes()
    return data


def encode_segment(data, codec: str | int, *, level: int | None = None,
                   owned_output: bool = False, **codec_kwargs) -> bytes | memoryview:
    """Compress ``data`` (bytes-like) using ``codec``.

    ``codec`` is either a name (``"zstd"``) or a numeric TIFF code.
    ``level`` is passed through to the underlying codec when the codec
    accepts a ``level`` kwarg (deflate, zstd, jxl, ...). Codec-specific
    extras can be passed via ``codec_kwargs``. ``owned_output=True`` may return
    a read-only view retaining its encoded allocation, avoiding a final copy.
    The caller must retain that view until the destination consumes it.

    A TIFF segment is always ``(rows, width[, samples])``, so JPEG 2000
    is called with ``planar=False`` unless the caller says otherwise:
    the codec's ``planar=None`` rule (imagecodecs') would read a strip
    of 4 or fewer rows with more than 4 samples as ``(C, H, W)``.
    """
    code = codec_name_to_code(codec)
    if code == NONE:
        return bytes(data) if not isinstance(data, bytes) else data
    fn = _lookup_fn(code, "encode_buffer" if owned_output and code == ZSTD else "encode")
    kw: dict[str, Any] = dict(codec_kwargs)
    if level is not None and "level" not in kw:
        kw["level"] = level
    if code == JPEG2000:
        kw.setdefault("planar", False)
    if code in (ZSTD, JPEG2000, JXL):
        from .pipeline import native_workers, in_worker
        # Zstandard counts background workers: zero means inline serial.
        threads = (0 if code == ZSTD and in_worker()
                   else native_workers(kw.get("numthreads")))
        if threads is not None:
            kw["numthreads"] = threads
    return fn(data, **kw)


def bind_segment_encoder(codec: str | int, *, level=None, owned_output=False,
                         verify=False, **codec_kwargs):
    """Bind repeated segment dispatch while retaining dynamic worker policy.

    The callable owns its immutable option dictionaries, not input pixels.
    Backend lookup and options are prepared once per writer or worker context.
    Nested calls still select a serial native encoder at invocation time.
    """
    from functools import partial
    from .pipeline import in_worker
    if verify:
        from .verification import verify_segment
        encoder = bind_segment_encoder(codec, level=level, owned_output=owned_output,
                                       **codec_kwargs)
        def checked(data):
            encoded = encoder(data)
            verify_segment(data, encoded, codec)
            return encoded
        return checked
    code = codec_name_to_code(codec)
    if code == NONE:
        return bytes
    fn = _lookup_fn(code, "encode_buffer" if owned_output and code == ZSTD else "encode")
    options = dict(codec_kwargs)
    if level is not None:
        options.setdefault("level", level)
    if code == JPEG2000:
        options.setdefault("planar", False)
    eager = partial(fn, **options)
    if code not in (ZSTD, JPEG2000, JXL):
        return eager
    nested_options = dict(options)
    nested_options["numthreads"] = 0 if code == ZSTD else 1
    nested = partial(fn, **nested_options)

    def encode(data):
        return nested(data) if in_worker() else eager(data)
    return encode


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
    if code in (JPEG2000, JXL):
        from .pipeline import native_workers
        threads = native_workers(codec_kwargs.get("numthreads"))
        if threads is not None:
            codec_kwargs["numthreads"] = threads
    return fn(data, **codec_kwargs)


__all__ = [
    "NONE", "LZW", "JPEG", "DEFLATE", "PACKBITS", "LZMA",
    "ZSTD", "WEBP", "JXL", "JPEG2000", "LERC", "LERC_LEGACY",
    "ADOBE_DEFLATE",
    "codec_name_to_code", "codec_code_to_name",
    "encode_segment", "bind_segment_encoder", "decode_segment", "segment_input_kind", "prepare_segment_input",
]
