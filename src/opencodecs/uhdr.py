"""Public Python API for the Ultra-HDR / ISO 21496-1 codec.

Encodes a single linear-light HDR raster (Display-P3 or Rec.2020) into
an Ultra-HDR JPEG (or HEIF / AVIF) containing an SDR base + a per-pixel
gain map. HDR-aware decoders (Chrome 116+, Safari 26+, Apple Photos,
libuhdr) composite to display headroom; older decoders see just the
SDR base -- which is the correctly-tonemapped fallback we want for
cross-browser HDR display.

This module is a thin re-export of the Cython binding at
``opencodecs.codecs._uhdr``, which calls libuhdr's own encode / decode
directly. The earlier ``encode_native`` Python+Cython "fast path" that
re-implemented gain-map computation outside libuhdr has been removed
to avoid silently diverging from libuhdr's reference formula on
upstream tweaks.

Examples
--------
Encode a linear-Display-P3 float HDR array as Ultra-HDR JPEG::

    import numpy as np
    import opencodecs
    # arr: HxWx3 or HxWx4 float, linear-light, 1.0 == 203 nits (SDR white).
    data = opencodecs.uhdr.encode(arr, gamut='display-p3')
    with open('out.jpg', 'wb') as f:
        f.write(data)

Decode back to fp16 HDR pixels::

    with open('out.jpg', 'rb') as f:
        info = opencodecs.uhdr.decode(f.read())
    hdr = info['hdr_fp16']   # (H, W, 4) fp16 RGBA, linear-light
"""

from __future__ import annotations

# Backend is optional: opencodecs still imports when libuhdr / the Cython
# extension isn't available; calling any uhdr function then raises a
# clear error instead of an ImportError at import time.
try:
    from .codecs._uhdr import (
        encode,
        encode_assembled,
        decode,
        is_uhdr,
        libuhdr_version,
        UhdrError,
    )
    _HAVE_BACKEND = True
except ImportError as _exc:  # pragma: no cover
    _HAVE_BACKEND = False
    _IMPORT_ERROR = _exc

    def _missing(*_a, **_kw):
        raise ImportError(
            "opencodecs.uhdr backend (libultrahdr Cython extension) is "
            f"not available: {_IMPORT_ERROR}. Install libultrahdr "
            "(macOS: `brew install libultrahdr`; Linux: build from "
            "https://github.com/google/libultrahdr) and reinstall opencodecs."
        )

    encode = encode_assembled = decode = is_uhdr = libuhdr_version = _missing  # type: ignore[assignment]
    UhdrError = type("UhdrError", (Exception,), {})  # type: ignore[assignment]


__all__ = [
    "encode",
    "encode_assembled",
    "decode",
    "is_uhdr",
    "libuhdr_version",
    "UhdrError",
    "read",
    "write",
]


def read(path, **kwargs) -> dict:
    """Decode an Ultra-HDR file from disk. Returns the same dict as
    :func:`decode`."""
    with open(path, "rb") as f:
        return decode(f.read(), **kwargs)


def write(path, hdr, **kwargs) -> None:
    """Encode an Ultra-HDR file to disk. ``hdr`` must be an HxWx3 or
    HxWx4 float array (linear-light, 1.0 == SDR-reference white = 203
    nits). All other kwargs forwarded to :func:`encode`."""
    data = encode(hdr, **kwargs)
    with open(path, "wb") as f:
        f.write(data)
