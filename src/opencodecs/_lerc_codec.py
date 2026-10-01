"""LercCodec — Codec adapter wrapping the native _lerc extension.

LERC = Limited Error Raster Compression. Self-describing blob format
that round-trips an ndarray with full shape and dtype, parameterized
by ``max_z_error`` (0 = lossless, > 0 = lossy with absolute error
budget).

Used heavily in geospatial / GIS pipelines (Esri ArcGIS, Cloud-Optimized
Raster MRF, COG with LERC compression). For float arrays, near-lossless
LERC routinely beats DEFLATE by 5-20× at minor precision cost.

Example::

    import numpy as np
    import opencodecs as oc

    # Lossless: max_z_error=0
    arr = np.random.rand(512, 512).astype(np.float32)
    blob = oc.write(None, arr, format="lerc")
    back = oc.read(blob, format="lerc")
    np.testing.assert_array_equal(arr, back)

    # Near-lossless: 0.001 absolute error budget
    blob = oc.write(None, arr, format="lerc", max_z_error=1e-3)
    back = oc.read(blob, format="lerc")
    assert (np.abs(arr - back) <= 1e-3).all()

Blobs are Lerc2 codec version 4 unless ``version=`` picks another (2 to
6); see ``opencodecs.codecs._lerc`` for why. imagecodecs' keywords
(``level``, ``version``, ``masks``, ``planar``, ``compression``,
``compressionargs``) mean the same here, and an unknown keyword raises
TypeError instead of being ignored.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core.buffers import array_output
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _lerc_encode, _lerc_decode, _lerc_info, _lerc_check_signature,
    _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._lerc",
    "encode", "decode", "info", "check_signature",
)


class LercCodec(Codec):
    """Native LERC codec — Esri Limited Error Raster Compression."""

    name = "lerc"
    file_extensions = (".lerc",)

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (
        np.int8, np.uint8,
        np.int16, np.uint16,
        np.int32, np.uint32,
        np.float32, np.float64,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return _lerc_check_signature(head)

    def encode(self, data: Any, *, dest=None,
               max_z_error: float | None = None,
               level: float | None = None,
               version: int | None = None,
               masks=None, planar: bool | None = None,
               compression: str | None = None,
               compressionargs: dict | None = None,
               **opts) -> bytes | None:
        """Encode as a LERC blob (Lerc2 codec version 4 by default).

        ``level`` is imagecodecs' name for ``max_z_error``.
        """
        if opts:
            raise TypeError(
                f"lerc encode: unsupported option(s) {', '.join(sorted(opts))}")
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        out = _lerc_encode(data, max_z_error=max_z_error, level=level,
                           version=version, masks=masks, planar=planar,
                           compression=compression,
                           compressionargs=compressionargs)
        return _write_dest(out, dest)

    def decode(self, src: Any, *, out=None, masks=None, **opts) -> np.ndarray:
        """Decode a LERC blob; ``masks=True`` also returns the masks.

        imagecodecs' lerc_decode takes ``masks`` and ``out``; any other
        keyword raises TypeError rather than being dropped.
        """
        if opts:
            raise TypeError(
                f"lerc decode: unsupported option(s) {', '.join(sorted(opts))}")
        if out is None:
            return _lerc_decode(_read_src(src), masks=masks)
        return _lerc_decode(_read_src(src), out=array_output(out), masks=masks)

    def info(self, src: Any) -> dict:
        """Return shape, dtype, value range, version without decoding."""
        return _lerc_info(_read_src(src))


__all__ = ["LercCodec"]
