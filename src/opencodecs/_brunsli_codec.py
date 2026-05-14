"""BrunsliCodec — JPEG transcoder that stores JPEGs ~20% smaller.

Brunsli (Google, 2019) repacks an existing JPEG bitstream into a
smaller container that decodes losslessly back to the same JPEG
bytestream. There is no quality change vs. the source JPEG —
Brunsli is a *container* reformatter, not a re-encoder.

This codec has two valid encode inputs:

  * a JPEG bitstream (``bytes``/``bytearray``) — the recommended path.
    Brunsli simply transcodes it.
  * an ndarray — opencodecs first JPEG-encodes it (via the standard
    ``jpeg`` codec) at quality ``level``, then transcodes that JPEG
    to Brunsli. Useful for "shrink-as-much-as-possible" pipelines
    that don't care about the intermediate JPEG.

Decode always returns an ndarray (the JPEG is recovered then decoded
via the standard ``jpeg`` codec). Raw-JPEG output (``asjpeg=True``) is
also supported for callers who want byte-identical JPEG recovery.

Example::

    import opencodecs as oc
    jpeg_bytes = open("photo.jpg", "rb").read()
    brn = oc.write(None, jpeg_bytes, format="brunsli")
    assert len(brn) < len(jpeg_bytes) * 0.95   # typically ~80% of input

    # Lossless JPEG recovery:
    same_jpeg = oc.read(brn, format="brunsli", asjpeg=True)
    assert same_jpeg == jpeg_bytes
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _brunsli_encode_jpeg, _brunsli_decode_jpeg, _brunsli_check_signature, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._brunsli",
    "encode_jpeg", "decode_jpeg", "check_signature",
)


class BrunsliCodec(Codec):
    """Native Brunsli — lossless JPEG transcoder."""

    name = "brunsli"
    file_extensions = (".brn",)

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.uint8,)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        return _brunsli_check_signature(head)

    def encode(self, data: Any, *, dest=None, level: int = 90, **opts) -> bytes | None:
        if isinstance(data, (bytes, bytearray, memoryview)):
            jpeg_bytes = bytes(data)
        else:
            # Re-encode array → JPEG first, then transcode.
            import opencodecs as _oc
            jpeg_bytes = _oc.write(None, data, format="jpeg", level=int(level))
        out = _brunsli_encode_jpeg(jpeg_bytes)
        return _write_dest(out, dest)

    def decode(self, src: Any, *, asjpeg: bool = False, **opts) -> Any:
        blob = _read_src(src)
        jpeg_bytes = _brunsli_decode_jpeg(blob)
        if asjpeg:
            return jpeg_bytes
        import opencodecs as _oc
        return _oc.read(jpeg_bytes, format="jpeg")


__all__ = ["BrunsliCodec"]
