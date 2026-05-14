"""GifCodec — GIF87a/89a via giflib.

Single-frame and animated GIF decode (composited to RGB); single-frame
encode from a palette-index array. For RGB-to-GIF encoding the caller
needs to quantize down to 256 colors first (we don't ship a quantizer
to avoid a heavy color-science dependency — use PIL's quantize() or
similar).

Returns RGB uint8 by default; pass ``asrgb=False`` to get raw palette
indices (single-frame only).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _gif_encode, _gif_decode, _gif_check_signature, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._gif",
    "encode", "decode", "check_signature",
)


class GifCodec(Codec):
    """GIF87a / GIF89a via giflib."""

    name = "gif"
    file_extensions = (".gif",)

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = True
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.uint8,)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        return _gif_check_signature(head)

    def encode(self, data: Any, *, dest=None,
               colormap=None, **opts) -> bytes | None:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        out = _gif_encode(data, colormap=colormap)
        return _write_dest(out, dest)

    def decode(self, src: Any, *, asrgb: bool = True, **opts) -> np.ndarray:
        return _gif_decode(_read_src(src), asrgb=bool(asrgb))


__all__ = ["GifCodec"]
