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

# Native module — also re-export GifReader / GifWriter for users who
# want the streaming API directly.
try:
    from .codecs import _gif as _gif_mod
    _gif_encode = _gif_mod.encode
    _gif_decode = _gif_mod.decode
    _gif_check_signature = _gif_mod.check_signature
    GifReader = _gif_mod.GifReader
    GifWriter = _gif_mod.GifWriter
    _HAVE_BACKEND = True
except Exception:  # pragma: no cover - extension not built
    (
        _gif_encode, _gif_decode, _gif_check_signature, _HAVE_BACKEND,
    ) = import_or_stubs(
        "opencodecs.codecs._gif",
        "encode", "decode", "check_signature",
    )
    GifReader = None  # type: ignore
    GifWriter = None  # type: ignore


class GifCodec(Codec):
    """GIF87a / GIF89a via giflib — full streaming Reader + Writer.

    ``open(src)`` returns a :class:`GifReader` that lazily composites
    frames to RGB on demand (memory cost is one frame, not N). For
    streaming encode (multi-frame animations), use :class:`GifWriter`
    directly::

        with GifWriter(width=320, height=200, loop=0) as w:
            for frame in frames:
                w.write_frame(frame, delay_centiseconds=10)
        blob = w.close()
    """

    name = "gif"
    file_extensions = (".gif",)

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = True
    streaming_decode = True
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
        # For RGB output (the common case) we route through GifReader,
        # which uses our custom oc_giflzw decoder (~1.5x faster than
        # libgif's reference + handles multi-frame). For asrgb=False
        # (raw palette indices, single-frame only) keep the original
        # libgif-based path — palette mode doesn't need compositing.
        if not asrgb:
            return _gif_decode(_read_src(src), asrgb=False)
        if GifReader is None:  # pragma: no cover
            return _gif_decode(_read_src(src), asrgb=True)
        with GifReader(_read_src(src)) as r:
            return r.read()

    def open(self, src: Any, **opts):
        """Return a streaming :class:`GifReader` for ``src``.

        ``src`` is bytes-like, a file path, or any object readable via
        :func:`opencodecs.core._io_helpers.read_src`. The returned
        reader honors :meth:`iter_frames`, ``[i]`` random access, and
        :meth:`read` (returns the stacked ndarray)."""
        if GifReader is None:  # pragma: no cover - extension missing
            raise RuntimeError("opencodecs._gif extension not built")
        return GifReader(_read_src(src))


__all__ = ["GifCodec", "GifReader", "GifWriter"]
