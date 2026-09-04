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

from .core.codec import Codec, Reader, Writer
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


if GifWriter is not None:
    # An extension type cannot inherit a Python ABC, and GifWriter
    # already has the whole surface (write_frame + close), so register
    # it as a virtual subclass rather than wrap it. The reader needs a
    # wrapper because it also had to normalize dtype; this does not.
    Writer.register(GifWriter)


class GifStreamReader(Reader):
    """Reader adapter wrapping the cdef GifReader.

    The compiled reader already has the whole surface -- read,
    iter_frames, shape, dtype, n_frames, ``[i]`` -- but a Cython
    extension type cannot inherit a Python ABC, so ``isinstance(r,
    Reader)`` was False for GIF and only for GIF. It also reported
    ``dtype`` as the scalar type ``numpy.uint8`` rather than an
    ``np.dtype``, which is what the Reader contract annotates and what
    every other format returns.

    Same shape as :class:`opencodecs._jxl_codec.JpegXLReader`, which
    wraps the cdef JXL reader for the same reason. Anything not named
    here forwards to the inner reader, so no GIF-specific attribute is
    lost by going through it.
    """

    is_chunked = True  # [i] replays from frame 0; GIF disposal forbids seek

    def __init__(self, data: Any):
        self._inner = GifReader(data)
        self.shape = self._inner.shape
        self.dtype = np.dtype(self._inner.dtype)
        self.n_frames = self._inner.n_frames

    def iter_frames(self):
        return self._inner.iter_frames()

    def read(self) -> np.ndarray:
        return self._inner.read()

    def __getitem__(self, idx) -> np.ndarray:
        return self._inner[idx]

    def __len__(self) -> int:
        return self._inner.n_frames

    def __getattr__(self, name: str):
        # Only reached for names this class does not define, so the
        # wrapper adds an interface without hiding one.
        return getattr(self._inner, name)

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if close is not None:
            close()

    def __repr__(self) -> str:
        return (f"<GifStreamReader shape={self.shape} "
                f"n_frames={self.n_frames}>")


class _LazyGifWriter(Writer):
    """GifWriter needs the canvas size before the first frame.

    Every other writer here takes a destination and learns the geometry
    from what it is given, so a caller driving writers generically has
    nothing to pass. This defers construction to the first
    ``write_frame`` and takes the canvas from that frame's shape, which
    is what the caller meant anyway -- GIF requires every frame to match
    it. Passing ``width``/``height`` explicitly still works and skips
    the inference.
    """

    def __init__(self, dest: Any = None, **opts):
        self._dest = dest
        self._opts = opts
        self._inner = None
        self._closed = False
        self._result: bytes | None = None

    def write_frame(self, arr, **opts) -> None:
        if self._closed:
            raise RuntimeError("gif: writer is closed")
        arr = np.asarray(arr)
        if self._inner is None:
            opts_ = dict(self._opts)
            opts_.setdefault("height", arr.shape[0])
            opts_.setdefault("width", arr.shape[1])
            self._inner = GifWriter(**opts_)
        self._inner.write_frame(arr, **opts)

    def close(self) -> bytes | None:
        if self._closed:
            return self._result
        self._closed = True
        if self._inner is None:
            raise ValueError("gif: closed without writing a frame")
        self._result = self._inner.close()
        if self._dest is not None and self._result is not None:
            self._result = _write_dest(self._result, self._dest)
        return self._result


class GifCodec(Codec):
    """GIF87a / GIF89a via giflib — full streaming Reader + Writer.

    ``open(src)`` returns a :class:`GifStreamReader` that lazily composites
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

    def writer(self, dest: Any = None, **opts):
        """A real streaming GIF writer, one composited frame at a time."""
        if GifWriter is None:  # pragma: no cover - extension missing
            raise RuntimeError("opencodecs._gif extension not built")
        return _LazyGifWriter(dest, **opts)

    def open(self, src: Any, **opts) -> "GifStreamReader":
        """Return a streaming :class:`GifStreamReader` for ``src``.

        ``src`` is bytes-like, a file path, or any object readable via
        :func:`opencodecs.core._io_helpers.read_src`. The returned
        reader honors :meth:`iter_frames`, ``[i]`` random access, and
        :meth:`read` (returns the stacked ndarray)."""
        if GifReader is None:  # pragma: no cover - extension missing
            raise RuntimeError("opencodecs._gif extension not built")
        return GifStreamReader(_read_src(src))


__all__ = ["GifCodec", "GifReader", "GifStreamReader", "GifWriter"]
