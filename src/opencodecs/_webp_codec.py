"""WebpCodec — Codec adapter wrapping the native _webp extension."""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec, Reader
from .core.buffers import array_output
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs
from .core.pipeline import native_workers
from ._png_codec import _no_encode_out

(
    _webp_encode, _webp_decode, _webp_check_signature,
    _webp_frame_count, _webp_decode_animation, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._webp",
    "encode", "decode", "check_signature",
    "frame_count", "decode_animation",
)


def decode_webp(data, *, out=None, index=None, hasalpha=None,
                numthreads=None):
    """Decode WebP ``data`` (bytes-like) with imagecodecs' parameters.

    Shared by ``WebpCodec.decode`` and the tifffile adapter; see
    ``WebpCodec.decode`` for what each parameter means.
    """
    if index is not None:
        index = operator.index(index)
    out = array_output(out)
    if index in (None, 0, -1):
        # A still decodes directly; libwebp refuses an animation here,
        # so the frame count is only read when that happens.
        try:
            return _webp_decode(data, hasalpha=hasalpha, out=out)
        except Exception:
            if _webp_frame_count(data) <= 1:
                raise
    n = _webp_frame_count(data)
    if index is None:
        frame = None
    else:
        frame = index + n if index < 0 else index
        if not 0 <= frame < n:
            raise IndexError(f"webp decode: index={index} out of range "
                             f"[0, {n - 1}]")
    if frame is None and out is not None:
        raise ValueError(
            "webp decode: out= cannot take a whole animation; pass index= "
            "or use open()")
    frames, _, _ = _webp_decode_animation(
        data, numthreads=native_workers(numthreads))
    arr = np.stack(frames) if frame is None else frames[frame]
    # libwebp composes every canvas as RGBA. With hasalpha=None,
    # imagecodecs keeps alpha only when a canvas it returns has a pixel
    # that is not fully opaque: any frame of the stack, or the one frame
    # index= picks. The VP8X alpha flag is not that rule: libwebp's
    # animation encoder sets it for opaque lossless frames too, since
    # the sub-frames it stores rely on blending.
    if hasalpha is None:
        hasalpha = int(arr[..., 3].min()) != 255
    if not hasalpha:
        arr = np.ascontiguousarray(arr[..., :3])
    if out is None:
        return arr
    if out.shape != arr.shape \
            or out.dtype != arr.dtype:
        raise ValueError(
            f"webp decode: out= must be a {arr.dtype} array of shape "
            f"{arr.shape}")
    out[...] = arr
    return out


class WebpCodec(Codec):
    """Native WebP codec via libwebp."""

    name = "webp"
    file_extensions = (".webp",)

    has_native = True
    has_delegate = False
    can_encode = True
    max_writer_frames = 1
    can_decode = True
    # Animated WebP, which we did not read at all: an animation
    # decoded as its first frame with nothing saying the rest existed.
    multi_frame = True
    # WebPAnimDecoderGetNext hands back one canvas at a time, so
    # iterating does not hold the whole animation.
    streaming_decode = True
    # chunked stays False on purpose, the same reasoning as GIF:
    # frames are sub-rectangles with disposal and blending rules, so
    # frame N really does require the frames before it. libwebp says
    # so in its own API -- GetNext only moves forward, and Reset
    # rewinds to the start.
    parallel_decode = False

    supported_dtypes = (np.uint8,)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        return _webp_check_signature(head)

    def encode(self, data: Any, *, dest=None, level: int | None = None,
               lossless: bool | None = True,
               numthreads: int | None = None,
               method: int | None = None,
               out: Any = None,
               **opts) -> bytes | None:
        """Encode a uint8 image as WebP.

        ``lossless=True`` by default to match ``imagecodecs.webp_encode``
        (see docs/codec_api_conventions.md "Default settings:
        Pareto-better than the reference, no cheating"). Lossless output
        is exact, including RGB values under fully transparent pixels.
        ``level`` is libwebp's ``WebPConfig.quality``: the quality factor
        when lossy, the compression effort (0 fastest, 100 smallest) when
        lossless; default 75. ``method`` is libwebp's 0-6 speed/size
        tradeoff (default 4), clamped to that range as in imagecodecs.
        For an RGB or RGBA array the bytes equal
        ``imagecodecs.webp_encode``'s for the same arguments when both
        link the same libwebp release. Callers who want a small lossy blob should
        pass ``lossless=False, level=N``. ``out``, imagecodecs' output
        buffer, may be ``None``; anything else raises ``TypeError``, since
        the encoded bytes are returned or written to ``dest``. Unknown
        options raise ``TypeError`` rather than being dropped.
        """
        if opts:
            raise TypeError(
                f"webp encode: unexpected option(s) {sorted(opts)}")
        _no_encode_out("webp", out)
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        encoded = _webp_encode(
            data, level=level, lossless=lossless,
            numthreads=native_workers(numthreads), method=method,
        )
        return _write_dest(encoded, dest)

    def decode(self, src: Any, *, out=None, index: int | None = None,
               hasalpha: bool | None = None,
               numthreads: int | None = None) -> np.ndarray:
        """Decode a still WebP, or every frame of an animation.

        The plain decoder cannot read an animation container at all: it
        failed with the bare message "WebP decode failed", which told a
        caller nothing about why.

        With ``index=None`` (default) an animation decodes to a
        ``(frames, H, W, C)`` stack, not to its first frame, as
        ``imagecodecs.webp_decode`` does; ``gif`` returns a time
        sequence the same way. Returning frame 0 would silently discard
        the rest. Pass ``index=`` for one frame; negative values count
        from the end, a still has the single frame 0, and anything out
        of range raises ``IndexError``. Every frame is the canvas the
        WebP container specification composes (sub-rectangle, blending
        and disposal applied), as in imagecodecs.

        ``hasalpha`` is imagecodecs' parameter: a true value returns
        RGBA and a false value RGB. With ``None`` the result has the
        same channels as imagecodecs: for a still, RGBA when its
        bitstream has alpha; for an animation, RGBA when a returned
        canvas has a pixel that is not fully opaque (any frame of the
        stack, or the frame ``index`` picks), RGB otherwise. ``open()``
        differs for an animation: its frames are always the RGBA canvas
        libwebp composes.

        ``numthreads`` caps the animation decoder's threads.

        A still is unchanged and still returns ``(H, W, C)``. Options
        imagecodecs does not define raise ``TypeError``.
        """
        return decode_webp(data=_read_src(src), out=out, index=index,
                           hasalpha=hasalpha, numthreads=numthreads)

    def frame_count(self, src: Any) -> int:
        """Frames in an animated WebP; 1 for a still.

        Reads the animation header only.
        """
        return _webp_frame_count(_read_src(src))

    def open(self, src: Any, *, numthreads: int | None = None):
        return WebpReader(_read_src(src), numthreads=native_workers(numthreads))



class WebpReader(Reader):
    """Reader over a WebP, animated or still.

    The whole animation is decoded on open and held. That is not
    laziness dressed up: WebP frames are sub-rectangles with disposal
    and blending rules, so reconstructing frame N requires replaying
    every frame before it, and libwebp's decoder only moves forward.
    Decoding once in a single forward pass is cheaper than re-running
    the prefix per frame, which is what any lazier arrangement would
    do. is_chunked is False to say so.

    A still is a one-frame animation rather than a special case.
    """

    is_chunked = False

    def __init__(self, data, *, numthreads: int | None = None):
        if _webp_frame_count(data) > 1:
            self._frames, self.timestamps, self.loop_count = \
                _webp_decode_animation(data, numthreads=native_workers(numthreads))
        else:
            self._frames = [_webp_decode(data)]
            self.timestamps = [0]
            self.loop_count = 0

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    @property
    def shape(self) -> tuple:
        return self._frames[0].shape

    @property
    def dtype(self):
        return self._frames[0].dtype

    def frame(self, index: int) -> np.ndarray:
        return self._frames[int(index)]

    def __getitem__(self, idx):
        i = int(idx)
        if i < 0:
            i += len(self._frames)
        if not 0 <= i < len(self._frames):
            raise IndexError(idx)
        return self._frames[i]

    def iter_frames(self):
        return iter(self._frames)

    def read(self) -> np.ndarray:
        """Every frame stacked, or the single image for a still.

        A still keeps the shape decode() returns for the same file.
        """
        if len(self._frames) == 1:
            return self._frames[0]
        return np.stack(self._frames)

    def close(self) -> None:
        self._frames = []

    def __enter__(self) -> "WebpReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()


__all__ = ["WebpCodec", "WebpReader", "decode_webp"]
