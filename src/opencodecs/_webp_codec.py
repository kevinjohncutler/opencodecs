"""WebpCodec — Codec adapter wrapping the native _webp extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec, Reader
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _webp_encode, _webp_decode, _webp_check_signature,
    _webp_frame_count, _webp_decode_animation, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._webp",
    "encode", "decode", "check_signature",
    "frame_count", "decode_animation",
)


class WebpCodec(Codec):
    """Native WebP codec via libwebp."""

    name = "webp"
    file_extensions = (".webp",)

    has_native = True
    has_delegate = False
    can_encode = True
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
               lossless: bool = True,
               numthreads: int | None = None,
               method: int = -1,
               **opts) -> bytes | None:
        # ``lossless=True`` by default to match ``imagecodecs.webp_encode``
        # — see docs/codec_api_conventions.md "Default settings:
        # Pareto-better than the reference, no cheating." Callers who
        # want a small lossy blob should pass ``lossless=False, level=N``.
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        encoded = _webp_encode(
            data, level=level, lossless=lossless,
            numthreads=numthreads, method=method,
        )
        return _write_dest(encoded, dest)

    def decode(self, src: Any, **opts) -> np.ndarray:
        """Decode a still WebP, or the first frame of an animation.

        The plain decoder cannot read an animation container at all --
        it failed with "WebP decode failed", which told a caller
        nothing about why. Falling back to the animation path gives the
        first frame, which is what the AVIF and HEIF codecs return for
        a multi-image file, so the three agree.
        """
        data = _read_src(src)
        try:
            return _webp_decode(data)
        except Exception:
            if _webp_frame_count(data) <= 1:
                raise
            frames, _, _ = _webp_decode_animation(data)
            return frames[0]

    def frame_count(self, src: Any) -> int:
        """Frames in an animated WebP; 1 for a still.

        Reads the animation header only.
        """
        return _webp_frame_count(_read_src(src))

    def open(self, src: Any, *, numthreads: int | None = None):
        return WebpReader(_read_src(src), numthreads=numthreads)



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
                _webp_decode_animation(data, numthreads=numthreads)
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


__all__ = ["WebpCodec", "WebpReader"]
