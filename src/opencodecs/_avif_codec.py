"""AvifCodec — Codec adapter wrapping the native _avif extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec, Reader
from .core.buffers import array_output
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs
from .core.pipeline import native_workers
from .core.native_source import NativeSource

(
    _avif_encode, _avif_decode, _avif_check_signature,
    _avif_read_icc, _avif_frame_count, _AvifSequence, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._avif",
    "encode", "decode", "check_signature", "read_icc_profile",
    "frame_count", "AvifSequence",
)


class AvifCodec(Codec):
    """Native AVIF codec via libavif."""

    name = "avif"
    file_extensions = (".avif",)

    has_native = True
    has_delegate = False
    can_encode = True
    max_writer_frames = 1
    can_decode = True
    # AVIF carries image sequences (the same machinery as AV1 video)
    # and progressive layers, both indexed through avifDecoderNthImage.
    # We used to decode only the primary image, so an animated AVIF
    # read back as its first frame with nothing saying the rest
    # existed.
    multi_frame = True
    # open() holds one decoder across frames, so the container is
    # parsed once and iterating does not materialize the sequence.
    streaming_decode = True
    # Frames are indexed in the sample table the parse already built,
    # so reaching frame N does not decode frames 0..N-1.
    chunked = True
    # libavif threads one image across tiles: decoder.maxThreads is
    # set in _avif.pyx decode(), and a 2048x2048 blob decodes 7.6x
    # faster at numthreads=8 than at 1.
    parallel_decode = True

    supported_dtypes = (np.uint8,)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        return _avif_check_signature(head)

    def encode(self, data: Any, *, dest=None, level: int | None = None,
               lossless: bool = True, speed: int = 0,
               color=None, bit_depth: int | None = None,
               numthreads: int | None = None,
               iccprofile: bytes | None = None,
               **opts) -> bytes | None:
        """Encode an array as AVIF.

        ``iccprofile`` embeds an ICC color profile.

        ``lossless=True`` by default to match
        ``imagecodecs.avif_encode``'s lossless-at-default behavior —
        see docs/codec_api_conventions.md "Default settings:
        Pareto-better than the reference, no cheating." For typical
        photo-storage workloads pass ``lossless=False, level=63`` (or
        similar) to get the smaller lossy blob.
        """
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        encoded = _avif_encode(
            data, level=level, lossless=lossless, speed=speed,
            color=color, bit_depth=bit_depth, numthreads=native_workers(numthreads),
            iccprofile=iccprofile,
        )
        return _write_dest(encoded, dest)

    def decode(self, src: Any, *, numthreads: int | None = None,
               out=None, **opts) -> np.ndarray:
        """Decode a still, or every image of a sequence.

        A sequence decodes to a ``(frames, H, W, C)`` stack, matching
        `gif` and `webp` here and matching what imagecodecs returns for
        the same file. This used to return the first image and say
        nothing about the rest, which is the thing reading sequences
        was meant to fix.

        A still is untouched: same shape, same single-image path, and
        `out=` still writes into the caller's buffer. That option has
        no meaning for a stack whose frame count is not known until the
        container is parsed, so it is refused there rather than
        silently ignored.
        """
        data = _read_src(src)
        if _avif_frame_count(data) <= 1:
            return _avif_decode(data, numthreads=native_workers(numthreads), out=out if out is None else array_output(out))
        if out is not None:
            raise ValueError(
                "avif decode: out= cannot take a multi-image sequence; "
                "decode without it, or use open() and write the frames "
                "where you want them")
        with AvifReader(data, numthreads=native_workers(numthreads)) as r:
            return np.stack([r.frame(i) for i in range(r.n_frames)])

    def frame_count(self, src: Any) -> int:
        """How many images the file holds; 1 for a plain still.

        Parses the container only, so asking does not cost a decode.
        """
        if isinstance(src, (bytes, bytearray, memoryview)):
            return _avif_frame_count(src)
        with self.open(src) as reader:
            return reader.n_frames

    def open(self, src: Any, *, numthreads: int | None = None):
        """A reader over an AVIF sequence.

        One decoder is held open across frames. Re-reading the file per
        frame would re-parse the container each time and turn N frames
        into N parses; here the parse happens once and a frame is a
        lookup in the sample table it built.
        """
        if isinstance(src, (bytes, bytearray, memoryview)):
            return AvifReader(src, numthreads=numthreads)
        source = NativeSource(src)
        try:
            return AvifReader(source, numthreads=numthreads)
        except BaseException:
            source.close()
            raise

    def read_icc_profile(self, src: Any) -> bytes | None:
        """Return the embedded ICC profile bytes, or ``None`` if absent."""
        return _avif_read_icc(_read_src(src))



class AvifReader(Reader):
    """Frame-oriented reader over an AVIF still, sequence or grid.

    A still is a one-frame sequence rather than a special case, so a
    caller that iterates gets one frame and does not have to ask which
    kind of file it opened first.
    """

    is_chunked = True

    def __init__(self, data, *, numthreads: int | None = None):
        import os
        import threading
        self._lock = threading.RLock()
        self._source = data if isinstance(data, NativeSource) else None
        self._numthreads = (os.cpu_count() or 4) if numthreads is None or numthreads <= 0 else numthreads
        self._seq = _AvifSequence(data, native_workers(self._numthreads))

    @property
    def n_frames(self) -> int:
        return self._seq.n_frames

    @property
    def shape(self) -> tuple:
        """``(H, W, C)`` of one frame."""
        c = 4 if self._seq.has_alpha else 3
        return (self._seq.height, self._seq.width, c)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype("u1" if self._seq.depth <= 8 else "u2")

    @property
    def duration(self) -> float:
        """Playback length in seconds; 0.0 for a still."""
        return self._seq.duration

    def frame(self, index: int) -> np.ndarray:
        with self._lock:
            if self._seq is None:
                raise ValueError("AVIF reader is closed")
            return self._seq.frame(int(index), numthreads=native_workers(self._numthreads))

    __getitem__ = frame

    def iter_frames(self):
        for i in range(self.n_frames):
            yield self.frame(i)

    def read(self) -> np.ndarray:
        """Every frame stacked, or the single image for a still.

        A still returns (H, W, C) rather than (1, H, W, C), matching
        what decode() has always returned for the same file.
        """
        if self.n_frames == 1:
            return self.frame(0)
        return np.stack([self.frame(i) for i in range(self.n_frames)])

    def close(self) -> None:
        with self._lock:
            self._seq = None
            if self._source is not None:
                self._source.close()

    def __del__(self):
        if hasattr(self, "_lock"):
            self.close()

    def __enter__(self) -> "AvifReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()


__all__ = ["AvifCodec", "AvifReader"]
