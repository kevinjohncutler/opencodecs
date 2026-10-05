"""AvifCodec — Codec adapter wrapping the native _avif extension."""

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
from .core.native_source import NativeSource
from .backends import select as _select_backend

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
               lossless: bool | None = None, speed: int | None = None,
               color=None, bit_depth: int | None = None,
               numthreads: int | None = None,
               iccprofile: bytes | None = None,
               codec: str | None = None,
               tile_cols_log2: int | None = None,
               tile_rows_log2: int | None = None,
               auto_tiling: bool = False,
               yuv_format: str | None = None,
               codec_options: dict | None = None,
               primaries: int | None = None,
               transfer: int | None = None,
               matrix: int | None = None,
               bitspersample: int | None = None,
               pixelformat: str | None = None,
               tilelog2: tuple | None = None,
               backend: str | None = None,
               **opts) -> bytes | None:
        """Encode an array as AVIF.

        ``level`` and ``lossless`` follow imagecodecs.avif_encode: with
        no level, or level=100, the file is lossless; a lower level is
        lossy at that quality, and a level of -1 or lower is lossy at
        libavif's own default quality. One difference: imagecodecs codes
        gray (1 or 2 sample) input lossless whatever the level, while
        here the level applies to gray as well. ``lossless=True`` with a
        lower level raises rather than ignoring one of them, and
        ``lossless=False`` with no level is lossy at quality 60. A bare
        ``encode(a)`` is lossless, so ``decode(encode(a))`` returns
        ``a``'s values. Gray is coded as 4:0:0 and decodes to (H, W) or
        (H, W, 2), so (H, W, 1) input comes back as (H, W), as in
        imagecodecs. Lossy color is coded
        4:4:4 unless ``yuv_format`` asks for subsampling, alpha is
        always lossless, as in imagecodecs. ``speed`` defaults to 6, as
        libavif's avifenc does, where imagecodecs keeps libavif's library
        default (libaom's slowest search): about 50x faster for about 10%
        larger files at equal perceived quality. Pass ``speed=0`` for
        imagecodecs' setting.

        uint16 data is coded at the smallest of 10 and 12 bits that
        holds it; data needing more than 12 bits raises, since AV1
        cannot store it. ``iccprofile`` embeds an ICC color profile.

        imagecodecs' keyword names are accepted as aliases:
        ``bitspersample`` (``bit_depth``), ``pixelformat``
        (``yuv_format``) and ``tilelog2`` (``(tile_cols_log2,
        tile_rows_log2)``). Any other unknown keyword raises TypeError.

        ``backend="nvvideocodec"`` (opt-in; NVIDIA GPU with
        PyNvVideoCodec and CuPy) encodes with NVENC's AV1 encoder:
        hundreds of times faster than aom, lossy only, 4:2:0, 8-bit
        (uint8) or 10-bit (uint16 up to 1023) RGB. It needs an explicit
        ``level`` (0 to 99), mapped to the same AV1 quantizer libavif
        gives aom for that level; ``speed`` picks the NVENC preset. Tile,
        codec and color options other than ``iccprofile`` raise.
        """
        if opts:
            raise TypeError(
                f"avif encode: unsupported option(s) {', '.join(sorted(opts))}")
        hardware = _select_backend(backend, self.name, "encode")
        bit_depth = _alias("bit_depth", bit_depth, "bitspersample", bitspersample)
        yuv_format = _alias("yuv_format", yuv_format, "pixelformat", pixelformat)
        if tilelog2 is not None:
            cols, rows = (int(v) for v in tilelog2)
            tile_cols_log2 = _alias("tile_cols_log2", tile_cols_log2,
                                    "tilelog2", cols)
            tile_rows_log2 = _alias("tile_rows_log2", tile_rows_log2,
                                    "tilelog2", rows)
        if hardware is not None:
            return _write_dest(hardware.encode_avif(
                data, level=level, lossless=lossless, speed=speed,
                bit_depth=bit_depth, yuv_format=yuv_format,
                iccprofile=iccprofile, color=color, codec=codec,
                tile_cols_log2=tile_cols_log2, tile_rows_log2=tile_rows_log2,
                auto_tiling=auto_tiling, codec_options=codec_options,
                primaries=primaries, transfer=transfer, matrix=matrix), dest)
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        encoded = _avif_encode(
            data, level=level, lossless=lossless, speed=speed,
            color=color, bit_depth=bit_depth, numthreads=native_workers(numthreads),
            iccprofile=iccprofile, codec=codec,
            tile_cols_log2=tile_cols_log2, tile_rows_log2=tile_rows_log2,
            auto_tiling=auto_tiling, yuv_format=yuv_format,
            codec_options=codec_options, primaries=primaries,
            transfer=transfer, matrix=matrix,
        )
        return _write_dest(encoded, dest)

    def decode(self, src: Any, *, numthreads: int | None = None,
               out=None, index: int | None = None,
               backend: str | None = None, **opts) -> np.ndarray:
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

        ``index`` (imagecodecs' keyword) decodes that one image of the
        file, 0 for a still; an index the file does not have raises
        IndexError. Negative values count from the end. Any other
        unknown keyword raises TypeError. ``backend`` takes None or
        ``"native"`` only (no hardware AVIF decoder is offered).
        """
        if opts:
            raise TypeError(
                f"avif decode: unsupported option(s) {', '.join(sorted(opts))}")
        # No hardware decoder is offered: this validates the name only.
        _select_backend(backend, self.name, "decode")
        data = _read_src(src)
        n_frames = _avif_frame_count(data)
        if index is not None:
            i = operator.index(index)
            if i < 0:
                i += n_frames
            if not 0 <= i < n_frames:
                raise IndexError(
                    f"avif decode: index {index} out of range for "
                    f"{n_frames} image(s)")
            if n_frames > 1:
                with AvifReader(data, numthreads=native_workers(numthreads)) as r:
                    frame = r.frame(i)
                if out is None:
                    return frame
                target = array_output(out)
                if (not isinstance(target, np.ndarray)
                        or target.shape != frame.shape
                        or target.dtype != frame.dtype):
                    raise ValueError(
                        f"avif decode: out= must be an ndarray of shape "
                        f"{frame.shape} and dtype {frame.dtype}")
                target[...] = frame
                return target
        if n_frames <= 1:
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



def _alias(name, value, alias, alias_value):
    """Merge an imagecodecs keyword into ours; raise if they disagree."""
    if alias_value is None:
        return value
    if value is not None and value != alias_value:
        raise ValueError(
            f"avif encode: {name}={value!r} and {alias}={alias_value!r} "
            f"disagree; pass one of them")
    return alias_value


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
        """``(H, W, C)`` of one frame, or ``(H, W)`` for monochrome."""
        if self._seq.monochrome:
            if self._seq.has_alpha:
                return (self._seq.height, self._seq.width, 2)
            return (self._seq.height, self._seq.width)
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
