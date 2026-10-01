"""HeifCodec — Codec adapter wrapping the native _heif extension."""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec, Reader
from .core.buffers import array_output, CallbackDestination
from .core._write_helpers import binary_destination
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs
from .core.pipeline import native_workers
from .core.native_source import NativeSource

(
    _heif_encode, _heif_decode, _heif_check_signature,
    _heif_read_icc, _heif_frame_count, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._heif",
    "encode", "decode", "check_signature", "read_icc_profile",
    "frame_count",
)


class HeifCodec(Codec):
    """Native HEIF/HEIC codec via libheif."""

    name = "heif"
    file_extensions = (".heif", ".heic")
    aliases = ("heic",)

    has_native = True
    has_delegate = False
    can_encode = True
    max_writer_frames = 1
    can_decode = True
    # A HEIF holds a SET of top-level images, one of which is primary.
    # A burst, a Live Photo's stills, a depth-plus-color capture all
    # put more than one there, and decoding only the primary dropped
    # the rest with nothing said.
    multi_frame = True
    streaming_decode = True
    # Images are addressed by item id out of the container's own list,
    # so reaching image N costs the metadata parse and image N, not
    # images 0..N-1.
    chunked = True
    # _heif.pyx does call heif_context_set_max_decoding_threads, but
    # decoding one untiled image measures 1.00x from one thread to
    # eight -- there is nothing to divide. Left False deliberately:
    # do not flip it back on the strength of the API call alone.
    parallel_decode = False

    supported_dtypes = (np.uint8, np.uint16)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        return _heif_check_signature(head)

    def encode(self, data: Any, *, dest=None, level: int | None = None,
               lossless: bool | None = None, color=None,
               bit_depth: int | None = None,
               numthreads: int | None = None,
               iccprofile: bytes | None = None,
               bitspersample: int | None = None,
               photometric=None, compression=None,
               **opts) -> bytes | None:
        """Encode an array as HEIF/HEIC.

        ``level`` and ``lossless`` follow imagecodecs.heif_encode: with
        no level, or a level above 100, the file is lossless; a level of
        0-100 is lossy at that quality. ``lossless=True`` with such a
        level raises rather than ignoring one of them, and
        ``lossless=False`` with no level is lossy at quality 50. A bare
        ``encode(a)`` is lossless, so ``decode(encode(a))`` returns
        ``a``. Gray input is coded monochrome; ``decode`` returns it as
        RGB, as imagecodecs does, and ``decode(...,
        photometric='monochrome')`` returns it with its original shape.

        uint16 data is coded at the smallest of 10 and 12 bits that
        holds it; data needing more than 12 bits raises instead of
        being clamped.

        ``iccprofile`` embeds an ICC color profile in the file's
        ``colr`` box (type ``prof``).

        imagecodecs' keywords: ``bitspersample`` is an alias of
        ``bit_depth``. ``compression`` may be None, 'hevc' or libheif's
        heif_compression_HEVC (1), the only encoder wired here; other
        values raise. ``photometric`` takes imagecodecs' values and must
        agree with the array: 'rgb', 'ycbcr' or libheif's colorspace 1
        or 0 for 3 or 4 samples; 'monochrome', 'minisblack' and the other
        gray names or colorspace 2 for 1 or 2; colorspace 99
        (undefined) for either. A disagreeing value raises, where
        imagecodecs would write a stack of gray images or ignore it. Any
        other unknown keyword raises TypeError.
        """
        if opts:
            raise TypeError(
                f"heif encode: unsupported option(s) {', '.join(sorted(opts))}")
        if bitspersample is not None:
            if bit_depth is not None and bit_depth != bitspersample:
                raise ValueError(
                    f"heif encode: bit_depth={bit_depth} and bitspersample="
                    f"{bitspersample} disagree; pass one of them")
            bit_depth = bitspersample
        if compression is not None and not _is_hevc(compression):
            raise ValueError(
                f"heif encode: compression={compression!r} is not "
                f"supported; only HEVC is")
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        if photometric is not None:
            _check_photometric(photometric, data)
        if dest is not None:
            with binary_destination(dest) as stream:
                return _heif_encode(data, level=level, lossless=lossless,
                    color=color, bit_depth=bit_depth,
                    numthreads=native_workers(numthreads), iccprofile=iccprofile,
                    destination=CallbackDestination(stream))
        encoded = _heif_encode(data, level=level, lossless=lossless,
                               color=color, bit_depth=bit_depth,
                               numthreads=native_workers(numthreads),
                               iccprofile=iccprofile)
        return _write_dest(encoded, dest)

    def decode(self, src: Any, *, numthreads: int | None = None,
               out=None, index=None, photometric=None,
               range_reads: bool = False, **opts) -> np.ndarray:
        """Decode one image; the primary one unless ``index`` is given.

        ``index`` counts top-level images in the order the container
        lists them, which is not necessarily primary-first.
        Ordinary full-image reads fetch encoded input once. ``range_reads``
        opts into bounded source callbacks; indexed and read_at sources use
        callbacks automatically.

        Output is RGB(A), monochrome images included, as in
        imagecodecs.heif_decode. ``photometric='monochrome'`` (imagecodecs'
        keyword) returns a monochrome image's gray plane instead, (H, W)
        or (H, W, 2) with alpha, and raises for a color image. Any other
        unknown keyword raises TypeError.
        """
        if opts:
            raise TypeError(
                f"heif decode: unsupported option(s) {', '.join(sorted(opts))}")
        kw = dict(numthreads=native_workers(numthreads),
                  out=out if out is None else array_output(out),
                  index=index, photometric=photometric)
        if isinstance(src, (bytes, bytearray, memoryview)):
            return _heif_decode(src, **kw)
        if index is None and not range_reads and not hasattr(src, "read_at"):
            return _heif_decode(_read_src(src), **kw)
        with NativeSource(src) as source:
            return _heif_decode(source, **kw)

    def frame_count(self, src: Any) -> int:
        """How many top-level images the file holds; 1 for a still.

        Metadata only: no image is decoded to answer it.
        """
        if isinstance(src, (bytes, bytearray, memoryview)):
            return _heif_frame_count(src)
        with NativeSource(src) as source:
            return _heif_frame_count(source)

    def open(self, src: Any, *, numthreads: int | None = None,
             photometric=None):
        """A reader over the top-level images; ``photometric`` as in decode."""
        if isinstance(src, (bytes, bytearray, memoryview)):
            return HeifReader(src, numthreads=numthreads,
                              photometric=photometric)
        source = NativeSource(src)
        try:
            return HeifReader(source, numthreads=numthreads,
                              photometric=photometric)
        except BaseException:
            source.close()
            raise

    def read_icc_profile(self, src: Any) -> bytes | None:
        """Return the embedded ICC profile bytes, or ``None`` if absent."""
        if isinstance(src, (bytes, bytearray, memoryview)):
            return _heif_read_icc(src)
        with NativeSource(src) as source:
            return _heif_read_icc(source)



# libheif's enum values, which imagecodecs accepts as integers.
_HEIF_COMPRESSION_HEVC = 1
_HEIF_COLORSPACE_YCBCR, _HEIF_COLORSPACE_RGB = 0, 1
_HEIF_COLORSPACE_MONOCHROME, _HEIF_COLORSPACE_UNDEFINED = 2, 99
_GRAY_NAMES = frozenset((
    "GRAY", "GREY", "BLACKISZERO", "MINISBLACK", "WHITEISZERO",
    "MINISWHITE", "MONOCHROME"))


def _enum_value(value):
    """``value`` as an int if it is an integer (not bool), else None."""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        return operator.index(value)
    except TypeError:
        return None


def _is_hevc(compression) -> bool:
    """True if ``compression`` names HEVC, by name or by libheif's enum."""
    if isinstance(compression, str):
        return compression.lower() in ("hevc", "h265", "heic")
    return _enum_value(compression) == _HEIF_COMPRESSION_HEVC


def _check_photometric(photometric, data) -> None:
    """Accept imagecodecs' ``photometric`` only where it matches ``data``.

    Gray input is always coded monochrome and color input as RGB, so the
    keyword can only confirm the layout. imagecodecs reads monochrome on
    a (H, W, 3) array as a stack of three gray images, which this
    encoder does not write, so a mismatch raises.
    """
    samples = 1 if data.ndim == 2 else int(data.shape[-1])
    gray = color = False
    if isinstance(photometric, str):
        name = photometric.upper()
        gray = name in _GRAY_NAMES
        color = name[:3] == "RGB" or name[:5] == "YCBCR"
    else:
        value = _enum_value(photometric)
        if value == _HEIF_COLORSPACE_UNDEFINED:
            return
        gray = value == _HEIF_COLORSPACE_MONOCHROME
        color = value in (_HEIF_COLORSPACE_YCBCR, _HEIF_COLORSPACE_RGB)
    if not (gray or color):
        raise ValueError(
            f"heif encode: photometric={photometric!r} is not supported")
    if gray != (samples <= 2):
        raise ValueError(
            f"heif encode: photometric={photometric!r} does not match "
            f"an array of shape {data.shape}")


class HeifReader(Reader):
    """Frame-oriented reader over a HEIF's top-level images.

    A single-image HEIF is a one-image set rather than a special case,
    so a caller that iterates need not first ask which kind of file it
    opened.

    Each image uses its own native context. Seekable sources are retained
    as bounded callbacks, so reading one item does not materialize the whole
    encoded container. Native decoded planes remain additional working memory.
    """

    is_chunked = True

    def __init__(self, data, *, numthreads: int | None = None,
                 photometric=None):
        import threading
        self._lock = threading.RLock()
        self._data = data
        self._numthreads = numthreads
        self._photometric = photometric
        self.closed = False
        self._n = _heif_frame_count(data)
        from .codecs._heif import decode_info
        info = decode_info(data, index=0, photometric=photometric)
        self._shape = info["shape"]
        self._dtype = info["dtype"]

    @property
    def n_frames(self) -> int:
        return self._n

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def frame(self, index: int) -> np.ndarray:
        with self._lock:
            if self.closed:
                raise ValueError("HEIF reader is closed")
            return _heif_decode(self._data, numthreads=native_workers(self._numthreads),
                                index=int(index), photometric=self._photometric)

    def close(self):
        with self._lock:
            if not self.closed:
                self.closed = True
                if isinstance(self._data, NativeSource):
                    self._data.close()

    def __del__(self):
        if hasattr(self, "closed"):
            self.close()

    __getitem__ = frame

    def iter_frames(self):
        for i in range(self._n):
            yield self.frame(i)

    def read(self) -> np.ndarray:
        """Every image stacked, or the single image for a plain still.

        Images in one HEIF need not share a shape (a depth map is
        smaller than its color image), so a stack is only attempted
        when they do; otherwise the frames come back as a list, which
        is the honest answer rather than a broadcast error.
        """
        if self._n == 1:
            return self.frame(0)
        frames = [self.frame(i) for i in range(self._n)]
        shapes = {f.shape for f in frames}
        if len(shapes) == 1:
            return np.stack(frames)
        return frames

    def __enter__(self) -> "HeifReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()


__all__ = ["HeifCodec", "HeifReader"]
