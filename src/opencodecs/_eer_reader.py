"""EER (Electron Event Representation) file reader.

EER files are produced by Thermo Fisher Falcon 4 / Selectris X direct
electron detectors. They wrap many short-exposure event-list frames in
a standard TIFF container — each IFD is one frame, the strip payload
is the variable-length bitstream, and three private tags (65007/8/9)
carry the per-strip ``skipbits / horzbits / vertbits`` widths.

The TIFF container is handled by :class:`opencodecs.TiffStream` and the
bitstream by :mod:`opencodecs.codecs._eer` — both already shipped. This
module is the convenience layer that closes the loop for cryo-EM users:
frame-by-frame iteration, dose-correction-style accumulation across
ranges, and ``oc.open(path)`` extension dispatch via ``.eer``.

Typical use::

    with oc.open("scan.eer") as r:
        # one frame at a time
        for frame in r.iter_frames():
            ...
        # accumulate the whole acquisition (dose-corrected average)
        total = r.sum(dtype=np.uint32)

The reader is also exposed through ``EerCodec.open(path)`` so it shows
up in the codec registry alongside TIFF / CZI / OME-Zarr.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from .core.codec import Codec, Reader
from ._tiff_codec import TiffStream


class EerReader(Reader):
    """Frame-oriented reader for a multi-frame EER file.

    Thin wrapper around :class:`TiffStream` — each IFD becomes one
    frame. The wrapped TiffStream already handles the EER bitstream
    decode internally (compression tags 65000/65001/65002 + private
    tags 65007/8/9), so this class only needs to expose the
    frame-oriented API surface.
    """

    is_chunked = True

    def __init__(self, src: Any, *, read_at=None):
        # TiffStream accepts paths, bytes, file-likes, or a read_at
        # callable (HTTPDataSource etc.) — same surface as for any other
        # TIFF-based reader.
        self._stream = TiffStream(src, read_at=read_at)

    @property
    def n_frames(self) -> int:
        return self._stream.n_frames

    @property
    def shape(self) -> tuple:
        """``(H, W)`` of one frame. All frames share the same shape in
        Falcon 4 acquisitions, so we just report frame 0."""
        return self._stream.page(0).shape

    @property
    def dtype(self) -> np.dtype:
        return self._stream.page(0).dtype

    def frame(self, i: int) -> np.ndarray:
        """Decode frame ``i`` to a 2-D event-count image (uint8)."""
        return self._stream.page(i).asarray()

    def iter_frames(self) -> Iterator[np.ndarray]:
        for i in range(self.n_frames):
            yield self.frame(i)

    def sum(
        self,
        start: int = 0,
        stop: int | None = None,
        *,
        dtype: np.dtype | type = np.uint16,
    ) -> np.ndarray:
        """Accumulate frames ``[start, stop)`` into one count image.

        The cryo-EM "dose-corrected average" primitive: sum the
        per-event counts from many short exposures into a higher-SNR
        composite. The default uint16 accumulator handles up to ~65k
        events per pixel; pass ``dtype=np.uint32`` if you're summing
        a long acquisition where any pixel might exceed that.

        Frame-by-frame decode + in-place accumulation, so peak memory
        is one frame's worth, not n_frames × frame.
        """
        if stop is None:
            stop = self.n_frames
        if start < 0 or stop > self.n_frames or start >= stop:
            raise ValueError(
                f"EerReader.sum: invalid range [{start}, {stop}) "
                f"for {self.n_frames} frames"
            )

        out = np.zeros(self.shape, dtype=dtype)
        for i in range(start, stop):
            # Each .frame() returns a fresh uint8 array; add via
            # broadcasting into the accumulator dtype.
            np.add(out, self.frame(i), out=out, casting="unsafe")
        return out

    def close(self) -> None:
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class EerCodec(Codec):
    """Codec entry for EER files.

    Routes ``.eer`` files through :class:`EerReader`. EER's encode path
    isn't implemented — the format is detector-output-only; users who
    want to *write* event lists should use a different tool. We still
    expose the codec entry so format detection and the ``oc.open()``
    extension dispatch work.
    """

    name = "eer"
    file_extensions = (".eer",)
    aliases = ()

    has_native = True
    has_delegate = False
    can_encode = False
    can_decode = True
    multi_frame = True
    chunked = True
    streaming_decode = True
    parallel_decode = False

    supported_dtypes = (np.uint8, np.uint16)
    supports_color = False

    def signature(self, head: bytes) -> bool:
        # EER files are TIFF-on-the-wire — same II*\0 / MM\0* magic.
        # Format detection by extension is more reliable than by header
        # since signature() can't distinguish EER from a generic TIFF.
        return False

    def decode(self, src: Any, *, frame: int = 0, **opts) -> np.ndarray:
        with self.open(src, **opts) as r:
            return r.frame(frame)

    def open(self, src: Any, **opts) -> EerReader:
        return EerReader(src, **opts)

    def encode(self, data: Any, **opts) -> bytes | None:
        raise NotImplementedError(
            "EER is a detector-output-only format; opencodecs doesn't "
            "ship an encoder. Convert to TIFF / MRC instead."
        )


__all__ = ["EerCodec", "EerReader"]
