"""Pyramids from a single codestream that decodes at reduced size.

The pyramid readers elsewhere in this package walk a container that
*stores* several resolutions: a COG's IFD chain, CZI's own pyramid,
Imaris's resolution groups. JPEG 2000, HTJ2K and JPEG store one image,
but all three can decode a smaller one directly out of it -- the
wavelet codecs by skipping their finest subbands, JPEG by running a
smaller inverse DCT. No downscale happens afterwards; the discarded
detail is never reconstructed.

That makes a pyramid without a pyramid on disk, and it is worth
having for the same reason as the stored kind: a viewer that wants a
1024-pixel-wide overview of a 20000-pixel image should not pay for the
20000-pixel decode. What it costs varies enormously by codec, and the
honest numbers are worth stating because they decide whether a caller
should reach for this at all (measured on a 2048x2048 RGB image):

    jpeg2k   reduce=3 -> 256x256 in 2.3 ms, against 211 ms full         ~90x
    htj2k    reduce=3 -> 256x256 in 1.4 ms, against  97 ms full         ~70x
    jpeg     scale=1/8 -> 256x256, 1.3x faster on noise, 2.0x on a
             photo-like image

JPEG is the outlier and the reason is structural: DCT scaling shrinks
the inverse DCT and the chroma upsampling, but the Huffman pass still
reads every coefficient in the bitstream, and on busy content that is
most of the decode. The win there is really the output -- a 1/8 decode
allocates 192 KB instead of 12 MB and hands downstream code 64x fewer
pixels -- not the decode time. The wavelet codecs skip the entropy
coding of the fine subbands too, which is why they scale properly.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from .core.pyramid import PyramidLevel, PyramidReader


class ScaledCodestreamPyramid(PyramidReader):
    """Pyramid over one codestream, one level per supported reduction.

    Subclasses supply ``_probe_levels`` (the shapes, from headers only)
    and ``_decode_level`` (pixels for one level). Everything else --
    level bookkeeping, region reads, the decoded-level cache -- is here
    because it is identical for every codec that can do this.
    """

    #: Human-readable codec name, used in error messages and repr.
    codec_name = "scaled"

    def __init__(self, data: Any, *, max_levels: int | None = None):
        if not isinstance(data, (bytes, bytearray, memoryview)):
            data = bytes(data)
        self._data = bytes(data)
        self._levels: list[PyramidLevel] | None = None
        self._max_levels = max_levels
        # One decoded level at a time. read_region on the same level is
        # the overwhelmingly common access pattern (a viewer panning at
        # one zoom), and decoding the whole level per region call would
        # make that quadratic. Deliberately not an unbounded cache: the
        # levels below 0 are small, but level 0 is the full image and
        # holding several of those is how a viewer runs out of memory.
        self._cache_key: int | None = None
        self._cache: np.ndarray | None = None

    # ---- Subclass hooks -----

    def _probe_levels(self) -> list[tuple[int, tuple[int, ...], np.dtype]]:
        """Return ``(reduction, shape, dtype)`` per level, coarsest last.

        Must read headers only. The whole point of the class is that
        enumerating levels is cheap; a subclass that decodes here would
        make ``best_level_for`` cost a full decode.
        """
        raise NotImplementedError

    def _decode_level(self, reduction: int) -> np.ndarray:
        """Decode the image at ``reduction`` and return it."""
        raise NotImplementedError

    # ---- PyramidReader -----

    @property
    def levels(self) -> list[PyramidLevel]:
        if self._levels is None:
            probed = self._probe_levels()
            if not probed:
                raise ValueError(
                    f"{self.codec_name}: codestream exposes no levels")
            base_h, base_w = probed[0][1][0], probed[0][1][1]
            out = []
            for reduction, shape, dtype in probed:
                h, w = shape[0], shape[1]
                # Integer downscale factors, rounded from the actual
                # level shapes rather than assumed to be 2**reduction:
                # odd dimensions do not halve exactly, and a viewer
                # mapping coordinates between levels needs the real
                # ratio.
                out.append(PyramidLevel(
                    reader=reduction,
                    downscale=(max(1, round(base_h / h)),
                               max(1, round(base_w / w))),
                    shape=shape,
                    dtype=np.dtype(dtype),
                ))
            self._levels = out
        return self._levels

    def _read_region(self, level: PyramidLevel, y0: int, y1: int,
                     x0: int, x1: int) -> np.ndarray:
        """Decode this level, then slice.

        Unlike the tiled backends, this cannot fetch only the storage
        units overlapping the bbox: none of these codecs expose a
        tile-addressable decode through the bindings here. The saving
        is the resolution, not the region, and pretending otherwise in
        the docstring would be the sort of claim this package keeps
        getting wrong. The decoded level is cached so repeated region
        reads at one zoom stay cheap.
        """
        reduction = level.reader
        if self._cache_key != reduction or self._cache is None:
            self._cache = self._decode_level(reduction)
            self._cache_key = reduction
        return self._cache[y0:y1, x0:x1]

    def read_level(self, level: int = 0) -> np.ndarray:
        """The whole of one level, as an array."""
        return self._decode_level(self.levels[level].reader)

    def close(self) -> None:
        self._cache = None
        self._cache_key = None

    def __repr__(self) -> str:
        try:
            shapes = " ".join(str(tuple(L.shape)) for L in self.levels)
        except Exception:                                    # noqa: BLE001
            shapes = "<unprobed>"
        return f"<{type(self).__name__} {self.codec_name} levels: {shapes}>"


def probe_by_reduction(
    info: Callable[[int], tuple[tuple[int, ...], Any]],
    *,
    max_reduction: int,
    min_size: int = 1,
) -> list[tuple[int, tuple[int, ...], Any]]:
    """Walk reductions 0..max_reduction, stopping where they stop working.

    ``info(reduction)`` returns ``(shape, dtype)`` or raises. Raising is
    the normal way a codestream says "that is more reductions than I
    have", so it ends the walk rather than propagating -- except at
    reduction 0, where a failure means the data itself is bad and the
    caller needs to see it.
    """
    out: list[tuple[int, tuple[int, ...], Any]] = []
    for r in range(max_reduction + 1):
        try:
            shape, dtype = info(r)
        except Exception:                                    # noqa: BLE001
            if r == 0:
                raise
            break
        if shape[0] < min_size or shape[1] < min_size:
            break
        out.append((r, shape, dtype))
    return out


def levels_from_shapes(
    shapes: Sequence[tuple[int, ...]],
) -> list[tuple[int, int]]:
    """``(y, x)`` downscale factors for a list of level shapes."""
    bh, bw = shapes[0][0], shapes[0][1]
    return [(max(1, round(bh / s[0])), max(1, round(bw / s[1])))
            for s in shapes]
