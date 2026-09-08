"""Pyramid over a single baseline JPEG.

libjpeg-turbo can run a smaller inverse DCT and emit the image at a
fraction of its stored size. The supported ratios come from the
library, and the useful ones for a pyramid are the powers of two --
1/1, 1/2, 1/4, 1/8 -- because a viewer picking a level wants halving
steps, not the sixteen ratios libjpeg-turbo also offers.

What this saves is worth stating plainly, because it is not what the
wavelet codecs save. The Huffman pass still reads every coefficient in
the bitstream; only the inverse DCT and the chroma upsampling shrink.
Measured on 2048x2048 RGB, a 1/8 decode runs 1.3x faster than full on
noisy content and 2.0x on photo-like content -- not the ~90x that
:class:`opencodecs.Jpeg2kPyramidReader` gets. The real win is the
output: 192 KB instead of 12 MB, and 64x fewer pixels for whatever
comes next. Use it for thumbnails and overviews, not as a way to make
JPEG decoding fast.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._scaled_pyramid import ScaledCodestreamPyramid, probe_by_reduction

# Halving steps only. libjpeg-turbo supports ratios like 7/8 and 11/8,
# but a pyramid whose levels are not related by a clean factor makes
# coordinate mapping between levels awkward for no benefit.
_POWER_OF_TWO_DENOMS = (1, 2, 4, 8)


class _JpegFamilyPyramid(ScaledCodestreamPyramid):
    """Shared body for the jpeg and mozjpeg pyramid readers."""

    #: Set by subclasses to the module providing decode/ scaling factors.
    _module_name = ""

    def __init__(self, data: Any, *, max_levels: int | None = None):
        super().__init__(data, max_levels=max_levels)
        self._denoms: tuple[int, ...] | None = None

    def _mod(self):
        from importlib import import_module
        return import_module(f".codecs.{self._module_name}", __package__)

    def denominators(self) -> tuple[int, ...]:
        """The 1/N factors this build supports, coarsest last.

        Asked of the library rather than assumed: libjpeg-turbo's set
        has changed across versions, and MozJPEG tracks its own. Cached
        because both probing and decoding need it, and computing it in
        each -- as an earlier version did -- is how the two drift into
        indexing different lists for the same level.
        """
        if self._denoms is None:
            supported = set(self._mod().supported_scaling_factors())
            self._denoms = tuple(
                d for d in _POWER_OF_TWO_DENOMS if (1, d) in supported)
        return self._denoms

    def _probe_levels(self):
        mod = self._mod()
        denoms = self.denominators()
        if not denoms:
            return []

        def info(i):
            # This index is into `denoms`, not a subband count, which is
            # why a level's `reader` field is opaque to callers: each
            # backend decides what it means and only ever hands it back
            # to its own decoder.
            a = mod.decode(self._data, scale=(1, denoms[i]))
            return a.shape, a.dtype

        levels = probe_by_reduction(info, max_reduction=len(denoms) - 1)
        if self._max_levels is not None:
            levels = levels[: self._max_levels]
        return levels

    def _decode_level(self, index: int) -> np.ndarray:
        return self._mod().decode(
            self._data, scale=(1, self.denominators()[index]))


class JpegPyramidReader(_JpegFamilyPyramid):
    """Multi-resolution view of one baseline JPEG, via DCT scaling."""

    codec_name = "jpeg"
    _module_name = "_jpeg"

