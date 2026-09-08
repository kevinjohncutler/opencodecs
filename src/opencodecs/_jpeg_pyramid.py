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

    def _mod(self):
        from importlib import import_module
        return import_module(f".codecs.{self._module_name}", __package__)

    def _probe_levels(self):
        mod = self._mod()
        supported = set(mod.supported_scaling_factors())
        denoms = [d for d in _POWER_OF_TWO_DENOMS if (1, d) in supported]
        if 1 not in denoms:
            denoms = [1] + denoms

        def info(i):
            # `reduction` here indexes the denominator list rather than
            # being a subband count, which is why the level's `reader`
            # field is opaque to callers: each backend decides what it
            # means and only ever hands it back to its own decoder.
            d = denoms[i]
            a = mod.decode(self._data, scale=(1, d))
            return a.shape, a.dtype

        levels = probe_by_reduction(info, max_reduction=len(denoms) - 1)
        if self._max_levels is not None:
            levels = levels[: self._max_levels]
        return levels

    def _decode_level(self, index: int) -> np.ndarray:
        mod = self._mod()
        supported = set(mod.supported_scaling_factors())
        denoms = [d for d in _POWER_OF_TWO_DENOMS if (1, d) in supported]
        if 1 not in denoms:
            denoms = [1] + denoms
        return mod.decode(self._data, scale=(1, denoms[index]))


class JpegPyramidReader(_JpegFamilyPyramid):
    """Multi-resolution view of one baseline JPEG, via DCT scaling."""

    codec_name = "jpeg"
    _module_name = "_jpeg"


class MozjpegPyramidReader(_JpegFamilyPyramid):
    """Multi-resolution view of one JPEG, decoded through MozJPEG.

    MozJPEG ships only TurboJPEG v2, which infers the scaling factor
    from the destination size rather than taking it directly; the
    difference is entirely inside ``_mozjpeg.decode`` and the levels
    here are identical to :class:`JpegPyramidReader`'s.
    """

    codec_name = "mozjpeg"
    _module_name = "_mozjpeg"
