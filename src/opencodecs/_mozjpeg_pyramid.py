"""Pyramid over one JPEG, decoded through MozJPEG.

The level logic is shared with :mod:`opencodecs._jpeg_pyramid`; what
differs is the decoder underneath. MozJPEG ships only TurboJPEG v2,
which has no SetScalingFactor and instead infers the scaling factor
from the destination size it is handed, so ``_mozjpeg.decode`` computes
the scaled extent itself. The pixels come out identical to the jpeg
codec at every supported ratio, and a test pins that -- two JPEG
decoders whose ``scale=`` meant subtly different things would be worse
than having only one.
"""

from __future__ import annotations

from ._jpeg_pyramid import _JpegFamilyPyramid


class MozjpegPyramidReader(_JpegFamilyPyramid):
    """Multi-resolution view of one JPEG, via MozJPEG's decoder."""

    codec_name = "mozjpeg"
    _module_name = "_mozjpeg"


__all__ = ["MozjpegPyramidReader"]
