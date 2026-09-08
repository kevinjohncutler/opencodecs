"""MozJPEG's pyramid reader.

The implementation is shared with the jpeg backend -- see
:mod:`opencodecs._jpeg_pyramid`. This module exists so the class is
importable under the name of the codec it belongs to, and so the
capability manifest attributes the pyramid to `mozjpeg` rather than
inferring it from a filename that merely contains "jpeg".
"""

from __future__ import annotations

from ._jpeg_pyramid import MozjpegPyramidReader

__all__ = ["MozjpegPyramidReader"]
