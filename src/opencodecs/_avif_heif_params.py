"""Encode-parameter rules shared by the AVIF and HEIF codecs.

Both formats code 8, 10 or 12 bits per sample from uint8 or uint16
input, and both take a quality ``level`` that imagecodecs gives a
lossless meaning at the top of its range. The two codecs used to make
these decisions separately and both got them wrong the same way: a
uint16 array defaulted to 10 bits with nothing checking the values, so
anything above 1023 was clamped by the library and came back changed,
and ``level=`` was dropped whenever ``lossless`` kept its default.

The rules here are the ones both codecs now follow:

* Bit depth. uint8 is 8 bits. For uint16 with no ``bit_depth`` the
  smallest of 10 and 12 that holds the largest value is used, and data
  that needs more than 12 bits raises, because AV1 (AOMedia AV1
  specification 5.5.2, color_config) stores at most 12 and the HEVC
  encoders behind libheif stop at 12 as well. An explicit
  ``bit_depth`` raises when a value does not fit, in lossy mode too:
  clamping loses range, which is not what a quality setting asks for.
* Lossless. ``lossless=None`` (the default) means lossless unless a
  ``level`` asks for less, using imagecodecs' thresholds: AVIF is
  lossless at level 100 (libavif's AVIF_QUALITY_LOSSLESS), HEIF only
  above 100 (imagecodecs' heif_encode). An explicit ``lossless=True``
  together with a lossy ``level`` is a contradiction and raises.
* Library default. imagecodecs maps an AVIF level of -1 or lower to
  libavif's AVIF_QUALITY_DEFAULT (-1), libavif's own default quality,
  and HEIF levels below 0 to quality 0; ``library_default_at`` carries
  that difference.
"""

from __future__ import annotations

import numpy as np


def resolve_bit_depth(arr: np.ndarray, bit_depth, *, name: str, error):
    """Return the coded bit depth for ``arr``, or raise ``error``.

    ``name`` is the format name used in messages ("AVIF", "HEIF").
    """
    if arr.dtype == np.uint8:
        if bit_depth is not None and int(bit_depth) != 8:
            raise error(
                f"{name}: uint8 input requires bit_depth=8 "
                f"(got {bit_depth})")
        return 8
    if arr.dtype != np.uint16:
        raise error(f"{name}: uint8 or uint16 input supported, "
                    f"got {arr.dtype}")
    top = int(arr.max()) if arr.size else 0
    if bit_depth is None:
        if top < 1024:
            return 10
        if top < 4096:
            return 12
        raise error(
            f"{name}: uint16 data with values up to {top} needs more than "
            f"12 bits, and {name} stores at most 12; rescale the data, or "
            f"use a codec with 16-bit support (jxl, png, jpeg2k)")
    depth = int(bit_depth)
    if depth not in (10, 12):
        raise error(
            f"{name}: uint16 input requires bit_depth 10 or 12 "
            f"(got {bit_depth})")
    if top >= (1 << depth):
        raise error(
            f"{name}: bit_depth={depth} holds values up to "
            f"{(1 << depth) - 1}, but the data reaches {top}; pass a "
            f"larger bit_depth (at most 12) or rescale the data")
    return depth


def resolve_lossless(level, lossless, *, name: str, lossless_from: int,
                     default_quality: int, library_default_at=None):
    """Return ``(lossless, quality)`` from the caller's ``level``/``lossless``.

    ``lossless_from`` is the lowest level that means lossless (100 for
    AVIF, 101 for HEIF). ``quality`` is the 0..100 lossy quality, or
    100 when lossless. When ``library_default_at`` is given, a level at
    or below it returns that value as the quality unchanged, the
    library's "use your default" sentinel (AVIF passes -1).
    """
    if lossless is None:
        lossless = level is None or int(level) >= lossless_from
    elif lossless and level is not None and int(level) < lossless_from:
        raise ValueError(
            f"{name} encode: lossless=True contradicts level={level}, "
            f"which asks for lossy output; pass one of them (a level "
            f"of {lossless_from} or more is lossless)")
    lossless = bool(lossless)
    if lossless:
        return True, 100
    quality = default_quality if level is None else int(level)
    if library_default_at is not None and quality <= library_default_at:
        return False, int(library_default_at)
    return False, max(0, min(100, quality))


def gray_layout(arr: np.ndarray):
    """``(samples, has_alpha)`` for an image array, or None if not one.

    (H, W) and (H, W, 1) are gray, (H, W, 2) gray plus alpha, (H, W, 3)
    RGB and (H, W, 4) RGBA, the same reading imagecodecs gives them.
    """
    if arr.ndim == 2:
        return 1, False
    if arr.ndim == 3 and arr.shape[2] in (1, 2, 3, 4):
        samples = int(arr.shape[2])
        return samples, samples in (2, 4)
    return None
