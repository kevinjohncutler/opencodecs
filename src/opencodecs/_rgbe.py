"""Radiance HDR (RGBE) top-level helpers: ``opencodecs.rgbe_encode`` etc.

These used to be a separate pure-Python implementation that disagreed
with the RGBE codec: it wrote an extra ``GAMMA=1.0`` header line (not a
Radiance header variable), and its reader ignored the resolution line's
orientation, so a ``+Y`` (bottom-to-top) or ``-X`` file decoded
unflipped and an X-first (transposed) file was rejected. The Radiance
file format makes the resolution string define the scan order, so they
now call the one implementation, the C-backed codec in
:mod:`opencodecs.codecs._rgbe`, and produce exactly its bytes and
pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .core._optional_backend import import_or_stubs

_encode, _decode, RgbeError, _HAVE_BACKEND = import_or_stubs(
    "opencodecs.codecs._rgbe", "encode", "decode", "RgbeError",
)
if not isinstance(RgbeError, type):  # pragma: no cover - extension missing
    class RgbeError(RuntimeError):
        """Raised on malformed RGBE input."""


def decode(data, *, header=None, rle=None, out=None) -> np.ndarray:
    """Decode RGBE bytes to an ``(H, W, 3)`` float32 array.

    The output is the linear-RGB color value at each pixel, oriented as
    the resolution line states. ``header=False`` reads a bare pixel
    stream into ``out``, as imagecodecs ``rgbe_decode`` does. A ``#?``
    header without a ``FORMAT`` line is read, as Radiance reads it.
    """
    return _decode(data, header=header, rle=rle, out=out)


def encode(rgb, *, header=None, rle=None) -> bytes:
    """Encode an ``(H, W, 3)`` float array as RGBE / Radiance HDR.

    By default a ``#?RADIANCE`` header and run-length-encoded scanlines,
    byte for byte what imagecodecs ``rgbe_encode`` writes.
    ``header=False`` writes only the pixels; ``rle=False`` writes flat
    RGBE quadruples, also after a header, where imagecodecs ignores
    ``rle=False`` and writes RLE (so those bytes differ from it).
    """
    return _encode(np.asarray(rgb, dtype=np.float32), header=header, rle=rle)


def imread(path: str | Path) -> np.ndarray:
    """Read an .hdr file as a float32 (H, W, 3) array."""
    return decode(Path(path).read_bytes())


def imwrite(path: str | Path, rgb: np.ndarray) -> None:
    """Write a float32 (H, W, 3) array as a Radiance .hdr file."""
    Path(path).write_bytes(encode(rgb))


__all__ = ["encode", "decode", "imread", "imwrite", "RgbeError"]
