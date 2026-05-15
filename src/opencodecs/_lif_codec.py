"""LIF Reader/Codec wrapping the ``readlif`` Python package.

Leica LIF (Leica Image File) is the proprietary container produced by
Leica LAS-X confocal / multi-photon software. A single LIF holds many
images, each with its own (X, Y, Z, T, M=mosaic) shape and channel
layout. ``readlif`` is the canonical pure-Python parser; we wrap it
the same way Nd2Codec / HdfCodec wrap their respective libraries.

The opencodecs ``decode(path)`` returns the **primary image** (index
0). Multi-image navigation is via ``LifReader``: ``r.n_images``,
``r.image(name_or_idx)``, ``r.iter_frames()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .core.codec import Codec, Reader

try:
    from readlif.reader import LifFile as _LifFile
    _HAVE_READLIF = True
except ImportError:  # pragma: no cover - readlif-missing branch
    _HAVE_READLIF = False


class LifError(RuntimeError):
    """Raised on LIF open / decode failures."""


def _image_to_array(lif_image) -> np.ndarray:
    """Materialize a readlif.LifImage as an ndarray. readlif's
    `as_array()` returns the (M?, Z, T, C, Y, X) ndarray when
    available, but raises on heterogeneous mosaic layouts. Fall back
    to assembling per-plane frames in that case."""
    if hasattr(lif_image, "as_array"):
        try:
            return np.asarray(lif_image.as_array())
        except (ValueError, TypeError):
            # readlif's as_array() builds a nested list then np.array()s
            # it — fails when mosaic tiles have inhomogeneous shapes.
            # Stitch per-plane manually instead.
            pass
    # Per-plane stitch. Returns (M, Z, T, C, Y, X) squeezed.
    d = lif_image.dims
    nm = max(1, getattr(lif_image, "n_mosaic", 1))
    nc = max(1, lif_image.channels)
    nz = max(1, d.z)
    nt = max(1, d.t)
    bd = lif_image.bit_depth[0] if isinstance(
        lif_image.bit_depth, (tuple, list)) else lif_image.bit_depth
    dtype = np.dtype(f"u{(bd + 7) // 8}")
    out = np.empty((nm, nz, nt, nc, d.y, d.x), dtype=dtype)
    for m in range(nm):
        for z in range(nz):
            for t in range(nt):
                for c in range(nc):
                    fr = lif_image.get_frame(z, t, c, m=m) \
                        if "m" in lif_image.get_frame.__code__.co_varnames \
                        else lif_image.get_frame(z, t, c)
                    out[m, z, t, c] = np.asarray(fr)
    # Squeeze singleton outer dims so a simple (Y, X) image stays (Y, X).
    return out.squeeze()


class LifReader(Reader):
    """Streaming Reader for Leica LIF files.

    Iterates the LIF's images in order; each image is one entry in
    ``iter_frames()`` (which means iteration unit is the IMAGE, not the
    plane within an image, matching how Leica LAS-X groups data).
    """

    def __init__(self, path: str | Path, image: int | str | None = None):
        if not _HAVE_READLIF:  # pragma: no cover - readlif-missing
            raise ImportError(
                "readlif is required for LIF support: pip install readlif")
        self._path = str(path)
        self._lif = _LifFile(self._path)
        self.n_images = int(self._lif.num_images)
        # Pick the requested image, or the first one.
        self._image_idx = self._resolve_image(image if image is not None else 0)
        self._image = self._lif.get_image(self._image_idx)
        self.shape = self._compute_shape(self._image)
        self.dtype = np.dtype(
            f"u{(self._image.bit_depth[0] + 7) // 8}"
        )
        self.n_frames = self.n_images
        self.is_chunked = False

    def _resolve_image(self, image: int | str) -> int:
        if isinstance(image, (int, np.integer)):
            return int(image)
        # Match by name across all images.
        for i, im in enumerate(self._lif.get_iter_image()):
            if im.name == image:
                return i
        raise KeyError(f"LIF: no image named {image!r}")

    @staticmethod
    def _compute_shape(im) -> tuple[int, ...]:
        """Compute (M?, T?, Z?, C?, Y, X) packed shape from LIF dims.
        Singletons are squeezed so a simple xy image is (Y, X)."""
        d = im.dims
        # Order matches readlif's as_array() output: (M, T, Z, C, Y, X)
        # Higher dims that equal 1 get squeezed out.
        full = (im.n_mosaic, d.t, d.z, im.channels, d.y, d.x)
        return tuple(s for s in full if s > 1)

    @property
    def image_names(self) -> list[str]:
        return [im.name for im in self._lif.get_iter_image()]

    def image(self, key: int | str) -> "LifReader":
        """Switch to a different image inside the same LIF."""
        self._image_idx = self._resolve_image(key)
        self._image = self._lif.get_image(self._image_idx)
        self.shape = self._compute_shape(self._image)
        self.dtype = np.dtype(
            f"u{(self._image.bit_depth[0] + 7) // 8}"
        )
        return self

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield each LIF image as one ndarray. The frame index thus
        navigates BETWEEN images; for plane-level iteration inside a
        single image, materialize with ``read()`` and slice."""
        for i in range(self.n_images):
            self.image(i)
            yield self.read()

    def read(self) -> np.ndarray:
        return _image_to_array(self._image)

    def close(self) -> None:
        # readlif.LifFile holds an mmap-like handle the gc will reap.
        self._lif = None

    def __enter__(self) -> "LifReader":
        return self

    def __exit__(self, *_) -> bool:
        self.close()
        return False


class LifCodec(Codec):
    """Leica LIF container codec — multi-image confocal microscopy."""

    name = "lif"
    file_extensions = (".lif",)
    aliases = ()

    has_native = False
    has_delegate = _HAVE_READLIF
    can_encode = False
    can_decode = _HAVE_READLIF
    multi_frame = True
    chunked = False
    streaming_decode = True
    parallel_decode = False

    supported_dtypes = (np.uint8, np.uint16)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        """LIF files begin with the 4-byte magic ``0x70 00 00 00``
        followed by an XML metadata header. The next bytes are
        ``\\x2A`` (ASCII *) and then the test string XML opening; we
        check the first byte tag because legacy and modern LIFs vary
        in their initial bytes after the magic."""
        return len(head) >= 8 and head[:4] == b"\x70\x00\x00\x00"

    def decode(self, src: Any, **opts) -> np.ndarray:
        with self.open(src, **opts) as reader:
            return reader.read()

    def open(self, src: Any, *, image: int | str | None = None,
             **opts) -> Reader:
        if not _HAVE_READLIF:
            raise ImportError(
                "readlif is required for LIF support: pip install readlif")
        if isinstance(src, (str, Path)):
            return LifReader(src, image=image)
        # readlif only takes paths; spill to a temp file otherwise.
        import os, tempfile
        if isinstance(src, (bytes, bytearray, memoryview)):
            fd, tmp = tempfile.mkstemp(suffix=".lif")
            os.write(fd, bytes(src))
            os.close(fd)
            return LifReader(tmp, image=image)
        if hasattr(src, "read"):
            data = src.read()
            fd, tmp = tempfile.mkstemp(suffix=".lif")
            os.write(fd, data)
            os.close(fd)
            return LifReader(tmp, image=image)
        raise TypeError(f"unsupported LIF source: {type(src).__name__}")


__all__ = ["LifCodec", "LifReader", "LifError"]
