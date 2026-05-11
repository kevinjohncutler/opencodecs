"""NDTiffPyramidDataset — pyramid view of Pycro-Manager NDTiff acquisitions.

ndstorage stores pyramidal NDTiff datasets as N sibling folders inside
a parent directory:

    /Acquisition/
        Full resolution/
            NDTiff.index
            NDTiffStack.tif
            NDTiffStack_1.tif
            ...
        Downsampled_x2/
            NDTiff.index
            NDTiffStack.tif
        Downsampled_x4/
            ...

Each subfolder is a complete NDTiff acquisition at its own resolution
level. This reader walks the folder list, opens an ``NDTiffDataset``
per level, and exposes them through the :class:`PyramidReader` ABC.

Unlike the TIFF pyramid where each level is one IFD's worth of
tiles, NDTiff "levels" are full datasets containing many frames per
axes-tuple. So the region API here works *per-frame at the chosen
level* — i.e. ``pyramid.read_region(level, y, x, z=0)`` picks the
frame at the given axes and crops to ``(y, x)``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ._ndtiff import NDTiffDataset
from .core.pyramid import PyramidLevel, PyramidReader


_FULL_RES_NAME = "Full resolution"
_DOWNSAMPLED_RE = re.compile(r"^Downsampled_x(\d+)$")


class NDTiffPyramidDataset(PyramidReader):
    """Pyramid view of an ndstorage NDTiffPyramidDataset folder.

    Examples
    --------
    Open a pyramidal lightsheet acquisition and render a low-res
    overview of a single z-slice::

        with NDTiffPyramidDataset("/path/to/Acquisition") as p:
            best = p.best_level_for(max_pixels_y=512)
            overview = p.read_region(best, z=42)
            # or crop a region from full resolution:
            crop = p.read_region(level=0, y=(1000, 1500), x=(2000, 2500), z=42)
    """

    def __init__(self, parent_path: str | Path):
        self._path = Path(parent_path)
        if not self._path.is_dir():
            raise FileNotFoundError(
                f"NDTiffPyramidDataset: not a directory: {self._path}"
            )
        # Find all level subdirs and their downscales.
        candidates: list[tuple[int, Path]] = []
        for child in sorted(self._path.iterdir()):
            if not child.is_dir():
                continue
            if child.name == _FULL_RES_NAME:
                candidates.append((1, child))
            else:
                m = _DOWNSAMPLED_RE.match(child.name)
                if m:
                    candidates.append((int(m.group(1)), child))
        if not candidates:
            raise FileNotFoundError(
                f"No NDTiff pyramid levels in {self._path}; expected "
                f"{_FULL_RES_NAME!r} and/or Downsampled_x* folders"
            )
        candidates.sort(key=lambda x: x[0])    # downscale ascending = level 0 first

        # Open each level's NDTiffDataset.
        self._datasets: list[NDTiffDataset] = []
        self._levels: list[PyramidLevel] = []
        for downscale, folder in candidates:
            ds = NDTiffDataset(folder)
            self._datasets.append(ds)
            # Frame shape at this level. NDTiff stores (h, w) per frame.
            shape = ds.shape
            self._levels.append(PyramidLevel(
                reader=ds,
                downscale=(downscale, downscale),
                shape=shape,
                dtype=ds.dtype,
            ))

    # ----- ABC contract -----

    @property
    def levels(self) -> list[PyramidLevel]:
        return self._levels

    def close(self) -> None:
        for ds in self._datasets:
            try:
                ds.close()
            except Exception:  # pragma: no cover - defensive
                pass

    # ----- Region read at a given level + axes -----

    def read_frame(self, level: int = 0, /, **axes) -> np.ndarray:
        """Read a full frame at ``level`` keyed by ``axes`` (z=, c=, ...).

        Equivalent to ``self.level(level).reader.read_frame(**axes)``.
        """
        return self.levels[level].reader.read_frame(**axes)

    def read_region(
        self,
        level: int = 0,
        *,
        y: slice | tuple[int, int] | None = None,
        x: slice | tuple[int, int] | None = None,
        **axes,
    ) -> np.ndarray:
        """Crop a region from the frame at ``level`` keyed by ``axes``.

        Unlike :class:`TiffPyramidReader.read_region` which fetches
        only the overlapping tiles, NDTiff frames are atomic — we
        always fetch the full frame at the chosen level, then crop
        in memory. The pyramid still pays off: a low-res overview
        level is far smaller than the full-res frame.
        """
        from .core.pyramid import _normalize_axis
        L = self.levels[level]
        full_h, full_w = L.shape[0], L.shape[1]
        y0, y1 = _normalize_axis(y, full_h)
        x0, x1 = _normalize_axis(x, full_w)
        frame = L.reader.read_frame(**axes)
        return frame[y0:y1, x0:x1]

    # ----- The ABC requires _read_region; we don't use it (we override
    # read_region above to thread axes through). Provide a stub that
    # raises if called directly. -----

    def _read_region(self, level, y0, y1, x0, x1):
        raise NotImplementedError(
            "NDTiffPyramidDataset.read_region needs axes kwargs (z=, c=, …); "
            "call read_region(level, y=, x=, **axes) directly."
        )


__all__ = ["NDTiffPyramidDataset"]
