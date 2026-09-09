"""Pyramid reader for VL Whole Slide Microscopy DICOM.

A DICOM whole-slide image is not one file. It is a SERIES of instances
that all describe the same physical area at different pixel extents,
sharing a SeriesInstanceUID and an Imaged Volume, and distinguished by
Total Pixel Matrix Columns / Rows. That is the pyramid: the levels are
separate files, and nothing inside any one of them says it is level 2
of 5.

So this takes a directory or a list of instances rather than a path to
"the file", which is the part that makes it a reader of its own rather
than a switch on DicomFile. Levels are ordered by extent, largest
first, which is the definition rather than an assumption about naming:
the standard does not require the files to be named or ordered
usefully, and real scanners do not oblige.

Two things this deliberately does not do. It does not resample: a level
is returned at the resolution it is stored at, and PyramidReader's
region API picks the level. And it does not stitch tiles across
instances -- a tiled WSI instance holds its tiles as frames, which
DicomFile already indexes by the Basic Offset Table.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ._dicom import DicomError, DicomFile
from .core.pyramid import PyramidLevel, PyramidReader


class DicomWsiLevel:
    """One resolution level: one DICOM instance of the series."""

    def __init__(self, path: Any, file: DicomFile):
        self.path = path
        self._file = file
        matrix = file.total_pixel_matrix
        # A single-frame instance need not carry a total pixel matrix;
        # then the instance IS the level and Rows/Columns are it.
        self._matrix = matrix or (file.rows, file.columns)

    @property
    def matrix(self) -> tuple[int, int]:
        """``(rows, columns)`` this level covers."""
        return self._matrix

    @property
    def shape(self) -> tuple[int, ...]:
        spp = self._file.samples_per_pixel
        r, c = self._matrix
        return (r, c) if spp == 1 else (r, c, spp)

    @property
    def dtype(self) -> np.dtype:
        return self._file.dtype

    @property
    def n_frames(self) -> int:
        return self._file.n_frames

    def asarray(self, **kw) -> np.ndarray:
        """This level's pixels.

        A tiled instance returns its frames stacked; the tile grid
        geometry needed to stitch them into one plane lives in the
        Per-Frame Functional Groups, which this reader does not parse.
        Returning the frames is the honest answer, and DicomFile
        indexes them by offset.
        """
        return self._file.asarray(**kw)

    def frame(self, index: int) -> np.ndarray:
        return self._file.frame(index)

    def close(self) -> None:
        self._file.close()

    def __repr__(self) -> str:
        r, c = self._matrix
        return f"<DicomWsiLevel {r}x{c} frames={self.n_frames}>"


class DicomWsiPyramid(PyramidReader):
    """The resolution levels of one whole-slide DICOM series.

    Construct from a directory, an iterable of paths, or an iterable of
    already-open :class:`DicomFile`.
    """

    def __init__(self, src: Any, *, series: str | None = None):
        paths = list(self._resolve(src))
        if not paths:
            raise DicomError(
                f"dicom: no DICOM instances found in {src!r}")

        opened: list[tuple[Any, DicomFile]] = []
        for p in paths:
            try:
                f = p if isinstance(p, DicomFile) else DicomFile(str(p))
            except Exception:                            # noqa: BLE001
                continue        # not a DICOM file; a slide directory
                                # routinely has a DICOMDIR and stray files
            if not f.is_whole_slide:
                f.close()
                continue
            opened.append((p, f))

        if not opened:
            for _, f in opened:
                f.close()
            raise DicomError(
                "dicom: none of these instances are VL Whole Slide "
                "Microscopy (SOP class 1.2.840.10008.5.1.4.1.1.77.1.6); "
                "a pyramid needs a whole-slide series")

        # One slide per SeriesInstanceUID. A directory can hold more
        # than one, and silently mixing them would build a pyramid out
        # of two different slides.
        by_series: dict[str | None, list[tuple[Any, DicomFile]]] = {}
        for p, f in opened:
            by_series.setdefault(f.series_instance_uid, []).append((p, f))
        if series is not None:
            if series not in by_series:
                for _, f in opened:
                    f.close()
                raise DicomError(
                    f"dicom: no series {series!r} here; found "
                    f"{sorted(k for k in by_series if k)}")
            chosen = series
        elif len(by_series) > 1:
            for _, f in opened:
                f.close()
            raise DicomError(
                f"dicom: {len(by_series)} whole-slide series here; pass "
                f"series= to choose one of "
                f"{sorted(k for k in by_series if k)}")
        else:
            chosen = next(iter(by_series))

        for uid, group in by_series.items():
            if uid != chosen:
                for _, f in group:
                    f.close()

        self.series_instance_uid = chosen
        # Largest extent first. The standard does not order or name the
        # instances usefully and real scanners do not oblige, so the
        # extent IS the ordering rather than a guess from a filename.
        self._levels_src = sorted(
            (DicomWsiLevel(p, f) for p, f in by_series[chosen]),
            key=lambda L: (-L.matrix[0] * L.matrix[1], str(L.path)))
        self._cache: list[PyramidLevel] | None = None

    @staticmethod
    def _resolve(src: Any) -> Iterable[Any]:
        if isinstance(src, DicomFile):
            return [src]
        if isinstance(src, (str, os.PathLike)):
            p = Path(src)
            if p.is_dir():
                return sorted(q for q in p.iterdir() if q.is_file())
            return [p]
        return list(src)

    # -- PyramidReader ----------------------------------------------

    @property
    def levels(self) -> list[PyramidLevel]:
        if self._cache is None:
            base = self._levels_src[0].matrix
            out = []
            for L in self._levels_src:
                r, c = L.matrix
                # Integer downscale relative to level 0. Rounded rather
                # than floored: a 1024-wide slide's half level is 512,
                # but a 1023-wide one's is 512 too, and flooring would
                # call that a factor of 1.
                out.append(PyramidLevel(
                    reader=L,
                    downscale=(max(1, round(base[0] / r)) if r else 1,
                               max(1, round(base[1] / c)) if c else 1),
                    shape=L.shape,
                    dtype=L.dtype,
                ))
            self._cache = out
        return self._cache

    def _read_region(self, level: PyramidLevel, y0: int, y1: int,
                     x0: int, x1: int) -> np.ndarray:
        """Crop a region out of a level.

        The backends that fetch only the storage units intersecting a
        box do so because their unit is a tile with its own offset. A
        WSI instance is tiled too -- its tiles are frames, indexed by
        the Basic Offset Table -- but which tile covers which part of
        the slide lives in the Per-Frame Functional Groups, and this
        reader does not parse those. Reading the level and cropping is
        the honest implementation of that: correct, and no faster than
        it claims to be.

        The level selection above it is where the saving actually is,
        and that is real: asking for a 512-pixel view of a slide reads
        the 512-pixel level, not the full-resolution one.
        """
        arr = level.reader.asarray()
        if arr.ndim == 2:
            return arr[y0:y1, x0:x1]
        if arr.shape[:2] == level.shape[:2]:
            return arr[y0:y1, x0:x1]
        # A tiled instance returns frames rather than a plane; a region
        # of that is not defined without the tile grid.
        raise DicomError(
            "dicom: this level is stored as tiles and the tile grid is in "
            "the Per-Frame Functional Groups, which this reader does not "
            "parse; use level(n).reader.frame(i) to reach frames directly")

    @property
    def imaged_volume(self) -> tuple[float, float] | None:
        """``(height, width)`` in millimetres, from level 0."""
        return self._levels_src[0]._file.imaged_volume

    def close(self) -> None:
        for L in self._levels_src:
            L.close()

    def __enter__(self) -> "DicomWsiPyramid":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"<DicomWsiPyramid levels={self.n_levels} "
                f"shapes={self.shapes}>")


__all__ = ["DicomWsiPyramid", "DicomWsiLevel"]
