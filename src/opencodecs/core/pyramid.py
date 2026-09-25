"""Pyramid (multi-resolution) reader abstraction.

A pyramid is an N-level stack of progressively-downscaled copies of
the same image. Level 0 is full resolution; level N-1 is the most
zoomed-out overview. Streaming workflows (web map tiles, COG viewers,
napari lazy loaders) need to:

  * Pick the right level for a given screen / output size — read
    a 100×100 overview instead of the full 100,000×100,000 image.
  * Read a region (bbox) from a specific level — fetch only the
    tiles that overlap.

This module is the shared abstraction. Each container/codec implements
its own pyramid discovery:

  * COG / OME-TIFF — :class:`opencodecs.TiffPyramidReader` walks the
    IFD chain, groups by SubfileType, sorts by area.
  * NDTiff (Pycro-Manager) — :class:`opencodecs.NDTiffPyramidDataset`
    (follow-up) walks the nested folder layout
    (``Full resolution/``, ``Downsampled_x2/``, …).
  * OME-Zarr v0.4+ multiscales — follow-up.

All backends produce a :class:`PyramidReader` with the same surface.
The :meth:`read_region` algorithm is implemented once here on top of
a per-format level-reader.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from .codec import Reader


# ---------------------------------------------------------------------------
# Level descriptor
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PyramidLevel:
    """One level of a pyramid.

    ``reader`` is whatever object the backend exposes for that level's
    pixel access — typically a :class:`Reader` subclass for the
    full-image-at-this-resolution. The pyramid's region API doesn't
    depend on it directly; it uses the level's tile/strip layout.

    Attributes
    ----------
    reader :
        Backend-specific accessor for this level's pixels. Has at
        minimum ``shape``, ``dtype``, and methods for reading the
        full level or a region of it. Concrete types: ``TiffPage``
        for COG, ``NDTiffDataset`` slice for NDTiff pyramid.
    downscale :
        ``(y_factor, x_factor)`` — how much smaller this level is
        than level 0. For typical 2× pyramids these are powers of 2.
    shape :
        Pixel shape at this level (``(h, w)`` or ``(h, w, c)``).
    dtype :
        Pixel dtype.
    """

    reader: Any
    downscale: tuple[int, int]
    shape: tuple[int, ...]
    dtype: np.dtype


# ---------------------------------------------------------------------------
# Pyramid ABC
# ---------------------------------------------------------------------------


class PyramidReader(ABC):
    """Multi-resolution reader. Subclasses fill in ``levels`` and the
    per-format ``_read_tile`` / ``_read_strip`` hooks; the rest of the
    public API is implemented here."""

    @property
    @abstractmethod
    def levels(self) -> list[PyramidLevel]:
        """All pyramid levels, level 0 (full res) first."""

    # ---- Convenience properties -----

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def __len__(self) -> int:
        return self.n_levels

    def level(self, n: int) -> PyramidLevel:
        if n < 0:
            n += self.n_levels
        return self.levels[n]

    @property
    def downscale_factors(self) -> tuple[tuple[int, int], ...]:
        return tuple(L.downscale for L in self.levels)

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(L.shape for L in self.levels)

    @property
    def dtype(self) -> np.dtype:
        return self.levels[0].dtype

    # ---- Level selection -----

    def best_level_for(
        self,
        max_pixels_y: int | None = None,
        max_pixels_x: int | None = None,
    ) -> int:
        """Pick the highest-resolution level whose ``(h, w)`` fits
        inside the requested envelope.

        For zoom-out / overview rendering, set the envelope to the
        viewport (or a few × the viewport so resampling has source
        pixels to work with). Returns the level index.

        With both axes left as ``None``, returns 0 (full resolution).
        """
        if max_pixels_y is None and max_pixels_x is None:
            return 0
        best = 0
        for i, L in enumerate(self.levels):
            h, w = L.shape[0], L.shape[1]
            if max_pixels_y is not None and h > max_pixels_y:
                continue
            if max_pixels_x is not None and w > max_pixels_x:
                continue
            best = i   # this level fits; remember and keep looking
            # No break: keep going to find the *highest-res* fit. Levels
            # are largest-first so once one fits, every later level
            # also fits but at lower resolution — that's not what we want.
            break
        return best

    # ---- Region read (shared algorithm) -----

    def read_region(
        self,
        level: int = 0,
        *,
        y: slice | tuple[int, int] | None = None,
        x: slice | tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Read a (y, x) bbox from the chosen pyramid level.

        Only the tiles/strips overlapping the bbox are read — this is
        the entire point of using a pyramid with a tile-based reader.
        With an HTTP-backed data source this means O(tiles in bbox)
        Range requests, not O(whole level).

        ``y`` and ``x`` accept either Python ``slice`` objects or
        ``(start, stop)`` tuples. ``None`` means "the whole axis".
        """
        L = self.levels[level]
        full_h, full_w = L.shape[0], L.shape[1]
        y0, y1 = _normalize_axis(y, full_h)
        x0, x1 = _normalize_axis(x, full_w)
        return self._read_region(L, y0, y1, x0, x1)

    @abstractmethod
    def _read_region(
        self,
        level: PyramidLevel,
        y0: int, y1: int,
        x0: int, x1: int,
    ) -> np.ndarray:
        """Backend hook: read the (y0:y1, x0:x1) region from level.

        Each backend implements this to fetch only the underlying
        storage units (TIFF tiles, NDTiff frames, etc.) that intersect
        the bbox and assemble them into the output array.
        """

    # ---- Batched regions (shared planner) -----
    #
    # A backend opts in by implementing the four hooks below. Tile ids are
    # backend-specific but must sort in the backend's composition order,
    # because the planner decodes the union of a batch's tiles once, in
    # sorted order, and pastes each into every output it intersects; that
    # is what keeps per-box results identical to read_region's.

    def _region_tiles(self, level, y0, y1, x0, x1):
        """Ids of the storage units covering the box, or NotImplementedError."""
        raise NotImplementedError

    def _decode_region_tiles(self, level, tile_ids):
        """Yield ``(tile_id, decoded)`` for ``tile_ids`` in the given order."""
        raise NotImplementedError

    def _paste_region_tile(self, level, tile_id, tile, out, y0, y1, x0, x1):
        """Place one decoded tile's intersection with the box into ``out``."""
        raise NotImplementedError

    def _region_tile_bytes(self, level, tile_id) -> int:
        """Decoded size of one tile, for batch budgeting. Zero if unknown."""
        return 0

    def _new_region_output(self, level, y0, y1, x0, x1) -> np.ndarray:
        """Zero-filled output for a box; uncovered pixels stay zero."""
        return np.zeros((y1 - y0, x1 - x0) + tuple(level.shape[2:]),
                        dtype=level.dtype)

    def iter_regions(self, level: int = 0, boxes=(), *,
                     max_batch_bytes: int = 64 << 20):
        """Yield one array per box, in request order, sharing tile work.

        ``boxes`` is an iterable of ``(y, x)`` pairs in the same forms
        ``read_region`` accepts. Boxes are grouped into batches bounded by
        ``max_batch_bytes`` of outputs plus newly required decoded tiles;
        within a batch every required tile is decoded once, however many
        boxes need it. Adjacent or overlapping requests therefore cost the
        union of their tiles; disjoint requests cost the same tiles as
        separate reads plus a small planning pass. Results are yielded
        batch by batch, so closing the iterator early never retains
        requests that were not reached. A backend without the batching
        hooks is served by ``read_region`` per box.

        ``batch_stats`` afterwards holds boxes, requested tiles (with
        repeats), unique tiles decoded, batches and fallback boxes.
        """
        import math

        L = self.levels[level]
        full_h, full_w = L.shape[0], L.shape[1]
        stats = {"boxes": 0, "requested_tiles": 0, "unique_tiles": 0,
                 "batches": 0, "fallback_boxes": 0}
        self.batch_stats = stats
        itemsize = np.dtype(L.dtype).itemsize
        extra = math.prod(L.shape[2:]) if len(L.shape) > 2 else 1

        pending: list = []          # (y0, y1, x0, x1, tile ids)
        pending_tiles: set = set()
        pending_bytes = 0

        def flush():
            stats["batches"] += 1
            outs = [self._new_region_output(L, y0, y1, x0, x1)
                    for y0, y1, x0, x1, _ in pending]
            owners: dict = {}
            for k, (_, _, _, _, tiles) in enumerate(pending):
                for t in tiles:
                    owners.setdefault(t, []).append(k)
            order = sorted(owners)
            stats["unique_tiles"] += len(order)
            for t, tile in self._decode_region_tiles(L, order):
                for k in owners[t]:
                    y0, y1, x0, x1, _ = pending[k]
                    self._paste_region_tile(L, t, tile, outs[k], y0, y1, x0, x1)
            return outs

        for y, x in boxes:
            y0, y1 = _normalize_axis(y, full_h)
            x0, x1 = _normalize_axis(x, full_w)
            stats["boxes"] += 1
            try:
                tiles = list(self._region_tiles(L, y0, y1, x0, x1))
            except NotImplementedError:
                if pending:
                    yield from flush()
                    pending.clear()
                    pending_tiles.clear()
                    pending_bytes = 0
                stats["fallback_boxes"] += 1
                yield self._read_region(L, y0, y1, x0, x1)
                continue
            stats["requested_tiles"] += len(tiles)
            cost = (y1 - y0) * (x1 - x0) * extra * itemsize
            cost += sum(self._region_tile_bytes(L, t)
                        for t in tiles if t not in pending_tiles)
            if pending and pending_bytes + cost > max_batch_bytes:
                yield from flush()
                pending.clear()
                pending_tiles.clear()
                pending_bytes = 0
            pending.append((y0, y1, x0, x1, tiles))
            pending_tiles.update(tiles)
            pending_bytes += cost
        if pending:
            yield from flush()

    def read_regions(self, level: int = 0, boxes=(), *,
                     max_batch_bytes: int = 64 << 20) -> list:
        """``list(iter_regions(...))``: every box's array, in request order."""
        return list(self.iter_regions(level, boxes, max_batch_bytes=max_batch_bytes))

    # ---- Iteration -----

    def iter_levels(self) -> Iterator[PyramidLevel]:
        return iter(self.levels)

    def __iter__(self) -> Iterator[PyramidLevel]:
        return self.iter_levels()

    # ---- Lifecycle -----

    def close(self) -> None:  # pragma: no cover - subclass override
        pass

    def __enter__(self) -> "PyramidReader":
        return self

    def __exit__(self, *_) -> bool:
        self.close()
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_axis(
    s: slice | tuple[int, int] | None,
    full: int,
) -> tuple[int, int]:
    """Normalize a slice / tuple / None into ``(start, stop)`` ints
    clipped to ``[0, full]``."""
    if s is None:
        return 0, full
    if isinstance(s, slice):
        start = 0 if s.start is None else int(s.start)
        stop = full if s.stop is None else int(s.stop)
    else:
        start, stop = int(s[0]), int(s[1])
    if start < 0:
        start += full
    if stop < 0:
        stop += full
    start = max(0, min(start, full))
    stop = max(start, min(stop, full))
    return start, stop


__all__ = ["PyramidReader", "PyramidLevel"]
