"""Pyramid reader for tiled VSI / ETS whole slides.

A CellSens ``.vsi`` is an index; the pixels live in sibling
``_NAME_/stackN/frame_t*.ets`` files. Each of those holds ONE image,
and holds it as a full resolution pyramid: the tile table's last
coordinate axis is the level, and level L covers the image halved L
times, tiled at the file's stored tile size.

That last sentence is the part that was unknown, and it is derived
rather than assumed. :func:`opencodecs._ets.parse_ets` only reports a
pyramid when the tile population of every level matches the grid
covering the extent halved that many times. On the two tiled files
this was mapped from, that is 13 independent predictions and all 13
are exact; on the untiled corpus file the check fails and the reader
reports a single level. Guessing a pyramid that is not there is the
mistake this format has already caused once here.

Separate stacks are separate IMAGES, not levels of one. An ``.ets``
whose tile table has no valid level axis is a one-level pyramid, which
keeps a caller from having to ask which kind of file it opened.

No proprietary documentation and no GPL reader were consulted, in
keeping with :mod:`opencodecs._ets`; the layout above comes from the
files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._ets import EtsInfo, parse_ets
from .core.io import coerce_data_source
from .core.pyramid import PyramidLevel, PyramidReader


class VsiPyramidError(Exception):
    """Raised when a VSI/ETS source cannot be read as a pyramid."""


def _tile_decoder(payload: bytes):
    """Pick a decoder from the tile's own first bytes.

    The tiled files here store JPEG. Sniffing rather than hardcoding
    costs nothing and means a file storing something else fails with
    the format named instead of decoding noise.
    """
    from . import get_codec

    if payload[:2] == b"\xff\xd8":
        return get_codec("jpeg").decode
    if payload[:4] == b"\xff\x4f\xff\x51" or payload[4:8] == b"jP  ":
        return get_codec("jpeg2k").decode
    if payload[:2] == b"\x89P":
        return get_codec("png").decode
    return None


class VsiEtsLevel:
    """One resolution level of one .ets image."""

    def __init__(self, parent: "VsiPyramidReader", level: int):
        self._parent = parent
        self.level = level
        h, w = parent.info.level_shape(level)
        self._shape2d = (h, w)

    @property
    def shape(self) -> tuple[int, ...]:
        c = self._parent.n_channels
        return self._shape2d if c == 1 else (*self._shape2d, c)

    @property
    def dtype(self) -> np.dtype:
        return self._parent.dtype

    @property
    def n_tiles(self) -> int:
        return len(self._parent.info.tiles_at(self.level))

    def asarray(self) -> np.ndarray:
        """The whole level, assembled from its tiles."""
        h, w = self._shape2d
        return self._parent._read_region(
            self._parent.levels[self.level], 0, h, 0, w)

    def __repr__(self) -> str:
        h, w = self._shape2d
        return f"<VsiEtsLevel {self.level} {w}x{h} tiles={self.n_tiles}>"


class VsiPyramidReader(PyramidReader):
    """Multi-resolution reader over one tiled ``.ets``.

    Takes the ``.ets`` directly, or a ``.vsi`` whose companion tree
    holds exactly one -- with more than one, ``stack=`` picks, since
    separate stacks are separate images and choosing silently would be
    choosing which picture to return.
    """

    def __init__(self, src: Any, *, stack: str | int | None = None,
                 numthreads: int | None = None, max_pending_bytes: int | None = None):
        # A URL names one .ets directly: there is no directory listing
        # over HTTP to find a companion tree with, so the caller has
        # already had to say which stack they mean. Running it through
        # Path() would collapse "http://" to "http:/".
        if isinstance(src, str) and src.startswith(("http://", "https://")):
            self._path = Path(src.rsplit("/", 1)[-1])
            source: Any = src
        else:
            self._path = self._resolve(src, stack)
            source = str(self._path)
        self._ds, self._owns, _ = coerce_data_source(source)
        self.info: EtsInfo = parse_ets(self._ds)
        if not self.info.magic_ok:
            raise VsiPyramidError(f"{self._path}: not a SIS/ETS file")
        if not self.info.records:
            raise VsiPyramidError(
                f"{self._path}: the tile table is empty or unreadable")
        # (level, tile_x, tile_y) -> record
        self._index = {
            (r.level, r.tile_x, r.tile_y): r for r in self.info.records}
        self._probe = None
        self._numthreads = numthreads
        self._max_pending_bytes = max_pending_bytes
        self._cache: list[PyramidLevel] | None = None

    # -- construction ------------------------------------------------

    @staticmethod
    def _resolve(src: Any, stack: str | int | None) -> Path:
        p = Path(str(src))
        if p.suffix.lower() == ".ets":
            return p
        companion = p.parent / f"_{p.stem}_"
        if not companion.is_dir():
            raise VsiPyramidError(
                f"VSI: no companion tree at {companion}; a .vsi index on "
                f"its own holds only the thumbnail")
        found: list[tuple[str, Path]] = []
        for sub in sorted(companion.iterdir()):
            if sub.is_dir():
                for ets in sorted(sub.glob("frame_t*.ets")):
                    found.append((sub.name, ets))
        if not found:
            raise VsiPyramidError(f"VSI: no .ets files under {companion}")
        if stack is not None:
            want = str(stack)
            for name, ets in found:
                if name == want or name == f"stack{want}":
                    return ets
            raise VsiPyramidError(
                f"VSI: no stack {stack!r}; have "
                f"{sorted({n for n, _ in found})}")
        if len({n for n, _ in found}) > 1:
            raise VsiPyramidError(
                f"VSI: {len(found)} stacks here and they are separate "
                f"images rather than levels of one; pass stack= to choose "
                f"from {sorted({n for n, _ in found})}")
        return found[0][1]

    # -- pixels ------------------------------------------------------

    def _decode_tile(self, rec, payload=None) -> np.ndarray:
        # The probe tile is decoded once to learn channels and dtype,
        # and it is usually the first tile a caller asks for too, so
        # reading the top-left region used to decode it twice.
        if self._probe is not None and rec is self.info.records[0]:
            return self._probe
        if payload is None:
            payload = self._ds.read_at(rec.offset, rec.size)
        if len(payload) != rec.size:
            raise VsiPyramidError("VSI: truncated tile payload")
        decode = _tile_decoder(bytes(payload[:8]))
        if decode is None:
            raise VsiPyramidError(
                f"VSI: tile at {rec.offset} starts with "
                f"{bytes(payload[:4]).hex()}, which is not a codestream "
                f"this reader recognizes")
        return decode(payload)

    def _first_tile(self) -> np.ndarray:
        if self._probe is None:
            self._probe = self._decode_tile(self.info.records[0])
        return self._probe

    @property
    def n_channels(self) -> int:
        t = self._first_tile()
        return 1 if t.ndim == 2 else t.shape[2]

    @property
    def dtype(self) -> np.dtype:
        return self._first_tile().dtype

    # -- PyramidReader ----------------------------------------------

    @property
    def levels(self) -> list[PyramidLevel]:
        if self._cache is None:
            n = self.info.n_levels if self.info.pyramid_ok else 1
            out = []
            for i in range(n):
                lv = VsiEtsLevel(self, i)
                out.append(PyramidLevel(
                    reader=lv,
                    downscale=(1 << i, 1 << i),
                    shape=lv.shape,
                    dtype=self.dtype,
                ))
            self._cache = out
        return self._cache

    def _read_region(self, level: PyramidLevel, y0: int, y1: int,
                     x0: int, x1: int) -> np.ndarray:
        """Assemble a region from the tiles that intersect it.

        Only those tiles are read and decoded, which is the point of a
        tiled pyramid and what separates this from a reader that
        decodes a level and crops.
        """
        lvl = level.reader.level
        tw, th = self.info.tile_width, self.info.tile_height
        lh, lw = self.info.level_shape(lvl)
        y0, y1 = max(0, y0), min(lh, y1)
        x0, x1 = max(0, x0), min(lw, x1)
        if y1 <= y0 or x1 <= x0:
            raise VsiPyramidError(
                f"VSI: empty region y[{y0}:{y1}] x[{x0}:{x1}] at level {lvl}")

        c = self.n_channels
        shape = (y1 - y0, x1 - x0) if c == 1 else (y1 - y0, x1 - x0, c)
        out = np.zeros(shape, dtype=self.dtype)

        records = [self._index[(lvl, tx, ty)]
                   for ty in range(y0 // th, (y1 - 1) // th + 1)
                   for tx in range(x0 // tw, (x1 - 1) // tw + 1)
                   if (lvl, tx, ty) in self._index]

        def place(rec, payload=None):
            tile = self._decode_tile(rec) if payload is None else self._decode_tile(rec, payload)
            ty0, tx0 = rec.tile_y * th, rec.tile_x * tw
            sy0, sy1 = max(y0, ty0), min(y1, ty0 + tile.shape[0])
            sx0, sx1 = max(x0, tx0), min(x1, tx0 + tile.shape[1])
            if sy1 > sy0 and sx1 > sx0:
                out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                    tile[sy0 - ty0:sy1 - ty0, sx0 - tx0:sx1 - tx0]

        if self._max_pending_bytes is None:
            for rec in records:
                place(rec)
        else:
            from .core.parallel import resolve_workers
            from .core.pipeline import map_bounded, range_batches
            workers = resolve_workers(self._numthreads, len(records),
                                      output_bytes=out.nbytes)
            cost = lambda rec: rec.size + 2 * th * tw * c * self.dtype.itemsize
            batches = range_batches(records, cost,
                                    max_bytes=max(1, min(4 << 20, self._max_pending_bytes // (workers + 1))), max_items=32)
            def fetch(batch):
                ranges = [(rec.offset, rec.size) for rec in batch]
                blobs = self._ds.read_many(ranges)
                return tuple(zip(batch, blobs))
            def place_batch(batch):
                for rec, blob in batch:
                    place(rec, blob)
            from .core.io import BufferDataSource
            from ._tiff_http import FileDataSource, HTTPDataSource
            if isinstance(self._ds, (BufferDataSource, FileDataSource, HTTPDataSource)):
                # Built-in sources support concurrent offset reads. Keep the
                # generic caller-owned source on its producer thread.
                def fetch_place(batch):
                    place_batch(fetch(batch))
                task, prepare = fetch_place, None
            else:
                task, prepare = place_batch, fetch
            for _ in map_bounded(task, batches, workers, prepare=prepare,
                                 size=lambda batch: sum(cost(rec) for rec in batch),
                                 max_bytes=self._max_pending_bytes, name="vsi"):
                pass
        return out

    def close(self) -> None:
        if self._owns and self._ds is not None:
            self._ds.close()
        self._ds = None

    def __enter__(self) -> "VsiPyramidReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"<VsiPyramidReader {self._path.name} "
                f"levels={self.n_levels} shapes={self.shapes}>")


__all__ = ["VsiPyramidReader", "VsiEtsLevel", "VsiPyramidError"]
