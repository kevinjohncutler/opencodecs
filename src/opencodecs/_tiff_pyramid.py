"""TiffPyramidReader — pyramid access on top of native TIFF.

Wraps a :class:`TiffStream` and treats each IFD as a pyramid level.
Levels are sorted largest-area-first so level 0 is full resolution.
Reduced-resolution overviews are identified by tag 254 (NewSubfileType)
bit 0; pages with that bit set are pure overviews, the others are the
"main" pages. Most COG and OME-TIFF pyramids fit this pattern.

The :meth:`PyramidReader.read_region` algorithm runs unchanged — this
class fills in :meth:`_read_region` to fetch only the tiles intersecting
the bbox. Combined with :class:`HTTPDataSource`, the same read_region
call streams a region from a remote COG with O(tiles-in-bbox) HTTP
Range requests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._tiff_codec import TiffStream, TiffPage, CMP_NONE
from .core.parallel import resolve_workers, run_batched
from .core.pyramid import PyramidLevel, PyramidReader


# NewSubfileType (TIFF tag 254):
#   bit 0 (0x1) = reduced-resolution overview of another image
#   bit 1 (0x2) = single page of a multi-page image
#   bit 2 (0x4) = transparency mask
#   bit 3 (0x8) = Aperio's "non-pyramid auxiliary" marker (label / macro
#                  / thumbnail). Not in the TIFF 6 spec but de-facto
#                  standard from SVS files.
_TAG_NEW_SUBFILE_TYPE = 254
_NSFT_REDUCED = 0x1
_NSFT_MASK = 0x4
_NSFT_APERIO_AUX = 0x8


def _nsft(page: TiffPage) -> int:
    """Read TIFF NewSubfileType (tag 254) for a page, returning 0 when absent."""
    raw = page.tags.get(_TAG_NEW_SUBFILE_TYPE)
    if raw is None:
        return 0
    # tags dict stores (dtype, count, value)
    v = raw[2]
    if isinstance(v, (tuple, list)):
        return int(v[0]) if v else 0
    return int(v)


class TiffPyramidReader(PyramidReader):
    """Pyramid view of a multi-IFD TIFF (COG / pyramidal OME-TIFF).

    Examples
    --------
    Open a remote COG and render a 1024-pixel-tall overview::

        from opencodecs._tiff_http import HTTPDataSource
        from opencodecs._tiff_pyramid import TiffPyramidReader

        src = HTTPDataSource("https://bucket/big.tif")
        with TiffPyramidReader(src) as p:
            level = p.best_level_for(max_pixels_y=1024)
            overview = p.read_region(level)

    Or do a tile-aware crop from the full-resolution level::

        crop = p.read_region(level=0, y=(2000, 4000), x=(8000, 10000))
        # — fetches only the COG tiles overlapping the 2000×2000 bbox
    """

    def __init__(
        self,
        src: Any,
        *,
        read_at=None,
        ifd_index: int | None = None,
        num_decode_workers: int | None = None,
        prefetch: bool | None = None,
        max_buffer_bytes: int = 64 << 20,
    ):
        """Open a TIFF and discover its pyramid structure.

        Parameters
        ----------
        src
            Same as :class:`TiffStream`: path, bytes, file-like, or a
            ``read_at`` callable like :class:`HTTPDataSource`.
        ifd_index : int or None
            How to discover pyramid levels.

            * ``None`` (default) — auto. If the first IFD has SubIFDs
              (tag 330, bioformats / OME-TIFF convention), use that IFD
              + its SubIFD chain as the pyramid. Otherwise fall back to
              grouping all top-level IFDs by area (the COG convention).
            * an int ``n`` — anchor on top-level IFD ``n``. Pyramid is
              ``[n]`` + that IFD's SubIFDs. Useful for multi-series
              OME-TIFFs where each top-level IFD is a different scene
              (T/C/Z plane), each with its own SubIFD pyramid.
        num_decode_workers : int or None
            Thread-pool size for parallel tile decoding inside
            :meth:`read_region`. Compressed TIFF tile decoders (JPEG,
            JPEG-2000, deflate, zstd) release the GIL during the C
            call, so threads scale decode across cores. The default uses
            the shared worker budget based on segment count and output size.
            Pass 1 for serial decode. Raw copy-only reads stay serial.
        prefetch : bool or None
            Overlap bounded range batches with decode. The default enables
            this for remote sources and preserves the local mapped fast path.
        max_buffer_bytes : int
            Reservation budget for queued compressed input and decoder scratch.
            A single oversized segment runs alone; output and source caches
            have separate lifetimes and are not included in this budget.
        """
        # Pass-through to TiffStream — it accepts paths, bytes,
        # file-likes, or a custom read_at callable (HTTPDataSource).
        self._stream = TiffStream(src, read_at=read_at) if read_at is not None \
            else TiffStream(src)
        self._ifd_index = ifd_index
        self._num_decode_workers = num_decode_workers
        if max_buffer_bytes < 1:
            raise ValueError("max_buffer_bytes must be positive")
        self._max_buffer_bytes = int(max_buffer_bytes)
        source = getattr(self._stream._read, "__self__", self._stream._read)
        if prefetch is None:
            from ._tiff_http import HTTPDataSource
            prefetch = isinstance(source, HTTPDataSource)
        self._prefetch = bool(prefetch)
        self._levels = self._build_levels()

    # ----- ABC contract -----

    @property
    def levels(self) -> list[PyramidLevel]:
        return self._levels

    def close(self) -> None:
        self._stream.close()

    # ----- Build pyramid levels from the IFD chain -----

    def _build_levels(self) -> list[PyramidLevel]:
        """Discover pyramid levels.

        Layout precedence:

        1. If ``ifd_index`` was given explicitly, the pyramid is the
           chosen top-level IFD followed by its SubIFDs.
        2. Otherwise, if the first IFD has SubIFDs (tag 330), assume
           bioformats / OME-TIFF layout: pyramid = IFD 0 + its SubIFDs.
        3. Otherwise (COG convention), pyramid = every top-level IFD,
           sorted by area descending.
        """
        n = self._stream.n_frames
        if n == 0:
            raise ValueError("TiffPyramidReader: no IFDs in TIFF")

        if self._ifd_index is not None:
            anchor = self._stream.page(self._ifd_index)
            pages = [anchor] + list(anchor.subifds)
        else:
            ifd0 = self._stream.page(0)
            if ifd0.subifds:
                # bioformats / OME-TIFF SubIFD-based pyramid layout
                pages = [ifd0] + list(ifd0.subifds)
            else:
                # COG / Aperio-style: walk top-level IFDs and pick the
                # ones that are pyramid levels. Page 0 is always
                # "level 0". Subsequent pages count only if they're
                # tagged as reduced-resolution overviews — that filters
                # out separate "main" images like SVS thumbnails (which
                # carry NewSubfileType=0) and Aperio macro/label pages
                # (NewSubfileType bit 3).
                main = self._stream.page(0)
                pages = [main]
                for i in range(1, n):
                    p = self._stream.page(i)
                    nsft = _nsft(p)
                    if nsft & _NSFT_APERIO_AUX:
                        continue   # Aperio label / macro
                    if nsft & _NSFT_MASK:
                        continue   # transparency mask
                    if not (nsft & _NSFT_REDUCED):
                        # not flagged as reduced — could be a separate
                        # image (SVS thumbnail). Skip.
                        continue
                    pages.append(p)

        # Largest-first. With the COG convention (full-res at IFD 0,
        # overviews after) this matches the natural order; with the
        # OME-TIFF SubIFDs convention sub-IFDs come after the main
        # page so sorting still produces the right order.
        pages.sort(key=lambda p: -(p.width * p.height))

        full_h = pages[0].height
        full_w = pages[0].width
        out = []
        for p in pages:
            # Downscale factor relative to level 0. We round here:
            # if the level is exactly N× smaller, downscale=N; if the
            # encoder used floor-rounding (common), we get the integer
            # ratio. Float ratios are reported as int via integer
            # division — the caller can compute precise scales from
            # full_shape / level.shape if needed.
            ds_y = max(1, full_h // p.height) if p.height else 1
            ds_x = max(1, full_w // p.width)  if p.width  else 1
            out.append(PyramidLevel(
                reader=p,
                downscale=(ds_y, ds_x),
                shape=p.shape,
                dtype=p.dtype,
            ))
        return out

    # ----- Region read -----

    def _read_region(
        self,
        level: PyramidLevel,
        y0: int, y1: int,
        x0: int, x1: int,
    ) -> np.ndarray:
        """Fetch the tiles/strips overlapping (y0:y1, x0:x1) and assemble."""
        page: TiffPage = level.reader
        out_h = y1 - y0
        out_w = x1 - x0
        if page.samples_per_pixel == 1:
            out_shape = (out_h, out_w)
        else:
            out_shape = (out_h, out_w, page.samples_per_pixel)
        if out_h == 0 or out_w == 0:
            return np.empty(out_shape, dtype=page.dtype)
        out = np.empty(out_shape, dtype=page.dtype)

        self._fill_segments(page, out, y0, y1, x0, x1)
        return out

    def _fill_segments(self, page, out, y0, y1, x0, x1):
        """Fetch intersecting segments, then decode directly into output.

        Tiles and strips share placement, prediction, and planar-channel
        handling. Workers own disjoint output rectangles. Decoded pixels
        live only for one segment per worker instead of a whole region.
        Batched source reads preserve range coalescing before decode.
        """
        tw, th = page.tile_width, page.tile_height
        ty_start, ty_stop = y0 // th, min((y1 + th - 1) // th, page.tiles_y)
        tx_start, tx_stop = x0 // tw, min((x1 + tw - 1) // tw, page.tiles_x)
        n_planes = page.samples_per_pixel if page.planar_config == 2 else 1
        per_plane = len(page.offsets) // n_planes
        coords = []
        ranges = []
        for plane in range(n_planes):
            for ty in range(ty_start, ty_stop):
                for tx in range(tx_start, tx_stop):
                    idx = plane * per_plane + ty * page.tiles_x + tx
                    coords.append((plane, ty, tx))
                    ranges.append((int(page.offsets[idx]), int(page.byte_counts[idx])))
        if not ranges:
            return
        source = getattr(self._stream._read, "__self__", self._stream._read)
        read_many = getattr(source, "read_many", None)
        workers = resolve_workers(
            self._num_decode_workers, len(coords), output_bytes=out.nbytes,
            max_workers=8,
            has_decode_work=page.compression != CMP_NONE or page.predictor != 1)
        blobs = None
        if not self._prefetch:
            blobs = (read_many(ranges) if read_many is not None and len(ranges) > 1
                     else [self._stream._read(o, n) for o, n in ranges])

        def place(i, raw=None):
            plane, ty, tx = coords[i]
            tile = page._decode_segment_pixels(blobs[i] if raw is None else raw, tx, ty)
            tile_y0, tile_x0 = ty * th, tx * tw
            in_y0, in_x0 = max(y0 - tile_y0, 0), max(x0 - tile_x0, 0)
            in_y1 = min(y1 - tile_y0, tile.shape[0])
            in_x1 = min(x1 - tile_x0, tile.shape[1])
            out_y0, out_x0 = tile_y0 + in_y0 - y0, tile_x0 + in_x0 - x0
            target = out[..., plane] if n_planes > 1 else out
            target[out_y0:out_y0 + in_y1 - in_y0,
                   out_x0:out_x0 + in_x1 - in_x0] = tile[in_y0:in_y1, in_x0:in_x1]

        if blobs is not None:
            run_batched(place, range(len(coords)), workers, name="tiff-region")
            return

        from contextlib import closing
        from .core.pipeline import map_bounded, range_batches
        segment_pixels = int(np.prod(page._padded_shape())) * page.dtype.itemsize
        # A worker decodes one tile at a time within its fetched batch, so
        # reserve one scratch tile, plus all retained compressed payloads.
        scratch_bytes = 3 * segment_pixels
        batches = range_batches(range(len(coords)), lambda i: ranges[i][1],
                                max_bytes=max(1, self._max_buffer_bytes // workers - scratch_bytes),
                                max_items=max(1, min(64, (len(coords) + workers - 1) // workers)))

        def fetch(batch):
            selected = [ranges[i] for i in batch]
            payloads = (read_many(selected) if read_many is not None and len(batch) > 1
                        else [self._stream._read(o, n) for o, n in selected])
            return batch, payloads

        def decode_batch(batch):
            batch, payloads = fetch(batch)
            for i, raw in zip(batch, payloads):
                place(i, raw)

        with closing(map_bounded(
                decode_batch, batches, workers,
                size=lambda batch: sum(ranges[i][1] for i in batch) + scratch_bytes,
                max_bytes=self._max_buffer_bytes, name="tiff-fetch-decode")) as results:
            for _ in results:
                pass



__all__ = ["TiffPyramidReader"]
