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

from ._tiff_codec import TiffStream, TiffPage
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
        """
        # Pass-through to TiffStream — it accepts paths, bytes,
        # file-likes, or a custom read_at callable (HTTPDataSource).
        self._stream = TiffStream(src, read_at=read_at) if read_at is not None \
            else TiffStream(src)
        self._ifd_index = ifd_index
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

        if page.is_tiled:
            self._fill_tiles(page, out, y0, y1, x0, x1)
        else:
            self._fill_strips(page, out, y0, y1, x0, x1)
        return out

    # ----- Tiled path -----

    def _fill_tiles(
        self, page: TiffPage, out: np.ndarray,
        y0: int, y1: int, x0: int, x1: int,
    ) -> None:
        """Assemble out from the tiles of page that intersect (y0:y1, x0:x1).

        When the underlying data source advertises ``read_many`` (HTTP
        range-requests or pread parallelism), all overlapping tiles are
        fetched in one batched call before decode. Otherwise tiles are
        pulled serially. This is the perf path that makes
        ``read_region`` over an HTTP COG do O(1) round-trips per tile
        cluster instead of O(N)."""
        tw, th = page.tile_width, page.tile_height
        ty_start = y0 // th
        ty_stop = (y1 + th - 1) // th
        tx_start = x0 // tw
        tx_stop = (x1 + tw - 1) // tw
        ty_stop = min(ty_stop, page.tiles_y)
        tx_stop = min(tx_stop, page.tiles_x)

        # Padded-tile native shape — every tile read decodes to this,
        # then we crop into out.
        full_tile_shape = page._padded_shape()

        # Build the (offset, nbytes) list for every tile we need.
        ranges: list[tuple[int, int]] = []
        coords: list[tuple[int, int]] = []
        for ty in range(ty_start, ty_stop):
            for tx in range(tx_start, tx_stop):
                idx = ty * page.tiles_x + tx
                ranges.append(
                    (int(page.offsets[idx]), int(page.byte_counts[idx]))
                )
                coords.append((ty, tx))
        if not ranges:
            return

        # Coalesced fetch path: one round-trip / one parallel batch for
        # the whole bbox, instead of N serial reads. Falls back to per-
        # tile reads when the data source doesn't expose read_many
        # (e.g. raw file handle, bytes, BytesIO).
        read_many = getattr(self._stream._read, "read_many", None)
        if read_many is not None and len(ranges) > 1:
            blobs = read_many(ranges)
        else:
            blobs = [self._stream._read(o, n) for (o, n) in ranges]

        for (ty, tx), raw in zip(coords, blobs):
            decoded = page._decode_segment(raw)
            # Byte-stream codecs return flat; image codecs return shaped.
            if decoded.ndim == 1:
                tile = decoded.reshape(full_tile_shape)
            else:
                tile = decoded

            tile_y0 = ty * th
            tile_x0 = tx * tw
            # Intersect tile rect with the requested bbox.
            in_y0 = max(y0 - tile_y0, 0)
            in_y1 = min(y1 - tile_y0, th)
            in_x0 = max(x0 - tile_x0, 0)
            in_x1 = min(x1 - tile_x0, tw)
            out_y0 = tile_y0 + in_y0 - y0
            out_x0 = tile_x0 + in_x0 - x0
            out[out_y0:out_y0 + (in_y1 - in_y0),
                out_x0:out_x0 + (in_x1 - in_x0)] = \
                tile[in_y0:in_y1, in_x0:in_x1]

    # ----- Striped path -----

    def _fill_strips(
        self, page: TiffPage, out: np.ndarray,
        y0: int, y1: int, x0: int, x1: int,
    ) -> None:
        """Assemble out from the strips of page that intersect (y0:y1, x0:x1).

        Strips span the full image width, so x clipping happens after
        decode. Only the strips overlapping (y0:y1) get fetched. Like
        :meth:`_fill_tiles`, batches the network reads through
        ``read_many`` when the data source advertises it.
        """
        rps = page.tile_height   # rows per strip (filed under tile_height for strips)
        h = page.height
        s_start = y0 // rps
        s_stop = (y1 + rps - 1) // rps
        s_stop = min(s_stop, len(page.offsets))

        ranges = [
            (int(page.offsets[s]), int(page.byte_counts[s]))
            for s in range(s_start, s_stop)
        ]
        if not ranges:
            return

        read_many = getattr(self._stream._read, "read_many", None)
        if read_many is not None and len(ranges) > 1:
            blobs = read_many(ranges)
        else:
            blobs = [self._stream._read(o, n) for (o, n) in ranges]

        for s, raw in zip(range(s_start, s_stop), blobs):
            decoded = page._decode_segment(raw)
            strip_y0 = s * rps
            strip_h = min(rps, h - strip_y0)
            strip_shape = (strip_h, page.width) if page.samples_per_pixel == 1 \
                else (strip_h, page.width, page.samples_per_pixel)
            strip = decoded.reshape(strip_shape) if decoded.ndim == 1 else decoded

            in_y0 = max(y0 - strip_y0, 0)
            in_y1 = min(y1 - strip_y0, strip_h)
            out_y0 = strip_y0 + in_y0 - y0
            # Clip x slice (full strip width → bbox columns).
            out[out_y0:out_y0 + (in_y1 - in_y0)] = \
                strip[in_y0:in_y1, x0:x1]


__all__ = ["TiffPyramidReader"]
