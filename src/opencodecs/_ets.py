"""SIS / ETS (Olympus Sequential Image Stream) parser.

Backs VsiCodec's full-resolution data path. ``frame_t*.ets`` files sit
alongside a top-level ``.vsi`` index and hold the pixels; each one is
ONE image, and a tiled one holds that image as a full pyramid. Both
header and tile-index sections are clean-room mapped from real files
(no proprietary docs, no GPL code read for this implementation).

Header (first 64 bytes of an .ets):
  @0..3   ASCII magic ``SIS\0``
  @4      u32 header_size (always 64 in observed files)
  @8      u32 version (3 in observed files)
  @12     u32 unknown (6)
  @16     u64 sub-header offset (=64)
  @24     u64 sub-header size (228 in observed files)
  @32     u64 tile-table offset
  @40     u64 tile-table RECORD COUNT  <- a count, not a byte size
  @48     u64 third chunk offset
  @56     u64 third chunk size

Sub-header at offset 64 (the "ETS\0" block):
  @0..3   ASCII magic ``ETS\0``
  @8      u32 component count
  @28     u32 stored TILE width
  @32     u32 stored TILE height
  @188    u32 full extent of level 0, width
  @192    u32 full extent of level 0, height

Tile table: ``count`` records, each self-describing and therefore
``4 * (ndims + 5)`` bytes ::

  u32      ndims          (4 in tiled files, 6 in the untiled one)
  u32[n]   coords         (tile_x, tile_y, ..., LEVEL)
  u64      offset
  u32      size
  u32      flag

The last coordinate is the pyramid level, and level L covers the
extent halved L times. That is checked rather than trusted: see
``_levels_check_out``, which requires the tile population of every
level to equal the grid covering that shrunken extent. It holds for
13 levels across the two tiled files mapped here, and fails for the
untiled corpus file, which is reported as a single level.

Two fields above were previously read wrongly, and together they made
a pyramidal file look flat. @40 was taken for a byte size, so a
413-record table was read as 413 bytes and the walk stopped after 11
entries. And records were assumed fixed-width at 44 bytes, which is
right only for the 6-axis untiled file and turns a 4-axis tiled one
into garbage. An earlier reading of this trailer as a pyramid level
count was also wrong, and a test pinned the wrong number; the check
above exists so that cannot happen quietly again.

Two decode paths, and they are not interchangeable. ``decode_ets`` /
``decode_ets_plane`` read uncompressed planes laid end to end from
offset 292, which is what the untiled files hold and what was verified
byte-identical to bftools. A tiled file stores JPEG tiles at the table
offsets instead, so those functions refuse it rather than return
noise; :mod:`opencodecs._vsi_pyramid` reads those.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .core.io import DataSource, coerce_data_source


_SIS_MAGIC = b"SIS\x00"
_ETS_MAGIC = b"ETS\x00"

# Trailing-table geometry, mapped from the OME corpus sample: a 28-byte
# preamble, then fixed-size records. Both are only as good as the files
# they were read from, which is why parse_ets stops at the first entry
# pointing outside the file rather than trusting the stride.
_ETS_TABLE_PREAMBLE = 28
_ETS_TABLE_RECORD = 44


@dataclass(frozen=True)
class EtsRecord:
    """One entry of the .ets trailing table: one stored tile.

    ``coords`` is the tile's position along the file's own axes. The
    number of axes is stated per record (see ``parse_ets``), and the
    LAST axis is the pyramid level: in a tiled file coords are
    ``(tile_x, tile_y, ..., level)``. ``offset`` and ``size`` locate
    the tile's bytes, which are a JPEG codestream in the tiled files
    seen here and a raw plane in the untiled one.
    """
    coords: tuple[int, ...]
    offset: int          # absolute byte offset of the tile
    size: int            # bytes of the stored tile
    flag: int            # meaning unknown; constant per file in some,
                         # a running count in others

    @property
    def level(self) -> int:
        """Pyramid level, the last coordinate. 0 is full resolution."""
        return self.coords[-1] if self.coords else 0

    @property
    def tile_x(self) -> int:
        return self.coords[0] if len(self.coords) > 0 else 0

    @property
    def tile_y(self) -> int:
        return self.coords[1] if len(self.coords) > 1 else 0


@dataclass
class EtsInfo:
    """Partial parse result for an .ets file."""
    file_size: int
    width: int                     # full extent of level 0, in pixels
    height: int
    tile_width: int                # one stored tile
    tile_height: int
    n_components: int
    n_dims: int                    # coordinate axes per record
    sub_chunk_offsets: list[int]   # 3 entries, last may be 0
    sub_chunk_sizes: list[int]     # [bytes, RECORD COUNT, bytes]
    records: tuple                 # EtsRecord entries from the trailer
    magic_ok: bool
    pyramid_ok: bool = False       # the level axis checks out (below)

    @property
    def n_records(self) -> int:
        return len(self.records)

    @property
    def n_levels(self) -> int:
        """Levels the table actually contains; 1 when there is no pyramid."""
        if not self.records:
            return 0
        return max(r.level for r in self.records) + 1

    def level_shape(self, level: int) -> tuple[int, int]:
        """``(height, width)`` at ``level``, halving per level."""
        return (-(-self.height // (1 << level)), -(-self.width // (1 << level)))

    def tiles_at(self, level: int) -> tuple:
        return tuple(r for r in self.records if r.level == level)

    @property
    def plane_stride(self) -> int:
        """Bytes between the blocks two consecutive records name."""
        if len(self.records) < 2:
            return 0
        return self.records[1].offset - self.records[0].offset


def _levels_check_out(info: "EtsInfo") -> bool:
    """Is the last coordinate really the pyramid level?

    Derived, then verified, rather than assumed. If the last axis is
    the level then the number of tiles at level L must be exactly the
    tile grid covering the extent halved L times, and that is a strong
    enough constraint that a wrong reading fails it immediately: on the
    two tiled files here it predicts 13 level populations, all exact.

    Returning False means "report one level and the whole image",
    which is what the untiled corpus file gets. Guessing a pyramid that
    is not there is the failure this module has already had once.
    """
    if not info.records or info.tile_width <= 0 or info.tile_height <= 0:
        return False
    if info.width <= 0 or info.height <= 0:
        return False
    counts: dict[int, int] = {}
    for r in info.records:
        counts[r.level] = counts.get(r.level, 0) + 1
    if sorted(counts) != list(range(len(counts))):
        return False            # levels must be 0..N with no holes
    if len(counts) < 2:
        return False            # one level is not a pyramid
    for level, got in counts.items():
        h, w = info.level_shape(level)
        want = (-(-w // info.tile_width)) * (-(-h // info.tile_height))
        if want != got:
            return False
    return True


def parse_ets(src: Any) -> EtsInfo:
    """Read just enough of an .ets to enumerate its structure.

    Accepts a path or a DataSource. HTTP-backed sources fetch the
    64-byte SIS header, the ETS sub-header, and the trailing tile
    table -- tens of kilobytes even for a 32 MB slide.

    Two things here were previously misread, and both are worth
    stating because they are what made a pyramidal file look flat.

    The SIS header's second chunk field is a RECORD COUNT, not a byte
    size. Reading it as bytes stopped the walk after a fraction of the
    table: a 413-record table read as 413 bytes yields 11 entries, and
    the first one that failed validation ended the loop.

    And a record is self-describing rather than fixed-width. It opens
    with the number of coordinate axes, so its length is
    ``4 * (ndims + 5)``: 44 bytes for the 6-axis untiled corpus file,
    36 for the 4-axis tiled ones. A fixed 44 read the tiled files as
    garbage, which is why they appeared to have no usable table.
    """
    ds, owns, file_size = coerce_data_source(src)
    try:
        hdr = ds.read_at(0, 64)
        if hdr[:4] != _SIS_MAGIC:
            return EtsInfo(
                file_size=file_size, width=0, height=0,
                tile_width=0, tile_height=0,
                n_components=0, n_dims=0, sub_chunk_offsets=[],
                sub_chunk_sizes=[], records=(), magic_ok=False,
            )
        ptr1 = struct.unpack_from("<Q", hdr, 16)[0]
        sz1  = struct.unpack_from("<Q", hdr, 24)[0]
        ptr2 = struct.unpack_from("<Q", hdr, 32)[0]
        n_rec = struct.unpack_from("<Q", hdr, 40)[0]
        ptr3 = struct.unpack_from("<Q", hdr, 48)[0]
        sz3  = struct.unpack_from("<Q", hdr, 56)[0]

        width = height = n_components = 0
        tile_width = tile_height = 0
        if sz1 >= 64:
            sub = ds.read_at(ptr1, min(sz1, 4096))
            if sub[:4] == _ETS_MAGIC:
                n_components = struct.unpack_from("<I", sub, 8)[0]
                # @28/@32 is the stored TILE size. On the untiled
                # corpus file it equals the image, which is why it was
                # taken for the image everywhere.
                tile_width  = struct.unpack_from("<I", sub, 28)[0]
                tile_height = struct.unpack_from("<I", sub, 32)[0]
                # @188/@192 is the full extent of level 0. On the
                # corpus file it reads 260x216, the same as the tile,
                # so nothing there could distinguish the two fields.
                if len(sub) >= 196:
                    width  = struct.unpack_from("<I", sub, 188)[0]
                    height = struct.unpack_from("<I", sub, 192)[0]
                if width <= 0 or height <= 0:
                    width, height = tile_width, tile_height

        records = []
        n_dims = 0
        if n_rec and ptr2 and ptr2 < file_size:
            head = ds.read_at(ptr2, 4)
            if len(head) == 4:
                n_dims = struct.unpack_from("<I", head, 0)[0]
                # A sane axis count. Anything else means the layout is
                # not the one this parser knows, and reading on would
                # invent a table.
                if 1 <= n_dims <= 16:
                    stride = 4 * (n_dims + 5)
                    want = n_rec * stride
                    table = ds.read_at(ptr2, min(want, file_size - ptr2))
                    for i in range(n_rec):
                        off = i * stride
                        if off + stride > len(table):
                            break
                        f = struct.unpack_from(
                            "<%dI" % (n_dims + 5), table, off)
                        # Every record restates the axis count. A
                        # record that disagrees means the stride is
                        # wrong, and stopping beats reinterpreting.
                        if f[0] != n_dims:
                            break
                        coords = tuple(f[1:1 + n_dims])
                        offset = f[1 + n_dims] | (f[2 + n_dims] << 32)
                        size = f[3 + n_dims]
                        if offset + size > file_size or size == 0:
                            break
                        records.append(EtsRecord(
                            coords=coords, offset=int(offset),
                            size=int(size), flag=int(f[4 + n_dims])))

        info = EtsInfo(
            file_size=file_size,
            width=width,
            height=height,
            tile_width=tile_width,
            tile_height=tile_height,
            n_components=n_components,
            n_dims=n_dims,
            sub_chunk_offsets=[ptr1, ptr2, ptr3],
            sub_chunk_sizes=[sz1, n_rec, sz3],
            records=tuple(records),
            magic_ok=True,
        )
        info.pyramid_ok = _levels_check_out(info)
        return info
    finally:
        if owns:
            ds.close()


def _refuse_if_tiled(info: EtsInfo) -> None:
    """Stop the contiguous-plane path from running on a tiled file.

    This path assumes uncompressed planes laid end to end from offset
    292, which is what the untiled files hold and what it was verified
    byte-identical to bftools on. A tiled file stores JPEG tiles at the
    offsets in its table instead, and reading it this way would return
    a plausible-looking array of noise rather than fail. Point the
    caller at the reader that can do it.
    """
    tiled = (info.tile_width and info.tile_height
             and (info.width > info.tile_width
                  or info.height > info.tile_height))
    if tiled or info.pyramid_ok:
        raise ValueError(
            f"ETS: this file is tiled ({info.width}x{info.height} in "
            f"{info.tile_width}x{info.tile_height} tiles"
            + (f", {info.n_levels} pyramid levels" if info.pyramid_ok else "")
            + "); the contiguous-plane path cannot read it. Use "
              "opencodecs._vsi_pyramid.VsiPyramidReader, or "
              "oc.open_pyramid(path, format='vsi').")


def decode_ets(src: Any) -> np.ndarray:
    """Decode an .ets source into a (planes, height, width) uint16 stack.

    Accepts a path or DataSource. The plane data is one contiguous
    sequential read from offset 292 through ``data_end`` (the
    trailing-index pointer in the SIS header). For an HTTP source
    this is one large range request rather than many small ones,
    efficient even at multi-GB file sizes.

    Verified byte-identical to bftools output on the OME
    zenodo-17590655 corpus sample.
    """
    ds, owns, file_size = coerce_data_source(src)
    try:
        info = parse_ets(ds)
        if not info.magic_ok:
            raise ValueError("not a SIS / ETS source")
        if info.width == 0 or info.height == 0:
            raise ValueError("ETS sub-header missing geometry")
        _refuse_if_tiled(info)
        plane_bytes = info.height * info.width * 2
        data_end = info.sub_chunk_offsets[1]
        data_start = 292   # header (64) + ETS sub-header (228)
        payload_bytes = data_end - data_start
        if payload_bytes % plane_bytes != 0:
            raise ValueError(
                f"ETS payload {payload_bytes} bytes isn't a "
                f"multiple of plane_bytes ({plane_bytes}); "
                f"cannot determine plane count")
        n_planes = payload_bytes // plane_bytes
        raw = ds.read_at(data_start, payload_bytes)
        return np.frombuffer(raw, dtype="<u2").reshape(
            n_planes, info.height, info.width)
    finally:
        if owns:
            ds.close()


def decode_ets_plane(src: Any, index: int) -> np.ndarray:
    """Decode just one plane from an .ets source. For HTTP sources
    this fetches only (height × width × 2) bytes rather than the
    whole stack."""
    ds, owns, _ = coerce_data_source(src)
    try:
        info = parse_ets(ds)
        if not info.magic_ok:
            raise ValueError("not a SIS / ETS source")
        _refuse_if_tiled(info)
        plane_bytes = info.height * info.width * 2
        data_start = 292
        offset = data_start + index * plane_bytes
        raw = ds.read_at(offset, plane_bytes)
        return np.frombuffer(raw, dtype="<u2").reshape(
            info.height, info.width)
    finally:
        if owns:
            ds.close()


__all__ = ["EtsInfo", "parse_ets", "decode_ets", "decode_ets_plane"]
