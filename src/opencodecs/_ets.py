"""SIS / ETS (Olympus Sequential Image Stream) partial parser.

Backs VsiCodec's full-resolution data path. ``frame_t_N.ets`` files
sit alongside a top-level ``.vsi`` index and hold the actual tile
data for one stack / time-point. Both header and tile-index
sections are clean-room mapped from real files (no proprietary
docs / no GPL code read for this implementation).

Header (first 64 bytes of an .ets):
  @0..3   ASCII magic ``SIS\\0``
  @4      u32 header_size (always 64 in observed files)
  @8      u32 version (3 in observed files)
  @12     u32 unknown (6)
  @16     u64 first chunk offset (=64 — sub-header starts immediately)
  @24     u64 first chunk size
  @32     u64 second chunk offset (typically near EOF; tile index)
  @40     u64 second chunk size
  @48     u64 third chunk offset (terminator or extension)
  @56     u64 third chunk size (0 in observed files)

Sub-header at offset 64 (the "ETS\\0" block):
  @0..3   ASCII magic ``ETS\\0``
  @4-7    u32 packed (lower byte: pixel-component count, upper: ?)
  @8      u32 component count / n_channels
  @24     u32 nominal width or tile width (observed 100 for some
          files, 216 for the test sample)
  @28     u32 nominal height (observed 260 for the test sample)
  @32     u32 depth or fourth axis

The second chunk (near EOF) appears to be a pyramid/tile index
with one record per pyramid level — each record holds a level
offset + level data size. Verifying this requires either:
  (a) cross-referencing a .ets with bioformats output (we don't
      do because of license contamination concerns), or
  (b) building decoding for individual tiles + checking against
      the .vsi thumbnail.

Current status: ``info(path)`` parses the header + sub-header and
returns ``{geometry, level_count, level_index, magic_ok}``.
Per-tile pixel decoding is a future native upgrade (single-session
work given a frame-of-known-content sample).
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_SIS_MAGIC = b"SIS\x00"
_ETS_MAGIC = b"ETS\x00"


@dataclass
class EtsInfo:
    """Partial parse result for an .ets file."""
    file_size: int
    width: int
    height: int
    n_components: int
    sub_chunk_offsets: list[int]   # 3 entries, last may be 0
    sub_chunk_sizes: list[int]
    level_count: int               # tile-pyramid level count
    magic_ok: bool


def parse_ets(path: str | Path) -> EtsInfo:
    """Read just enough of an .ets file to enumerate its structure.

    Does NOT decode any tile pixel data — full-decode work is
    deferred until we have a known-content sample to verify
    against (clean-room development requires ground truth).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    with open(p, "rb") as f:
        hdr = f.read(64)
    file_size = p.stat().st_size

    if hdr[:4] != _SIS_MAGIC:
        return EtsInfo(
            file_size=file_size, width=0, height=0, n_components=0,
            sub_chunk_offsets=[], sub_chunk_sizes=[], level_count=0,
            magic_ok=False,
        )

    ptr1 = struct.unpack_from("<Q", hdr, 16)[0]
    sz1  = struct.unpack_from("<Q", hdr, 24)[0]
    ptr2 = struct.unpack_from("<Q", hdr, 32)[0]
    sz2  = struct.unpack_from("<Q", hdr, 40)[0]
    ptr3 = struct.unpack_from("<Q", hdr, 48)[0]
    sz3  = struct.unpack_from("<Q", hdr, 56)[0]

    width = height = n_components = 0
    if sz1 >= 64:
        with open(p, "rb") as f:
            f.seek(ptr1)
            sub = f.read(min(sz1, 256))
        if sub[:4] == _ETS_MAGIC:
            # The geometry-bearing u32s are at fixed offsets in
            # observed files. We've only confirmed two real samples,
            # so this may need adjustment for variants.
            n_components = struct.unpack_from("<I", sub, 8)[0]
            width  = struct.unpack_from("<I", sub, 32)[0]
            height = struct.unpack_from("<I", sub, 28)[0]

    level_count = 0
    if sz2 >= 4:
        with open(p, "rb") as f:
            f.seek(ptr2)
            idx_head = f.read(min(sz2, 64))
        level_count = struct.unpack_from("<I", idx_head, 0)[0]

    return EtsInfo(
        file_size=file_size,
        width=width,
        height=height,
        n_components=n_components,
        sub_chunk_offsets=[ptr1, ptr2, ptr3],
        sub_chunk_sizes=[sz1, sz2, sz3],
        level_count=level_count,
        magic_ok=True,
    )


def read_ets_levels(path: str | Path) -> list[np.ndarray]:
    """**Experimental** — read each level entry's payload from the
    .ets pyramid index. Returns one ndarray per level, reshaped
    using the ETS sub-header geometry. Useful for inspecting what's
    in a .ets companion before we have a verified full-decode.

    For the corpus file (216×260, 4 components, 6 levels):
      - Each level entry is 112,320 bytes = 216×260×u16
      - Total of 6 level entries = 673,920 bytes
      - The rest of the 20 MB file is presumed tile-pyramid data
        we don't yet decode
    """
    import numpy as np
    info = parse_ets(path)
    out: list[np.ndarray] = []
    if not info.magic_ok or info.width == 0:
        return out
    # The level index at sub_chunk_offsets[1] lists per-level offsets.
    # Each level record is 44 bytes (observed in the corpus file):
    #   u32 const=6, u32 zeros×5, u32 level_idx, u32 zeros×3,
    #   u64 file_offset, u64 size, u32 extra
    idx_off = info.sub_chunk_offsets[1]
    idx_size = info.sub_chunk_sizes[1]
    with open(path, "rb") as f:
        f.seek(idx_off)
        idx_bytes = f.read(idx_size)
        # Decode each level entry. Walk the index looking for the
        # pattern that emerged from inspecting real bytes: each
        # entry's u32 file_offset lands at a predictable index.
        # Use a simple approach: scan for u32 values that point into
        # the data region (< sub_chunk_offsets[1]) and are followed
        # by a size of 112320 (the observed level data size).
        n = info.level_count
        for i in range(n):
            # Each record is 44 bytes (observed)
            rec_off = 4 + i * 44     # skip the leading u32 count
            if rec_off + 44 > len(idx_bytes):
                break
            level_offset = (
                idx_bytes[rec_off + 24]
                | (idx_bytes[rec_off + 25] << 8)
                | (idx_bytes[rec_off + 26] << 16)
                | (idx_bytes[rec_off + 27] << 24)
            )
            level_size = (
                idx_bytes[rec_off + 32]
                | (idx_bytes[rec_off + 33] << 8)
                | (idx_bytes[rec_off + 34] << 16)
                | (idx_bytes[rec_off + 35] << 24)
            )
            if level_offset == 0 or level_size == 0:
                continue
            f.seek(level_offset)
            payload = f.read(level_size)
            pixels = np.frombuffer(payload, dtype="<u2").copy()
            # Reshape via the ETS sub-header dimensions when total matches
            if pixels.size == info.width * info.height:
                pixels = pixels.reshape(info.height, info.width)
            out.append(pixels)
    return out


__all__ = ["EtsInfo", "parse_ets", "read_ets_levels"]
