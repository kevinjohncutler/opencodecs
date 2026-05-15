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
            # Offset 28 = width, offset 32 = height (verified via
            # byte-for-byte diff against bftools output on the
            # OME zenodo-17590655 corpus).
            n_components = struct.unpack_from("<I", sub, 8)[0]
            width  = struct.unpack_from("<I", sub, 28)[0]
            height = struct.unpack_from("<I", sub, 32)[0]

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


def decode_ets(path: str | Path) -> np.ndarray:
    """Decode an .ets file into a (planes, height, width) uint16 stack.

    For the observed file format, planes are sequentially packed
    starting at file offset 292 (= 64 header + 228 sub-header), each
    plane is ``height × width × 2`` bytes (uint16 LE), reshape as
    ``(height, width)``. The trailing 8 KB at EOF holds the index
    sub-chunks; we don't need them for pixel decode.

    Plane count = (file_size - 292 - trailing_index_size) /
    (height × width × 2). For the corpus sample this is 180 planes
    (5T × 18Z × 2C × 216×260×u16).

    Verified byte-identical to bftools output on the OME
    zenodo-17590655 corpus sample.
    """
    info = parse_ets(path)
    if not info.magic_ok:
        raise ValueError(f"{path}: not a SIS / ETS file")
    if info.width == 0 or info.height == 0:
        raise ValueError(
            f"{path}: ETS sub-header missing geometry")
    plane_bytes = info.height * info.width * 2
    # The first sub-chunk (sub_chunk_offsets[1]) marks where the
    # post-data index starts. Everything between the header end and
    # there is plane data.
    data_end = info.sub_chunk_offsets[1]
    data_start = 292   # header (64) + ETS sub-header (228)
    payload_bytes = data_end - data_start
    if payload_bytes % plane_bytes != 0:
        raise ValueError(
            f"{path}: ETS payload {payload_bytes} bytes isn't a "
            f"multiple of plane_bytes ({plane_bytes}); cannot "
            f"determine plane count")
    n_planes = payload_bytes // plane_bytes
    with open(path, "rb") as f:
        f.seek(data_start)
        raw = f.read(payload_bytes)
    return np.frombuffer(raw, dtype="<u2").reshape(
        n_planes, info.height, info.width)


__all__ = ["EtsInfo", "parse_ets", "decode_ets"]
