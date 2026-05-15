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

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .core.io import DataSource, coerce_data_source


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


def parse_ets(src: Any) -> EtsInfo:
    """Read just enough of an .ets to enumerate its structure.

    Accepts a path or a DataSource. HTTP-backed sources fetch
    only the 64-byte SIS header, ~256 bytes of the ETS sub-header,
    and 4 bytes of the trailing index — typically under 1 KB
    regardless of file size.
    """
    ds, owns, file_size = coerce_data_source(src)
    try:
        hdr = ds.read_at(0, 64)
        if hdr[:4] != _SIS_MAGIC:
            return EtsInfo(
                file_size=file_size, width=0, height=0,
                n_components=0, sub_chunk_offsets=[],
                sub_chunk_sizes=[], level_count=0, magic_ok=False,
            )
        ptr1 = struct.unpack_from("<Q", hdr, 16)[0]
        sz1  = struct.unpack_from("<Q", hdr, 24)[0]
        ptr2 = struct.unpack_from("<Q", hdr, 32)[0]
        sz2  = struct.unpack_from("<Q", hdr, 40)[0]
        ptr3 = struct.unpack_from("<Q", hdr, 48)[0]
        sz3  = struct.unpack_from("<Q", hdr, 56)[0]

        width = height = n_components = 0
        if sz1 >= 64:
            sub = ds.read_at(ptr1, min(sz1, 256))
            if sub[:4] == _ETS_MAGIC:
                # Offset 28 = width, offset 32 = height (verified
                # against bftools on the OME zenodo-17590655 corpus).
                n_components = struct.unpack_from("<I", sub, 8)[0]
                width  = struct.unpack_from("<I", sub, 28)[0]
                height = struct.unpack_from("<I", sub, 32)[0]

        level_count = 0
        if sz2 >= 4:
            idx_head = ds.read_at(ptr2, min(sz2, 64))
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
    finally:
        if owns:
            ds.close()


def decode_ets(src: Any) -> np.ndarray:
    """Decode an .ets source into a (planes, height, width) uint16 stack.

    Accepts a path or DataSource. The plane data is one contiguous
    sequential read from offset 292 through ``data_end`` (the
    trailing-index pointer in the SIS header). For an HTTP source
    this is one large range request rather than many small ones —
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
