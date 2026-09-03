"""MRC2014 writer.

Reading MRC without writing it makes opencodecs a dead end in a cryo-EM
pipeline: you can open a motion-corrected stack but not save the result.
The format is a 1024-byte header and raw voxels, so writing it is mostly
a matter of filling the header in honestly, which is where the care goes.

Two fields are worth filling rather than zeroing, because downstream
tools read them and a zeroed header is a subtly broken file:

* ``CELLA`` divided by the grid gives voxel size in Angstroms, which is
  what every consumer uses for scale. A zero cell means "no scale".
* ``DMIN`` / ``DMAX`` / ``DMEAN`` / ``RMS`` are the data statistics.
  Writers that leave them zero produce files that display as blank in
  Chimera and IMOD, because those use the range to set contrast.
"""

from __future__ import annotations

import os
import struct
from typing import Any

import numpy as np

from ._mrc import HEADER_SIZE, MrcError

# numpy dtype -> MRC MODE. int8 maps to 0, which the format defines as
# signed; unsigned 8-bit has no mode of its own, so it is widened rather
# than silently reinterpreted.
_MODE_FOR = {
    np.dtype("i1"): 0,
    np.dtype("i2"): 1,
    np.dtype("f4"): 2,
    np.dtype("u2"): 6,
    np.dtype("f2"): 12,
    np.dtype("c8"): 4,
}


def mrc_header(shape, dtype, *, voxel_size=None, stats=None,
               nstart=(0, 0, 0), ispg: int = 0) -> bytes:
    """Build the 1024-byte MRC2014 header for an array."""
    dtype = np.dtype(dtype)
    if dtype not in _MODE_FOR:
        raise MrcError(
            f"mrc: cannot write dtype {dtype}; MRC modes cover "
            f"{sorted(str(d) for d in _MODE_FOR)}")
    if len(shape) == 2:
        nz, ny, nx = 1, shape[0], shape[1]
    elif len(shape) == 3:
        nz, ny, nx = shape
    else:
        raise MrcError(f"mrc: expected a 2-D or 3-D array, got shape {shape}")

    h = bytearray(HEADER_SIZE)
    struct.pack_into("<iiii", h, 0, nx, ny, nz, _MODE_FOR[dtype])
    struct.pack_into("<iii", h, 16, *nstart)
    struct.pack_into("<iii", h, 28, nx, ny, nz)              # MX MY MZ
    vs = voxel_size if voxel_size is not None else (1.0, 1.0, 1.0)
    if np.isscalar(vs):
        vs = (float(vs),) * 3
    # CELLA is the cell size, so voxel size times the grid.
    struct.pack_into("<fff", h, 40, float(vs[0]) * nx,
                     float(vs[1]) * ny, float(vs[2]) * nz)
    struct.pack_into("<fff", h, 52, 90.0, 90.0, 90.0)        # CELLB angles
    struct.pack_into("<iii", h, 64, 1, 2, 3)                 # MAPC/MAPR/MAPS
    dmin, dmax, dmean, rms = stats if stats else (0.0, 0.0, 0.0, 0.0)
    struct.pack_into("<fff", h, 76, dmin, dmax, dmean)
    struct.pack_into("<i", h, 88, ispg)
    struct.pack_into("<i", h, 92, 0)                         # NSYMBT
    struct.pack_into("<i", h, 108, 20140)                    # NVERSION
    h[208:212] = b"MAP "
    h[212:216] = b"\x44\x44\x00\x00"                         # little-endian
    struct.pack_into("<f", h, 216, rms)
    struct.pack_into("<i", h, 220, 1)                        # NLABL
    label = b"Created by opencodecs".ljust(80, b" ")
    h[224:304] = label
    return bytes(h)


def encode_mrc(data: Any, *, voxel_size=None, nstart=(0, 0, 0),
               ispg: int = 0) -> bytes:
    """Serialize an array as a complete MRC file."""
    arr = np.ascontiguousarray(data)
    if arr.dtype == np.dtype("u1"):
        # MODE 0 is signed. Widening is the honest choice: reinterpreting
        # 200 as -56 would round-trip through our own reader and be wrong
        # in everyone else's.
        arr = arr.astype("i2")
    if arr.dtype.byteorder == ">" or (
            arr.dtype.byteorder == "=" and np.little_endian is False):
        arr = arr.astype(arr.dtype.newbyteorder("<"))

    if arr.size:
        finite = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr
        if finite.size:
            stats = (float(finite.min()), float(finite.max()),
                     float(finite.mean()), float(finite.std()))
        else:
            stats = (0.0, 0.0, 0.0, 0.0)
    else:
        stats = (0.0, 0.0, 0.0, 0.0)

    header = mrc_header(arr.shape, arr.dtype, voxel_size=voxel_size,
                        stats=stats, nstart=nstart, ispg=ispg)
    return header + arr.tobytes()


def write_mrc(path: Any, data: Any, **kwargs) -> None:
    """Write an array to an MRC file."""
    blob = encode_mrc(data, **kwargs)
    if hasattr(path, "write"):
        path.write(blob)
        return
    with open(os.fspath(path), "wb") as fh:
        fh.write(blob)


__all__ = ["encode_mrc", "write_mrc", "mrc_header"]
