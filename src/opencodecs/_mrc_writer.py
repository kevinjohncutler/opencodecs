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


def _bounded_statistics(arr):
    """Merge finite block statistics without a volume-sized mask or copy."""
    from .core._write_helpers import iter_array_buffers
    accumulation_dtype = np.complex128 if arr.dtype.kind == "c" else np.float64
    count = 0
    mean = moment = 0.0
    minimum, maximum = np.inf, -np.inf
    for raw in iter_array_buffers(arr):
        block = np.frombuffer(raw, dtype=arr.dtype)
        if arr.dtype.kind == "f":
            finite = np.isfinite(block)
            if not finite.all():
                block = block[finite]
        n = block.size
        if not n:
            continue
        block_mean = block.mean(dtype=accumulation_dtype).item()
        block_moment = float(block.var(dtype=accumulation_dtype).real) * n
        delta = block_mean - mean
        total = count + n
        moment += block_moment + abs(delta) ** 2 * count * n / total
        mean += delta * n / total
        count = total
        minimum = float(np.minimum(minimum, float(block.min().real)))
        maximum = float(np.maximum(maximum, float(block.max().real)))
    return (minimum, maximum, float(np.real(mean)), (moment / count) ** 0.5) if count else (0.0,) * 4


def _mrc_parts(data, *, voxel_size=None, nstart=(0, 0, 0), ispg=0):
    """Prepare a header without materializing the serialized volume."""
    arr = np.asarray(data)
    dtype = np.dtype("i2") if arr.dtype == np.dtype("u1") else arr.dtype.newbyteorder("<")
    # Validate before computing statistics or creating a destination.
    mrc_header(arr.shape, dtype.newbyteorder("="), voxel_size=voxel_size,
               nstart=nstart, ispg=ispg)
    stats = _bounded_statistics(arr)
    header = mrc_header(arr.shape, dtype.newbyteorder("="), voxel_size=voxel_size,
                        stats=stats, nstart=nstart, ispg=ispg)
    return header, arr, dtype


def encode_mrc(data: Any, **kwargs) -> bytes:
    """Serialize an array as a complete MRC file."""
    import io
    dest = io.BytesIO()
    write_mrc(dest, data, **kwargs)
    return dest.getvalue()


def write_mrc(path: Any, data: Any, **kwargs) -> None:
    """Write header and bounded pixel blocks without an encoded-volume copy."""
    from .core._write_helpers import binary_destination, iter_array_buffers, write_all
    header, arr, dtype = _mrc_parts(data, **kwargs)
    with binary_destination(path) as dest:
        write_all(dest, header)
        for block in iter_array_buffers(arr, dtype=dtype):
            write_all(dest, block)


__all__ = ["encode_mrc", "write_mrc", "mrc_header"]
