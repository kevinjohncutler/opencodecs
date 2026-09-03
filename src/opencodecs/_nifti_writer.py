"""NIfTI-1 writer.

Same reasoning as the MRC writer: a reader alone cannot be an output
stage. NIfTI-1 rather than NIfTI-2 because every tool reads NIfTI-1 and
the 32767-per-axis limit it carries is not a constraint for anything
that would be produced here; writing NIfTI-2 for a small volume would
narrow, not widen, who can open the result.

The affine matters more than the voxels. A NIfTI with no orientation is
a stack of numbers: FSL, SPM and FreeSurfer all read qform/sform to know
which way is left. We write an sform (and a matching qform code of 0,
meaning "not usable") built from the voxel sizes, which is the honest
minimum: a diagonal RAS affine that says "these are the voxel sizes and
nothing has been reoriented".
"""

from __future__ import annotations

import gzip
import os
import struct
from typing import Any

import numpy as np

from ._nifti import NIFTI1_HEADER_SIZE, NiftiError

_CODE_FOR = {
    np.dtype("u1"): (2, 8), np.dtype("i2"): (4, 16), np.dtype("i4"): (8, 32),
    np.dtype("f4"): (16, 32), np.dtype("c8"): (32, 64),
    np.dtype("f8"): (64, 64), np.dtype("i1"): (256, 8),
    np.dtype("u2"): (512, 16), np.dtype("u4"): (768, 32),
    np.dtype("i8"): (1024, 64), np.dtype("u8"): (1280, 64),
}

# xyzt_units: 2 = millimetres, 8 = seconds. Packed into one byte.
UNITS_MM_SEC = 2 | 8


def nifti1_header(shape, dtype, *, voxel_size=None, vox_offset: int = 352,
                  units: int = UNITS_MM_SEC) -> bytes:
    dtype = np.dtype(dtype)
    if dtype not in _CODE_FOR:
        raise NiftiError(
            f"nifti: cannot write dtype {dtype}; supported are "
            f"{sorted(str(d) for d in _CODE_FOR)}")
    if not 1 <= len(shape) <= 7:
        raise NiftiError(f"nifti: rank {len(shape)} outside the legal 1..7")
    if any(d > 32767 for d in shape):
        raise NiftiError(
            f"nifti: dimension {max(shape)} exceeds NIfTI-1's int16 limit of "
            f"32767; this volume needs NIfTI-2")

    code, bitpix = _CODE_FOR[dtype]
    h = bytearray(NIFTI1_HEADER_SIZE)
    struct.pack_into("<i", h, 0, NIFTI1_HEADER_SIZE)
    dim = [len(shape)] + list(shape) + [1] * (7 - len(shape))
    struct.pack_into("<8h", h, 40, *dim)
    struct.pack_into("<2h", h, 70, code, bitpix)

    vs = voxel_size if voxel_size is not None else (1.0,) * len(shape)
    if np.isscalar(vs):
        vs = (float(vs),) * len(shape)
    pixdim = [1.0] + [float(v) for v in vs] + [0.0] * (7 - len(vs))
    struct.pack_into("<8f", h, 76, *pixdim[:8])
    struct.pack_into("<f", h, 108, float(vox_offset))
    # scl_slope 0 means "no scaling", which is what we want: the values
    # written are the values meant.
    struct.pack_into("<2f", h, 112, 0.0, 0.0)
    h[123] = units
    struct.pack_into("<2h", h, 252, 0, 2)                # qform_code, sform_code
    # A diagonal RAS srow: voxel sizes on the diagonal, no rotation.
    sx, sy, sz = (list(vs) + [1.0, 1.0, 1.0])[:3]
    struct.pack_into("<4f", h, 280, float(sx), 0.0, 0.0, 0.0)
    struct.pack_into("<4f", h, 296, 0.0, float(sy), 0.0, 0.0)
    struct.pack_into("<4f", h, 312, 0.0, 0.0, float(sz), 0.0)
    h[344:348] = b"n+1\x00"
    return bytes(h)


def encode_nifti(data: Any, *, voxel_size=None, compress: bool = False) -> bytes:
    """Serialize an array as a complete single-file NIfTI-1 (.nii)."""
    arr = np.asarray(data)
    if arr.dtype.byteorder == ">":
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    header = nifti1_header(arr.shape, arr.dtype, voxel_size=voxel_size)
    # vox_offset is 352: the 348-byte header plus the 4-byte extender
    # that says "no header extensions follow".
    blob = header + b"\x00\x00\x00\x00" + arr.tobytes(order="F")
    return gzip.compress(blob) if compress else blob


def write_nifti(path: Any, data: Any, *, compress: bool | None = None,
                **kwargs) -> None:
    """Write a NIfTI-1 file, gzipping when the name ends in .gz."""
    if compress is None:
        name = getattr(path, "name", path)
        compress = isinstance(name, (str, os.PathLike)) and \
            str(name).endswith(".gz")
    blob = encode_nifti(data, compress=compress, **kwargs)
    if hasattr(path, "write"):
        path.write(blob)
        return
    with open(os.fspath(path), "wb") as fh:
        fh.write(blob)


__all__ = ["encode_nifti", "write_nifti", "nifti1_header"]
