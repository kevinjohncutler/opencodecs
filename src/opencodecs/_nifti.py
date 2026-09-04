"""NIfTI-1 and NIfTI-2 reader (neuroimaging volumes).

NIfTI is what every brain-imaging pipeline speaks: FSL, SPM, AFNI,
FreeSurfer, nilearn, and the whole BIDS ecosystem. It is Analyze 7.5 with
the spatial orientation problem fixed, which is why the header is mostly
a fixed struct with an affine bolted on.

Two versions are in circulation and both are common. NIfTI-1 has a
348-byte header with 16-bit dimensions, which caps an axis at 32767 and
is why NIfTI-2 exists; NIfTI-2 has a 540-byte header with 64-bit
dimensions and doubles. They are distinguished by the size field and the
magic, and this reader handles both plus either byte order.

Almost every NIfTI in the wild is gzipped, so ``.nii.gz`` is handled
transparently rather than being the caller's problem.

Reference: the NIfTI-1.1 and NIfTI-2 header definitions published by the
NIfTI Data Format Working Group (nifti.nimh.nih.gov). The struct layouts
below are transcribed from those specifications; no code is derived from
any implementation.
"""

from __future__ import annotations

import gzip
import os
import struct
from typing import Any

import numpy as np

from .core._io_helpers import read_src as _read_src
from .core.codec import ArrayReader

NIFTI1_HEADER_SIZE = 348
NIFTI2_HEADER_SIZE = 540

# NIfTI datatype codes. The spec's own names are in the comments; the
# gaps are codes the standard reserves or deprecates from Analyze.
_DTYPES: dict[int, str] = {
    2: "u1",        # UINT8
    4: "i2",        # INT16
    8: "i4",        # INT32
    16: "f4",       # FLOAT32
    32: "c8",       # COMPLEX64
    64: "f8",       # FLOAT64
    256: "i1",      # INT8
    512: "u2",      # UINT16
    768: "u4",      # UINT32
    1024: "i8",     # INT64
    1280: "u8",     # UINT64
    1536: "f16",    # FLOAT128, only where the platform has it
    1792: "c16",    # COMPLEX128
}

# Stored as 3 or 4 bytes per voxel rather than a scalar type. Reported
# rather than silently reinterpreted, because a caller expecting a
# scalar volume should not receive a struct array by surprise.
_RGB_TYPES = {128: "RGB24", 2304: "RGBA32"}

_UNSUPPORTED = {
    1: "binary (1 bit per voxel)",
    **_RGB_TYPES,
}


class NiftiError(Exception):
    """Raised when a file is not valid NIfTI, or uses a type we cannot read."""


def _maybe_gunzip(raw: bytes) -> bytes:
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


class NiftiStream(ArrayReader):
    """Reader over one NIfTI-1 or NIfTI-2 file.

    Unlike the MRC and TIFF readers this holds the file in memory, because
    the dominant on-disk form is gzip and a gzip member has no usable
    random access: reaching the last slice means inflating everything
    before it. For an uncompressed ``.nii`` the whole-file read is one
    syscall anyway.
    """

    def __init__(self, src: Any):
        self._raw = _maybe_gunzip(_read_src(src))
        if len(self._raw) < 4:
            raise NiftiError("NIfTI: file is too short to hold a header")
        self._header = self._parse_header(self._raw)

    # -- header ------------------------------------------------------

    @staticmethod
    def _version_and_order(raw: bytes) -> tuple[int, str]:
        """Decide NIfTI-1 vs NIfTI-2, and the byte order, from sizeof_hdr.

        The size field is the format's own version marker: 348 for
        NIfTI-1, 540 for NIfTI-2. Reading it byte-swapped and finding one
        of those values is exactly how the spec says to detect a
        foreign-endian file.
        """
        for order in ("<", ">"):
            size = struct.unpack_from(order + "i", raw, 0)[0]
            if size == NIFTI1_HEADER_SIZE:
                return 1, order
            if size == NIFTI2_HEADER_SIZE:
                return 2, order
        raise NiftiError(
            "NIfTI: sizeof_hdr is neither 348 nor 540 in either byte "
            f"order (got {struct.unpack_from('<i', raw, 0)[0]}); this is "
            f"not a NIfTI file")

    def _parse_header(self, raw: bytes) -> dict[str, Any]:
        version, order = self._version_and_order(raw)
        need = NIFTI1_HEADER_SIZE if version == 1 else NIFTI2_HEADER_SIZE
        if len(raw) < need:
            raise NiftiError(
                f"NIfTI-{version}: file is {len(raw)} bytes, shorter than "
                f"its {need}-byte header")

        if version == 1:
            # NIfTI-1: dim is int16, pixdim/vox_offset are float32, and
            # the magic sits at the very end of the header.
            dim = struct.unpack_from(order + "8h", raw, 40)
            datatype, bitpix = struct.unpack_from(order + "2h", raw, 70)
            pixdim = struct.unpack_from(order + "8f", raw, 76)
            vox_offset = struct.unpack_from(order + "f", raw, 108)[0]
            scl_slope, scl_inter = struct.unpack_from(order + "2f", raw, 112)
            xyzt_units = raw[123]
            magic = raw[344:348]
        else:
            # NIfTI-2 reorders the struct and widens the numeric fields;
            # the magic moves to the front, right after sizeof_hdr.
            magic = raw[4:8]
            datatype, bitpix = struct.unpack_from(order + "2h", raw, 12)
            dim = struct.unpack_from(order + "8q", raw, 16)
            pixdim = struct.unpack_from(order + "8d", raw, 104)
            vox_offset = struct.unpack_from(order + "q", raw, 168)[0]
            scl_slope, scl_inter = struct.unpack_from(order + "2d", raw, 176)
            xyzt_units = struct.unpack_from(order + "i", raw, 500)[0]

        if magic[:3] not in (b"n+1", b"ni1", b"n+2", b"ni2"):
            raise NiftiError(f"NIfTI-{version}: bad magic {magic!r}")

        ndim = dim[0]
        if not 1 <= ndim <= 7:
            raise NiftiError(
                f"NIfTI: dim[0] is {ndim}, outside the legal range 1..7")
        shape = tuple(int(d) for d in dim[1:ndim + 1])
        if any(d <= 0 for d in shape):
            raise NiftiError(f"NIfTI: non-positive dimension in {shape}")

        if datatype in _UNSUPPORTED:
            raise NiftiError(
                f"NIfTI: datatype {datatype} ({_UNSUPPORTED[datatype]}) "
                f"is not supported")
        if datatype not in _DTYPES:
            raise NiftiError(f"NIfTI: unknown datatype code {datatype}")

        return {
            "version": version,
            "byteorder": order,
            "magic": magic.rstrip(b"\x00").decode("ascii", "replace"),
            "single_file": magic[:3] in (b"n+1", b"n+2"),
            "datatype": datatype,
            "bitpix": bitpix,
            "ndim": int(ndim),
            "shape": shape,
            "pixdim": tuple(float(p) for p in pixdim),
            "vox_offset": int(vox_offset),
            "scl_slope": float(scl_slope),
            "scl_inter": float(scl_inter),
            "xyzt_units": int(xyzt_units),
        }

    @property
    def header(self) -> dict[str, Any]:
        return dict(self._header)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape in NIfTI axis order, fastest-varying axis first.

        Matches what nibabel reports, so ``(x, y, z)`` or ``(x, y, z, t)``
        rather than the reversed order a naive C-order reshape produces.
        """
        return self._header["shape"]

    @property
    def dtype(self) -> np.dtype:
        h = self._header
        return np.dtype(_DTYPES[h["datatype"]]).newbyteorder(h["byteorder"])

    @property
    def voxel_size(self) -> tuple[float, ...]:
        """pixdim[1..ndim]: the spatial (and temporal) sampling."""
        h = self._header
        return h["pixdim"][1:h["ndim"] + 1]

    @property
    def has_scaling(self) -> bool:
        """Whether scl_slope / scl_inter actually change the values.

        A slope of 0 means "no scaling" per the spec, and slope 1 with
        intercept 0 is the identity, so neither counts.
        """
        s, i = self._header["scl_slope"], self._header["scl_inter"]
        return s != 0.0 and not (s == 1.0 and i == 0.0)

    # -- data --------------------------------------------------------

    def asarray(self, *, scaled: bool = True) -> np.ndarray:
        """Read the volume.

        ``scaled`` applies scl_slope and scl_inter, which is what the
        header means by them: the stored integers are a compressed
        representation and the real values are slope * stored + inter.
        Scaling only happens when it would change something, so an
        unscaled integer volume stays integer rather than being widened
        to float for nothing. Pass ``scaled=False`` for raw stored values.
        """
        h = self._header
        offset = h["vox_offset"]
        if not h["single_file"]:
            raise NiftiError(
                "NIfTI: this is the header of a .hdr/.img pair (magic "
                f"{h['magic']!r}); the voxels live in the .img file")
        count = 1
        for d in h["shape"]:
            count *= d
        itemsize = self.dtype.itemsize
        need = offset + count * itemsize
        if len(self._raw) < need:
            raise NiftiError(
                f"NIfTI: truncated file; needs {need} bytes for "
                f"{h['shape']} at offset {offset}, have {len(self._raw)}")

        arr = np.frombuffer(self._raw, dtype=self.dtype,
                            count=count, offset=offset)
        # NIfTI stores the first dimension fastest, which is Fortran
        # order for the shape as reported.
        arr = arr.reshape(h["shape"], order="F")

        if scaled and self.has_scaling:
            arr = arr * np.float32(h["scl_slope"]) + np.float32(h["scl_inter"])
        return arr

    def close(self) -> None:
        self._raw = b""

    def __enter__(self) -> "NiftiStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["NiftiStream", "NiftiError",
           "NIFTI1_HEADER_SIZE", "NIFTI2_HEADER_SIZE"]
