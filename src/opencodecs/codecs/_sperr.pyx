# opencodecs/codecs/_sperr.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native SPERR codec — wavelet-based error-bounded lossy compression.

SPERR (2022+) is a SPECK/wavelet-based compressor for scientific
floating-point arrays. It often hits much smaller bitstreams than ZFP
or SZ3 at the same PSNR target on smooth fields (climate, CFD,
seismic, scientific simulation).

C API (``SPERR_C_API.h``) accepts 2D slices and 3D volumes in either
float32 (``is_float=1``) or float64 (``is_float=0``). 2D output can
optionally embed a 10-byte header carrying dims + dtype; 3D output
always embeds its own header.

We wrap the SPERR bitstream with a 48-byte opencodecs preamble so we
can sniff the format and round-trip arbitrary ndim/dtype without
trusting SPERR's internal header alone::

    bytes  0..3   ASCII magic 'SPRR'
    byte   4      dtype: 1 = float32, 0 = float64
    byte   5      ndim:  2 or 3
    bytes  6..7   reserved
    bytes  8..47  shape: 5 × uint64 little-endian
                  (dimx fastest, dimy, dimz, _, _)
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t
from libc.string cimport memcpy

import struct as _struct

import numpy as np
cimport numpy as cnp

from sperr cimport (
    sperr_comp_2d, sperr_decomp_2d,
    sperr_comp_3d, sperr_decomp_3d,
    sperr_parse_header,
    free,
)


cnp.import_array()


_HEADER_LEN = 48
_HEADER_MAGIC = b'SPRR'
_HEADER_FMT = '<4sBB2x5Q'


class SperrError(RuntimeError):
    """Raised on SPERR encode/decode failures."""


_MODE_ALIASES = {
    "bpp": 1,
    "psnr": 2,
    "pwe": 3,
}


def _pack_header(int is_float, int ndim, dimx, dimy, dimz):
    return _struct.pack(
        _HEADER_FMT, _HEADER_MAGIC,
        int(is_float) & 0xff, int(ndim) & 0xff,
        int(dimx), int(dimy), int(dimz), 0, 0,
    )


def _unpack_header(buf):
    if len(buf) < _HEADER_LEN:
        raise SperrError("sperr blob too short to contain header")
    magic, is_float, ndim, dimx, dimy, dimz, _r4, _r5 = _struct.unpack(
        _HEADER_FMT, bytes(buf[:_HEADER_LEN]),
    )
    if magic != _HEADER_MAGIC:
        raise SperrError(f"sperr blob has wrong magic {magic!r}")
    return is_float, ndim, dimx, dimy, dimz


def encode(arr, *,
           mode: str = "psnr",
           psnr: float = 80.0,
           bpp: float = 4.0,
           pwe: float = 1e-3,
           chunk=(256, 256, 256),
           nthreads: int = 0) -> bytes:
    """Compress ndarray with SPERR.

    Parameters
    ----------
    arr : np.ndarray
        2D or 3D float32 / float64 array.
    mode : {"psnr", "bpp", "pwe"}
        Quality control mode.
        - "psnr": target peak signal-to-noise ratio in dB (default 80).
        - "bpp" : target bits per pixel (default 4.0).
        - "pwe" : point-wise absolute error bound (default 1e-3).
    psnr, bpp, pwe : float
        Target value for the chosen mode.
    chunk : tuple of 3 ints
        Chunk dims for 3D mode (ignored for 2D).
    nthreads : int
        OpenMP threads for 3D mode (0 = auto). 2D is single-threaded.
    """
    cdef:
        cnp.ndarray contig
        int is_float
        int err_mode
        size_t dimx = 0, dimy = 0, dimz = 0
        size_t cx, cy, cz
        size_t out_size = 0
        void* dst = NULL
        int rc
        bytes payload
        bytes header
        double quality

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    contig = np.ascontiguousarray(arr)
    if contig.dtype == np.dtype(np.float32):
        is_float = 1
    elif contig.dtype == np.dtype(np.float64):
        is_float = 0
    else:
        raise ValueError(
            f"sperr encode: only float32/float64 supported "
            f"(got {contig.dtype!r}); use 'zfp' or 'sz3' for integer arrays"
        )

    if mode not in _MODE_ALIASES:
        raise ValueError(
            f"sperr encode: unknown mode {mode!r}; expected one of "
            f"{sorted(_MODE_ALIASES.keys())}"
        )
    err_mode = _MODE_ALIASES[mode]
    if mode == "psnr":
        quality = float(psnr)
    elif mode == "bpp":
        quality = float(bpp)
    else:  # pwe
        quality = float(pwe)

    if contig.ndim not in (2, 3):
        raise ValueError(f"sperr encode: ndim must be 2 or 3, got {contig.ndim}")

    # numpy is row-major: outermost dim is shape[0] = slowest. SPERR takes
    # dimx as the fastest-varying.
    if contig.ndim == 2:
        dimx = <size_t> contig.shape[1]
        dimy = <size_t> contig.shape[0]
        dimz = 1
    else:
        dimx = <size_t> contig.shape[2]
        dimy = <size_t> contig.shape[1]
        dimz = <size_t> contig.shape[0]

    cdef void* data_ptr = <void*> contig.data
    cdef size_t nthreads_c = <size_t> int(nthreads)

    if contig.ndim == 2:
        with nogil:
            rc = sperr_comp_2d(
                data_ptr, is_float, dimx, dimy,
                err_mode, quality, 0,  # no embedded header — we use our own
                &dst, &out_size,
            )
    else:
        cx = <size_t> min(int(chunk[0]), contig.shape[0])
        cy = <size_t> min(int(chunk[1]), contig.shape[1])
        cz = <size_t> min(int(chunk[2]), contig.shape[2])
        # Map (chunk_z, chunk_y, chunk_x) onto our axis order (chunk[0] is
        # outermost). The user passes chunk as (z, y, x) matching numpy.
        with nogil:
            rc = sperr_comp_3d(
                data_ptr, is_float,
                dimx, dimy, dimz,
                cz, cy, cx,
                err_mode, quality, nthreads_c,
                &dst, &out_size,
            )

    if rc != 0 or dst == NULL or out_size == 0:
        if dst != NULL:
            free(dst)
        raise SperrError(f"SPERR compression failed (rc={rc})")

    try:
        payload = PyBytes_FromStringAndSize(<const char*> dst, <Py_ssize_t> out_size)
    finally:
        free(dst)

    header = _pack_header(is_float, contig.ndim, dimx, dimy, dimz)
    return header + payload


def decode(data) -> 'np.ndarray':
    """Decode an SPERR blob (preamble + bitstream) to an ndarray."""
    cdef:
        const uint8_t[::1] src
        Py_ssize_t srcsize
        Py_ssize_t header_off = <Py_ssize_t> _HEADER_LEN
        Py_ssize_t payload_len
        void* dst = NULL
        size_t out_dimx = 0, out_dimy = 0, out_dimz = 0
        int rc
        cnp.ndarray out_arr

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = src.shape[0]
    if srcsize < _HEADER_LEN:
        raise SperrError("sperr blob too short to contain header")

    is_float, ndim, dimx, dimy, dimz = _unpack_header(bytes(src[:_HEADER_LEN]))
    payload_len = srcsize - _HEADER_LEN
    if payload_len <= 0:
        raise SperrError("sperr blob payload missing")

    dtype = np.float32 if is_float else np.float64
    if ndim == 2:
        shape = (int(dimy), int(dimx))
    elif ndim == 3:
        shape = (int(dimz), int(dimy), int(dimx))
    else:
        raise SperrError(f"sperr header has invalid ndim={ndim}")

    out = np.empty(shape, dtype=dtype)
    out_arr = out

    cdef const void* src_ptr = <const void*> &src[header_off]
    cdef size_t pl = <size_t> payload_len
    cdef size_t dx = <size_t> dimx, dy = <size_t> dimy
    cdef int isf = int(is_float)

    if ndim == 2:
        with nogil:
            rc = sperr_decomp_2d(src_ptr, pl, isf, dx, dy, &dst)
    else:
        with nogil:
            rc = sperr_decomp_3d(
                src_ptr, pl, isf, 0,
                &out_dimx, &out_dimy, &out_dimz,
                &dst,
            )

    if rc != 0 or dst == NULL:
        if dst != NULL:
            free(dst)
        raise SperrError(f"SPERR decompression failed (rc={rc})")

    try:
        memcpy(<void*> out_arr.data, dst, <size_t>(out.size * out.dtype.itemsize))
    finally:
        free(dst)
    return out


def check_signature(data) -> bool:
    """Match opencodecs SPERR preamble magic 'SPRR'."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:4])
    else:
        try:
            head = bytes(data)[:4]
        except Exception:
            return False
    return head == _HEADER_MAGIC
