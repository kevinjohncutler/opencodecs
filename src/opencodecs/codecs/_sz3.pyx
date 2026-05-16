# opencodecs/codecs/_sz3.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native SZ3 codec — error-bounded lossy compression for scientific arrays.

SZ3 is a modern (2022+) prediction-based compressor that often
beats ZFP at the same error budget for scientifically-correlated data
(time series, simulation snapshots). The C API is ``SZ_compress_args``;
shape and dtype are *not* stored in the encoded payload, so we wrap
the SZ3 stream with a small opencodecs preamble:

    bytes  0..3   ASCII magic 'SZ3O'  (= "SZ3 + Open codecs")
    byte   4      dtype enum (SZ_FLOAT/SZ_DOUBLE/...)
    byte   5      ndim (1..5)
    bytes  6..7   reserved
    bytes  8..47  shape: 5 × uint64 little-endian (r1, r2, r3, r4, r5)
    bytes 48..    SZ3 raw stream
"""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdint cimport uint8_t
from libc.string cimport memcpy

import numpy as np
cimport numpy as cnp

from sz3c cimport (
    ABS, REL, ABS_AND_REL, ABS_OR_REL, PSNR, NORM,
    SZ_FLOAT, SZ_DOUBLE,
    SZ_UINT8, SZ_INT8, SZ_UINT16, SZ_INT16,
    SZ_UINT32, SZ_INT32, SZ_UINT64, SZ_INT64,
    SZ_compress_args, SZ_decompress, free_buf,
)


cnp.import_array()


_HEADER_LEN = 48
_HEADER_MAGIC = b'SZ3O'
import struct as _struct
_HEADER_FMT = '<4sBB2x5Q'


class Sz3Error(RuntimeError):
    """Raised on SZ3 encode/decode failures."""


_DTYPE_TO_SZ = {
    np.dtype(np.float32): SZ_FLOAT,
    np.dtype(np.float64): SZ_DOUBLE,
    np.dtype(np.uint8):   SZ_UINT8,
    np.dtype(np.int8):    SZ_INT8,
    np.dtype(np.uint16):  SZ_UINT16,
    np.dtype(np.int16):   SZ_INT16,
    np.dtype(np.uint32):  SZ_UINT32,
    np.dtype(np.int32):   SZ_INT32,
    np.dtype(np.uint64):  SZ_UINT64,
    np.dtype(np.int64):   SZ_INT64,
}
_SZ_TO_DTYPE = {v: k for k, v in _DTYPE_TO_SZ.items()}


_MODE_ALIASES = {
    "abs": ABS, "rel": REL, "abs_and_rel": ABS_AND_REL,
    "abs_or_rel": ABS_OR_REL, "psnr": PSNR, "norm": NORM,
}


def _pack_header(dtype_enum, ndim, shape5):
    return _struct.pack(_HEADER_FMT, _HEADER_MAGIC,
                        int(dtype_enum) & 0xff, int(ndim) & 0xff,
                        *[int(s) for s in shape5])


def _unpack_header(buf):
    if len(buf) < _HEADER_LEN:
        raise Sz3Error("sz3 blob too short to contain header")
    magic, dtype_enum, ndim, *shape5 = _struct.unpack(_HEADER_FMT,
                                                      bytes(buf[:_HEADER_LEN]))
    if magic != _HEADER_MAGIC:
        raise Sz3Error(f"sz3 blob has wrong magic {magic!r}")
    return dtype_enum, ndim, shape5


def encode(arr, *,
           mode: str = "abs",
           abs_err: float = 1e-3,
           rel_err: float = 0.0,
           psnr: float = 0.0) -> bytes:
    """Compress ndarray with SZ3.

    Parameters
    ----------
    arr : np.ndarray
        1D-5D array. Float32/Float64/int{8,16,32,64} (signed/unsigned).
    mode : str
        "abs" (absolute err bound), "rel" (value-range relative),
        "abs_and_rel", "abs_or_rel", "psnr", "norm".
    abs_err : float
        Used in "abs" / mixed modes. Absolute error per pixel.
    rel_err : float
        Used in "rel" / mixed modes. Fraction of value range.
    psnr : float
        Used in "psnr" mode (target dB).
    """
    cdef:
        cnp.ndarray contig
        int dtype_enum
        int err_mode
        size_t r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0
        size_t out_size = 0
        unsigned char* sz_buf
        bytes payload
        bytes header

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    contig = np.ascontiguousarray(arr)
    if contig.dtype not in _DTYPE_TO_SZ:
        raise ValueError(f"sz3 encode: unsupported dtype {contig.dtype!r}")
    dtype_enum = _DTYPE_TO_SZ[contig.dtype]

    if mode not in _MODE_ALIASES:
        raise ValueError(
            f"sz3 encode: unknown mode {mode!r}; expected one of "
            f"{sorted(_MODE_ALIASES.keys())}"
        )
    err_mode = _MODE_ALIASES[mode]

    if contig.ndim < 1 or contig.ndim > 5:
        raise ValueError(f"sz3 encode: ndim must be 1..5, got {contig.ndim}")

    # SZ3 takes (r5, r4, r3, r2, r1) — innermost dim is r1 (fastest-varying).
    # numpy is row-major: outermost dim is shape[0]. Map: r1 = shape[ndim-1],
    # r2 = shape[ndim-2], ..., r5 = shape[0]. Unused dims pass 0.
    if contig.ndim >= 1: r1 = <size_t> contig.shape[contig.ndim - 1]
    if contig.ndim >= 2: r2 = <size_t> contig.shape[contig.ndim - 2]
    if contig.ndim >= 3: r3 = <size_t> contig.shape[contig.ndim - 3]
    if contig.ndim >= 4: r4 = <size_t> contig.shape[contig.ndim - 4]
    if contig.ndim >= 5: r5 = <size_t> contig.shape[contig.ndim - 5]

    cdef void* data_ptr = <void*> contig.data
    cdef double abs_e = float(abs_err)
    cdef double rel_e = float(rel_err)
    cdef double psnr_v = float(psnr)
    # SZ3 uses pwrBoundRatio for PSNR mode in some legacy paths; pass 0.
    cdef double pwr_e = 0.0

    with nogil:
        sz_buf = SZ_compress_args(
            dtype_enum, data_ptr, &out_size,
            err_mode,
            abs_e, rel_e, pwr_e,
            r5, r4, r3, r2, r1,
        )
    if sz_buf == NULL or out_size == 0:
        raise Sz3Error("SZ_compress_args returned NULL")

    try:
        payload = PyBytes_FromStringAndSize(<const char*> sz_buf, <Py_ssize_t> out_size)
    finally:
        free_buf(<void*> sz_buf)

    shape5 = (r1, r2, r3, r4, r5)
    header = _pack_header(dtype_enum, contig.ndim, shape5)
    return header + payload


def decode(data, *, out=None) -> 'np.ndarray':
    """Decode an SZ3 blob (header + SZ3 stream) to an ndarray.

    ``out=`` is a preallocated ndarray; see ``_png.decode`` for the full
    contract. SZ3's library decompresses into its own malloc'd buffer
    which we memcpy into the destination, so out= saves the second
    alloc (not the SZ3-internal one).
    """
    cdef:
        const uint8_t[::1] src
        Py_ssize_t srcsize
        int dtype_enum
        int ndim
        size_t r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0
        Py_ssize_t payload_len
        void* sz_out
        size_t total_elems = 1
        cnp.ndarray out_arr

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = src.shape[0]
    if srcsize < _HEADER_LEN:
        raise Sz3Error("sz3 blob too short to contain header")

    dtype_enum, ndim, shape5 = _unpack_header(bytes(src[:_HEADER_LEN]))
    r1, r2, r3, r4, r5 = shape5

    if dtype_enum not in _SZ_TO_DTYPE:
        raise Sz3Error(f"sz3 blob has unsupported dtype enum {dtype_enum}")
    dtype = _SZ_TO_DTYPE[dtype_enum]

    # Reconstruct numpy shape from r1..r5 and ndim.
    if ndim == 1:
        shape = (int(r1),)
    elif ndim == 2:
        shape = (int(r2), int(r1))
    elif ndim == 3:
        shape = (int(r3), int(r2), int(r1))
    elif ndim == 4:
        shape = (int(r4), int(r3), int(r2), int(r1))
    elif ndim == 5:
        shape = (int(r5), int(r4), int(r3), int(r2), int(r1))
    else:
        raise Sz3Error(f"sz3 header has invalid ndim={ndim}")

    payload_len = srcsize - _HEADER_LEN
    if payload_len <= 0:
        raise Sz3Error("sz3 blob payload missing")

    if out is not None:
        if not isinstance(out, np.ndarray):
            raise TypeError(
                f"sz3 decode: out= must be an ndarray, "
                f"got {type(out).__name__}")
        if out.shape != shape:
            raise ValueError(
                f"sz3 decode: out= shape {out.shape} does not match "
                f"expected {shape}")
        if out.dtype != dtype:
            raise ValueError(
                f"sz3 decode: out= dtype {out.dtype} does not match "
                f"expected {dtype}")
        if not out.flags['C_CONTIGUOUS']:
            raise ValueError("sz3 decode: out= must be C-contiguous")
        out_arr = out
    else:
        out_arr = np.empty(shape, dtype=dtype)
    total_elems = <size_t> out_arr.size

    cdef Py_ssize_t header_off = <Py_ssize_t> _HEADER_LEN
    with nogil:
        sz_out = SZ_decompress(
            dtype_enum,
            <unsigned char*> &src[header_off], <size_t> payload_len,
            r5, r4, r3, r2, r1,
        )
    if sz_out == NULL:
        raise Sz3Error("SZ_decompress returned NULL")
    try:
        memcpy(<void*> out_arr.data, sz_out, total_elems * out_arr.dtype.itemsize)
    finally:
        free_buf(sz_out)
    return out_arr


def check_signature(data) -> bool:
    """Match opencodecs SZ3 preamble magic 'SZ3O'."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:4])
    else:
        try:
            head = bytes(data)[:4]
        except Exception:
            return False
    return head == _HEADER_MAGIC
