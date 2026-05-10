# opencodecs/codecs/_pcodec.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native pcodec — modern numerical-array compressor.

pcodec (https://github.com/mwlon/pcodec) is a recent (2024+) lossless
numerical compressor that beats zstd on float / int arrays by 1.5-3×
without filtering. It's particularly strong on time-series and
sensor data thanks to its statistical-modelling design.

The C API (``pco_standalone_simple_*``) operates on flat typed buffers;
the encoded payload contains the dtype + element count, but we still
prepend a small opencodecs preamble so we can reconstruct the *shape*
of multidimensional inputs:

    bytes  0..3   ASCII magic 'PCOO'  (= "pcodec + Open codecs")
    byte   4      dtype enum (PCO_TYPE_*)
    byte   5      ndim (1..8)
    bytes  6..7   reserved (zero)
    bytes  8..71  shape: 8 × uint64 little-endian
    bytes 72..    pcodec standalone bytes
"""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdint cimport uint8_t

import numpy as np
cimport numpy as cnp

from pcodec cimport (
    PCO_TYPE_U8, PCO_TYPE_I8,
    PCO_TYPE_U16, PCO_TYPE_I16, PCO_TYPE_F16,
    PCO_TYPE_U32, PCO_TYPE_I32, PCO_TYPE_F32,
    PCO_TYPE_U64, PCO_TYPE_I64, PCO_TYPE_F64,
    PcoError, PcoSuccess, PcoInvalidType, PcoCompressionError, PcoDecompressionError,
    PcoChunkConfig,
    pco_standalone_guarantee_file_size,
    pco_standalone_simple_compress_into,
    pco_standalone_simple_decompress_into,
)


cnp.import_array()


_HEADER_LEN = 72
_HEADER_MAGIC = b'PCOO'

import struct as _struct
_HEADER_FMT = '<4sBB2x8Q'


class PcodecError(RuntimeError):
    """Raised on pcodec encode/decode failures."""


_DTYPE_TO_PCO = {
    np.dtype(np.uint8):   PCO_TYPE_U8,
    np.dtype(np.int8):    PCO_TYPE_I8,
    np.dtype(np.uint16):  PCO_TYPE_U16,
    np.dtype(np.int16):   PCO_TYPE_I16,
    np.dtype(np.float16): PCO_TYPE_F16,
    np.dtype(np.uint32):  PCO_TYPE_U32,
    np.dtype(np.int32):   PCO_TYPE_I32,
    np.dtype(np.float32): PCO_TYPE_F32,
    np.dtype(np.uint64):  PCO_TYPE_U64,
    np.dtype(np.int64):   PCO_TYPE_I64,
    np.dtype(np.float64): PCO_TYPE_F64,
}
_PCO_TO_DTYPE = {v: k for k, v in _DTYPE_TO_PCO.items()}


_PCO_ERROR_MSG = {
    PcoInvalidType: "PcoInvalidType",
    PcoCompressionError: "PcoCompressionError",
    PcoDecompressionError: "PcoDecompressionError",
}


def _pack_header(dtype_enum, ndim, shape8):
    return _struct.pack(_HEADER_FMT, _HEADER_MAGIC,
                        int(dtype_enum) & 0xff, int(ndim) & 0xff,
                        *[int(s) for s in shape8])


def _unpack_header(buf):
    if len(buf) < _HEADER_LEN:
        raise PcodecError("pcodec blob too short to contain header")
    magic, dtype_enum, ndim, *shape8 = _struct.unpack(_HEADER_FMT,
                                                     bytes(buf[:_HEADER_LEN]))
    if magic != _HEADER_MAGIC:
        raise PcodecError(f"pcodec blob has wrong magic {magic!r}")
    return dtype_enum, ndim, shape8


def encode(arr, *, level: int = 8, max_page_n: int = 0) -> bytes:
    """Compress an ndarray with pcodec.

    Parameters
    ----------
    arr : np.ndarray
        Any 1D-8D array of a supported dtype (i8/u8/i16/u16/f16/i32/u32/
        f32/i64/u64/f64).
    level : int
        Compression level 0..12 (default 8). Higher = better ratio,
        slower encode.
    max_page_n : int
        Maximum elements per internal page; 0 = library default (262144).
        Smaller pages = more random-access friendly, slightly worse ratio.
    """
    cdef:
        cnp.ndarray contig
        unsigned char dtype_enum
        size_t n_elems
        size_t cap
        size_t written = 0
        bytes payload
        unsigned char* out_ptr
        PcoChunkConfig cfg
        PcoError rc

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    contig = np.ascontiguousarray(arr)
    if contig.dtype not in _DTYPE_TO_PCO:
        raise ValueError(f"pcodec encode: unsupported dtype {contig.dtype!r}")
    dtype_enum = <unsigned char> _DTYPE_TO_PCO[contig.dtype]

    if contig.ndim < 1 or contig.ndim > 8:
        raise ValueError(f"pcodec encode: ndim must be 1..8, got {contig.ndim}")

    n_elems = <size_t> contig.size
    cfg.compression_level = <unsigned int> int(level)
    cfg.max_page_n = <size_t> int(max_page_n)

    cap = pco_standalone_guarantee_file_size(n_elems, dtype_enum)
    if cap == 0:
        raise PcodecError("pco_standalone_guarantee_file_size returned 0 (invalid type?)")
    payload = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> cap)
    out_ptr = <unsigned char*> PyBytes_AsString(payload)

    cdef const void* data_ptr = <const void*> contig.data

    with nogil:
        rc = pco_standalone_simple_compress_into(
            data_ptr, n_elems, dtype_enum, &cfg,
            <void*> out_ptr, cap, &written,
        )
    if rc != PcoSuccess:
        raise PcodecError(
            f"pcodec compress failed: {_PCO_ERROR_MSG.get(int(rc), int(rc))}"
        )

    # Pack the original shape into 8 slots so decode reconstructs it.
    # contig.shape is a npy_intp* C array, not a Python tuple — index into it.
    shape8 = [int(contig.shape[i]) for i in range(contig.ndim)]
    shape8 += [0] * (8 - contig.ndim)
    header = _pack_header(dtype_enum, contig.ndim, shape8)
    return header + payload[:written]


def decode(data) -> 'np.ndarray':
    """Decode a pcodec blob (header + standalone bytes) to an ndarray."""
    cdef:
        const uint8_t[::1] src
        Py_ssize_t srcsize
        unsigned char dtype_enum
        int ndim
        Py_ssize_t header_off
        Py_ssize_t payload_len
        size_t written = 0
        PcoError rc
        cnp.ndarray out_arr

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = src.shape[0]
    if srcsize < _HEADER_LEN:
        raise PcodecError("pcodec blob too short to contain header")

    dtype_enum_py, ndim, shape8 = _unpack_header(bytes(src[:_HEADER_LEN]))
    dtype_enum = <unsigned char> dtype_enum_py
    if dtype_enum_py not in _PCO_TO_DTYPE:
        raise PcodecError(f"pcodec blob has unsupported dtype enum {dtype_enum_py}")
    dtype = _PCO_TO_DTYPE[dtype_enum_py]

    shape = tuple(int(shape8[i]) for i in range(ndim))
    out = np.empty(shape, dtype=dtype)
    out_arr = out

    cdef size_t out_n = <size_t> out.size
    header_off = <Py_ssize_t> _HEADER_LEN
    payload_len = srcsize - header_off

    with nogil:
        rc = pco_standalone_simple_decompress_into(
            <const void*> &src[header_off], <size_t> payload_len, dtype_enum,
            <void*> out_arr.data, out_n, &written,
        )
    if rc != PcoSuccess:
        raise PcodecError(
            f"pcodec decompress failed: {_PCO_ERROR_MSG.get(int(rc), int(rc))}"
        )
    if <Py_ssize_t> written != <Py_ssize_t> out.size:
        raise PcodecError(
            f"pcodec decompress wrote {written} elements, expected {out.size}"
        )
    return out


def check_signature(data) -> bool:
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:4])
    else:
        try:
            head = bytes(data)[:4]
        except Exception:
            return False
    return head == _HEADER_MAGIC
