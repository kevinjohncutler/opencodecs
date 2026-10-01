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
sensor data thanks to its statistical-modeling design.

``encode`` writes the pcodec standalone format (``pco_standalone_*``)
as it is: magic ``pco!``, then the standalone version, the number
type and a hint of the element count (the count, or 0 if unknown),
then the chunks and a termination byte (pcodec docs/format.md). That
is the stream the pcodec package, ``numcodecs.PCodec`` and
``imagecodecs.pcodec_encode`` write. It records no N-D shape, so
``decode`` returns 1-D unless the caller passes ``shape``.

Releases up to 0.4.0 wrote a private 72-byte preamble in front of the
standalone bytes to carry the shape::

    bytes  0..3   ASCII magic 'PCOO'
    byte   4      dtype enum (PCO_TYPE_*)
    byte   5      ndim (1..8)
    bytes  6..7   reserved (zero)
    bytes  8..71  shape: 8 × uint64 little-endian

``decode`` still reads those blobs, recognized by the magic.
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
_STANDALONE_MAGIC = b'pco!'

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


# Shared with sz3 and sperr; pcodec reserves eight dimension slots.
from opencodecs.core._sidecar_header import make_sidecar_header as \
    _make_sidecar_header
_HEADER_LEN, _pack_header, _unpack_header = _make_sidecar_header(
    _HEADER_MAGIC, 8, PcodecError, "pcodec")


def encode(arr, *, level: int = 8, max_page_n: int = 0) -> bytes:
    """Compress an ndarray with pcodec into a standalone stream.

    Parameters
    ----------
    arr : np.ndarray
        An array of a supported dtype (i8/u8/i16/u16/f16/i32/u32/f32/
        i64/u64/f64), compressed in C order. The stream records the
        element count (as a hint), not the shape.
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

    return payload[:written]


def decode_framed(data, *, out=None) -> 'np.ndarray':
    """Decode a blob in the framed layout releases up to 0.4.0 wrote
    ('PCOO' preamble + standalone bytes) to an ndarray of its shape.

    ``out=`` is a preallocated ndarray; pcodec writes directly into the
    caller's buffer via pco_standalone_simple_decompress_into. See
    ``_png.decode`` for the full contract.
    """
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
    if out is not None:
        if not isinstance(out, np.ndarray):
            raise TypeError(
                f"pcodec decode: out= must be an ndarray, "
                f"got {type(out).__name__}")
        if out.shape != shape:
            raise ValueError(
                f"pcodec decode: out= shape {out.shape} does not match "
                f"expected {shape}")
        if out.dtype != dtype:
            raise ValueError(
                f"pcodec decode: out= dtype {out.dtype} does not match "
                f"expected {dtype}")
        if not out.flags['C_CONTIGUOUS']:
            raise ValueError("pcodec decode: out= must be C-contiguous")
        out_arr = out
    else:
        out_arr = np.empty(shape, dtype=dtype)

    cdef size_t out_n = <size_t> out_arr.size
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
    if <Py_ssize_t> written != <Py_ssize_t> out_arr.size:
        raise PcodecError(
            f"pcodec decompress wrote {written} elements, expected {out_arr.size}"
        )
    return out_arr


def standalone_info(data):
    """Return ``(version, dtype or None, n or None)`` from the header of
    a standalone stream.

    The header is the magic, a version byte and, from version 2, the
    number type (0 when the file does not commit to one). Version 3
    follows with the element count hint: six bits ``p``, then ``n``
    itself in ``p + 1`` bits, packed least significant bit first. The
    format makes ``n`` a hint, 0 when the writer did not know the
    count. Fields this reader cannot place come back None.
    """
    head = bytes(data[:16])
    if len(head) < 5 or head[:4] != _STANDALONE_MAGIC:
        raise PcodecError("pcodec: not a standalone stream (no 'pco!' magic)")
    version = head[4]
    dtype = None
    n = None
    if version >= 2 and len(head) >= 6 and head[5]:
        if head[5] not in _PCO_TO_DTYPE:
            raise PcodecError(f"pcodec stream has unsupported number type {head[5]}")
        dtype = _PCO_TO_DTYPE[head[5]]
    if version == 3 and len(head) >= 7:
        bits = int.from_bytes(head[6:], 'little')
        p = bits & 63
        if 6 + p + 1 <= 8 * (len(head) - 6):
            n = (bits >> 6) & ((1 << (p + 1)) - 1)
    return version, dtype, n


def decode_standalone(data, *, dtype, Py_ssize_t n, out=None,
                      bint exact=True) -> 'np.ndarray':
    """Decode a standalone stream of ``n`` elements of ``dtype``.

    Returns a 1-D array, or fills ``out`` (C-contiguous, ``n`` elements
    of ``dtype``, any shape) and returns it. pcodec fails when the
    stream holds more than ``n`` elements. With ``exact`` (and always
    with ``out``) fewer than ``n`` raise too; without it ``n`` is only
    a capacity, and the result holds the elements the stream has.
    """
    cdef:
        const uint8_t[::1] src
        Py_ssize_t srcsize
        unsigned char dtype_enum
        size_t written = 0
        size_t out_n
        PcoError rc
        cnp.ndarray out_arr

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = src.shape[0]
    dt = np.dtype(dtype)
    if dt not in _DTYPE_TO_PCO:
        raise ValueError(f"pcodec decode: unsupported dtype {dt!r}")
    dtype_enum = <unsigned char> _DTYPE_TO_PCO[dt]
    if n < 0:
        raise ValueError("pcodec decode: element count must be nonnegative")
    if out is not None:
        if not isinstance(out, np.ndarray):
            raise TypeError(
                f"pcodec decode: out= must be an ndarray, "
                f"got {type(out).__name__}")
        if out.size != n or out.dtype != dt:
            raise ValueError(
                f"pcodec decode: out= holds {out.size} {out.dtype} "
                f"elements, expected {n} {dt}")
        if not out.flags['C_CONTIGUOUS']:
            raise ValueError("pcodec decode: out= must be C-contiguous")
        out_arr = out
    else:
        out_arr = np.empty(n, dtype=dt)
    if srcsize == 0:
        raise PcodecError("pcodec decode: empty input")
    out_n = <size_t> n
    with nogil:
        rc = pco_standalone_simple_decompress_into(
            <const void*> &src[0], <size_t> srcsize, dtype_enum,
            <void*> out_arr.data, out_n, &written,
        )
    if rc != PcoSuccess:
        raise PcodecError(
            f"pcodec decompress failed: {_PCO_ERROR_MSG.get(int(rc), int(rc))} "
            f"(a damaged stream, or one holding more than {n} elements)"
        )
    if <Py_ssize_t> written != n:
        if not exact and out is None:
            return out_arr[:written].copy()
        raise ValueError(
            f"pcodec decode: the stream holds {written} elements, "
            f"expected {n}"
        )
    return out_arr


def check_signature(data) -> bool:
    """Match the pcodec standalone magic 'pco!', or the 'PCOO'
    preamble of releases up to 0.4.0."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:4])
    else:
        try:
            head = bytes(data)[:4]
        except Exception:
            return False
    return head == _STANDALONE_MAGIC or head == _HEADER_MAGIC
