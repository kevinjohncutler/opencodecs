# opencodecs/codecs/_zfp.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native ZFP codec — fast lossy compression for 1D-4D float / int arrays.

ZFP is the de-facto standard for HPC scientific compression. It supports
three lossy modes plus reversible (lossless):

* **rate**: fixed bits-per-value (predictable size)
* **precision**: fixed bits-of-precision (predictable accuracy)
* **accuracy**: fixed absolute error tolerance (predictable error)
* **reversible**: lossless (use for round-trip integrity)

The encoded blob is fully self-describing: shape, dtype, and mode
metadata are written via ``zfp_write_header(ZFP_HEADER_FULL)``, so
decode reconstructs the array without out-of-band parameters.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t

import numpy as np
cimport numpy as cnp

from zfp cimport (
    zfp_type, zfp_type_int32, zfp_type_int64, zfp_type_float, zfp_type_double,
    zfp_field, zfp_field_alloc,
    zfp_field_1d, zfp_field_2d, zfp_field_3d, zfp_field_4d,
    zfp_field_free, zfp_field_set_pointer,
    zfp_stream, zfp_stream_open, zfp_stream_close,
    zfp_stream_maximum_size, zfp_stream_rewind,
    zfp_stream_set_bit_stream, zfp_stream_set_reversible,
    zfp_stream_set_rate, zfp_stream_set_precision, zfp_stream_set_accuracy,
    zfp_compress, zfp_decompress, zfp_write_header, zfp_read_header,
    ZFP_HEADER_FULL,
    bitstream, stream_open, stream_close, stream_size,
)


cnp.import_array()


class ZfpError(RuntimeError):
    """Raised on ZFP encode/decode failures."""


_DTYPE_TO_ZFP = {
    np.dtype(np.int32):   zfp_type_int32,
    np.dtype(np.int64):   zfp_type_int64,
    np.dtype(np.float32): zfp_type_float,
    np.dtype(np.float64): zfp_type_double,
}
_ZFP_TO_DTYPE = {v: k for k, v in _DTYPE_TO_ZFP.items()}


cdef int _ndim_from_field(zfp_field* f) nogil:
    if f.nw > 0: return 4
    if f.nz > 0: return 3
    if f.ny > 0: return 2
    return 1


def encode(arr, *,
           mode: str = "reversible",
           rate=None, precision=None, accuracy=None) -> bytes:
    """ZFP-encode a 1D-4D float or int array.

    Parameters
    ----------
    arr : np.ndarray
        Shape (nx,), (nx, ny), (nx, ny, nz), or (nx, ny, nz, nw).
    mode : str
        "reversible" (lossless), "rate", "precision", or "accuracy".
    rate, precision, accuracy : numeric
        Mode-specific parameter. Required when ``mode`` matches.

    Returns
    -------
    bytes
        Self-describing ZFP stream (with full header).
    """
    cdef:
        cnp.ndarray contig
        zfp_type ztype
        zfp_field* field = NULL
        zfp_stream* zstream = NULL
        bitstream* bs = NULL
        bytes out
        const unsigned char[::1] dst_mv
        size_t cap
        size_t nbytes

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    contig = np.ascontiguousarray(arr)
    if contig.dtype not in _DTYPE_TO_ZFP:
        raise ValueError(
            f"zfp encode: unsupported dtype {contig.dtype!r}; "
            f"expected int32/int64/float32/float64"
        )
    if contig.ndim < 1 or contig.ndim > 4:
        raise ValueError(
            f"zfp encode: ndim must be 1..4, got {contig.ndim}"
        )

    ztype = _DTYPE_TO_ZFP[contig.dtype]

    cdef unsigned int n0, n1, n2, n3
    cdef void* data_ptr = <void*> contig.data

    if contig.ndim == 1:
        n0 = <unsigned int> contig.shape[0]
        field = zfp_field_1d(data_ptr, ztype, n0)
    elif contig.ndim == 2:
        n0 = <unsigned int> contig.shape[1]   # ZFP uses (nx, ny) where nx is fastest-varying axis
        n1 = <unsigned int> contig.shape[0]
        field = zfp_field_2d(data_ptr, ztype, n0, n1)
    elif contig.ndim == 3:
        n0 = <unsigned int> contig.shape[2]
        n1 = <unsigned int> contig.shape[1]
        n2 = <unsigned int> contig.shape[0]
        field = zfp_field_3d(data_ptr, ztype, n0, n1, n2)
    else:  # ndim == 4
        n0 = <unsigned int> contig.shape[3]
        n1 = <unsigned int> contig.shape[2]
        n2 = <unsigned int> contig.shape[1]
        n3 = <unsigned int> contig.shape[0]
        field = zfp_field_4d(data_ptr, ztype, n0, n1, n2, n3)
    if field == NULL:
        raise ZfpError("zfp_field_*d returned NULL")

    zstream = zfp_stream_open(NULL)
    if zstream == NULL:
        zfp_field_free(field)
        raise ZfpError("zfp_stream_open returned NULL")

    try:
        if mode == "reversible":
            zfp_stream_set_reversible(zstream)
        elif mode == "rate":
            if rate is None:
                raise ValueError("zfp encode: mode='rate' requires rate=...")
            zfp_stream_set_rate(zstream, float(rate), ztype, contig.ndim, 0)
        elif mode == "precision":
            if precision is None:
                raise ValueError("zfp encode: mode='precision' requires precision=...")
            zfp_stream_set_precision(zstream, int(precision))
        elif mode == "accuracy":
            if accuracy is None:
                raise ValueError("zfp encode: mode='accuracy' requires accuracy=...")
            zfp_stream_set_accuracy(zstream, float(accuracy))
        else:
            raise ValueError(
                f"zfp encode: unknown mode {mode!r}; expected reversible/"
                f"rate/precision/accuracy"
            )

        cap = zfp_stream_maximum_size(zstream, field)
        out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> cap)
        # Same memoryview-cast trick that _zstd.pyx uses: a uint8
        # memoryview onto the bytes object lets the Cython compiler
        # route writes through the buffer-export path, which lines
        # up better with libzfp's page-fault pattern than the raw
        # PyBytes_AsString pointer. ~5-10 us / encode on a 56 KB
        # output (M1 Ultra).
        dst_mv = out
        bs = stream_open(<void*> &dst_mv[0], cap)
        if bs == NULL:
            raise ZfpError("stream_open returned NULL")
        zfp_stream_set_bit_stream(zstream, bs)
        zfp_stream_rewind(zstream)

        if zfp_write_header(zstream, field, ZFP_HEADER_FULL) == 0:
            raise ZfpError("zfp_write_header failed")
        nbytes = zfp_compress(zstream, field)
        if nbytes == 0:
            raise ZfpError("zfp_compress failed")
        out_size = <Py_ssize_t> stream_size(bs)
    finally:
        if bs != NULL:
            stream_close(bs)
        if zstream != NULL:
            zfp_stream_close(zstream)
        if field != NULL:
            zfp_field_free(field)
    # Drop the memoryview before the slice so it doesn't keep an
    # export alive across the slice copy.
    del dst_mv
    return out[:out_size]


def decode(data, *, out=None) -> 'np.ndarray':
    """Decode a self-describing ZFP stream.

    ``out=`` is a preallocated ndarray; zfp writes into the caller's
    buffer directly via zfp_field_set_pointer. See ``_png.decode`` for
    the full contract.
    """
    cdef:
        const uint8_t[::1] src
        Py_ssize_t srcsize
        zfp_field* field = NULL
        zfp_stream* zstream = NULL
        bitstream* bs = NULL
        size_t rc
        cnp.ndarray out_arr

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = src.shape[0]
    if srcsize == 0:
        raise ZfpError("empty zfp stream")

    bs = stream_open(<void*> &src[0], <size_t> srcsize)
    if bs == NULL:
        raise ZfpError("stream_open returned NULL")

    zstream = zfp_stream_open(bs)
    if zstream == NULL:
        stream_close(bs)
        raise ZfpError("zfp_stream_open returned NULL")

    field = zfp_field_alloc()
    if field == NULL:
        zfp_stream_close(zstream)
        stream_close(bs)
        raise ZfpError("zfp_field_alloc returned NULL")

    try:
        if zfp_read_header(zstream, field, ZFP_HEADER_FULL) == 0:
            raise ZfpError("zfp_read_header failed (full header missing)")

        # Translate ZFP field's (nx, ny, nz, nw) — fastest-axis-first — to
        # numpy shape (slowest-axis-first).
        ndim = _ndim_from_field(field)
        if ndim == 1:
            shape = (int(field.nx),)
        elif ndim == 2:
            shape = (int(field.ny), int(field.nx))
        elif ndim == 3:
            shape = (int(field.nz), int(field.ny), int(field.nx))
        else:
            shape = (int(field.nw), int(field.nz), int(field.ny), int(field.nx))

        if field.type not in _ZFP_TO_DTYPE:
            raise ZfpError(f"zfp stream has unsupported type {int(field.type)}")
        dtype = _ZFP_TO_DTYPE[field.type]

        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError(
                    f"zfp decode: out= must be an ndarray, "
                    f"got {type(out).__name__}")
            if out.shape != shape:
                raise ValueError(
                    f"zfp decode: out= shape {out.shape} does not match "
                    f"expected {shape}")
            if out.dtype != dtype:
                raise ValueError(
                    f"zfp decode: out= dtype {out.dtype} does not match "
                    f"expected {dtype}")
            if not out.flags['C_CONTIGUOUS']:
                raise ValueError("zfp decode: out= must be C-contiguous")
            out_arr = out
        else:
            out_arr = np.empty(shape, dtype=dtype)
        zfp_field_set_pointer(field, <void*> out_arr.data)

        rc = zfp_decompress(zstream, field)
        if rc == 0:
            raise ZfpError("zfp_decompress failed")
    finally:
        if field != NULL:
            zfp_field_free(field)
        if zstream != NULL:
            zfp_stream_close(zstream)
        if bs != NULL:
            stream_close(bs)
    return out_arr


def check_signature(data) -> bool:
    """ZFP magic word: bytes 'zfp' (0x7A, 0x66, 0x70) at the start."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:3])
    else:
        try:
            head = bytes(data)[:3]
        except Exception:
            return False
    # ZFP_MAGIC_BITS = 32; the magic word has "zfp" + version byte.
    return head == b"zfp"
