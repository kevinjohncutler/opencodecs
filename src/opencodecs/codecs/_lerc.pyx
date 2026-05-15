# opencodecs/codecs/_lerc.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native LERC codec — Esri Limited Error Raster Compression.

LERC is a fast lossless / near-lossless raster codec used heavily in
geospatial pipelines. The blob is self-describing (shape, dtype, value
range and codec version are all in the header), so decode reconstructs
the array without out-of-band info.

Encoding is parameterized by ``maxZErr``: 0 means lossless, > 0 caps the
absolute reconstruction error at that value (per pixel). For floats,
this is a fast way to trade a known error budget for much better
compression.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdint cimport uint8_t

import numpy as np
cimport numpy as cnp

from lerc cimport (
    lerc_status,
    lerc_computeCompressedSize, lerc_encode,
    lerc_getBlobInfo, lerc_decode,
)


cnp.import_array()


class LercError(RuntimeError):
    """Raised on LERC encode/decode failures."""


# numpy dtype -> LERC enum  (Lerc_types.h DataType)
_DTYPE_TO_LERC = {
    np.dtype(np.int8):    0,   # dt_char
    np.dtype(np.uint8):   1,   # dt_uchar
    np.dtype(np.int16):   2,   # dt_short
    np.dtype(np.uint16):  3,   # dt_ushort
    np.dtype(np.int32):   4,   # dt_int
    np.dtype(np.uint32):  5,   # dt_uint
    np.dtype(np.float32): 6,   # dt_float
    np.dtype(np.float64): 7,   # dt_double
}

_LERC_TO_DTYPE = {v: k for k, v in _DTYPE_TO_LERC.items()}


def _err(func, code):
    return LercError(f'{func} returned LERC status {code}')


def encode(arr, *, max_z_error=0.0) -> bytes:
    """LERC-encode an ndarray.

    Parameters
    ----------
    arr : np.ndarray
        2D (rows, cols), 3D (rows, cols, depth) — depth is interleaved per
        pixel (e.g. RGB triplets) — or 4D (bands, rows, cols, depth).
    max_z_error : float
        0 = lossless. > 0 = lossy with absolute error <= this value
        per pixel. For float dtypes this is the main lossy knob.

    Returns
    -------
    bytes
        Self-describing LERC blob.
    """
    cdef:
        cnp.ndarray contig
        unsigned int data_type
        int n_depth = 1
        int n_cols
        int n_rows
        int n_bands = 1
        unsigned int needed = 0
        unsigned int written = 0
        bytes out
        unsigned char* out_ptr
        lerc_status rc

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    contig = np.ascontiguousarray(arr)

    if contig.dtype not in _DTYPE_TO_LERC:
        raise ValueError(
            f"lerc encode: unsupported dtype {contig.dtype!r}; "
            f"expected int8/uint8/int16/uint16/int32/uint32/float32/float64"
        )
    data_type = _DTYPE_TO_LERC[contig.dtype]

    # Layout: 2D = (rows, cols); 3D = (rows, cols, depth); 4D = (bands, rows, cols, depth)
    if contig.ndim == 2:
        n_rows = <int> contig.shape[0]
        n_cols = <int> contig.shape[1]
    elif contig.ndim == 3:
        n_rows = <int> contig.shape[0]
        n_cols = <int> contig.shape[1]
        n_depth = <int> contig.shape[2]
    elif contig.ndim == 4:
        n_bands = <int> contig.shape[0]
        n_rows = <int> contig.shape[1]
        n_cols = <int> contig.shape[2]
        n_depth = <int> contig.shape[3]
    else:
        raise ValueError(
            f"lerc encode: ndim must be 2/3/4, got {contig.ndim}"
        )

    cdef const void* data_ptr = <const void*> contig.data
    cdef double zerr = float(max_z_error)
    cdef Py_ssize_t raw_bytes = <Py_ssize_t> contig.nbytes
    # Skip ``lerc_computeCompressedSize`` — it runs a full encode-pass
    # internally to compute the exact size (measured at 12 ms on a
    # 4 MP u16 image, ~28% of total encode time). Instead, allocate a
    # generous upper bound and slice after. LERC's lossless blob never
    # exceeds raw + ~256 bytes of header (per-tile metadata is sub-1%
    # even on incompressible noise); 5% + 4 KiB is a hard upper bound.
    cdef unsigned int cap = <unsigned int>(
        raw_bytes + (raw_bytes // 20) + 4096
    )
    out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> cap)
    out_ptr = <unsigned char*> PyBytes_AsString(out)

    with nogil:
        rc = lerc_encode(
            data_ptr, data_type, n_depth, n_cols, n_rows, n_bands,
            0, NULL, zerr, out_ptr, cap, &written,
        )
    if rc != 0:
        # Buffer-too-small is the only realistic failure for the
        # upper-bound path. Retry with exact-size precompute.
        rc = lerc_computeCompressedSize(
            data_ptr, data_type, n_depth, n_cols, n_rows, n_bands,
            0, NULL, zerr, &needed,
        )
        if rc != 0:
            raise _err('lerc_computeCompressedSize', rc)
        out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> needed)
        out_ptr = <unsigned char*> PyBytes_AsString(out)
        with nogil:
            rc = lerc_encode(
                data_ptr, data_type, n_depth, n_cols, n_rows, n_bands,
                0, NULL, zerr, out_ptr, needed, &written,
            )
        if rc != 0:
            raise _err('lerc_encode', rc)

    # Slice to the actual written size. An earlier version used
    # ``_PyBytes_Resize`` in place to avoid a copy, but that pattern was
    # use-after-free: on shrink, realloc can move the bytes object, and
    # the Cython-managed Python ref to the pre-resize pointer would
    # decref freed memory. GuardMalloc surfaced this as an abort during
    # encode under heap-tight conditions.
    return out[:written]


def decode(data) -> 'np.ndarray':
    """Decode a LERC blob to an ndarray (shape and dtype reconstructed)."""
    cdef:
        const uint8_t[::1] src
        unsigned int blobsize
        unsigned int info[11]   # Lerc_types.h InfoArrOrder ::_last == 11
        double rng[3]
        lerc_status rc
        int n_masks
        int n_depth
        int n_cols
        int n_rows
        int n_bands
        unsigned int data_type

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    blobsize = <unsigned int> src.shape[0]
    if blobsize == 0:
        raise LercError("empty lerc blob")

    rc = lerc_getBlobInfo(<const unsigned char*> &src[0], blobsize,
                          info, rng, 11, 3)
    if rc != 0:
        raise _err('lerc_getBlobInfo', rc)

    data_type = info[1]
    n_depth = <int> info[2]
    n_cols = <int> info[3]
    n_rows = <int> info[4]
    n_bands = <int> info[5]
    n_masks = <int> info[8]

    if data_type not in _LERC_TO_DTYPE:
        raise LercError(f"lerc blob has unknown data type {data_type}")
    dtype = _LERC_TO_DTYPE[data_type]

    if n_bands > 1 and n_depth > 1:
        shape = (n_bands, n_rows, n_cols, n_depth)
    elif n_bands > 1:
        shape = (n_bands, n_rows, n_cols)
    elif n_depth > 1:
        shape = (n_rows, n_cols, n_depth)
    else:
        shape = (n_rows, n_cols)

    out = np.empty(shape, dtype=dtype)
    cdef cnp.ndarray out_arr = out

    # LERC blobs can carry validity masks (n_masks > 0). When present,
    # the decoder needs a destination buffer of n_cols*n_rows*n_masks
    # bytes; passing NULL with a masked blob returns status=2
    # (ErrCode_BufferTooSmall). Allocate locally and discard — masks
    # are useful info but our decode() contract is "decode pixel data".
    # If a future caller wants the mask, add an `out_mask=` keyword.
    cdef cnp.ndarray mask_arr = None
    cdef unsigned char* mask_ptr = NULL
    if n_masks > 0:
        mask_arr = np.empty(
            (n_masks, n_rows, n_cols), dtype=np.uint8)
        mask_ptr = <unsigned char*> mask_arr.data

    with nogil:
        rc = lerc_decode(
            <const unsigned char*> &src[0], blobsize,
            n_masks, mask_ptr,
            n_depth, n_cols, n_rows, n_bands, data_type,
            <void*> out_arr.data,
        )
    if rc != 0:
        raise _err('lerc_decode', rc)
    return out


def info(data) -> dict:
    """Return shape/dtype/value-range info for a LERC blob (no decode)."""
    cdef:
        const uint8_t[::1] src
        unsigned int blobsize
        unsigned int infoArr[11]
        double rng[3]
        lerc_status rc

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    blobsize = <unsigned int> src.shape[0]
    if blobsize == 0:
        raise LercError("empty lerc blob")

    rc = lerc_getBlobInfo(<const unsigned char*> &src[0], blobsize,
                          infoArr, rng, 11, 3)
    if rc != 0:
        raise _err('lerc_getBlobInfo', rc)

    return {
        "version": int(infoArr[0]),
        "dtype": _LERC_TO_DTYPE.get(infoArr[1], None),
        "n_depth": int(infoArr[2]),
        "n_cols": int(infoArr[3]),
        "n_rows": int(infoArr[4]),
        "n_bands": int(infoArr[5]),
        "n_valid_pixels": int(infoArr[6]),
        "blob_size": int(infoArr[7]),
        "n_masks": int(infoArr[8]),
        "z_min": float(rng[0]),
        "z_max": float(rng[1]),
        "max_z_err_used": float(rng[2]),
    }


def check_signature(data) -> bool:
    """LERC v2+ blobs start with the ASCII magic 'Lerc2 '."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:6])
    else:
        try:
            head = bytes(data)[:6]
        except Exception:
            return False
    return head.startswith(b"Lerc2 ") or head.startswith(b"CntZImag")
