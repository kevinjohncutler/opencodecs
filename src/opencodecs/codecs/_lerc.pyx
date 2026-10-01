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

Codec version
=============

Blobs are written as Lerc2 codec version 4 (v2.4) unless ``version=``
asks for another. That is the version imagecodecs writes by default and
the one libtiff's LERC codec (TIFF compression 34887, as in GDAL)
fixes for TIFF, where a different version draws an "Unexpected version
number" warning. Version 6 (v2.6, libLerc 4.0 and later) adds lossless
float compression that is often smaller; readers built on older libLerc
cannot open it, so it is opt-in. Every version decodes here.

Outer compression
=================

``compression='zstd'`` or ``'deflate'`` wraps the blob the way
imagecodecs and TIFF LercParameters do (Lerc then zstd / zlib), and
decode recognizes either wrapper by its magic bytes.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdint cimport uint8_t

import numpy as np
cimport numpy as cnp

from lerc cimport (
    lerc_status,
    lerc_computeCompressedSizeForVersion, lerc_encodeForVersion,
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


#: Lerc2 codec version written when the caller does not choose one.
DEFAULT_VERSION = 4


def _resolve_max_z_error(max_z_error, level):
    """``level`` is imagecodecs' name for the error bound; merge the two."""
    if level is not None:
        if max_z_error is not None and float(max_z_error) != float(level):
            raise ValueError(
                f"lerc encode: level={level} and max_z_error={max_z_error} "
                f"disagree; pass one of them")
        max_z_error = level
    if max_z_error is None:
        return 0.0
    return max(0.0, float(max_z_error))


def encode(arr, *, max_z_error=None, level=None, version=None, masks=None,
           planar=None, compression=None, compressionargs=None) -> bytes:
    """LERC-encode an ndarray.

    Parameters
    ----------
    arr : np.ndarray
        1D (cols), coded as one row; 2D (rows, cols); 3D (rows, cols,
        depth), depth interleaved per pixel (e.g. RGB triplets), or with
        ``planar=True`` (bands, rows, cols); 4D (bands, rows, cols, depth).
    max_z_error : float
        0 (default) = lossless. > 0 = lossy with absolute error <= this
        value per pixel. For float dtypes this is the main lossy knob.
    level : float, optional
        imagecodecs' name for ``max_z_error``.
    version : int, optional
        Lerc2 codec version, 2 to 6, or -1 for the latest. Default 4,
        the version imagecodecs and libtiff write (see module docs).
    masks : bool ndarray, optional
        Valid-pixel masks, (rows, cols) or (n_masks, rows, cols); False
        marks an invalid pixel. Invalid pixels decode as 0.
    planar : bool, optional
        Read a 3D array as (bands, rows, cols) instead of (rows, cols,
        depth).
    compression : {None, 'zstd', 'deflate'}
        Outer compression of the blob, as imagecodecs and TIFF
        LercParameters use it. ``compressionargs`` goes to that codec;
        for deflate it may hold only ``level``, clamped to -1..9 as
        imagecodecs clamps it.

    Returns
    -------
    bytes
        Self-describing LERC blob.
    """
    cdef:
        cnp.ndarray contig
        cnp.ndarray msk
        unsigned int data_type
        int n_depth = 1
        int n_cols
        int n_rows = 1
        int n_bands = 1
        int n_masks = 0
        int iversion
        const unsigned char* valid_ptr = NULL
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
    iversion = DEFAULT_VERSION if version is None else int(version)
    if iversion != -1 and not 2 <= iversion <= 6:
        raise ValueError(
            f"lerc encode: version must be 2..6 or -1 (got {version})")

    # Layout: 1D = (cols), one row; 2D = (rows, cols); 3D = (rows, cols,
    # depth) or, planar, (bands, rows, cols); 4D = (bands, rows, cols, depth)
    if contig.ndim == 1:
        n_cols = <int> contig.shape[0]
    elif contig.ndim == 2:
        n_rows = <int> contig.shape[0]
        n_cols = <int> contig.shape[1]
    elif contig.ndim == 3 and planar:
        n_bands = <int> contig.shape[0]
        n_rows = <int> contig.shape[1]
        n_cols = <int> contig.shape[2]
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
            f"lerc encode: ndim must be 1/2/3/4, got {contig.ndim}"
        )

    if masks is not None:
        msk = np.ascontiguousarray(masks)
        if msk.dtype != np.bool_:
            raise ValueError("lerc encode: masks must be a bool array")
        mshape = tuple((<object> msk).shape)
        if len(mshape) == 2 and mshape == (n_rows, n_cols):
            n_masks = 1
        elif len(mshape) == 3 and mshape[1:] == (n_rows, n_cols):
            n_masks = <int> mshape[0]
        else:
            raise ValueError(
                f"lerc encode: masks shape {mshape} does not match "
                f"{n_rows} rows x {n_cols} cols")
        if n_masks != 1 and n_masks != n_bands:
            raise ValueError(
                f"lerc encode: {n_masks} masks for {n_bands} bands; "
                f"give one mask or one per band")
        valid_ptr = <const unsigned char*> msk.data

    cdef const void* data_ptr = <const void*> contig.data
    cdef double zerr = _resolve_max_z_error(max_z_error, level)
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
        rc = lerc_encodeForVersion(
            data_ptr, iversion, data_type, n_depth, n_cols, n_rows, n_bands,
            n_masks, valid_ptr, zerr, out_ptr, cap, &written,
        )
    if rc != 0:
        # Buffer-too-small is the only realistic failure for the
        # upper-bound path. Retry with exact-size precompute.
        rc = lerc_computeCompressedSizeForVersion(
            data_ptr, iversion, data_type, n_depth, n_cols, n_rows, n_bands,
            n_masks, valid_ptr, zerr, &needed,
        )
        if rc != 0:
            raise _err('lerc_computeCompressedSizeForVersion', rc)
        out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> needed)
        out_ptr = <unsigned char*> PyBytes_AsString(out)
        with nogil:
            rc = lerc_encodeForVersion(
                data_ptr, iversion, data_type, n_depth, n_cols, n_rows,
                n_bands, n_masks, valid_ptr, zerr, out_ptr, needed, &written,
            )
        if rc != 0:
            raise _err('lerc_encodeForVersion', rc)

    # Slice to the actual written size. An earlier version used
    # ``_PyBytes_Resize`` in place to avoid a copy, but that pattern was
    # use-after-free: on shrink, realloc can move the bytes object, and
    # the Cython-managed Python ref to the pre-resize pointer would
    # decref freed memory. GuardMalloc surfaced this as an abort during
    # encode under heap-tight conditions.
    out = out[:written]
    if compression is None:
        return out
    return _outer_compress(out, compression, compressionargs)


def _outer_compress(bytes blob, compression, compressionargs):
    kwargs = dict(compressionargs or {})
    name = str(compression).lower()
    if name == 'zstd':
        from opencodecs import get_codec
        return bytes(get_codec('zstd').encode(blob, **kwargs))
    if name in ('deflate', 'zlib'):
        import zlib
        # imagecodecs' zlib_encode takes only ``level`` and clamps it to
        # zlib's range (-1, the default, to 9) instead of failing.
        level = kwargs.pop('level', None)
        if kwargs:
            raise TypeError(
                f"lerc encode: compressionargs {sorted(kwargs)} not "
                f"supported for deflate (only 'level')")
        level = -1 if level is None else max(-1, min(9, int(level)))
        return zlib.compress(blob, level)
    raise ValueError(
        f"lerc encode: compression={compression!r} not supported "
        f"(None, 'zstd' or 'deflate')")


def _outer_decompress(data):
    """Undo a zstd or zlib wrapper around a Lerc blob, if there is one."""
    head = bytes(data[:4])
    if head == b'\x28\xb5\x2f\xfd':
        from opencodecs import get_codec
        return bytes(get_codec('zstd').decode(bytes(data)))
    if len(head) >= 2 and head[0] == 0x78 and (head[0] * 256 + head[1]) % 31 == 0:
        import zlib
        return zlib.decompress(bytes(data))
    return data


def decode(data, *, out=None, masks=None):
    """Decode a LERC blob to an ndarray (shape and dtype reconstructed).

    A blob wrapped in zstd or zlib (imagecodecs' ``compression=``, TIFF
    LercParameters) is unwrapped first. Invalid pixels, where the blob
    carries a mask, decode as 0.

    Parameters
    ----------
    out : np.ndarray | None, optional
        Preallocated output array. Must match the shape and dtype
        encoded in the LERC blob — same contract as ``_png.decode``.
        ``lerc_decode`` writes directly into the provided buffer
        when out= is supplied (true zero-alloc fast path).
    masks : bool or ndarray, optional
        As imagecodecs: None/False returns the array alone. True returns
        ``(array, masks)``, where masks is a bool (rows, cols) or
        (n_masks, rows, cols) array, or None when the blob has no mask.
        A bool ndarray receives the mask in place and is returned the
        same way.
    """
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
        int n_valid
        unsigned int data_type

    data = _outer_decompress(data)
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
    n_valid = <int> info[6]
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

    # Pixels a mask marks invalid are not written by lerc_decode; they
    # are set to 0 here so their value does not depend on the buffer.
    cdef bint has_invalid = n_masks > 1 or (
        n_masks == 1 and n_valid != n_rows * n_cols)
    cdef cnp.ndarray out_arr
    if out is not None:
        if not isinstance(out, np.ndarray):
            raise TypeError(
                f"lerc decode: out= must be an ndarray, "
                f"got {type(out).__name__}")
        if out.shape != shape:
            raise ValueError(
                f"lerc decode: out= shape {out.shape} does not match "
                f"expected {shape}")
        if out.dtype != dtype:
            raise ValueError(
                f"lerc decode: out= dtype {out.dtype} does not match "
                f"expected {dtype}")
        if not out.flags['C_CONTIGUOUS']:
            raise ValueError("lerc decode: out= must be C-contiguous")
        out_arr = out
        if has_invalid:
            out_arr[...] = 0
    elif has_invalid:
        out_arr = np.zeros(shape, dtype=dtype)
    else:
        out_arr = np.empty(shape, dtype=dtype)

    # LERC blobs can carry validity masks (n_masks > 0). When present,
    # the decoder needs a destination buffer of n_cols*n_rows*n_masks
    # bytes; passing NULL with a masked blob returns status=2
    # (ErrCode_BufferTooSmall).
    cdef cnp.ndarray mask_arr = None
    cdef unsigned char* mask_ptr = NULL
    if n_masks == 1:
        mask_shape = (int(n_rows), int(n_cols))
    else:
        mask_shape = (int(n_masks), int(n_rows), int(n_cols))
    if n_masks > 0:
        if isinstance(masks, np.ndarray):
            if (masks.shape != mask_shape or masks.dtype != np.bool_
                    or not masks.flags['C_CONTIGUOUS']):
                raise ValueError(
                    f"lerc decode: masks= must be a C-contiguous bool "
                    f"array of shape {mask_shape}")
            mask_arr = masks
        else:
            mask_arr = np.empty(mask_shape, dtype=np.bool_)
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
    if masks is None or masks is False:
        return out_arr
    return out_arr, mask_arr


def info(data) -> dict:
    """Return shape/dtype/value-range info for a LERC blob (no decode)."""
    cdef:
        const uint8_t[::1] src
        unsigned int blobsize
        unsigned int infoArr[11]
        double rng[3]
        lerc_status rc

    data = _outer_decompress(data)
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
