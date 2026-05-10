# opencodecs/codecs/_b2nd.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native Blosc2 NDim (b2nd) codec — ndarray ↔ self-describing cframe.

The b2nd format is c-blosc2's multidimensional layer. A serialized "cframe"
is a single contiguous byte buffer that encodes shape, dtype, chunk layout,
and the compressed chunks. Decoding reconstructs the ndarray from the cframe
alone — no out-of-band metadata.

Pairs naturally with:
- LZ77 codecs (zstd, lz4, blosclz) for the inner compressor
- Shuffle / bitshuffle filters
- Multidim chunking for partial-array reads (future: open-style API)
"""

from libc.stdlib cimport free
from libc.stdint cimport int8_t, int32_t, int64_t, uint8_t
from libc.string cimport strlen
from cpython.bytes cimport PyBytes_FromStringAndSize

import numpy as np
cimport numpy as cnp

from b2nd cimport (
    OC_B2ND_MAX_DIM,
    oc_b2nd_encode, oc_b2nd_inspect, oc_b2nd_release, oc_b2nd_decode,
)


cnp.import_array()


class B2ndError(RuntimeError):
    """Raised on b2nd encode/decode failures."""


_SHUFFLE_VALUES = {
    None: -1,    # explicit "no shuffle"
    "none": -1,
    "byte": 0,
    "shuffle": 0,
    True: 0,     # legacy: blosc2 "shuffle" arg
    "bit": 1,
    "bitshuffle": 1,
}


def _shuffle_to_int(shuffle):
    if shuffle in _SHUFFLE_VALUES:
        return _SHUFFLE_VALUES[shuffle]
    if shuffle is False:
        return -1
    raise ValueError(
        f"unknown shuffle value {shuffle!r}; "
        f"expected one of: 'byte', 'bit', 'none', None, True, False"
    )


def encode(arr,
           *,
           level: int = 5,
           compressor: str | None = "zstd",
           shuffle="bit") -> bytes:
    """Encode an ndarray as a self-contained b2nd cframe (bytes).

    Parameters
    ----------
    arr : np.ndarray
        Any numpy ndarray (contiguous; non-contiguous is silently copied).
    level : int
        Compression level 0..9 (default 5).
    compressor : str
        One of "zstd", "lz4", "lz4hc", "blosclz", "zlib".
    shuffle : str or None
        "bit" (bitshuffle filter), "byte" (byte shuffle), "none"/None
        for no filter.

    Returns
    -------
    bytes
        Self-describing b2nd cframe. Pass to decode() to reconstruct
        the ndarray with full shape and dtype.
    """
    cdef:
        cnp.ndarray contig
        int8_t ndim
        int64_t shape_buf[8]
        int32_t itemsize
        int level_c = int(level)
        int shuffle_c = int(_shuffle_to_int(shuffle))
        bytes dtype_bytes
        bytes comp_bytes
        const char* dtype_cstr
        const char* comp_cstr
        uint8_t* out_buf = NULL
        int64_t out_len = 0
        int rc
        bytes out

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    contig = np.ascontiguousarray(arr)
    if contig.ndim < 1 or contig.ndim > OC_B2ND_MAX_DIM:
        raise ValueError(
            f"unsupported ndim={contig.ndim}; b2nd accepts 1..{OC_B2ND_MAX_DIM}"
        )

    ndim = <int8_t> contig.ndim
    for i in range(contig.ndim):
        shape_buf[i] = <int64_t> contig.shape[i]
    itemsize = <int32_t> contig.dtype.itemsize

    dtype_bytes = contig.dtype.str.encode("ascii")
    # Implicit-assign converts bytes -> char* via PyBytes_AsString.
    # (`<const char*> bytes_obj` would yield the PyObject pointer.)
    dtype_cstr = dtype_bytes

    if compressor is None:
        comp_cstr = NULL
    else:
        comp_bytes = compressor.encode("ascii")
        comp_cstr = comp_bytes

    cdef Py_ssize_t data_size = <Py_ssize_t> contig.nbytes
    cdef const void* data_ptr = <const void*> contig.data

    with nogil:
        rc = oc_b2nd_encode(
            data_ptr, data_size,
            ndim, shape_buf,
            itemsize, dtype_cstr,
            level_c, comp_cstr, shuffle_c,
            &out_buf, &out_len,
        )
    if rc != 0:
        raise B2ndError(f"oc_b2nd_encode failed: blosc2 error {rc}")
    try:
        out = PyBytes_FromStringAndSize(<const char*> out_buf, <Py_ssize_t> out_len)
    finally:
        free(out_buf)
    return out


def decode(data) -> np.ndarray:
    """Decode a b2nd cframe back to a fully-typed ndarray."""
    cdef:
        const uint8_t[::1] src
        int8_t ndim = 0
        int64_t shape_buf[8]
        int32_t itemsize = 0
        char* dtype_cstr = NULL
        void* handle = NULL
        int rc

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    cdef Py_ssize_t srclen = src.shape[0]
    if srclen == 0:
        raise B2ndError("empty b2nd cframe")

    rc = oc_b2nd_inspect(
        <const void*> &src[0], <int64_t> srclen,
        &ndim, shape_buf, &itemsize, &dtype_cstr, &handle,
    )
    if rc != 0:
        if handle != NULL:
            oc_b2nd_release(handle)
        raise B2ndError(f"oc_b2nd_inspect failed: blosc2 error {rc}")

    try:
        py_shape = tuple(int(shape_buf[i]) for i in range(ndim))
        if dtype_cstr != NULL and strlen(dtype_cstr) > 0:
            dtype_str = (<bytes> dtype_cstr).decode("ascii", "replace")
            try:
                dtype = np.dtype(dtype_str)
            except TypeError:
                dtype = np.dtype((np.uint8, itemsize))
        else:
            dtype = np.dtype((np.uint8, itemsize))
    finally:
        oc_b2nd_release(handle)

    out = np.empty(py_shape, dtype=dtype)
    cdef cnp.ndarray out_arr = out
    cdef int64_t out_size = <int64_t> out.nbytes

    with nogil:
        rc = oc_b2nd_decode(
            <const void*> &src[0], <int64_t> srclen,
            <void*> out_arr.data, out_size,
        )
    if rc != 0:
        raise B2ndError(f"oc_b2nd_decode failed: blosc2 error {rc}")
    return out


def inspect(data) -> dict:
    """Return shape, dtype, itemsize, ndim from a cframe (no decompression)."""
    cdef:
        const uint8_t[::1] src
        int8_t ndim = 0
        int64_t shape_buf[8]
        int32_t itemsize = 0
        char* dtype_cstr = NULL
        void* handle = NULL
        int rc

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    cdef Py_ssize_t srclen = src.shape[0]
    if srclen == 0:
        raise B2ndError("empty b2nd cframe")

    rc = oc_b2nd_inspect(
        <const void*> &src[0], <int64_t> srclen,
        &ndim, shape_buf, &itemsize, &dtype_cstr, &handle,
    )
    if rc != 0:
        if handle != NULL:
            oc_b2nd_release(handle)
        raise B2ndError(f"oc_b2nd_inspect failed: blosc2 error {rc}")

    try:
        py_shape = tuple(int(shape_buf[i]) for i in range(ndim))
        if dtype_cstr != NULL and strlen(dtype_cstr) > 0:
            dtype_str = (<bytes> dtype_cstr).decode("ascii", "replace")
        else:
            dtype_str = ""
    finally:
        oc_b2nd_release(handle)

    return {
        "ndim": int(ndim),
        "shape": py_shape,
        "itemsize": int(itemsize),
        "dtype": dtype_str,
    }


def check_signature(data) -> bool:
    """A b2nd cframe starts with the blosc2 magic 0x62 0x32 ("b2"). Approx."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:2])
    else:
        try:
            head = bytes(data)[:2]
        except Exception:
            return False
    # b2nd cframe magic: first byte is the blosc2 frame magic byte 0x62.
    return len(head) >= 1 and head[:1] == b"\x62"
