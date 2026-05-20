# opencodecs/codecs/_hcomp.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""H-compress decode — Cython binding to cfitsio's
``fits_hdecompress`` / ``fits_hdecompress64`` for use in the FITS
compressed-image reader.

H-compress (Richard White, STScI) is the older lossy-or-lossless
quadtree compressor used in legacy HST and some ground-survey FITS
data. It writes integer tiles after a Haar-like 2-D transform and
optional quantization. We only need the decode side — modern FITS
encoders prefer RICE_1 / GZIP_2 — and only for 16/32-bit signed
integer (or float-via-quantization) tiles.

The vendored cfitsio sources are at
``3rdparty/cfitsio/fits_hdecompress.c`` with a minimal
``3rdparty/cfitsio/fitsio2.h`` stub that defines the four symbols
the file needs from cfitsio internals (LONGLONG, ffpmsg, FFLOCK,
FFUNLOCK).

NOTE on axis convention: cfitsio's ``fits_hdecompress`` returns
``(ny, nx)`` where ``ny`` is the fastest-varying axis (cols in
FITS) and ``nx`` is the slowest-varying axis (rows). Callers
should reshape via ``arr.reshape(nx, ny)`` to get numpy's slow-
first order.
"""

from libc.stdlib cimport free, malloc

import numpy as np
cimport numpy as cnp


cnp.import_array()


cdef extern from "fits_hdecompress_api.h" nogil:
    int fits_hdecompress(
        unsigned char* input, int smooth, int* a, int na,
        int* ny, int* nx, int* scale, int* status,
    )
    # cfitsio uses ``LONGLONG`` (= long long in our fitsio2.h stub) for
    # the 64-bit variant. ``long long`` here matches both the C ABI
    # and Cython's strict pointer typing.
    int fits_hdecompress64 "fits_hdecompress64" (
        unsigned char* input, int smooth, long long* a, int na,
        int* ny, int* nx, int* scale, int* status,
    )


class HcompError(RuntimeError):
    """Raised on H-compress decode failures."""


def decode_raw(
    data,
    *,
    int smooth=0,
    int bytes_per_pixel=4,
    int max_pixels=16 * 1024 * 1024,
):
    """Decode a raw H-compress tile bitstream (no FITS framing).

    Parameters
    ----------
    data : bytes-like
        Compressed payload (as it appears in a BINTABLE
        ``COMPRESSED_DATA`` cell for ZCMPTYPE='HCOMPRESS_1').
    smooth : int
        0 or 1. cfitsio defaults to 0 (no smoothing). The FITS
        compressed-image spec stores the smoothing flag in
        ``ZNAME``/``ZVAL`` cards — caller pulls it out and passes
        here.
    bytes_per_pixel : int
        4 (int32, the common case) or 8 (int64). H-compress doesn't
        natively support 1/2-byte input; for ZBITPIX=16 tiles the
        FITS encoder up-casts to int32 before encoding and we cast
        back on return.
    max_pixels : int
        Safety cap on the staging buffer size — bigger tiles raise.
        Default 16 megapixels covers any tile size that makes sense
        for HCOMPRESS_1 in real FITS files.

    Returns
    -------
    (ndarray, ny, nx)
        Decoded tile as an ``int32`` or ``int64`` ndarray of length
        ``ny * nx``. Caller reshapes to ``(nx, ny)`` (slow-first,
        numpy convention).
    """
    cdef:
        const unsigned char[::1] src_mv
        unsigned char* src_ptr
        Py_ssize_t srcsize
        int* a32
        long long* a64
        cnp.ndarray out
        int* dst32
        long long* dst64
        Py_ssize_t i
        int ny = 0
        int nx = 0
        int scale = 0
        int status = 0
        int total

    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    src_mv = data
    srcsize = src_mv.shape[0]
    if srcsize < 4:
        raise HcompError("hcompress decode: input too short")
    src_ptr = <unsigned char*> &src_mv[0]

    if bytes_per_pixel == 4 or bytes_per_pixel == 2 or bytes_per_pixel == 1:
        a32 = <int*> malloc(<size_t> max_pixels * sizeof(int))
        if a32 == NULL:
            raise MemoryError("hcompress decode: malloc(int32) failed")
        try:
            with nogil:
                status = fits_hdecompress(
                    src_ptr, smooth, a32, max_pixels,
                    &ny, &nx, &scale, &status,
                )
            if status != 0:
                raise HcompError(
                    f"fits_hdecompress returned status={status}"
                )
            if ny <= 0 or nx <= 0:
                raise HcompError(
                    f"fits_hdecompress: invalid shape ({ny}, {nx})"
                )
            total = ny * nx
            out = np.empty(total, dtype=np.int32)
            dst32 = <int*> cnp.PyArray_DATA(out)
            for i in range(total):
                dst32[i] = a32[i]
        finally:
            free(a32)
        return out, ny, nx
    elif bytes_per_pixel == 8:
        a64 = <long long*> malloc(<size_t> max_pixels * sizeof(long long))
        if a64 == NULL:
            raise MemoryError("hcompress decode: malloc(int64) failed")
        try:
            with nogil:
                status = fits_hdecompress64(
                    src_ptr, smooth, <long long*> a64, max_pixels,
                    &ny, &nx, &scale, &status,
                )
            if status != 0:
                raise HcompError(
                    f"fits_hdecompress64 returned status={status}"
                )
            if ny <= 0 or nx <= 0:
                raise HcompError(
                    f"fits_hdecompress64: invalid shape ({ny}, {nx})"
                )
            total = ny * nx
            out = np.empty(total, dtype=np.int64)
            dst64 = <long long*> cnp.PyArray_DATA(out)
            for i in range(total):
                dst64[i] = a64[i]
        finally:
            free(a64)
        return out, ny, nx
    else:
        raise HcompError(
            f"hcompress decode: bytes_per_pixel must be 1/2/4/8, "
            f"got {bytes_per_pixel}"
        )
