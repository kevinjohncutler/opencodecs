# opencodecs/codecs/_deflate.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native zlib / deflate codec — bytes-in / bytes-out compression.

Uses the zlib `compress2` and `uncompress` helpers, which produce/consume
zlib-format streams (deflate data with a 2-byte header + 4-byte adler32).
This matches imagecodecs's ``zlib_encode`` / ``zlib_decode``.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdint cimport uint8_t
from libc.stdlib cimport realloc, free

from zlib_h cimport (
    Z_OK, Z_DEFAULT_COMPRESSION,
    compress2, uncompress, compressBound,
    uLongf, uLong,
)


class ZlibError(RuntimeError):
    """Raised on zlib encode/decode failures."""


def encode(data, *, level: int | None = None) -> bytes:
    """Encode bytes-like input as a zlib stream."""
    cdef:
        const uint8_t[::1] src
        uLong srcsize
        uLongf dstsize
        int rc, lvl
        bytes out
        const uint8_t* src_ptr = NULL
        uint8_t* dst_ptr

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <uLong> src.shape[0]

    if level is None:
        # Z_DEFAULT_COMPRESSION (-1) is a sentinel; zlib maps it to its
        # internal default (level 6). Don't clamp it to 0 — that means
        # *uncompressed* and silently produces output larger than input.
        lvl = Z_DEFAULT_COMPRESSION
    else:
        lvl = int(level)
        if lvl < 0: lvl = 0
        if lvl > 9: lvl = 9

    dstsize = compressBound(srcsize)
    out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dstsize)
    dst_ptr = <uint8_t*> PyBytes_AsString(out)
    if srcsize > 0:
        src_ptr = <const uint8_t*> &src[0]

    with nogil:
        rc = compress2(dst_ptr, &dstsize, src_ptr, srcsize, lvl)
    if rc != Z_OK:
        raise ZlibError(f'compress2 failed: {rc}')
    return out[:dstsize]


def decode(data) -> bytes:
    """Decode a zlib stream to bytes."""
    cdef:
        const uint8_t[::1] src
        uLong srcsize
        uLongf dstcap, dstsize
        int rc
        uint8_t* buf = NULL
        bytes out

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <uLong> src.shape[0]
    if srcsize == 0:
        return b''

    # Start at 4x source; grow on Z_BUF_ERROR (== -5).
    dstcap = max(<uLong> 4 * srcsize, <uLong> 65536)
    while True:
        buf = <uint8_t*> realloc(buf, dstcap)
        if buf == NULL:
            raise MemoryError()
        dstsize = dstcap
        with nogil:
            rc = uncompress(buf, &dstsize, &src[0], srcsize)
        if rc == Z_OK:
            try:
                out = PyBytes_FromStringAndSize(<char*> buf, <Py_ssize_t> dstsize)
                return out
            finally:
                free(buf)
        # Z_BUF_ERROR (-5) means the output buffer was too small. Grow and
        # retry. Cap at 1 GiB so a wildly wrong stream can't loop forever;
        # also `uLong` is 32-bit on MSVC, so any literal larger than 2^31
        # overflows to 0 in the cast and breaks the bound.
        if rc == -5 and dstcap < <uLong>(1 << 30):
            dstcap *= 2
            continue
        free(buf)
        raise ZlibError(f'uncompress failed: {rc}')


def check_signature(data) -> bool:
    """True if `data` looks like a zlib stream (CMF byte 0x78 most common)."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:2])
    else:
        try:
            head = bytes(data)[:2]
        except Exception:
            return False
    if len(head) < 2:
        return False
    # zlib: CMF (0x78 typical) + FLG; (CMF*256 + FLG) % 31 == 0.
    return (head[0] & 0x0F) == 0x08 and ((head[0] * 256 + head[1]) % 31 == 0)
