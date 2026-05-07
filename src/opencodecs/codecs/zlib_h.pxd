# Minimal Cython declarations for zlib (deflate / inflate).

from libc.stdint cimport uint8_t

cdef extern from 'zlib.h' nogil:
    ctypedef unsigned long uLongf
    ctypedef unsigned long uLong
    int Z_OK
    int Z_BEST_SPEED
    int Z_BEST_COMPRESSION
    int Z_DEFAULT_COMPRESSION

    int compress2(
        uint8_t* dest, uLongf* destLen,
        const uint8_t* source, uLong sourceLen,
        int level,
    )

    int uncompress(
        uint8_t* dest, uLongf* destLen,
        const uint8_t* source, uLong sourceLen,
    )

    uLong compressBound(uLong sourceLen)
