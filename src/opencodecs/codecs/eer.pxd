# opencodecs/codecs/eer.pxd
# cython: language_level = 3
"""Cython declarations for opencodecs' EER decoder (3rdparty/oc_eer)."""

from libc.stdint cimport uint8_t, uint16_t


cdef extern from "oc_eer.h" nogil:

    ptrdiff_t oc_eer_decode_u8(
        const uint8_t* src, size_t srcsize,
        uint8_t* dst, size_t height, size_t width,
        unsigned skipbits, unsigned horzbits, unsigned vertbits,
        unsigned superres,
    )

    ptrdiff_t oc_eer_decode_u16(
        const uint8_t* src, size_t srcsize,
        uint16_t* dst, size_t height, size_t width,
        unsigned skipbits, unsigned horzbits, unsigned vertbits,
        unsigned superres,
    )
