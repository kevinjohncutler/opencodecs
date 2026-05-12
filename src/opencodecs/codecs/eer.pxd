# Cython declarations for the vendored EER decoder shim.

from libc.stdint cimport uint8_t, uint16_t, uint32_t


cdef extern from "eer.h" nogil:
    int EER_OK
    int EER_VALUE_ERROR
    int EER_INPUT_CORRUPT
    int EER_OUTPUT_TOO_SMALL

    ssize_t opencodecs_eer_decode_u1(
        const uint8_t* src, ssize_t srcsize,
        uint8_t* dst, ssize_t height, ssize_t width,
        uint32_t skipbits, uint32_t horzbits, uint32_t vertbits,
        uint32_t superres,
    )

    ssize_t opencodecs_eer_decode_u2(
        const uint8_t* src, ssize_t srcsize,
        uint16_t* dst, ssize_t height, ssize_t width,
        uint32_t skipbits, uint32_t horzbits, uint32_t vertbits,
        uint32_t superres,
    )
