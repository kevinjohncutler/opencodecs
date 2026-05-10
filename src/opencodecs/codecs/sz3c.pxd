# cython: language_level = 3
"""Cython header for the SZ3 C API (sz3c)."""

from libc.stddef cimport size_t


cdef extern from "SZ3c/sz3c.h" nogil:

    # Error-bound modes
    int ABS
    int REL
    int VR_REL
    int ABS_AND_REL
    int ABS_OR_REL
    int PSNR
    int NORM

    # Data types
    int SZ_FLOAT
    int SZ_DOUBLE
    int SZ_UINT8
    int SZ_INT8
    int SZ_UINT16
    int SZ_INT16
    int SZ_UINT32
    int SZ_INT32
    int SZ_UINT64
    int SZ_INT64

    unsigned char* SZ_compress_args(
        int dataType, void* data, size_t* outSize,
        int errBoundMode,
        double absErrBound, double relBoundRatio, double pwrBoundRatio,
        size_t r5, size_t r4, size_t r3, size_t r2, size_t r1,
    )

    void* SZ_decompress(
        int dataType, unsigned char* bytes_, size_t byteLength,
        size_t r5, size_t r4, size_t r3, size_t r2, size_t r1,
    )

    void free_buf(void* p)
