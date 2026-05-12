# Cython declarations for the OpenJPH HTJ2K shim.

from libc.stddef cimport size_t


cdef extern from "openjph_shim.h" nogil:
    int opencodecs_htj2k_encode(
        const void* src,
        int width,
        int height,
        int components,
        int bit_depth,
        int is_signed,
        int bytes_per_sample,
        int reversible,
        float irrev_delta,
        int num_decomp,
        void** out_buf,
        size_t* out_size,
    )

    int opencodecs_htj2k_decode_info(
        const void* src,
        size_t srcsize,
        int* width,
        int* height,
        int* components,
        int* bit_depth,
        int* is_signed,
    )

    int opencodecs_htj2k_decode(
        const void* src,
        size_t srcsize,
        void* dst,
        size_t dst_size,
        int bytes_per_sample,
    )

    void opencodecs_htj2k_free(void* buf)
    const char* opencodecs_htj2k_last_error()
