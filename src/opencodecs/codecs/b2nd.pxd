# cython: language_level = 3
"""Cython header for the small oc_b2nd_helpers shim around c-blosc2's b2nd."""

from libc.stdint cimport int8_t, int32_t, int64_t, uint8_t


cdef extern from "b2nd_helpers.h" nogil:

    int OC_B2ND_MAX_DIM

    int oc_b2nd_encode(
        const void* data,
        int64_t data_size,
        int8_t ndim,
        const int64_t* shape,
        int32_t itemsize,
        const char* dtype_str,
        int level,
        const char* compressor,
        int do_bitshuffle,
        uint8_t** out_cframe,
        int64_t* out_cframe_len,
    )

    int oc_b2nd_inspect(
        const void* cframe,
        int64_t cframe_len,
        int8_t* out_ndim,
        int64_t* out_shape,
        int32_t* out_itemsize,
        char** out_dtype,
        void** handle,
    )

    void oc_b2nd_release(void* handle)

    int oc_b2nd_decode(
        const void* cframe,
        int64_t cframe_len,
        void* dest_buffer,
        int64_t dest_buffer_size,
    )
