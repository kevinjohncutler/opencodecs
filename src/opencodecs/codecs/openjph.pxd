# Cython declarations for the OpenJPH HTJ2K shim.

from libc.stddef cimport size_t


cdef extern from "openjph_shim.h" nogil:
    ctypedef struct opencodecs_htj2k_encode_params:
        int width
        int height
        int components
        int bit_depth
        int is_signed
        int bytes_per_sample
        int src_planar
        int reversible
        float irrev_delta
        int qfactor
        int num_decomp
        int color_transform
        int nlt_binary_complement
        int tile_w
        int tile_h
        int tlm
        int tilepart_resolutions
        int tilepart_components
        int block_w
        int block_h
        const char* prog_order
        const char* profile

    ctypedef struct opencodecs_htj2k_info:
        int width
        int height
        int components
        int bit_depth
        int is_signed
        int num_decompositions
        int color_transform
        int nlt_type
        int uniform

    int opencodecs_htj2k_encode(
        const void* src,
        const opencodecs_htj2k_encode_params* params,
        void** out_buf,
        size_t* out_size,
    )

    int opencodecs_htj2k_decode_info(
        const void* src,
        size_t srcsize,
        int reduce_data,
        int reduce_recon,
        int resilient,
        opencodecs_htj2k_info* info,
    )

    int opencodecs_htj2k_decode(
        const void* src,
        size_t srcsize,
        void* dst,
        size_t dst_size,
        int bytes_per_sample,
        int reduce_data,
        int reduce_recon,
        int resilient,
        int dst_planar,
    )

    void opencodecs_htj2k_free(void* buf)
    const char* opencodecs_htj2k_last_error()
    const char* opencodecs_htj2k_last_warnings()
    void opencodecs_htj2k_clear_warnings()
