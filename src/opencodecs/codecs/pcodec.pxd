# cython: language_level = 3
"""Cython header for pcodec's C API (cpcodec)."""

# Transcribed from upstream 'cpcodec.h'. imagecodecs declares the same C
# API in its own pcodec.pxd; the two overlap because the header fixes the
# names and signatures, not because either was copied from the other.

from libc.stddef cimport size_t


cdef extern from "cpcodec.h" nogil:

    # Dtype enum values (from cpcodec.h #define).
    int PCO_TYPE_U32
    int PCO_TYPE_U64
    int PCO_TYPE_I32
    int PCO_TYPE_I64
    int PCO_TYPE_F32
    int PCO_TYPE_F64
    int PCO_TYPE_U16
    int PCO_TYPE_I16
    int PCO_TYPE_F16
    int PCO_TYPE_U8
    int PCO_TYPE_I8

    ctypedef enum PcoError:
        PcoSuccess
        PcoInvalidType
        PcoCompressionError
        PcoDecompressionError

    ctypedef struct PcoChunkConfig:
        unsigned int compression_level
        size_t max_page_n

    size_t pco_standalone_guarantee_file_size(size_t n, unsigned char dtype)

    PcoError pco_standalone_simple_compress_into(
        const void* nums,
        size_t n,
        unsigned char dtype,
        const PcoChunkConfig* config,
        void* dst,
        size_t dst_cap,
        size_t* n_written,
    )

    PcoError pco_standalone_simple_decompress_into(
        const void* compressed,
        size_t compressed_len,
        unsigned char dtype,
        void* dst,
        size_t dst_cap,
        size_t* n_written,
    )
