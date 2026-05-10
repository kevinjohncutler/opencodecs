# cython: language_level = 3
"""Cython header for the vendored bitshuffle C library."""

from libc.stdint cimport int64_t


cdef extern from "bitshuffle_core.h" nogil:

    int BSHUF_VERSION_MAJOR
    int BSHUF_VERSION_MINOR
    int BSHUF_VERSION_POINT

    size_t bshuf_default_block_size(const size_t elem_size)

    int64_t bshuf_bitshuffle(
        const void* in_,
        void* out,
        const size_t size,
        const size_t elem_size,
        size_t block_size,
    )

    int64_t bshuf_bitunshuffle(
        const void* in_,
        void* out,
        const size_t size,
        const size_t elem_size,
        size_t block_size,
    )


# bitshuffle.h itself wraps an LZ4-coupled API we don't link against
# in opencodecs (use the blosc2 codec for compress-with-bitshuffle).
