# Minimal Cython declarations for c-blosc2.

# These declarations transcribe upstream 'blosc2.h' and therefore overlap
# substantially with imagecodecs/blosc2.pxd (BSD-3-Clause, Copyright (c)
# 2008-2026 Christoph Gohlke), which transcribes the same header.
# Credited here out of caution; see THIRD-PARTY.md.

from libc.stdint cimport int32_t

cdef extern from 'blosc2.h' nogil:
    int BLOSC2_MAX_OVERHEAD
    int BLOSC_NOSHUFFLE
    int BLOSC_SHUFFLE
    int BLOSC_BITSHUFFLE

    void blosc2_init()
    void blosc2_destroy()

    int blosc2_compress(
        int clevel, int doshuffle, int32_t typesize,
        const void* src, int32_t srcsize,
        void* dest, int32_t destsize,
    )

    int blosc2_decompress(
        const void* src, int32_t srcsize,
        void* dest, int32_t destsize,
    )

    int blosc2_cbuffer_sizes(
        const void* cbuffer,
        int32_t* nbytes, int32_t* cbytes, int32_t* blocksize,
    )

    int blosc1_set_compressor(const char* compname)
    const char* blosc1_get_compressor()
