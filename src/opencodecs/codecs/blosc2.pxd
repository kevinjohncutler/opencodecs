# Minimal Cython declarations for c-blosc2.

# Transcribed from upstream 'blosc2.h'. imagecodecs declares the same C
# API in its own blosc2.pxd; the two overlap because the header fixes the
# names and signatures, not because either was copied from the other.

from libc.stddef cimport size_t
from libc.stdint cimport int16_t, int32_t

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

    # Partial decompression: pull items [start, start+nitems) out of a
    # compressed buffer without expanding the rest. Blosc2 splits a
    # chunk into blocks and only touches the blocks the range covers,
    # which is what makes a blosc2 buffer randomly addressable at all.
    # The chunk carries its own typesize; a caller cannot override it,
    # because blosc2_getitem interprets start/nitems in units of the
    # stored value. Reading it back is the only way to size the
    # destination correctly.
    void blosc1_cbuffer_metainfo(
        const void* cbuffer, size_t* typesize, int* flags,
    )

    int blosc2_getitem(
        const void* src, int32_t srcsize, int start, int nitems,
        void* dest, int32_t destsize,
    )

    # Context API. The nthreads knob lives on a context rather than in
    # a global, and that matters: blosc2_set_nthreads() is process-wide
    # state, so a library that used it would be changing the setting
    # under everyone else in the process.
    ctypedef struct blosc2_context:
        pass

    ctypedef struct blosc2_dparams:
        int16_t nthreads
        void* schunk
        void* postfilter
        void* postparams
        int32_t typesize

    blosc2_context* blosc2_create_dctx(blosc2_dparams dparams)
    void blosc2_free_ctx(blosc2_context* context)

    int blosc2_decompress_ctx(
        blosc2_context* context, const void* src, int32_t srcsize,
        void* dest, int32_t destsize,
    )

    int blosc2_getitem_ctx(
        blosc2_context* context, const void* src, int32_t srcsize,
        int start, int nitems, void* dest, int32_t destsize,
    )
