# Minimal Cython declarations for libspng.

from libc.stdint cimport uint8_t, uint32_t

cdef extern from 'spng.h' nogil:
    ctypedef struct spng_ctx:
        pass

    cdef enum spng_color_type:
        SPNG_COLOR_TYPE_GRAYSCALE = 0
        SPNG_COLOR_TYPE_TRUECOLOR = 2
        SPNG_COLOR_TYPE_INDEXED = 3
        SPNG_COLOR_TYPE_GRAYSCALE_ALPHA = 4
        SPNG_COLOR_TYPE_TRUECOLOR_ALPHA = 6

    cdef enum spng_format:
        SPNG_FMT_RGBA8 = 1
        SPNG_FMT_RGBA16 = 2
        SPNG_FMT_RGB8 = 4
        SPNG_FMT_GA8 = 16
        SPNG_FMT_GA16 = 32
        SPNG_FMT_G8 = 64
        SPNG_FMT_PNG = 256
        SPNG_FMT_RAW = 512

    cdef enum spng_ctx_flags:
        SPNG_CTX_IGNORE_ADLER32 = 1
        SPNG_CTX_ENCODER = 2

    cdef enum spng_encode_flags:
        SPNG_ENCODE_PROGRESSIVE = 1
        SPNG_ENCODE_FINALIZE = 2

    cdef enum spng_option:
        SPNG_KEEP_UNKNOWN_CHUNKS = 1
        SPNG_IMG_COMPRESSION_LEVEL
        SPNG_IMG_WINDOW_BITS
        SPNG_IMG_MEM_LEVEL
        SPNG_IMG_COMPRESSION_STRATEGY
        SPNG_TEXT_COMPRESSION_LEVEL
        SPNG_TEXT_WINDOW_BITS
        SPNG_TEXT_MEM_LEVEL
        SPNG_TEXT_COMPRESSION_STRATEGY
        SPNG_FILTER_CHOICE
        SPNG_CHUNK_COUNT_LIMIT
        SPNG_ENCODE_TO_BUFFER

    ctypedef struct spng_ihdr "struct spng_ihdr":
        uint32_t width
        uint32_t height
        uint8_t bit_depth
        uint8_t color_type
        uint8_t compression_method
        uint8_t filter_method
        uint8_t interlace_method

    spng_ctx* spng_ctx_new(int flags)
    void spng_ctx_free(spng_ctx* ctx)
    int spng_set_png_buffer(spng_ctx* ctx, const void* buf, size_t size)
    void* spng_get_png_buffer(spng_ctx* ctx, size_t* length, int* error)
    int spng_set_image_limits(
        spng_ctx* ctx, uint32_t width, uint32_t height)
    int spng_set_chunk_limits(
        spng_ctx* ctx, size_t chunk_size, size_t cache_size)
    int spng_set_option(spng_ctx* ctx, spng_option option, int value)
    int spng_decoded_image_size(
        spng_ctx* ctx, int fmt, size_t* len)
    int spng_decode_image(
        spng_ctx* ctx, void* out, size_t length, int fmt, int flags)
    int spng_encode_image(
        spng_ctx* ctx, const void* img, size_t length, int fmt, int flags)
    int spng_get_ihdr(spng_ctx* ctx, spng_ihdr* ihdr)
    int spng_set_ihdr(spng_ctx* ctx, spng_ihdr* ihdr)
    const char* spng_strerror(int err)
