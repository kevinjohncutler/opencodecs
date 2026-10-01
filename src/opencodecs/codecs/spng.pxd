# Minimal Cython declarations for libspng.

from libc.stdint cimport uint8_t, uint16_t, uint32_t

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

    # filter_choice values are a bitmask of these. Pass to
    # spng_set_option(ctx, SPNG_FILTER_CHOICE, <bitmask>).
    cdef enum spng_filter_choice:
        SPNG_DISABLE_FILTERING = 0
        SPNG_FILTER_CHOICE_NONE = 8
        SPNG_FILTER_CHOICE_SUB = 16
        SPNG_FILTER_CHOICE_UP = 32
        SPNG_FILTER_CHOICE_AVG = 64
        SPNG_FILTER_CHOICE_PAETH = 128
        SPNG_FILTER_CHOICE_ALL = 248    # 8|16|32|64|128

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

    # ICC profile chunk (iCCP). The profile_name slot is a 79-char
    # null-terminated identifier; libspng copies what the caller
    # provides on set_iccp and returns the in-stream name on get_iccp.
    ctypedef struct spng_iccp "struct spng_iccp":
        char profile_name[80]
        size_t profile_len
        char* profile

    int spng_get_iccp(spng_ctx* ctx, spng_iccp* iccp)
    int spng_set_iccp(spng_ctx* ctx, spng_iccp* iccp)

    const char* spng_strerror(int err)

    cdef enum spng_decode_flags:
        SPNG_DECODE_TRNS
        SPNG_DECODE_PROGRESSIVE
    cdef enum spng_errno:
        SPNG_EOI
        SPNG_ECHUNKAVAIL

    # Transparency chunk (tRNS). Only its presence is used: libspng
    # applies it itself when SPNG_DECODE_TRNS is passed to decode.
    ctypedef struct spng_trns "struct spng_trns":
        uint16_t gray
        uint16_t red
        uint16_t green
        uint16_t blue
        uint32_t n_type3_entries
        uint8_t type3_alpha[256]

    int spng_get_trns(spng_ctx* ctx, spng_trns* trns)
    ctypedef struct spng_row_info "struct spng_row_info":
        uint32_t scanline_idx
        uint32_t row_num
        int pass_num "pass"
        uint8_t filter
    ctypedef int spng_rw_fn(spng_ctx* ctx, void* user, void* data, size_t length) noexcept
    int spng_set_png_stream(spng_ctx* ctx, spng_rw_fn* callback, void* user)
    int spng_get_row_info(spng_ctx* ctx, spng_row_info* row_info)
    int spng_decode_scanline(spng_ctx* ctx, void* out, size_t length)
    int spng_decode_chunks(spng_ctx* ctx)
    int spng_encode_row(spng_ctx* ctx, const void* row, size_t length)
