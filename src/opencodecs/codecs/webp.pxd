# Minimal Cython declarations for libwebp.

from libc.stdint cimport uint8_t
from libc.stddef cimport size_t

cdef extern from 'webp/encode.h' nogil:
    size_t WebPEncodeRGB(
        const uint8_t* rgb, int width, int height, int stride,
        float quality_factor, uint8_t** output,
    )
    size_t WebPEncodeRGBA(
        const uint8_t* rgba, int width, int height, int stride,
        float quality_factor, uint8_t** output,
    )
    size_t WebPEncodeLosslessRGB(
        const uint8_t* rgb, int width, int height, int stride,
        uint8_t** output,
    )
    size_t WebPEncodeLosslessRGBA(
        const uint8_t* rgba, int width, int height, int stride,
        uint8_t** output,
    )
    void WebPFree(void* ptr)

cdef extern from 'webp/decode.h' nogil:
    int WebPGetInfo(
        const uint8_t* data, size_t data_size,
        int* width, int* height,
    )

    uint8_t* WebPDecodeRGB(
        const uint8_t* data, size_t data_size,
        int* width, int* height,
    )
    uint8_t* WebPDecodeRGBA(
        const uint8_t* data, size_t data_size,
        int* width, int* height,
    )

    # Need richer feature info to know if alpha is present.
    ctypedef struct WebPBitstreamFeatures:
        int width
        int height
        int has_alpha
        int has_animation
        int format
        unsigned int[5] pad

    ctypedef enum VP8StatusCode:
        VP8_STATUS_OK
        VP8_STATUS_OUT_OF_MEMORY
        VP8_STATUS_INVALID_PARAM
        VP8_STATUS_BITSTREAM_ERROR
        VP8_STATUS_UNSUPPORTED_FEATURE
        VP8_STATUS_SUSPENDED
        VP8_STATUS_USER_ABORT
        VP8_STATUS_NOT_ENOUGH_DATA

    VP8StatusCode WebPGetFeatures(
        const uint8_t* data, size_t data_size,
        WebPBitstreamFeatures* features,
    )

