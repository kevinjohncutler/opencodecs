# Cython declarations for CharLS (libcharls) — C JPEG-LS API.

from libc.stdint cimport uint8_t, int32_t, uint32_t
from libc.stddef cimport size_t


cdef extern from "charls/charls.h" nogil:
    ctypedef struct charls_jpegls_encoder:
        pass
    ctypedef struct charls_jpegls_decoder:
        pass

    cdef struct charls_frame_info:
        uint32_t width
        uint32_t height
        int32_t bits_per_sample
        int32_t component_count

    # jpegls_errc enum — we treat any non-zero as error.
    ctypedef int charls_jpegls_errc

    # Interleave-mode enum (RGB encoding layout).
    cdef enum:
        CHARLS_INTERLEAVE_MODE_NONE = 0    # planar (RRRGGGBBB)
        CHARLS_INTERLEAVE_MODE_LINE = 1    # per-line interleaved
        CHARLS_INTERLEAVE_MODE_SAMPLE = 2  # interleaved (RGBRGBRGB)

    # Encoder
    charls_jpegls_encoder* charls_jpegls_encoder_create()
    void charls_jpegls_encoder_destroy(const charls_jpegls_encoder* enc)
    charls_jpegls_errc charls_jpegls_encoder_set_frame_info(
        charls_jpegls_encoder* enc, const charls_frame_info* info)
    charls_jpegls_errc charls_jpegls_encoder_set_near_lossless(
        charls_jpegls_encoder* enc, int32_t near_lossless)
    charls_jpegls_errc charls_jpegls_encoder_set_interleave_mode(
        charls_jpegls_encoder* enc, int mode)
    charls_jpegls_errc charls_jpegls_decoder_get_interleave_mode(
        const charls_jpegls_decoder* dec, int* mode)
    charls_jpegls_errc charls_jpegls_encoder_get_estimated_destination_size(
        const charls_jpegls_encoder* enc, size_t* size)
    charls_jpegls_errc charls_jpegls_encoder_set_destination_buffer(
        charls_jpegls_encoder* enc, void* dst, size_t size)
    charls_jpegls_errc charls_jpegls_encoder_encode_from_buffer(
        charls_jpegls_encoder* enc, const void* src, size_t src_size,
        uint32_t stride)
    charls_jpegls_errc charls_jpegls_encoder_get_bytes_written(
        const charls_jpegls_encoder* enc, size_t* bytes_written)

    # Decoder
    charls_jpegls_decoder* charls_jpegls_decoder_create()
    void charls_jpegls_decoder_destroy(const charls_jpegls_decoder* dec)
    charls_jpegls_errc charls_jpegls_decoder_set_source_buffer(
        charls_jpegls_decoder* dec, const void* src, size_t size)
    charls_jpegls_errc charls_jpegls_decoder_read_header(
        charls_jpegls_decoder* dec)
    charls_jpegls_errc charls_jpegls_decoder_get_frame_info(
        const charls_jpegls_decoder* dec, charls_frame_info* info)
    charls_jpegls_errc charls_jpegls_decoder_get_destination_size(
        const charls_jpegls_decoder* dec, uint32_t stride, size_t* size)
    charls_jpegls_errc charls_jpegls_decoder_decode_to_buffer(
        charls_jpegls_decoder* dec, void* dst, size_t dst_size,
        uint32_t stride)

    # Error string lookup
    const char* charls_get_error_message(charls_jpegls_errc errc)
