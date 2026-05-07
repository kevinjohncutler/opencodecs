# Minimal Cython declarations for libheif (decode-only here; encode is
# deferred to a follow-up — encode requires walking encoder descriptors).

from libc.stdint cimport uint8_t

cdef extern from 'heif_shim.h' nogil:
    ctypedef struct heif_context:
        pass
    ctypedef struct heif_image_handle:
        pass
    ctypedef struct heif_image:
        pass
    ctypedef struct heif_decoding_options:
        pass
    ctypedef struct heif_encoder:
        pass
    ctypedef struct heif_encoding_options:
        pass

    cdef enum heif_error_code:
        heif_error_Ok = 0

    cdef enum heif_colorspace:
        heif_colorspace_undefined = 99
        heif_colorspace_YCbCr = 0
        heif_colorspace_RGB = 1
        heif_colorspace_monochrome = 2

    cdef enum heif_chroma:
        heif_chroma_undefined = 99
        heif_chroma_monochrome = 0
        heif_chroma_420 = 1
        heif_chroma_422 = 2
        heif_chroma_444 = 3
        heif_chroma_interleaved_RGB = 10
        heif_chroma_interleaved_RGBA = 11
        heif_chroma_interleaved_RRGGBB_BE = 12
        heif_chroma_interleaved_RRGGBBAA_BE = 13
        heif_chroma_interleaved_RRGGBB_LE = 14
        heif_chroma_interleaved_RRGGBBAA_LE = 15

    cdef enum heif_channel:
        heif_channel_Y = 0
        heif_channel_Cb = 1
        heif_channel_Cr = 2
        heif_channel_R = 3
        heif_channel_G = 4
        heif_channel_B = 5
        heif_channel_Alpha = 6
        heif_channel_interleaved = 10

    cdef enum heif_compression_format:
        heif_compression_undefined = 0
        heif_compression_HEVC = 1
        heif_compression_AVC = 2
        heif_compression_JPEG = 3
        heif_compression_AV1 = 4
        heif_compression_VVC = 5

    ctypedef struct heif_error:
        heif_error_code code
        int subcode
        const char* message

    void heif_init(void* options)
    void heif_deinit()
    heif_context* heif_context_alloc()
    void heif_context_free(heif_context*)
    heif_error heif_context_read_from_memory_without_copy(
        heif_context*, const void* mem, size_t size,
        const void* options)
    heif_error heif_context_get_primary_image_handle(
        heif_context*, heif_image_handle**)
    heif_error heif_decode_image(
        const heif_image_handle*, heif_image** out,
        heif_colorspace colorspace, heif_chroma chroma,
        const heif_decoding_options* options,
    )
    int heif_image_handle_get_width(const heif_image_handle*)
    int heif_image_handle_get_height(const heif_image_handle*)
    int heif_image_handle_has_alpha_channel(const heif_image_handle*)
    void heif_image_handle_release(const heif_image_handle*)

    int heif_image_get_width(const heif_image*, heif_channel channel)
    int heif_image_get_height(const heif_image*, heif_channel channel)
    const uint8_t* heif_image_get_plane_readonly(
        const heif_image*, heif_channel, int* out_stride)
    uint8_t* heif_image_get_plane(
        heif_image*, heif_channel, int* out_stride)
    void heif_image_release(const heif_image*)

    heif_error heif_image_create(
        int width, int height,
        heif_colorspace, heif_chroma, heif_image** out,
    )
    heif_error heif_image_add_plane(
        heif_image*, heif_channel,
        int width, int height, int bit_depth,
    )

    heif_error heif_context_get_encoder_for_format(
        heif_context*, heif_compression_format, heif_encoder** out)
    void heif_encoder_release(heif_encoder*)
    heif_error heif_encoder_set_lossy_quality(heif_encoder*, int)
    heif_error heif_encoder_set_lossless(heif_encoder*, int)
    heif_error heif_context_encode_image(
        heif_context*, const heif_image*, heif_encoder*,
        const heif_encoding_options*, heif_image_handle** out_handle,
    )

    ctypedef struct heif_writer:
        int writer_api_version
        heif_error (*write)(heif_context*, const void* data,
                            size_t size, void* userdata)

    heif_error heif_context_write(
        heif_context*, heif_writer*, void* userdata,
    )
