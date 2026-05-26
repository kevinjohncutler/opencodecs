# opencodecs/codecs/libuhdr.pxd
#
# Cython declarations for Google's libultrahdr (libuhdr) 1.4.x
# https://github.com/google/libultrahdr
#
# Spec: ISO/IEC 21496-1:2025 ("Gain map metadata for image conversion").
# Carries an SDR base image plus a per-pixel gain map in a JPEG / HEIF /
# AVIF container; HDR-aware decoders composite to display headroom, dumb
# decoders see just the SDR base.
#
# We declare only what we use; the full header (~50 functions) has more
# decoder-side helpers, GPU acceleration toggles, and effect ops that
# are out of scope for the v1 binding.

from libc.stdint cimport uint8_t, uint32_t, uint64_t


cdef extern from "ultrahdr_api.h" nogil:

    # All enums and structs use libuhdr's ``_t`` typedef names (defined
    # in the header) as the C identifier -- avoids the C-side "must use
    # 'enum' tag" error that bare ``uhdr_codec_err`` (no typedef) trips.

    # ---- Enums -------------------------------------------------------

    ctypedef enum uhdr_img_fmt_t:
        UHDR_IMG_FMT_UNSPECIFIED
        UHDR_IMG_FMT_24bppYCbCrP010
        UHDR_IMG_FMT_12bppYCbCr420
        UHDR_IMG_FMT_8bppYCbCr400
        UHDR_IMG_FMT_32bppRGBA8888
        UHDR_IMG_FMT_64bppRGBAHalfFloat
        UHDR_IMG_FMT_32bppRGBA1010102
        UHDR_IMG_FMT_24bppYCbCr444
        UHDR_IMG_FMT_16bppYCbCr422
        UHDR_IMG_FMT_16bppYCbCr440
        UHDR_IMG_FMT_12bppYCbCr411
        UHDR_IMG_FMT_12bppYCbCr410
        UHDR_IMG_FMT_24bppRGB888

    ctypedef enum uhdr_color_gamut_t:
        UHDR_CG_UNSPECIFIED
        UHDR_CG_BT_709
        UHDR_CG_DISPLAY_P3
        UHDR_CG_BT_2100

    ctypedef enum uhdr_color_transfer_t:
        UHDR_CT_UNSPECIFIED
        UHDR_CT_LINEAR
        UHDR_CT_HLG
        UHDR_CT_PQ
        UHDR_CT_SRGB

    ctypedef enum uhdr_color_range_t:
        UHDR_CR_UNSPECIFIED
        UHDR_CR_LIMITED_RANGE
        UHDR_CR_FULL_RANGE

    ctypedef enum uhdr_codec_t:
        UHDR_CODEC_JPG
        UHDR_CODEC_HEIF
        UHDR_CODEC_AVIF

    ctypedef enum uhdr_img_label_t:
        UHDR_HDR_IMG
        UHDR_SDR_IMG
        UHDR_BASE_IMG
        UHDR_GAIN_MAP_IMG

    ctypedef enum uhdr_enc_preset_t:
        UHDR_USAGE_REALTIME
        UHDR_USAGE_BEST_QUALITY

    ctypedef enum uhdr_codec_err_t:
        UHDR_CODEC_OK
        UHDR_CODEC_ERROR
        UHDR_CODEC_UNKNOWN_ERROR
        UHDR_CODEC_INVALID_PARAM
        UHDR_CODEC_MEM_ERROR
        UHDR_CODEC_INVALID_OPERATION
        UHDR_CODEC_UNSUPPORTED_FEATURE
        UHDR_CODEC_LIST_END

    # ---- Structs -----------------------------------------------------

    ctypedef struct uhdr_error_info_t:
        uhdr_codec_err_t error_code
        int has_detail
        char detail[256]

    ctypedef struct uhdr_raw_image_t:
        uhdr_img_fmt_t fmt
        uhdr_color_gamut_t cg
        uhdr_color_transfer_t ct
        uhdr_color_range_t range
        unsigned int w
        unsigned int h
        void* planes[3]
        unsigned int stride[3]

    ctypedef struct uhdr_compressed_image_t:
        void* data
        size_t data_sz
        size_t capacity
        uhdr_color_gamut_t cg
        uhdr_color_transfer_t ct
        uhdr_color_range_t range

    ctypedef struct uhdr_mem_block_t:
        void* data
        size_t data_sz
        size_t capacity

    ctypedef struct uhdr_gainmap_metadata_t:
        float max_content_boost[3]
        float min_content_boost[3]
        float gamma[3]
        float offset_sdr[3]
        float offset_hdr[3]
        float hdr_capacity_min
        float hdr_capacity_max
        int use_base_cg

    # Opaque handle for both encoder and decoder contexts.
    ctypedef struct uhdr_codec_private_t:
        pass

    # ---- Encoder API -------------------------------------------------

    uhdr_codec_private_t* uhdr_create_encoder()
    void uhdr_release_encoder(uhdr_codec_private_t* enc)

    uhdr_error_info_t uhdr_enc_set_raw_image(
        uhdr_codec_private_t* enc,
        uhdr_raw_image_t* img,
        uhdr_img_label_t intent)
    uhdr_error_info_t uhdr_enc_set_compressed_image(
        uhdr_codec_private_t* enc,
        uhdr_compressed_image_t* img,
        uhdr_img_label_t intent)
    uhdr_error_info_t uhdr_enc_set_gainmap_image(
        uhdr_codec_private_t* enc,
        uhdr_compressed_image_t* img,
        uhdr_gainmap_metadata_t* metadata)

    uhdr_error_info_t uhdr_enc_set_quality(
        uhdr_codec_private_t* enc, int quality, uhdr_img_label_t intent)
    uhdr_error_info_t uhdr_enc_set_exif_data(
        uhdr_codec_private_t* enc, uhdr_mem_block_t* exif)
    uhdr_error_info_t uhdr_enc_set_using_multi_channel_gainmap(
        uhdr_codec_private_t* enc, int use_multi_channel)
    uhdr_error_info_t uhdr_enc_set_gainmap_scale_factor(
        uhdr_codec_private_t* enc, int factor)
    uhdr_error_info_t uhdr_enc_set_gainmap_gamma(
        uhdr_codec_private_t* enc, float gamma)
    uhdr_error_info_t uhdr_enc_set_min_max_content_boost(
        uhdr_codec_private_t* enc, float min_boost, float max_boost)
    uhdr_error_info_t uhdr_enc_set_target_display_peak_brightness(
        uhdr_codec_private_t* enc, float nits)
    uhdr_error_info_t uhdr_enc_set_preset(
        uhdr_codec_private_t* enc, uhdr_enc_preset_t preset)
    uhdr_error_info_t uhdr_enc_set_output_format(
        uhdr_codec_private_t* enc, uhdr_codec_t output_codec)

    uhdr_error_info_t uhdr_encode(uhdr_codec_private_t* enc)
    uhdr_compressed_image_t* uhdr_get_encoded_stream(uhdr_codec_private_t* enc)
    void uhdr_reset_encoder(uhdr_codec_private_t* enc)

    # ---- Decoder API -------------------------------------------------

    int is_uhdr_image(void* data, int size)

    uhdr_codec_private_t* uhdr_create_decoder()
    void uhdr_release_decoder(uhdr_codec_private_t* dec)

    uhdr_error_info_t uhdr_dec_set_image(
        uhdr_codec_private_t* dec, uhdr_compressed_image_t* img)
    uhdr_error_info_t uhdr_dec_set_out_img_format(
        uhdr_codec_private_t* dec, uhdr_img_fmt_t fmt)
    uhdr_error_info_t uhdr_dec_set_out_color_transfer(
        uhdr_codec_private_t* dec, uhdr_color_transfer_t ct)
    uhdr_error_info_t uhdr_dec_set_out_max_display_boost(
        uhdr_codec_private_t* dec, float display_boost)
    uhdr_error_info_t uhdr_dec_probe(uhdr_codec_private_t* dec)
    int uhdr_dec_get_image_width(uhdr_codec_private_t* dec)
    int uhdr_dec_get_image_height(uhdr_codec_private_t* dec)
    int uhdr_dec_get_gainmap_width(uhdr_codec_private_t* dec)
    int uhdr_dec_get_gainmap_height(uhdr_codec_private_t* dec)
    uhdr_mem_block_t* uhdr_dec_get_exif(uhdr_codec_private_t* dec)
    uhdr_mem_block_t* uhdr_dec_get_icc(uhdr_codec_private_t* dec)
    uhdr_mem_block_t* uhdr_dec_get_base_image(uhdr_codec_private_t* dec)
    uhdr_mem_block_t* uhdr_dec_get_gainmap_image(uhdr_codec_private_t* dec)
    uhdr_gainmap_metadata_t* uhdr_dec_get_gainmap_metadata(
        uhdr_codec_private_t* dec)
    uhdr_error_info_t uhdr_decode(uhdr_codec_private_t* dec)
    uhdr_raw_image_t* uhdr_get_decoded_image(uhdr_codec_private_t* dec)
    uhdr_raw_image_t* uhdr_get_decoded_gainmap_image(uhdr_codec_private_t* dec)
    void uhdr_reset_decoder(uhdr_codec_private_t* dec)
