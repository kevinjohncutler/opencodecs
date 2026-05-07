# Minimal Cython declarations for OpenJPEG 2.5+ memory-stream API.

from libc.stdint cimport int32_t, uint32_t, uint8_t
from libc.stddef cimport size_t

cdef extern from 'openjpeg.h' nogil:
    ctypedef int OPJ_BOOL
    ctypedef int OPJ_INT32
    ctypedef unsigned int OPJ_UINT32
    ctypedef size_t OPJ_SIZE_T
    ctypedef long long OPJ_OFF_T

    cdef enum CODEC_FORMAT:
        OPJ_CODEC_UNKNOWN
        OPJ_CODEC_J2K
        OPJ_CODEC_JPT
        OPJ_CODEC_JP2
        OPJ_CODEC_JPP
        OPJ_CODEC_JPX

    cdef enum COLOR_SPACE:
        OPJ_CLRSPC_UNKNOWN = -1
        OPJ_CLRSPC_UNSPECIFIED = 0
        OPJ_CLRSPC_SRGB = 1
        OPJ_CLRSPC_GRAY = 2
        OPJ_CLRSPC_SYCC = 3
        OPJ_CLRSPC_EYCC = 4
        OPJ_CLRSPC_CMYK = 5

    ctypedef struct opj_image_comp:
        OPJ_UINT32 dx
        OPJ_UINT32 dy
        OPJ_UINT32 w
        OPJ_UINT32 h
        OPJ_UINT32 x0
        OPJ_UINT32 y0
        OPJ_UINT32 prec
        OPJ_UINT32 bpp
        OPJ_UINT32 sgnd
        OPJ_UINT32 resno_decoded
        OPJ_UINT32 factor
        OPJ_INT32* data
        OPJ_UINT32 alpha

    ctypedef struct opj_image_cmptparm "opj_image_cmptparm_t":
        OPJ_UINT32 dx
        OPJ_UINT32 dy
        OPJ_UINT32 w
        OPJ_UINT32 h
        OPJ_UINT32 x0
        OPJ_UINT32 y0
        OPJ_UINT32 prec
        OPJ_UINT32 bpp
        OPJ_UINT32 sgnd

    ctypedef struct opj_image_t:
        OPJ_UINT32 x0
        OPJ_UINT32 y0
        OPJ_UINT32 x1
        OPJ_UINT32 y1
        OPJ_UINT32 numcomps
        COLOR_SPACE color_space
        opj_image_comp* comps
        # ... more fields we don't need

    ctypedef struct opj_codec_t:
        pass

    ctypedef struct opj_stream_t:
        pass

    ctypedef struct opj_dparameters_t:
        OPJ_UINT32 cp_reduce
        OPJ_UINT32 cp_layer
        # ... more fields

    ctypedef struct opj_cparameters_t:
        OPJ_BOOL tile_size_on
        int cp_tx0
        int cp_ty0
        int cp_tdx
        int cp_tdy
        int cp_disto_alloc
        int cp_fixed_alloc
        int cp_fixed_quality
        int* cp_matrice
        char* cp_comment
        int csty
        int prog_order
        # ... more fields - opaque size handled by openjpeg
        # We rely on opj_set_default_encoder_parameters to fill these.
        OPJ_UINT32 numresolution
        OPJ_UINT32 cblockw_init
        OPJ_UINT32 cblockh_init
        OPJ_UINT32 mode
        OPJ_BOOL irreversible
        int roi_compno
        int roi_shift
        int res_spec
        int tcp_numlayers
        float tcp_rates[100]
        float tcp_distoratio[100]
        OPJ_UINT32 max_comp_size

    OPJ_BOOL opj_has_thread_support()

    opj_image_t* opj_image_create(
        OPJ_UINT32 numcmpts, opj_image_cmptparm* cmptparms,
        COLOR_SPACE clrspc,
    )
    void opj_image_destroy(opj_image_t* image)

    opj_stream_t* opj_stream_default_create(OPJ_BOOL p_is_input)
    void opj_stream_destroy(opj_stream_t* p_stream)

    ctypedef OPJ_SIZE_T (*opj_stream_read_fn)(
        void* p_buffer, OPJ_SIZE_T p_nb_bytes, void* p_user_data) noexcept nogil
    ctypedef OPJ_SIZE_T (*opj_stream_write_fn)(
        void* p_buffer, OPJ_SIZE_T p_nb_bytes, void* p_user_data) noexcept nogil
    ctypedef OPJ_OFF_T (*opj_stream_skip_fn)(
        OPJ_OFF_T p_nb_bytes, void* p_user_data) noexcept nogil
    ctypedef OPJ_BOOL (*opj_stream_seek_fn)(
        OPJ_OFF_T p_nb_bytes, void* p_user_data) noexcept nogil
    ctypedef void (*opj_stream_free_user_data_fn)(void* p_user_data) noexcept nogil

    void opj_stream_set_read_function(
        opj_stream_t* p_stream, opj_stream_read_fn p_function)
    void opj_stream_set_write_function(
        opj_stream_t* p_stream, opj_stream_write_fn p_function)
    void opj_stream_set_skip_function(
        opj_stream_t* p_stream, opj_stream_skip_fn p_function)
    void opj_stream_set_seek_function(
        opj_stream_t* p_stream, opj_stream_seek_fn p_function)
    void opj_stream_set_user_data(
        opj_stream_t* p_stream, void* p_data,
        opj_stream_free_user_data_fn p_function)
    void opj_stream_set_user_data_length(
        opj_stream_t* p_stream, OPJ_UINT32 data_length)

    opj_codec_t* opj_create_decompress(CODEC_FORMAT format)
    opj_codec_t* opj_create_compress(CODEC_FORMAT format)
    void opj_destroy_codec(opj_codec_t* p_codec)

    void opj_set_default_decoder_parameters(opj_dparameters_t* parameters)
    void opj_set_default_encoder_parameters(opj_cparameters_t* parameters)
    OPJ_BOOL opj_setup_decoder(
        opj_codec_t* p_codec, opj_dparameters_t* parameters)
    OPJ_BOOL opj_setup_encoder(
        opj_codec_t* p_codec, opj_cparameters_t* parameters,
        opj_image_t* image)

    OPJ_BOOL opj_read_header(
        opj_stream_t* p_stream, opj_codec_t* p_codec, opj_image_t** p_image)
    OPJ_BOOL opj_decode(
        opj_codec_t* p_decompressor, opj_stream_t* p_stream,
        opj_image_t* p_image)
    OPJ_BOOL opj_end_decompress(
        opj_codec_t* p_codec, opj_stream_t* p_stream)

    OPJ_BOOL opj_start_compress(
        opj_codec_t* p_codec, opj_image_t* p_image, opj_stream_t* p_stream)
    OPJ_BOOL opj_encode(
        opj_codec_t* p_codec, opj_stream_t* p_stream)
    OPJ_BOOL opj_end_compress(
        opj_codec_t* p_codec, opj_stream_t* p_stream)
