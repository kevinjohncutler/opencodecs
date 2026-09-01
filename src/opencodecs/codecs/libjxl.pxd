# opencodecs/codecs/libjxl.pxd
# cython: language_level = 3
"""Cython declarations for libjxl 0.11.2 (JPEG XL).

Transcribed from the upstream public headers that ship with libjxl
0.11.2: ``jxl/types.h``, ``color_encoding.h``, ``codestream_header.h``,
``decode.h``, ``encode.h``, ``parallel_runner.h`` and
``thread_parallel_runner.h``.

Scope is deliberately narrow. This declares only the surface
``opencodecs.codecs._jxl`` actually calls, roughly a third of libjxl's
public C API, and structs carry only the fields we touch. Cython emits
field accesses against the real C struct from the header above, so a
partial declaration is safe and keeps this file to what we use rather
than mirroring the whole library.

Two naming notes. ``JxlBitDepth.type`` is spelled ``dtype`` here
because ``type`` is not usable as a Cython attribute name; the quoted
form keeps the real C name in the generated code. Enums whose members
we never name are still declared in full, since they are short and it
avoids casting through ``int``.

https://github.com/libjxl/libjxl
"""

from libc.stdint cimport int32_t, int64_t, uint8_t, uint32_t


cdef extern from "jxl/types.h" nogil:

    ctypedef int JXL_BOOL
    int JXL_TRUE
    int JXL_FALSE

    ctypedef enum JxlDataType:
        JXL_TYPE_FLOAT
        JXL_TYPE_UINT8
        JXL_TYPE_UINT16
        JXL_TYPE_FLOAT16

    ctypedef enum JxlEndianness:
        JXL_NATIVE_ENDIAN
        JXL_LITTLE_ENDIAN
        JXL_BIG_ENDIAN

    ctypedef enum JxlBitDepthType:
        JXL_BIT_DEPTH_FROM_PIXEL_FORMAT
        JXL_BIT_DEPTH_FROM_CODESTREAM
        JXL_BIT_DEPTH_CUSTOM

    ctypedef struct JxlPixelFormat:
        uint32_t num_channels
        JxlDataType data_type
        JxlEndianness endianness
        size_t align

    ctypedef struct JxlBitDepth:
        JxlBitDepthType dtype "type"
        uint32_t bits_per_sample
        uint32_t exponent_bits_per_sample


cdef extern from "jxl/color_encoding.h" nogil:

    ctypedef enum JxlColorSpace:
        JXL_COLOR_SPACE_RGB
        JXL_COLOR_SPACE_GRAY
        JXL_COLOR_SPACE_XYB
        JXL_COLOR_SPACE_UNKNOWN

    ctypedef enum JxlWhitePoint:
        JXL_WHITE_POINT_D65
        JXL_WHITE_POINT_CUSTOM
        JXL_WHITE_POINT_E
        JXL_WHITE_POINT_DCI

    ctypedef enum JxlPrimaries:
        JXL_PRIMARIES_SRGB
        JXL_PRIMARIES_CUSTOM
        JXL_PRIMARIES_2100
        JXL_PRIMARIES_P3

    ctypedef enum JxlTransferFunction:
        JXL_TRANSFER_FUNCTION_709
        JXL_TRANSFER_FUNCTION_UNKNOWN
        JXL_TRANSFER_FUNCTION_LINEAR
        JXL_TRANSFER_FUNCTION_SRGB
        JXL_TRANSFER_FUNCTION_PQ
        JXL_TRANSFER_FUNCTION_DCI
        JXL_TRANSFER_FUNCTION_HLG
        JXL_TRANSFER_FUNCTION_GAMMA

    ctypedef enum JxlRenderingIntent:
        JXL_RENDERING_INTENT_PERCEPTUAL
        JXL_RENDERING_INTENT_RELATIVE
        JXL_RENDERING_INTENT_SATURATION
        JXL_RENDERING_INTENT_ABSOLUTE

    ctypedef struct JxlColorEncoding:
        JxlColorSpace color_space
        JxlWhitePoint white_point
        JxlPrimaries primaries
        JxlTransferFunction transfer_function
        double gamma
        JxlRenderingIntent rendering_intent


cdef extern from "jxl/codestream_header.h" nogil:

    ctypedef enum JxlOrientation:
        JXL_ORIENT_IDENTITY
        JXL_ORIENT_FLIP_HORIZONTAL
        JXL_ORIENT_ROTATE_180
        JXL_ORIENT_FLIP_VERTICAL
        JXL_ORIENT_TRANSPOSE
        JXL_ORIENT_ROTATE_90_CW
        JXL_ORIENT_ANTI_TRANSPOSE
        JXL_ORIENT_ROTATE_90_CCW

    ctypedef struct JxlAnimationHeader:
        uint32_t tps_numerator
        uint32_t tps_denominator
        uint32_t num_loops

    ctypedef struct JxlBasicInfo:
        uint32_t xsize
        uint32_t ysize
        uint32_t bits_per_sample
        uint32_t exponent_bits_per_sample
        float intensity_target
        float min_nits
        JXL_BOOL relative_to_max_display
        float linear_below
        JXL_BOOL uses_original_profile
        JXL_BOOL have_animation
        JxlOrientation orientation
        uint32_t num_color_channels
        uint32_t num_extra_channels
        uint32_t alpha_bits
        uint32_t alpha_exponent_bits
        JxlAnimationHeader animation

    ctypedef struct JxlFrameHeader:
        uint32_t duration
        JXL_BOOL is_last


cdef extern from "jxl/parallel_runner.h" nogil:

    ctypedef int (*JxlParallelRunner)(
        void* runner_opaque, void* jpegxl_opaque,
        void* init, void* func,
        uint32_t start_range, uint32_t end_range,
    )


cdef extern from "jxl/thread_parallel_runner.h" nogil:

    int JxlThreadParallelRunner(
        void* runner_opaque, void* jpegxl_opaque,
        void* init, void* func,
        uint32_t start_range, uint32_t end_range,
    )
    void* JxlThreadParallelRunnerCreate(void* memory_manager,
                                        size_t num_worker_threads)
    void JxlThreadParallelRunnerDestroy(void* runner_opaque)
    size_t JxlThreadParallelRunnerDefaultNumWorkerThreads()


cdef extern from "jxl/decode.h" nogil:

    ctypedef struct JxlDecoder:
        pass

    ctypedef enum JxlSignature:
        JXL_SIG_NOT_ENOUGH_BYTES
        JXL_SIG_INVALID
        JXL_SIG_CODESTREAM
        JXL_SIG_CONTAINER

    ctypedef enum JxlColorProfileTarget:
        JXL_COLOR_PROFILE_TARGET_ORIGINAL
        JXL_COLOR_PROFILE_TARGET_DATA

    ctypedef enum JxlProgressiveDetail:
        kFrames
        kDC
        kLastPasses
        kPasses
        kDCProgressive
        kDCGroups
        kGroups

    ctypedef enum JxlDecoderStatus:
        JXL_DEC_SUCCESS
        JXL_DEC_ERROR
        JXL_DEC_NEED_MORE_INPUT
        JXL_DEC_NEED_PREVIEW_OUT_BUFFER
        JXL_DEC_NEED_IMAGE_OUT_BUFFER
        JXL_DEC_JPEG_NEED_MORE_OUTPUT
        JXL_DEC_BASIC_INFO
        JXL_DEC_COLOR_ENCODING
        JXL_DEC_PREVIEW_IMAGE
        JXL_DEC_FRAME
        JXL_DEC_FULL_IMAGE
        JXL_DEC_JPEG_RECONSTRUCTION
        JXL_DEC_BOX
        JXL_DEC_FRAME_PROGRESSION
        JXL_DEC_BOX_COMPLETE

    uint32_t JxlDecoderVersion()
    JxlSignature JxlSignatureCheck(const uint8_t* buf, size_t len)

    JxlDecoder* JxlDecoderCreate(void* memory_manager)
    void JxlDecoderDestroy(JxlDecoder* dec)

    JxlDecoderStatus JxlDecoderSubscribeEvents(JxlDecoder* dec,
                                               int events_wanted)
    JxlDecoderStatus JxlDecoderSetParallelRunner(
        JxlDecoder* dec, JxlParallelRunner parallel_runner,
        void* parallel_runner_opaque,
    )
    JxlDecoderStatus JxlDecoderSetInput(JxlDecoder* dec,
                                        const uint8_t* data, size_t size)
    void JxlDecoderCloseInput(JxlDecoder* dec)
    size_t JxlDecoderReleaseInput(JxlDecoder* dec)
    JxlDecoderStatus JxlDecoderProcessInput(JxlDecoder* dec)

    JxlDecoderStatus JxlDecoderGetBasicInfo(const JxlDecoder* dec,
                                            JxlBasicInfo* info)
    JxlDecoderStatus JxlDecoderGetColorAsEncodedProfile(
        const JxlDecoder* dec, JxlColorProfileTarget target,
        JxlColorEncoding* color_encoding,
    )
    JxlDecoderStatus JxlDecoderGetICCProfileSize(
        const JxlDecoder* dec, JxlColorProfileTarget target, size_t* size)
    JxlDecoderStatus JxlDecoderGetColorAsICCProfile(
        const JxlDecoder* dec, JxlColorProfileTarget target,
        uint8_t* icc_profile, size_t size,
    )

    JxlDecoderStatus JxlDecoderImageOutBufferSize(
        const JxlDecoder* dec, const JxlPixelFormat* format, size_t* size)
    JxlDecoderStatus JxlDecoderSetImageOutBuffer(
        const JxlDecoder* dec, const JxlPixelFormat* format,
        void* buffer, size_t size,
    )
    JxlDecoderStatus JxlDecoderSetImageOutBitDepth(
        JxlDecoder* dec, const JxlBitDepth* bit_depth)
    JxlDecoderStatus JxlDecoderFlushImage(JxlDecoder* dec)

    JxlDecoderStatus JxlDecoderSetKeepOrientation(JxlDecoder* dec,
                                                  JXL_BOOL skip_reorientation)
    JxlDecoderStatus JxlDecoderSetCoalescing(JxlDecoder* dec,
                                             JXL_BOOL coalescing)
    JxlDecoderStatus JxlDecoderSetProgressiveDetail(
        JxlDecoder* dec, JxlProgressiveDetail detail)
    size_t JxlDecoderGetIntendedDownsamplingRatio(JxlDecoder* dec)
    JxlDecoderStatus JxlDecoderSkipCurrentFrame(JxlDecoder* dec)
    void JxlDecoderSkipFrames(JxlDecoder* dec, size_t amount)


cdef extern from "jxl/encode.h" nogil:

    ctypedef struct JxlEncoder:
        pass

    ctypedef struct JxlEncoderFrameSettings:
        pass

    ctypedef enum JxlEncoderStatus:
        JXL_ENC_SUCCESS
        JXL_ENC_ERROR
        JXL_ENC_NEED_MORE_OUTPUT

    ctypedef enum JxlEncoderError:
        JXL_ENC_ERR_OK
        JXL_ENC_ERR_GENERIC
        JXL_ENC_ERR_OOM
        JXL_ENC_ERR_JBRD
        JXL_ENC_ERR_BAD_INPUT
        JXL_ENC_ERR_NOT_SUPPORTED
        JXL_ENC_ERR_API_USAGE

    ctypedef enum JxlEncoderFrameSettingId:
        JXL_ENC_FRAME_SETTING_EFFORT
        JXL_ENC_FRAME_SETTING_DECODING_SPEED
        JXL_ENC_FRAME_SETTING_RESPONSIVE
        JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC
        JXL_ENC_FRAME_SETTING_MODULAR
        JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM

    JxlEncoder* JxlEncoderCreate(void* memory_manager)
    void JxlEncoderDestroy(JxlEncoder* enc)
    JxlEncoderError JxlEncoderGetError(JxlEncoder* enc)

    JxlEncoderStatus JxlEncoderSetParallelRunner(
        JxlEncoder* enc, JxlParallelRunner parallel_runner,
        void* parallel_runner_opaque,
    )
    JxlEncoderStatus JxlEncoderUseContainer(JxlEncoder* enc,
                                            JXL_BOOL use_container)

    void JxlEncoderInitBasicInfo(JxlBasicInfo* info)
    JxlEncoderStatus JxlEncoderSetBasicInfo(JxlEncoder* enc,
                                            const JxlBasicInfo* info)
    void JxlEncoderInitFrameHeader(JxlFrameHeader* frame_header)
    JxlEncoderStatus JxlEncoderSetFrameHeader(
        JxlEncoderFrameSettings* frame_settings,
        const JxlFrameHeader* frame_header,
    )

    void JxlColorEncodingSetToSRGB(JxlColorEncoding* color_encoding,
                                   JXL_BOOL is_gray)
    void JxlColorEncodingSetToLinearSRGB(JxlColorEncoding* color_encoding,
                                         JXL_BOOL is_gray)
    JxlEncoderStatus JxlEncoderSetColorEncoding(JxlEncoder* enc,
                                                const JxlColorEncoding* color)
    JxlEncoderStatus JxlEncoderSetICCProfile(JxlEncoder* enc,
                                             const uint8_t* icc_profile,
                                             size_t size)

    JxlEncoderFrameSettings* JxlEncoderFrameSettingsCreate(
        JxlEncoder* enc, const JxlEncoderFrameSettings* source)
    JxlEncoderStatus JxlEncoderFrameSettingsSetOption(
        JxlEncoderFrameSettings* frame_settings,
        JxlEncoderFrameSettingId option, int64_t value,
    )
    JxlEncoderStatus JxlEncoderSetFrameDistance(
        JxlEncoderFrameSettings* frame_settings, float distance)
    JxlEncoderStatus JxlEncoderSetFrameLossless(
        JxlEncoderFrameSettings* frame_settings, JXL_BOOL lossless)
    JxlEncoderStatus JxlEncoderSetFrameBitDepth(
        JxlEncoderFrameSettings* frame_settings, const JxlBitDepth* bit_depth)
    float JxlEncoderDistanceFromQuality(float quality)

    JxlEncoderStatus JxlEncoderAddImageFrame(
        const JxlEncoderFrameSettings* frame_settings,
        const JxlPixelFormat* pixel_format,
        const void* buffer, size_t size,
    )
    void JxlEncoderCloseInput(JxlEncoder* enc)
    JxlEncoderStatus JxlEncoderFlushInput(JxlEncoder* enc)
    JxlEncoderStatus JxlEncoderProcessOutput(JxlEncoder* enc,
                                             uint8_t** next_out,
                                             size_t* avail_out)
