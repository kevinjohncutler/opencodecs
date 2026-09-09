# Minimal Cython declarations for libavif.

from libc.stdint cimport uint8_t, uint16_t, uint32_t

cdef extern from 'avif/avif.h' nogil:
    int AVIF_QUALITY_LOSSLESS
    int AVIF_RESULT_OK

    ctypedef uint16_t avifColorPrimaries
    ctypedef uint16_t avifTransferCharacteristics
    ctypedef uint16_t avifMatrixCoefficients

    cdef enum avifPixelFormat:
        AVIF_PIXEL_FORMAT_NONE = 0
        AVIF_PIXEL_FORMAT_YUV444
        AVIF_PIXEL_FORMAT_YUV422
        AVIF_PIXEL_FORMAT_YUV420
        AVIF_PIXEL_FORMAT_YUV400

    cdef enum avifRGBFormat:
        AVIF_RGB_FORMAT_RGB = 0
        AVIF_RGB_FORMAT_RGBA
        AVIF_RGB_FORMAT_ARGB
        AVIF_RGB_FORMAT_BGR
        AVIF_RGB_FORMAT_BGRA
        AVIF_RGB_FORMAT_ABGR
        AVIF_RGB_FORMAT_RGB_565
        AVIF_RGB_FORMAT_GRAY
        AVIF_RGB_FORMAT_GRAYA

    ctypedef int avifResult

    ctypedef struct avifRWData:
        uint8_t* data
        size_t size

    void avifRWDataFree(avifRWData* raw)

    ctypedef struct avifImage:
        uint32_t width
        uint32_t height
        uint32_t depth
        avifPixelFormat yuvFormat
        uint8_t* alphaPlane
        uint8_t alphaPremultiplied
        avifColorPrimaries colorPrimaries
        avifTransferCharacteristics transferCharacteristics
        avifMatrixCoefficients matrixCoefficients
        # ICC profile attached to the image. libavif populates this
        # field on decode if the file carries an ICC profile; on
        # encode it's the buffer avifImageSetProfileICC manages.
        avifRWData icc

    ctypedef struct avifRGBImage:
        uint32_t width
        uint32_t height
        uint32_t depth
        avifRGBFormat format
        # ...
        uint32_t rowBytes
        uint8_t* pixels

    avifImage* avifImageCreate(uint32_t width, uint32_t height,
                               uint32_t depth, avifPixelFormat yuvFormat)
    avifImage* avifImageCreateEmpty()
    void avifImageDestroy(avifImage* image)

    # ICC profile attach. avifImageSetProfileICC copies the bytes
    # into image->icc — caller can free immediately after the call.
    # The retrieve side reads image->icc directly via the struct field.
    void avifImageSetProfileICC(
        avifImage* image, const uint8_t* icc, size_t iccSize)

    void avifRGBImageSetDefaults(avifRGBImage* rgb, const avifImage* image)
    avifResult avifRGBImageAllocatePixels(avifRGBImage* rgb)
    void avifRGBImageFreePixels(avifRGBImage* rgb)

    avifResult avifImageRGBToYUV(avifImage* image, const avifRGBImage* rgb)
    avifResult avifImageYUVToRGB(const avifImage* image, avifRGBImage* rgb)

    ctypedef enum avifCodecChoice:
        AVIF_CODEC_CHOICE_AUTO   = 0
        AVIF_CODEC_CHOICE_AOM    = 1
        AVIF_CODEC_CHOICE_DAV1D  = 2
        AVIF_CODEC_CHOICE_LIBGAV1 = 3
        AVIF_CODEC_CHOICE_RAV1E  = 4
        AVIF_CODEC_CHOICE_SVT    = 5
        AVIF_CODEC_CHOICE_AVM    = 6

    ctypedef struct avifEncoder:
        avifCodecChoice codecChoice
        int maxThreads
        int speed
        int quality
        int qualityAlpha
        int tileRowsLog2
        int tileColsLog2
        int autoTiling

    avifEncoder* avifEncoderCreate()
    void avifEncoderDestroy(avifEncoder* encoder)
    avifResult avifEncoderSetCodecSpecificOption(
        avifEncoder* encoder, const char* key, const char* value)
    avifResult avifEncoderWrite(
        avifEncoder* encoder, const avifImage* image, avifRWData* output)

    ctypedef struct avifDecoder:
        int maxThreads
        # Sequence outputs, valid after avifDecoderParse(). imageCount
        # is 1 for a plain still, so it doubles as "is this a
        # sequence". `image` is owned by the decoder and its contents
        # are replaced by the next NextImage/NthImage call, which is
        # why the reader copies out of it rather than holding it.
        avifImage* image
        int imageIndex
        int imageCount
        unsigned long long timescale
        unsigned long long durationInTimescales
        int repetitionCount
        int alphaPresent
        # ...

    avifDecoder* avifDecoderCreate()
    void avifDecoderDestroy(avifDecoder* decoder)
    avifResult avifDecoderReadMemory(
        avifDecoder* decoder, avifImage* image,
        const uint8_t* data, size_t size,
    )

    # Sequence path: point the decoder at the buffer once, parse the
    # container once, then pull frames. SetIOMemory does NOT copy, so
    # the caller has to keep the bytes alive for the decoder's life.
    avifResult avifDecoderSetIOMemory(
        avifDecoder* decoder, const uint8_t* data, size_t size)
    avifResult avifDecoderParse(avifDecoder* decoder)
    avifResult avifDecoderNextImage(avifDecoder* decoder)
    avifResult avifDecoderNthImage(
        avifDecoder* decoder, unsigned int frameIndex)

    const char* avifResultToString(avifResult result)
