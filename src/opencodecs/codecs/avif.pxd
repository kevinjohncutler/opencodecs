# Minimal Cython declarations for libavif.

from libc.stdint cimport uint8_t, uint32_t

cdef extern from 'avif/avif.h' nogil:
    int AVIF_QUALITY_LOSSLESS
    int AVIF_RESULT_OK

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
        int matrixCoefficients

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

    void avifRGBImageSetDefaults(avifRGBImage* rgb, const avifImage* image)
    avifResult avifRGBImageAllocatePixels(avifRGBImage* rgb)
    void avifRGBImageFreePixels(avifRGBImage* rgb)

    avifResult avifImageRGBToYUV(avifImage* image, const avifRGBImage* rgb)
    avifResult avifImageYUVToRGB(const avifImage* image, avifRGBImage* rgb)

    ctypedef struct avifEncoder:
        int maxThreads
        int speed
        int quality
        int qualityAlpha

    avifEncoder* avifEncoderCreate()
    void avifEncoderDestroy(avifEncoder* encoder)
    avifResult avifEncoderWrite(
        avifEncoder* encoder, const avifImage* image, avifRWData* output)

    ctypedef struct avifDecoder:
        int maxThreads
        # ...

    avifDecoder* avifDecoderCreate()
    void avifDecoderDestroy(avifDecoder* decoder)
    avifResult avifDecoderReadMemory(
        avifDecoder* decoder, avifImage* image,
        const uint8_t* data, size_t size,
    )

    const char* avifResultToString(avifResult result)
