# Cython declarations for the vendored Bruce Walter / Greg Ward RGBE C
# library (``3rdparty/rgbe/rgbe.{c,h}``).

# Transcribed from upstream 'rgbe.h'. imagecodecs declares the same C
# API in its own rgbe.pxd; the two overlap because the header fixes the
# names and signatures, not because either was copied from the other.

from libc.stddef cimport size_t


cdef extern from "rgbe.h" nogil:
    int RGBE_RETURN_SUCCESS
    int RGBE_RETURN_FAILURE
    int RGBE_READ_ERROR
    int RGBE_WRITE_ERROR
    int RGBE_FORMAT_ERROR
    int RGBE_MEMORY_ERROR

    ctypedef struct rgbe_stream_t:
        char* data
        size_t size
        size_t pos
        int owner

    ctypedef struct rgbe_header_info:
        int valid
        char programtype[16]
        float gamma
        float exposure

    rgbe_stream_t* rgbe_stream_new(size_t size, char* data)
    void rgbe_stream_del(rgbe_stream_t* stream)

    int RGBE_WriteHeader(
        rgbe_stream_t* fp, int width, int height,
        const rgbe_header_info* info
    )
    int RGBE_ORIENT_NONE
    int RGBE_ORIENT_FLIP_X
    int RGBE_ORIENT_FLIP_Y
    int RGBE_ORIENT_TRANSPOSE

    int RGBE_ReadHeaderOriented(
        rgbe_stream_t* fp, int* width, int* height,
        rgbe_header_info* info, int* orientation,
    ) nogil

    int RGBE_ReadHeader(
        rgbe_stream_t* fp, int* width, int* height,
        rgbe_header_info* info
    )

    int RGBE_WritePixels(
        rgbe_stream_t* fp, const float* data, int numpixels
    )
    int RGBE_ReadPixels(
        rgbe_stream_t* fp, float* data, int numpixels
    )

    int RGBE_WritePixels_RLE(
        rgbe_stream_t* fp, const float* data,
        int scanline_width, int num_scanlines
    )
    int RGBE_ReadPixels_RLE(
        rgbe_stream_t* fp, float* data,
        int scanline_width, int num_scanlines
    )
