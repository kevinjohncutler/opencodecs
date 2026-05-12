# MozJPEG (libjpeg-turbo fork) — TurboJPEG v2 C API.
#
# MozJPEG ships only the older v2 ``tj*`` symbols; the v3 ``tj3*``
# API is libjpeg-turbo 3.0+. We bind v2 here as a separate codec
# from the regular ``_jpeg.pyx`` (which uses v3).

from libc.stdint cimport uint8_t


cdef extern from 'turbojpeg.h' nogil:

    # Pixel formats
    cdef enum:
        TJPF_RGB = 0
        TJPF_BGR = 1
        TJPF_RGBX = 2
        TJPF_BGRX = 3
        TJPF_XBGR = 4
        TJPF_XRGB = 5
        TJPF_GRAY = 6
        TJPF_RGBA = 7
        TJPF_BGRA = 8
        TJPF_ABGR = 9
        TJPF_ARGB = 10
        TJPF_CMYK = 11

    # Subsampling
    cdef enum:
        TJSAMP_444 = 0
        TJSAMP_422 = 1
        TJSAMP_420 = 2
        TJSAMP_GRAY = 3
        TJSAMP_440 = 4
        TJSAMP_411 = 5

    # Flags for tjCompress2 / tjDecompress2
    cdef enum:
        TJFLAG_ACCURATEDCT = 4096
        TJFLAG_PROGRESSIVE = 16384

    ctypedef void* tjhandle

    tjhandle tjInitCompress()
    tjhandle tjInitDecompress()
    int tjDestroy(tjhandle handle)

    char* tjGetErrorStr2(tjhandle handle)

    int tjCompress2(
        tjhandle handle,
        const unsigned char* src,
        int width, int pitch, int height,
        int pixel_format,
        unsigned char** jpeg_buf, unsigned long* jpeg_size,
        int subsamp, int quality, int flags,
    )

    int tjDecompressHeader3(
        tjhandle handle,
        const unsigned char* jpeg_buf, unsigned long jpeg_size,
        int* width, int* height, int* subsamp, int* colorspace,
    )

    int tjDecompress2(
        tjhandle handle,
        const unsigned char* jpeg_buf, unsigned long jpeg_size,
        unsigned char* dst, int width, int pitch, int height,
        int pixel_format, int flags,
    )

    void tjFree(unsigned char* buffer)
