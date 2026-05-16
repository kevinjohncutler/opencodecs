# Minimal Cython declarations for libjpeg-turbo's TurboJPEG API (v3).

from libc.stddef cimport size_t

cdef extern from 'turbojpeg.h' nogil:
    ctypedef void* tjhandle

    cdef enum:
        TJINIT_COMPRESS
        TJINIT_DECOMPRESS

    cdef enum:
        TJPF_RGB
        TJPF_BGR
        TJPF_RGBX
        TJPF_BGRX
        TJPF_XBGR
        TJPF_XRGB
        TJPF_GRAY
        TJPF_RGBA
        TJPF_BGRA
        TJPF_ABGR
        TJPF_ARGB

    cdef enum:
        TJSAMP_444
        TJSAMP_422
        TJSAMP_420
        TJSAMP_GRAY
        TJSAMP_440
        TJSAMP_411

    cdef enum:
        TJPARAM_QUALITY
        TJPARAM_SUBSAMP
        TJPARAM_JPEGWIDTH
        TJPARAM_JPEGHEIGHT
        TJPARAM_PRECISION
        TJPARAM_COLORSPACE

    tjhandle tj3Init(int initType)
    void tj3Destroy(tjhandle handle)
    char* tj3GetErrorStr(tjhandle handle)
    int tj3Set(tjhandle handle, int param, int value)
    int tj3Get(tjhandle handle, int param)
    void* tj3Alloc(size_t bytes)
    void tj3Free(void* buffer)
    size_t tj3JPEGBufSize(int width, int height, int jpegSubsamp)

    int tj3Compress8(
        tjhandle handle, const unsigned char* srcBuf,
        int width, int pitch, int height, int pixelFormat,
        unsigned char** jpegBuf, size_t* jpegSize,
    )

    int tj3DecompressHeader(
        tjhandle handle,
        const unsigned char* jpegBuf, size_t jpegSize,
    )

    int tj3Decompress8(
        tjhandle handle, const unsigned char* jpegBuf, size_t jpegSize,
        unsigned char* dstBuf, int pitch, int pixelFormat,
    )

    # ICC profile attach/retrieve. tj3SetICCProfile takes the bytes
    # the next tj3Compress8 should embed as an APP2 marker; libjpeg-
    # turbo copies them so the caller can free immediately.
    # tj3GetICCProfile allocates with tj3Alloc — caller must tj3Free.
    int tj3SetICCProfile(
        tjhandle handle, const unsigned char* iccBuf, size_t iccSize,
    )
    int tj3GetICCProfile(
        tjhandle handle, unsigned char** iccBuf, size_t* iccSize,
    )
