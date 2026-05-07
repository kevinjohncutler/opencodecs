# Cython declarations for the QOI single-header library
# (https://github.com/phoboslab/qoi). The .h is vendored at
# 3rdparty/qoi/qoi.h and compiled into our extension via
# QOI_IMPLEMENTATION.

cdef extern from 'qoi.h' nogil:
    int QOI_SRGB
    int QOI_LINEAR

    ctypedef struct qoi_desc:
        unsigned int width
        unsigned int height
        unsigned char channels
        unsigned char colorspace

    void* qoi_encode(
        const void* data,
        const qoi_desc* desc,
        int* out_len
    )

    void* qoi_decode(
        const void* data,
        int size,
        qoi_desc* desc,
        int channels
    )
