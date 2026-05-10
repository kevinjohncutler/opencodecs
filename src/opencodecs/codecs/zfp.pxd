# cython: language_level = 3
"""Cython header for libzfp + bitstream."""

from libc.stddef cimport size_t


cdef extern from "zfp/bitstream.h" nogil:

    ctypedef struct bitstream:
        pass

    bitstream* stream_open(void* buffer, size_t bytes)
    void stream_close(bitstream* stream)
    size_t stream_size(const bitstream* stream)


cdef extern from "zfp.h" nogil:

    ctypedef enum zfp_type:
        zfp_type_none   = 0
        zfp_type_int32  = 1
        zfp_type_int64  = 2
        zfp_type_float  = 3
        zfp_type_double = 4

    ctypedef struct zfp_field:
        zfp_type type
        unsigned int nx
        unsigned int ny
        unsigned int nz
        unsigned int nw
        int sx
        int sy
        int sz
        int sw
        void* data

    ctypedef struct zfp_stream:
        pass

    int ZFP_HEADER_NONE
    int ZFP_HEADER_MAGIC
    int ZFP_HEADER_META
    int ZFP_HEADER_MODE
    int ZFP_HEADER_FULL

    zfp_field* zfp_field_alloc()
    zfp_field* zfp_field_1d(void* pointer, zfp_type type, unsigned int nx)
    zfp_field* zfp_field_2d(void* pointer, zfp_type type,
                            unsigned int nx, unsigned int ny)
    zfp_field* zfp_field_3d(void* pointer, zfp_type type,
                            unsigned int nx, unsigned int ny, unsigned int nz)
    zfp_field* zfp_field_4d(void* pointer, zfp_type type,
                            unsigned int nx, unsigned int ny,
                            unsigned int nz, unsigned int nw)
    void zfp_field_free(zfp_field* field)
    void zfp_field_set_pointer(zfp_field* field, void* pointer)

    zfp_stream* zfp_stream_open(bitstream* stream)
    void zfp_stream_close(zfp_stream* stream)
    size_t zfp_stream_maximum_size(const zfp_stream* stream,
                                   const zfp_field* field)
    void zfp_stream_set_bit_stream(zfp_stream* stream, bitstream* bs)
    void zfp_stream_rewind(zfp_stream* stream)
    int zfp_stream_set_reversible(zfp_stream* stream)
    double zfp_stream_set_rate(zfp_stream* stream, double rate, zfp_type type,
                               unsigned int dims, int wra)
    unsigned int zfp_stream_set_precision(zfp_stream* stream,
                                          unsigned int precision)
    double zfp_stream_set_accuracy(zfp_stream* stream, double tolerance)

    size_t zfp_compress(zfp_stream* stream, const zfp_field* field)
    size_t zfp_decompress(zfp_stream* stream, zfp_field* field)
    size_t zfp_write_header(zfp_stream* stream, const zfp_field* field, unsigned int mask)
    size_t zfp_read_header(zfp_stream* stream, zfp_field* field, unsigned int mask)
