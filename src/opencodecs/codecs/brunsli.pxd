# cython: language_level = 3
"""Cython header for the Brunsli C API (libbrunslienc-c / libbrunslidec-c)."""

# These declarations transcribe upstream 'brunsli/decode.h' and therefore overlap
# substantially with imagecodecs/brunsli.pxd (BSD-3-Clause, Copyright (c)
# 2008-2026 Christoph Gohlke), which transcribes the same header.
# Credited here out of caution; see THIRD-PARTY.md.

from libc.stdint cimport uint8_t


cdef extern from "brunsli/decode.h" nogil:
    ctypedef size_t (*DecodeBrunsliSink)(
        void* sink, const uint8_t* buf, size_t size,
    ) nogil

    int DecodeBrunsli(
        size_t size, const uint8_t* data,
        void* sink, DecodeBrunsliSink out_fun,
    )


cdef extern from "brunsli/encode.h" nogil:
    int EncodeBrunsli(
        size_t size, const unsigned char* data,
        void* sink, DecodeBrunsliSink out_fun,
    )
