# cython: language_level = 3
"""Cython header for libaec (CCSDS 121.0-B-2 adaptive entropy coding)."""

# Transcribed from upstream 'libaec.h'. imagecodecs declares the same C
# API in its own libaec.pxd; the two overlap because the header fixes the
# names and signatures, not because either was copied from the other.

from libc.stddef cimport size_t


cdef extern from "libaec.h" nogil:

    int AEC_OK
    int AEC_CONF_ERROR
    int AEC_STREAM_ERROR
    int AEC_DATA_ERROR
    int AEC_MEM_ERROR

    int AEC_DATA_SIGNED
    int AEC_DATA_3BYTE
    int AEC_DATA_MSB
    int AEC_DATA_PREPROCESS
    int AEC_RESTRICTED
    int AEC_PAD_RSI
    int AEC_NOT_ENFORCE

    int AEC_NO_FLUSH
    int AEC_FLUSH

    struct aec_stream:
        const unsigned char* next_in
        size_t avail_in
        size_t total_in
        unsigned char* next_out
        size_t avail_out
        size_t total_out
        unsigned int bits_per_sample
        unsigned int block_size
        unsigned int rsi
        unsigned int flags
        void* state

    int aec_buffer_encode(aec_stream* strm)
    int aec_buffer_decode(aec_stream* strm)

    # Streaming API — measurably faster than ``aec_buffer_encode`` because
    # the one-shot wrapper allocates and frees the internal state on every
    # call. Use init / encode(AEC_FLUSH) / end for the lowest-overhead
    # single-shot encode (the buffer path is just a convenience layer
    # around this trio internally).
    int aec_encode_init(aec_stream* strm)
    int aec_encode(aec_stream* strm, int flush)
    int aec_encode_end(aec_stream* strm)
