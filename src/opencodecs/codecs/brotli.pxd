# Minimal Cython declarations for libbrotli (encode + decode).

from libc.stdint cimport uint8_t

cdef extern from 'brotli/encode.h' nogil:
    ctypedef enum BROTLI_BOOL:
        BROTLI_FALSE
        BROTLI_TRUE

    ctypedef enum BrotliEncoderMode:
        BROTLI_MODE_GENERIC
        BROTLI_MODE_TEXT
        BROTLI_MODE_FONT

    int BROTLI_MIN_QUALITY
    int BROTLI_MAX_QUALITY
    int BROTLI_DEFAULT_QUALITY
    int BROTLI_MIN_WINDOW_BITS
    int BROTLI_MAX_WINDOW_BITS
    int BROTLI_DEFAULT_WINDOW
    int BROTLI_DEFAULT_MODE

    size_t BrotliEncoderMaxCompressedSize(size_t input_size)

    BROTLI_BOOL BrotliEncoderCompress(
        int quality, int lgwin, BrotliEncoderMode mode,
        size_t input_size, const uint8_t* input_buffer,
        size_t* encoded_size, uint8_t* encoded_buffer,
    )

cdef extern from 'brotli/decode.h' nogil:
    ctypedef enum BrotliDecoderResult:
        BROTLI_DECODER_RESULT_ERROR
        BROTLI_DECODER_RESULT_SUCCESS
        BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT
        BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT

    ctypedef struct BrotliDecoderState:
        pass

    BrotliDecoderState* BrotliDecoderCreateInstance(
        void* alloc_func, void* free_func, void* opaque,
    )
    void BrotliDecoderDestroyInstance(BrotliDecoderState* state)

    BrotliDecoderResult BrotliDecoderDecompressStream(
        BrotliDecoderState* state,
        size_t* available_in, const uint8_t** next_in,
        size_t* available_out, uint8_t** next_out,
        size_t* total_out,
    )

    BROTLI_BOOL BrotliDecoderIsFinished(const BrotliDecoderState* state)
    int BrotliDecoderGetErrorCode(const BrotliDecoderState* state)
    const char* BrotliDecoderErrorString(int c)
