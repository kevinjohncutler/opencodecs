# opencodecs/codecs/_brotli.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native brotli codec — bytes-in / bytes-out compression."""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdlib cimport realloc, free
from libc.string cimport memcpy
from libc.stdint cimport uint8_t

from brotli cimport (
    BROTLI_BOOL, BROTLI_TRUE, BROTLI_FALSE,
    BrotliEncoderMode, BROTLI_MODE_GENERIC,
    BROTLI_DEFAULT_QUALITY, BROTLI_DEFAULT_WINDOW, BROTLI_MAX_QUALITY,
    BROTLI_MIN_QUALITY,
    BrotliEncoderMaxCompressedSize, BrotliEncoderCompress,
    BrotliDecoderState, BrotliDecoderResult,
    BROTLI_DECODER_RESULT_SUCCESS, BROTLI_DECODER_RESULT_ERROR,
    BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT,
    BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT,
    BrotliDecoderCreateInstance, BrotliDecoderDestroyInstance,
    BrotliDecoderDecompressStream,
    BrotliDecoderGetErrorCode, BrotliDecoderErrorString,
)


class BrotliError(RuntimeError):
    """Raised on brotli encode/decode failures."""


def encode(data, *, level: int | None = None) -> bytes:
    """Encode bytes-like input as a brotli stream."""
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        size_t dstcap
        size_t encoded_size
        int quality
        BROTLI_BOOL ok
        bytes out
        const uint8_t* src_ptr = NULL
        uint8_t* dst_ptr

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <size_t> src.shape[0]

    if level is None:
        quality = BROTLI_DEFAULT_QUALITY
    else:
        quality = int(level)
    if quality < BROTLI_MIN_QUALITY: quality = BROTLI_MIN_QUALITY
    if quality > BROTLI_MAX_QUALITY: quality = BROTLI_MAX_QUALITY

    dstcap = BrotliEncoderMaxCompressedSize(srcsize)
    if dstcap == 0:
        # Per docs, returns 0 if input too large for a single call. Fall back
        # to a generous bound; brotli rarely expands by more than ~0.04%.
        dstcap = srcsize + (srcsize >> 4) + 64
    out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dstcap)
    dst_ptr = <uint8_t*> PyBytes_AsString(out)
    if srcsize > 0:
        src_ptr = <const uint8_t*> &src[0]
    encoded_size = dstcap
    with nogil:
        ok = BrotliEncoderCompress(
            quality, BROTLI_DEFAULT_WINDOW, BROTLI_MODE_GENERIC,
            srcsize, src_ptr, &encoded_size, dst_ptr,
        )
    if ok == BROTLI_FALSE:
        raise BrotliError('BrotliEncoderCompress failed')
    return out[:encoded_size]


def decode(data) -> bytes:
    """Decode a brotli stream to bytes."""
    cdef:
        const uint8_t[::1] src
        size_t srcsize, available_in, available_out, total_out
        const uint8_t* next_in
        uint8_t* next_out
        uint8_t* buf = NULL
        size_t bufcap
        BrotliDecoderState* state = NULL
        BrotliDecoderResult res
        int err
        bytes out

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if srcsize == 0:
        return b''

    state = BrotliDecoderCreateInstance(NULL, NULL, NULL)
    if state == NULL:
        raise BrotliError('BrotliDecoderCreateInstance failed')

    bufcap = max(<size_t> 4 * srcsize, <size_t> 65536)
    buf = <uint8_t*> realloc(NULL, bufcap)
    if buf == NULL:
        BrotliDecoderDestroyInstance(state)
        raise MemoryError()
    try:
        available_in = srcsize
        next_in = <const uint8_t*> &src[0]
        total_out = 0
        while True:
            available_out = bufcap - total_out
            next_out = buf + total_out
            with nogil:
                res = BrotliDecoderDecompressStream(
                    state, &available_in, &next_in,
                    &available_out, &next_out, &total_out,
                )
            if res == BROTLI_DECODER_RESULT_SUCCESS:
                break
            if res == BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT:
                # Grow output buffer.
                bufcap *= 2
                tmp = <uint8_t*> realloc(buf, bufcap)
                if tmp == NULL:
                    raise MemoryError()
                buf = tmp
                continue
            if res == BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT:
                raise BrotliError('truncated brotli stream')
            err = BrotliDecoderGetErrorCode(state)
            raise BrotliError(
                f'BrotliDecoderDecompressStream: '
                f'{BrotliDecoderErrorString(err).decode()}')

        out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> total_out)
        if total_out > 0:
            memcpy(<void*> PyBytes_AsString(out), buf, total_out)
        return out
    finally:
        free(buf)
        BrotliDecoderDestroyInstance(state)


def check_signature(data) -> bool:
    """Brotli has no fixed magic — always returns False (not auto-detectable)."""
    # Brotli streams have no magic bytes. Return False to avoid false-positives
    # in `codec_for_bytes`. Callers must use extension/format= dispatch.
    return False
