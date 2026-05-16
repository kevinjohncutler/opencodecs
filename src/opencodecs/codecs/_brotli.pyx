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
        const uint8_t[::1] dst   # memoryview view onto the output bytes
        size_t srcsize
        size_t dstcap
        size_t encoded_size
        int quality
        BROTLI_BOOL ok
        bytes out
        const uint8_t* src_ptr = NULL

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
    # bytes → uint8_t[::1] memoryview cast (instead of PyBytes_AsString);
    # combined with the ``out[:encoded_size]`` slice + ``del dst`` below
    # this matches imagecodecs's encode-output pattern and is measurably
    # faster than the PyBytes_AsString + _PyBytes_Resize alternative on
    # 10 MB+ payloads. See _zstd.encode() for the empirical comparison.
    dst = out
    if srcsize > 0:
        src_ptr = <const uint8_t*> &src[0]
    encoded_size = dstcap
    with nogil:
        ok = BrotliEncoderCompress(
            quality, BROTLI_DEFAULT_WINDOW, BROTLI_MODE_GENERIC,
            srcsize, src_ptr, &encoded_size,
            <uint8_t*> &dst[0],
        )
    if ok == BROTLI_FALSE:
        raise BrotliError('BrotliEncoderCompress failed')
    del dst
    return out[:encoded_size]


def decode(data, *, out=None):
    """Decode a brotli stream.

    Parameters
    ----------
    out : int | bytearray | memoryview | None, optional
        See ``_zstd.decode`` for the full ``out=`` contract.
    """
    cdef:
        const uint8_t[::1] src
        uint8_t[::1] out_view             # writable view of caller buffer
        size_t srcsize, available_in, available_out, total_out
        const uint8_t* next_in
        uint8_t* next_out
        uint8_t* buf = NULL
        size_t bufcap
        bint can_grow
        BrotliDecoderState* state = NULL
        BrotliDecoderResult res
        int err
        bytes out_bytes

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if srcsize == 0:
        if out is None or isinstance(out, int):
            return b''
        return out[:0]

    state = BrotliDecoderCreateInstance(NULL, NULL, NULL)
    if state == NULL:
        raise BrotliError('BrotliDecoderCreateInstance failed')

    # ----- caller-supplied writable buffer (zero-alloc path) -----
    if out is not None and not isinstance(out, int):
        try:
            out_view = out
        except (TypeError, ValueError, BufferError) as e:
            BrotliDecoderDestroyInstance(state)
            raise TypeError(
                f"brotli decode: out= must be int or writable buffer, "
                f"got {type(out).__name__}"
            ) from e
        try:
            available_in = srcsize
            next_in = <const uint8_t*> &src[0]
            total_out = 0
            available_out = <size_t> out_view.shape[0]
            next_out = &out_view[0]
            with nogil:
                res = BrotliDecoderDecompressStream(
                    state, &available_in, &next_in,
                    &available_out, &next_out, &total_out,
                )
            if res == BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT:
                raise BrotliError(
                    'brotli decode: out= buffer too small')
            if res == BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT:
                raise BrotliError('truncated brotli stream')
            if res != BROTLI_DECODER_RESULT_SUCCESS:
                err = BrotliDecoderGetErrorCode(state)
                raise BrotliError(
                    f'BrotliDecoderDecompressStream: '
                    f'{BrotliDecoderErrorString(err).decode()}')
            del out_view
            return out[:total_out]
        finally:
            BrotliDecoderDestroyInstance(state)

    # ----- fresh bytes allocation -----
    if isinstance(out, int):
        if out < 0:
            BrotliDecoderDestroyInstance(state)
            raise ValueError("brotli decode: out=int(N) requires N >= 0")
        bufcap = <size_t> out
        can_grow = False
    else:
        bufcap = max(<size_t> 4 * srcsize, <size_t> 65536)
        can_grow = True

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
                if not can_grow:
                    raise BrotliError(
                        'brotli decode: out= int hint too small')
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

        out_bytes = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> total_out)
        if total_out > 0:
            memcpy(<void*> PyBytes_AsString(out_bytes), buf, total_out)
        return out_bytes
    finally:
        free(buf)
        BrotliDecoderDestroyInstance(state)


def check_signature(data) -> bool:
    """Brotli has no fixed magic — always returns False (not auto-detectable)."""
    # Brotli streams have no magic bytes. Return False to avoid false-positives
    # in `codec_for_bytes`. Callers must use extension/format= dispatch.
    return False
