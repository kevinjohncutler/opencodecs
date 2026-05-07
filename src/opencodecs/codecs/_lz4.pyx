# opencodecs/codecs/_lz4.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native LZ4 codec — frame format (.lz4 file format)."""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.string cimport memset
from libc.stdint cimport uint8_t

from lz4 cimport (
    LZ4F_VERSION,
    LZ4F_preferences_t, LZ4F_frameInfo_t,
    LZ4F_compressFrame, LZ4F_compressFrameBound,
    LZ4F_dctx, LZ4F_createDecompressionContext, LZ4F_freeDecompressionContext,
    LZ4F_decompress, LZ4F_getFrameInfo,
    LZ4F_decompressOptions_t,
    LZ4F_isError, LZ4F_getErrorName,
)


class Lz4Error(RuntimeError):
    """Raised on LZ4 encode/decode failures."""


def encode(data, *, level: int | None = None) -> bytes:
    """Encode bytes-like input as an LZ4 frame."""
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        size_t dstcap
        size_t ret
        LZ4F_preferences_t prefs
        bytes out
        void* dst_ptr
        const void* src_ptr = NULL

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <size_t> src.shape[0]

    memset(<void*> &prefs, 0, sizeof(LZ4F_preferences_t))
    prefs.compressionLevel = 0 if level is None else int(level)

    if srcsize > 0:
        src_ptr = <const void*> &src[0]

    dstcap = LZ4F_compressFrameBound(srcsize, &prefs)
    out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dstcap)
    dst_ptr = <void*> PyBytes_AsString(out)
    with nogil:
        ret = LZ4F_compressFrame(dst_ptr, dstcap, src_ptr, srcsize, &prefs)
    if LZ4F_isError(ret):
        raise Lz4Error(
            f'LZ4F_compressFrame: {LZ4F_getErrorName(ret).decode()}')
    return out[:ret]


def decode(data) -> bytes:
    """Decode an LZ4 frame to bytes."""
    cdef:
        const uint8_t[::1] src
        size_t srcsize, src_consumed, src_remaining
        size_t dst_size, dst_used
        size_t ret
        LZ4F_dctx* dctx = NULL
        LZ4F_frameInfo_t info
        const void* src_ptr
        void* dst_ptr
        unsigned long long content_size
        bytes out
        size_t total_consumed
        Py_ssize_t chunk_cap
        size_t src_pos, this_dst, this_src
        bytes chunk
        bytearray out_buf

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if srcsize == 0:
        return b''

    ret = LZ4F_createDecompressionContext(&dctx, LZ4F_VERSION)
    if LZ4F_isError(ret):
        raise Lz4Error(
            f'LZ4F_createDecompressionContext: '
            f'{LZ4F_getErrorName(ret).decode()}')
    try:
        # Peek frame info to learn the content size if it's recorded.
        memset(<void*> &info, 0, sizeof(LZ4F_frameInfo_t))
        src_consumed = srcsize
        src_ptr = <const void*> &src[0]
        ret = LZ4F_getFrameInfo(dctx, &info, src_ptr, &src_consumed)
        if LZ4F_isError(ret):
            raise Lz4Error(
                f'LZ4F_getFrameInfo: {LZ4F_getErrorName(ret).decode()}')

        # If contentSize is in the frame header, allocate exactly that
        # size and decode in one shot (the common case for files
        # encoded with content-size enabled).
        # Otherwise, decompress in chunks of `chunk_cap` bytes into a
        # growing bytearray. The previous "4× source" heuristic broke
        # for highly-compressible inputs (16 MB of zeros → 69 KB
        # encoded → 4× = 276 KB, but actual output is 16 MB).
        src_pos = src_consumed

        if info.contentSize > 0:
            dst_size = <size_t> info.contentSize
            out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dst_size)
            dst_ptr = <void*> PyBytes_AsString(out)
            dst_used = dst_size
            src_remaining = srcsize - src_consumed
            with nogil:
                ret = LZ4F_decompress(
                    dctx, dst_ptr, &dst_used,
                    <const void*> (&src[0] + src_consumed), &src_remaining,
                    NULL)
            if LZ4F_isError(ret):
                raise Lz4Error(
                    f'LZ4F_decompress: {LZ4F_getErrorName(ret).decode()}')
            if ret != 0:
                raise Lz4Error(
                    'LZ4F_decompress: contentSize header was wrong '
                    '(decoder requests more input)')
            return out[:dst_used]

        # No content-size hint — chunked decode into a bytearray. Use a
        # 1 MB working chunk; the loop handles inputs of any expansion
        # ratio without needing to pre-size the output.
        chunk_cap = 1 << 20
        out_buf = bytearray()
        chunk = PyBytes_FromStringAndSize(NULL, chunk_cap)
        dst_ptr = <void*> PyBytes_AsString(chunk)

        while True:
            this_dst = chunk_cap
            this_src = srcsize - src_pos
            with nogil:
                ret = LZ4F_decompress(
                    dctx, dst_ptr, &this_dst,
                    <const void*> (&src[0] + src_pos), &this_src,
                    NULL)
            if LZ4F_isError(ret):
                raise Lz4Error(
                    f'LZ4F_decompress: {LZ4F_getErrorName(ret).decode()}')
            if this_dst:
                out_buf.extend(chunk[:this_dst])
            src_pos += this_src
            if ret == 0:
                # Frame complete.
                break
            if this_src == 0 and this_dst == 0:
                # Decoder consumed nothing and produced nothing — stuck.
                raise Lz4Error('LZ4F_decompress: stalled (no progress)')
        return bytes(out_buf)
    finally:
        LZ4F_freeDecompressionContext(dctx)


def check_signature(data) -> bool:
    """True if `data` starts with the LZ4 frame magic 0x184D2204."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:4])
    else:
        try:
            head = bytes(data)[:4]
        except Exception:
            return False
    return head == b'\x04\x22\x4d\x18'
