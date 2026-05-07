# opencodecs/codecs/_zstd.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native zstd codec — bytes-in / bytes-out compression."""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdint cimport uint8_t

from zstd cimport (
    ZSTD_compress, ZSTD_decompress,
    ZSTD_compressBound, ZSTD_getFrameContentSize,
    ZSTD_minCLevel, ZSTD_maxCLevel,
    ZSTD_CLEVEL_DEFAULT, ZSTD_isError, ZSTD_getErrorName,
    ZSTD_CONTENTSIZE_UNKNOWN, ZSTD_CONTENTSIZE_ERROR,
    ZSTD_VERSION_MAJOR, ZSTD_VERSION_MINOR, ZSTD_VERSION_RELEASE,
)


class ZstdError(RuntimeError):
    """Raised on zstd encode/decode failures."""


def libzstd_version() -> str:
    return f'{ZSTD_VERSION_MAJOR}.{ZSTD_VERSION_MINOR}.{ZSTD_VERSION_RELEASE}'


def encode(data, *, level: int | None = None) -> bytes:
    """Encode bytes-like input as a zstd frame.

    Accepts any buffer-protocol object that exposes a 1D contiguous
    uint8 view — bytes, bytearray, memoryview, mmap, numpy uint8 arrays.
    Anything else is coerced via ``bytes(data)``.
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        size_t dstcap
        size_t ret
        int lvl
        int min_l, max_l
        bytes out

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <size_t> src.shape[0]

    min_l = ZSTD_minCLevel()
    max_l = ZSTD_maxCLevel()
    if level is None:
        lvl = ZSTD_CLEVEL_DEFAULT
    else:
        lvl = int(level)
    if lvl < min_l: lvl = min_l
    if lvl > max_l: lvl = max_l

    dstcap = ZSTD_compressBound(srcsize)
    out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dstcap)
    cdef void* dst_ptr = <void*> PyBytes_AsString(out)
    cdef const void* src_ptr = NULL
    if srcsize > 0:
        src_ptr = <const void*> &src[0]

    with nogil:
        ret = ZSTD_compress(dst_ptr, dstcap, src_ptr, srcsize, lvl)
    if ZSTD_isError(ret):
        raise ZstdError(
            f'ZSTD_compress: {ZSTD_getErrorName(ret).decode()}')
    return out[:ret]


def decode(data) -> bytes:
    """Decode a zstd frame to bytes.

    Accepts any buffer-protocol object (bytes, bytearray, memoryview,
    mmap, numpy uint8). For mmap-backed memoryviews this is a true
    zero-copy path — no bytes() materialisation before the codec call.
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        unsigned long long content_size
        size_t dstcap
        size_t ret
        bytes out

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if srcsize == 0:
        return b''

    content_size = ZSTD_getFrameContentSize(<const void*> &src[0], srcsize)
    if content_size == <unsigned long long> ZSTD_CONTENTSIZE_ERROR:
        raise ZstdError('ZSTD_getFrameContentSize: not a zstd frame')
    if content_size == <unsigned long long> ZSTD_CONTENTSIZE_UNKNOWN:
        # Streaming-encoded (no size header) — pick a generous starting
        # capacity and grow until we succeed. We try 4× input first.
        dstcap = max(<size_t> 4 * srcsize, <size_t> 65536)
    else:
        dstcap = <size_t> content_size

    cdef void* dst_ptr
    cdef const void* src_ptr = <const void*> &src[0]
    while True:
        out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dstcap)
        dst_ptr = <void*> PyBytes_AsString(out)
        with nogil:
            ret = ZSTD_decompress(dst_ptr, dstcap, src_ptr, srcsize)
        if not ZSTD_isError(ret):
            return out[:ret]
        # If content_size was unknown and our guess was too small, grow.
        if content_size == <unsigned long long> ZSTD_CONTENTSIZE_UNKNOWN \
                and dstcap < <size_t> (1 << 32):
            dstcap *= 2
            continue
        raise ZstdError(
            f'ZSTD_decompress: {ZSTD_getErrorName(ret).decode()}')


def check_signature(data) -> bool:
    """True if `data` starts with the zstd frame magic 0x28B52FFD."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:4])
    else:
        try:
            head = bytes(data)[:4]
        except Exception:
            return False
    return head == b'\x28\xb5\x2f\xfd'
