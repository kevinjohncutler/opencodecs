# opencodecs/codecs/_bitshuffle.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native bitshuffle filter — bit-level transpose for typed numerical data.

bitshuffle is a *pre-filter* (not a stand-alone compressor). For each element
of an N-element typed array, the bits at bit-position k are gathered together
across the array. The output has the same size as the input but is far more
LZ77-friendly when there is bit-level correlation (most scientific data).

Pair with zstd / lz4 / blosc2 for actual compression. The blosc2 codec
already wraps bitshuffle internally via `shuffle="bit"`; this codec
exposes the raw transform for use with non-blosc compressors and for
ndarray reshaping pipelines.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdint cimport int64_t, uint8_t

from bitshuffle cimport (
    BSHUF_VERSION_MAJOR, BSHUF_VERSION_MINOR, BSHUF_VERSION_POINT,
    bshuf_default_block_size,
    bshuf_bitshuffle, bshuf_bitunshuffle,
)


class BitshuffleError(RuntimeError):
    """Raised on bitshuffle encode/decode failures."""


_ERROR_MESSAGES = {
    -1: 'failed to allocate memory',
    -11: 'missing SSE',
    -12: 'missing AVX',
    -13: 'missing ARM Neon',
    -14: 'missing AVX512',
    -80: 'input size not a multiple of 8',
    -81: 'block_size not a multiple of 8',
    -91: 'decompression error, wrong number of bytes processed',
}


def _err(func, code):
    msg = _ERROR_MESSAGES.get(int(code), f'unknown error {int(code)}')
    return BitshuffleError(f'{func} returned {msg}')


def version() -> str:
    """Return bitshuffle library version string."""
    return f'{BSHUF_VERSION_MAJOR}.{BSHUF_VERSION_MINOR}.{BSHUF_VERSION_POINT}'


def encode(data, *, itemsize: int = 1, blocksize: int = 0) -> bytes:
    """Bitshuffle a typed byte buffer.

    Parameters
    ----------
    data : bytes-like
        Raw bytes representing ``N`` elements of ``itemsize`` bytes each.
    itemsize : int
        Element size in bytes (1, 2, 4, 8 are typical).
    blocksize : int
        Number of elements per processing block. 0 = library default.

    Returns
    -------
    bytes
        Bitshuffled output, same length as the input.
    """
    cdef:
        const uint8_t[::1] src
        Py_ssize_t srcsize
        size_t elem_size
        size_t block_size
        size_t nelem
        bytes out
        void* dst_ptr
        int64_t ret

    if itemsize < 1:
        raise ValueError(f'invalid itemsize {itemsize!r}')
    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = src.shape[0]
    if srcsize == 0:
        return b''
    if srcsize % itemsize != 0:
        raise ValueError(
            f'input size {srcsize} is not a multiple of itemsize {itemsize}'
        )

    elem_size = <size_t> itemsize
    block_size = <size_t> max(int(blocksize), 0)
    nelem = <size_t> (srcsize // itemsize)

    out = PyBytes_FromStringAndSize(NULL, srcsize)
    dst_ptr = <void*> PyBytes_AsString(out)

    with nogil:
        ret = bshuf_bitshuffle(
            <const void*> &src[0], dst_ptr,
            nelem, elem_size, block_size,
        )
    if ret < 0:
        raise _err('bshuf_bitshuffle', ret)
    return out


def decode(data, *, itemsize: int = 1, blocksize: int = 0) -> bytes:
    """Reverse a bitshuffle. Output has same length as input."""
    cdef:
        const uint8_t[::1] src
        Py_ssize_t srcsize
        size_t elem_size
        size_t block_size
        size_t nelem
        bytes out
        void* dst_ptr
        int64_t ret

    if itemsize < 1:
        raise ValueError(f'invalid itemsize {itemsize!r}')
    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = src.shape[0]
    if srcsize == 0:
        return b''
    if srcsize % itemsize != 0:
        raise ValueError(
            f'input size {srcsize} is not a multiple of itemsize {itemsize}'
        )

    elem_size = <size_t> itemsize
    block_size = <size_t> max(int(blocksize), 0)
    nelem = <size_t> (srcsize // itemsize)

    out = PyBytes_FromStringAndSize(NULL, srcsize)
    dst_ptr = <void*> PyBytes_AsString(out)

    with nogil:
        ret = bshuf_bitunshuffle(
            <const void*> &src[0], dst_ptr,
            nelem, elem_size, block_size,
        )
    if ret < 0:
        raise _err('bshuf_bitunshuffle', ret)
    return out


def default_blocksize(itemsize: int) -> int:
    """Return the library's default block_size for a given itemsize."""
    if itemsize < 1:
        raise ValueError(f'invalid itemsize {itemsize!r}')
    return int(bshuf_default_block_size(<size_t> itemsize))


def check_signature(data) -> bool:
    """Bitshuffle is a filter, not a container — no magic. Always False."""
    return False
