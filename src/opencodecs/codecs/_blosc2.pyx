# opencodecs/codecs/_blosc2.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native blosc2 codec — bytes-in / bytes-out compression."""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdint cimport int32_t, uint8_t

from blosc2 cimport (
    BLOSC2_MAX_OVERHEAD, BLOSC_SHUFFLE, BLOSC_NOSHUFFLE,
    blosc2_compress, blosc2_decompress, blosc2_cbuffer_sizes,
    blosc1_set_compressor,
)


class Blosc2Error(RuntimeError):
    """Raised on blosc2 encode/decode failures."""


def encode(data, *, level: int | None = None,
           compressor: str | None = None,
           typesize: int | None = None,
           shuffle: bool | None = None) -> bytes:
    """Encode bytes-like input as a blosc2 chunk.

    ``compressor`` selects the inner codec ("blosclz", "lz4", "lz4hc",
    "zlib", "zstd"). Default is blosc2's compile-time default (zstd as
    of c-blosc2 2.x).
    """
    cdef:
        const uint8_t[::1] src
        int32_t srcsize
        int32_t dstcap
        int ret
        int clevel
        int do_shuffle
        int32_t tsize
        bytes out
        const void* src_ptr = NULL
        void* dst_ptr

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <int32_t> src.shape[0]

    if compressor is not None:
        cname = compressor.encode() if isinstance(compressor, str) else compressor
        if blosc1_set_compressor(cname) < 0:
            raise Blosc2Error(f'unknown blosc2 compressor: {compressor!r}')

    clevel = 5 if level is None else int(level)
    if clevel < 0: clevel = 0
    if clevel > 9: clevel = 9

    do_shuffle = BLOSC_SHUFFLE if (shuffle is None or shuffle) else BLOSC_NOSHUFFLE
    tsize = 1 if typesize is None else int(typesize)
    if tsize < 1: tsize = 1

    dstcap = srcsize + BLOSC2_MAX_OVERHEAD
    out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dstcap)
    dst_ptr = <void*> PyBytes_AsString(out)
    if srcsize > 0:
        src_ptr = <const void*> &src[0]

    with nogil:
        ret = blosc2_compress(
            clevel, do_shuffle, tsize,
            src_ptr, srcsize, dst_ptr, dstcap,
        )
    if ret < 0:
        raise Blosc2Error(f'blosc2_compress failed: {ret}')
    return out[:ret]


def decode(data, *, out=None):
    """Decode a blosc2 chunk.

    Parameters
    ----------
    out : int | bytearray | memoryview | None, optional
        See ``_zstd.decode`` for the full ``out=`` contract.
    """
    cdef:
        const uint8_t[::1] src
        uint8_t[::1] out_view             # writable view of caller buffer
        int32_t srcsize
        int32_t nbytes = 0, cbytes = 0, blocksize = 0
        int ret
        bytes out_bytes
        void* dst_ptr

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <int32_t> src.shape[0]
    if srcsize == 0:
        if out is None or isinstance(out, int):
            return b''
        return out[:0]
    if srcsize < BLOSC2_MAX_OVERHEAD:
        # Not strictly required to be >= overhead; tiny chunks are still
        # legal — but we still need the header to learn nbytes.
        if srcsize < 32:
            raise Blosc2Error('blosc2 chunk too small to read header')

    ret = blosc2_cbuffer_sizes(<const void*> &src[0], &nbytes, &cbytes, &blocksize)
    if ret < 0:
        raise Blosc2Error(f'blosc2_cbuffer_sizes failed: {ret}')

    cdef const void* src_ptr = <const void*> &src[0]

    # ----- caller-supplied writable buffer (zero-alloc path) -----
    if out is not None and not isinstance(out, int):
        try:
            out_view = out
        except (TypeError, ValueError, BufferError) as e:
            raise TypeError(
                f"blosc2 decode: out= must be int or writable buffer, "
                f"got {type(out).__name__}"
            ) from e
        if out_view.shape[0] < nbytes:
            raise Blosc2Error(
                f"blosc2 decode: out= buffer is {out_view.shape[0]} bytes "
                f"but the blosc2 header declares {nbytes} bytes")
        dst_ptr = <void*> &out_view[0]
        with nogil:
            ret = blosc2_decompress(src_ptr, srcsize, dst_ptr, nbytes)
        if ret < 0:
            raise Blosc2Error(f'blosc2_decompress failed: {ret}')
        del out_view
        return out[:ret]

    # ----- fresh bytes allocation -----
    if isinstance(out, int):
        if out < nbytes:
            raise Blosc2Error(
                f"blosc2 decode: out=int({out}) is less than the "
                f"blosc2 header's declared {nbytes} bytes")
    out_bytes = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> nbytes)
    dst_ptr = <void*> PyBytes_AsString(out_bytes)
    with nogil:
        ret = blosc2_decompress(src_ptr, srcsize, dst_ptr, nbytes)
    if ret < 0:
        raise Blosc2Error(f'blosc2_decompress failed: {ret}')
    return out_bytes[:ret]


def check_signature(data) -> bool:
    """True if `data` starts with a blosc2 frame/chunk magic byte (0x02)."""
    # blosc2 chunks start with version byte = 0x02. Not unique enough for
    # reliable auto-detection, so callers should use format=/extension.
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:1])
    else:
        try:
            head = bytes(data)[:1]
        except Exception:
            return False
    return head == b'\x02'
