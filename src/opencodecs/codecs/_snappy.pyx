# opencodecs/codecs/_snappy.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native Snappy codec — Google's fast block compression.

Bytes-in / bytes-out wrapper around libsnappy's C API (snappy-c.h).
Snappy targets the ~500 MB/s throughput regime with ~2x compression
ratios — useful in Parquet/Hadoop/Bigtable pipelines where speed
matters more than tight ratios.

This is raw block format (no framing / no checksums); for the
``.sz`` framing format used by Hadoop SequenceFiles, layer your own
framing on top.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t

from snappy cimport (
    snappy_status, SNAPPY_OK, SNAPPY_INVALID_INPUT, SNAPPY_BUFFER_TOO_SMALL,
    snappy_compress, snappy_uncompress,
    snappy_max_compressed_length, snappy_uncompressed_length,
    snappy_validate_compressed_buffer,
)


class SnappyError(RuntimeError):
    """Raised on Snappy encode/decode failures."""


def encode(data) -> bytes:
    """Compress bytes-like input as a raw Snappy block.

    Accepts any buffer-protocol object (bytes, bytearray, memoryview,
    numpy uint8 arrays). Returns the compressed bytes; raises
    :class:`SnappyError` on internal failure (rare — Snappy doesn't
    have many failure modes on the encode side).
    """
    cdef:
        const uint8_t[::1] src
        const uint8_t[::1] dst
        size_t srcsize, dstcap, dstlen
        bytes out
        snappy_status rc

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <size_t> src.shape[0]

    dstcap = snappy_max_compressed_length(srcsize)
    out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dstcap)
    # Memoryview output pattern (matches _zstd / _brotli / _lerc) —
    # measurably faster than PyBytes_AsString + _PyBytes_Resize on
    # multi-MB outputs. See _zstd.encode for the empirical rationale.
    dst = out
    dstlen = dstcap

    cdef const char* src_p = NULL
    if srcsize > 0:
        src_p = <const char*> &src[0]

    with nogil:
        rc = snappy_compress(src_p, srcsize, <char*> &dst[0], &dstlen)
    if rc != SNAPPY_OK:
        raise SnappyError(f"snappy_compress failed: status={rc}")
    del dst
    return out[:dstlen]


def decode(data, *, out=None):
    """Decompress a Snappy block.

    The uncompressed size is stored in the block header (Snappy's
    first varint), so we can pre-allocate the exact output buffer
    in one shot — no growing-buffer retry loop.

    Parameters
    ----------
    out : int | bytearray | memoryview | None, optional
        See ``_zstd.decode`` for the full ``out=`` contract.
    """
    cdef:
        const uint8_t[::1] src
        const uint8_t[::1] dst                # output bytes view
        uint8_t[::1] out_view                 # writable view of caller buffer
        size_t srcsize
        size_t dstlen = 0
        size_t out_len
        bytes out_bytes
        snappy_status rc
        char* dst_ptr

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if srcsize == 0:
        if out is None or isinstance(out, int):
            return b''
        return out[:0]

    cdef const char* src_p = <const char*> &src[0]
    rc = snappy_uncompressed_length(src_p, srcsize, &dstlen)
    if rc != SNAPPY_OK:
        raise SnappyError(
            f"snappy_uncompressed_length failed: status={rc} "
            f"(not a valid Snappy block?)"
        )

    # ----- caller-supplied writable buffer (zero-alloc path) -----
    if out is not None and not isinstance(out, int):
        try:
            out_view = out
        except (TypeError, ValueError, BufferError) as e:
            raise TypeError(
                f"snappy decode: out= must be int or writable buffer, "
                f"got {type(out).__name__}"
            ) from e
        if out_view.shape[0] < dstlen:
            raise SnappyError(
                f"snappy decode: out= buffer is {out_view.shape[0]} bytes "
                f"but the Snappy header declares {dstlen} bytes")
        out_len = <size_t> out_view.shape[0]
        dst_ptr = <char*> &out_view[0]
        with nogil:
            rc = snappy_uncompress(src_p, srcsize, dst_ptr, &out_len)
        if rc != SNAPPY_OK:
            raise SnappyError(f"snappy_uncompress failed: status={rc}")
        del out_view
        return out[:out_len]

    # ----- fresh bytes allocation -----
    if isinstance(out, int):
        if out < dstlen:
            raise SnappyError(
                f"snappy decode: out=int({out}) is less than the "
                f"Snappy header's declared {dstlen} bytes")
    out_bytes = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dstlen)
    dst = out_bytes
    out_len = dstlen   # snappy_uncompress overwrites
    with nogil:
        rc = snappy_uncompress(src_p, srcsize, <char*> &dst[0], &out_len)
    if rc != SNAPPY_OK:
        raise SnappyError(f"snappy_uncompress failed: status={rc}")
    del dst
    if out_len != dstlen:
        # Shouldn't happen for valid input — header says X bytes, we
        # got Y. Slice down to be safe.
        return out_bytes[:out_len]
    return out_bytes


def check_signature(data) -> bool:
    """Best-effort Snappy block detection.

    Snappy raw blocks have no fixed magic — they start with a varint
    encoding the uncompressed length. We validate the full buffer
    via ``snappy_validate_compressed_buffer`` (single linear scan,
    no allocation). It's not free, but for the byte sizes typical of
    a sniff (a few KB), the cost is negligible — and a header-only
    check is wrong far too often to be useful.
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        snappy_status rc

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        try:
            src = bytes(data)
        except Exception:
            return False
    srcsize = <size_t> src.shape[0]
    if srcsize < 1 or srcsize > 0x7fffffff:
        return False
    rc = snappy_validate_compressed_buffer(
        <const char*> &src[0], srcsize)
    return rc == SNAPPY_OK


__all__ = ["encode", "decode", "check_signature", "SnappyError"]
