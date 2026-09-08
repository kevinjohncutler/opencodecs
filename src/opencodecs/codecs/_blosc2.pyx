# opencodecs/codecs/_blosc2.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native blosc2 codec — bytes-in / bytes-out compression."""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdint cimport int16_t, int32_t, uint8_t
from libc.string cimport memset

from blosc2 cimport (
    BLOSC2_MAX_OVERHEAD, BLOSC_SHUFFLE, BLOSC_NOSHUFFLE,
    blosc2_compress, blosc2_decompress, blosc2_cbuffer_sizes,
    blosc1_set_compressor,
    blosc2_context, blosc2_dparams,
    blosc2_create_dctx, blosc2_free_ctx, blosc2_decompress_ctx,
    blosc2_getitem, blosc2_getitem_ctx, blosc1_cbuffer_metainfo,
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

    # Always set the compressor explicitly. The blosc1_set_compressor
    # API is *process-global state*; without an explicit set here, the
    # active compressor leaks across calls and the output depends on
    # whatever was last chosen elsewhere. Default to ``zstd``, matching
    # both blosc2 2.x's compile-time default and
    # ``imagecodecs.blosc2_encode``.
    cname = (compressor or 'zstd').encode() if isinstance(compressor or 'zstd', str) else compressor
    if blosc1_set_compressor(cname) < 0:
        raise Blosc2Error(f'unknown blosc2 compressor: {compressor!r}')

    # Default clevel=1: matches imagecodecs.blosc2_encode.
    clevel = 1 if level is None else int(level)
    if clevel < 0: clevel = 0
    if clevel > 9: clevel = 9

    do_shuffle = BLOSC_SHUFFLE if (shuffle is None or shuffle) else BLOSC_NOSHUFFLE

    # Default typesize=8 — matches ic.blosc2_encode for bytes input.
    # The shuffle filter rearranges bytes within typesize-byte groups
    # before handing off to zstd; typesize=8 (treat data as int64
    # lanes) gives zstd a tight cache-line-aligned stride to find
    # repeats. typesize=1 reduces shuffle to a no-op and is
    # surprisingly ~2x slower on natural-image bytes (the literal
    # byte stream is harder for zstd than the lane-shuffled version,
    # even though it produces ~9% smaller output for some workloads).
    # See docs/codec_api_conventions.md "Default settings" — Pareto
    # default trades size parity for speed parity since most callers
    # use blosc2 for throughput, not for the size-tightest path.
    tsize = 8 if typesize is None else int(typesize)
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


cdef int _decompress_threaded(
    const void* src_ptr, int32_t srcsize, void* dst_ptr,
    int32_t nbytes, int nthreads,
) except? -1:
    """Whole-chunk decompress through a context with nthreads set.

    Separate from the inline single-threaded call so the common path
    keeps paying nothing for the context create/free round trip.
    """
    cdef blosc2_dparams dparams
    cdef blosc2_context* dctx = NULL
    cdef int ret
    # The struct has to be zeroed: blosc2 treats 0 in most fields as
    # "use the default", but `schunk` and the postfilter pointers are
    # read as pointers, and a stack-garbage value there is a crash.
    memset(&dparams, 0, sizeof(blosc2_dparams))
    dparams.nthreads = <int16_t> nthreads
    dparams.typesize = 8
    dctx = blosc2_create_dctx(dparams)
    if dctx == NULL:
        raise Blosc2Error('blosc2_create_dctx failed')
    try:
        with nogil:
            ret = blosc2_decompress_ctx(dctx, src_ptr, srcsize,
                                        dst_ptr, nbytes)
    finally:
        blosc2_free_ctx(dctx)
    return ret


def chunk_typesize(data) -> int:
    """The item width a blosc2 chunk was compressed with, in bytes.

    Item width is not a decode option: the shuffle filters are defined
    over items, so the value is fixed when the chunk is written and
    every later read has to use it.
    """
    cdef const uint8_t[::1] src
    cdef size_t ts = 0
    cdef int flags = 0
    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    if src.shape[0] < 32:
        raise Blosc2Error('blosc2 chunk too small to read header')
    blosc1_cbuffer_metainfo(<const void*> &src[0], &ts, &flags)
    return int(ts)


def decode_partial(data, start: int, nitems: int, *,
                   typesize: int | None = None,
                   numthreads: int | None = None):
    """Decompress ``nitems`` items starting at ``start``, and no more.

    This is the capability that makes a blosc2 buffer randomly
    addressable: blosc2 stores a chunk as a series of independently
    compressed blocks, so a range touches only the blocks it overlaps
    instead of expanding the whole chunk and slicing. Reading a small
    window out of a large chunk is proportionally cheaper -- measured
    on an 8 MB chunk, pulling 1/64th of it costs a fraction of a full
    decode rather than the same.

    ``start`` and ``nitems`` are in ITEMS, not bytes. That is blosc2's
    own convention -- its shuffle filters are defined over items, so a
    byte-addressed API would not line up with the block boundaries --
    and the item width comes from the chunk itself, not from the
    caller. Passing ``typesize`` asserts what you believe it to be and
    raises if the chunk disagrees; it cannot change the interpretation,
    because blosc2_getitem reads the stored value either way. An
    earlier version of this function took ``typesize`` as an override,
    which merely mis-sized the destination buffer and failed with a
    bare error code.

    Returns the requested items as ``bytes``. Use
    :func:`chunk_typesize` to ask what the width is.
    """
    cdef:
        const uint8_t[::1] src
        int32_t srcsize
        int32_t nbytes = 0, cbytes = 0, blocksize = 0
        int ts = 1
        int c_start, c_nitems
        int32_t destsize
        int ret
        int nthreads = 1
        bytes out_bytes
        void* dst_ptr
        blosc2_dparams dparams
        blosc2_context* dctx = NULL

    if numthreads is not None:
        nthreads = int(numthreads)
        if nthreads < 1:
            nthreads = 1

    c_start = int(start)
    c_nitems = int(nitems)
    if c_start < 0:
        raise ValueError(f'start must be >= 0, got {start}')
    if c_nitems < 0:
        raise ValueError(f'nitems must be >= 0, got {nitems}')
    if c_nitems == 0:
        return b''

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <int32_t> src.shape[0]
    if srcsize < 32:
        raise Blosc2Error('blosc2 chunk too small to read header')

    ret = blosc2_cbuffer_sizes(<const void*> &src[0],
                               &nbytes, &cbytes, &blocksize)
    if ret < 0:
        raise Blosc2Error(f'blosc2_cbuffer_sizes failed: {ret}')

    cdef size_t stored_ts = 0
    cdef int meta_flags = 0
    blosc1_cbuffer_metainfo(<const void*> &src[0], &stored_ts, &meta_flags)
    ts = int(stored_ts) if stored_ts > 0 else 1
    if typesize is not None and int(typesize) != ts:
        raise ValueError(
            f'blosc2 decode_partial: typesize={typesize} does not match '
            f'the {ts} this chunk was written with. blosc2 addresses '
            f'items of the stored width, so the value cannot be '
            f'overridden here.')

    # Bounds-check here rather than trusting the library: blosc2
    # returns a negative code for an out-of-range request, but the
    # message that comes back is a bare number and the caller cannot
    # tell an over-long range from a corrupt chunk.
    if (<long long> c_start + c_nitems) * ts > <long long> nbytes:
        raise ValueError(
            f'blosc2 decode_partial: items [{c_start}, '
            f'{c_start + c_nitems}) of {ts} bytes run past the chunk\'s '
            f'{nbytes} uncompressed bytes '
            f'({nbytes // ts} items at that typesize)')

    destsize = <int32_t> (c_nitems * ts)
    out_bytes = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> destsize)
    dst_ptr = <void*> PyBytes_AsString(out_bytes)
    cdef const void* src_ptr = <const void*> &src[0]

    if nthreads > 1:
        memset(&dparams, 0, sizeof(blosc2_dparams))
        dparams.nthreads = <int16_t> nthreads
        dparams.typesize = ts
        dctx = blosc2_create_dctx(dparams)
        if dctx == NULL:
            raise Blosc2Error('blosc2_create_dctx failed')
        try:
            with nogil:
                ret = blosc2_getitem_ctx(dctx, src_ptr, srcsize,
                                         c_start, c_nitems,
                                         dst_ptr, destsize)
        finally:
            blosc2_free_ctx(dctx)
    else:
        with nogil:
            ret = blosc2_getitem(src_ptr, srcsize, c_start, c_nitems,
                                 dst_ptr, destsize)
    if ret < 0:
        raise Blosc2Error(f'blosc2_getitem failed: {ret}')
    return out_bytes


def decode(data, *, out=None, numthreads: int | None = None):
    """Decode a blosc2 chunk.

    Parameters
    ----------
    out : int | bytearray | memoryview | None, optional
        See ``_zstd.decode`` for the full ``out=`` contract.
    numthreads : int, optional
        Blosc2 splits a chunk into blocks and can decompress them on
        separate threads. ``None`` or ``1`` uses the single-threaded
        call; anything higher goes through a decompression *context*,
        never ``blosc2_set_nthreads``, which is process-global state a
        library has no business changing.

        Worth knowing before reaching for it: this only pays on chunks
        big enough to hold several blocks, and on small ones it costs.
        Measured at 8 threads against the single-threaded call:

            1 MB chunk    0.42x   -- slower; context setup dominates
            8 MB chunk    2.68x
            64 MB chunk   5.92x

        Which is why it is opt-in rather than automatic: the default
        stays on the call that is never worse.
    """
    cdef:
        const uint8_t[::1] src
        uint8_t[::1] out_view             # writable view of caller buffer
        int32_t srcsize
        int32_t nbytes = 0, cbytes = 0, blocksize = 0
        int ret
        int nthreads = 1
        bytes out_bytes
        void* dst_ptr
        blosc2_context* dctx = NULL
        blosc2_dparams dparams

    if numthreads is not None:
        nthreads = int(numthreads)
        if nthreads < 1:
            nthreads = 1

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
        if nthreads > 1:
            ret = _decompress_threaded(
                src_ptr, srcsize, dst_ptr, nbytes, nthreads)
        else:
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
    if nthreads > 1:
        ret = _decompress_threaded(
            src_ptr, srcsize, dst_ptr, nbytes, nthreads)
    else:
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
