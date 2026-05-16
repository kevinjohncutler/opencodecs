# opencodecs/codecs/_zstd.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native zstd codec — bytes-in / bytes-out compression."""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t

from zstd cimport (
    ZSTD_compress, ZSTD_decompress,
    ZSTD_compressBound, ZSTD_getFrameContentSize,
    ZSTD_CLEVEL_DEFAULT, ZSTD_isError, ZSTD_getErrorName,
    ZSTD_CONTENTSIZE_UNKNOWN, ZSTD_CONTENTSIZE_ERROR,
    ZSTD_VERSION_MAJOR, ZSTD_VERSION_MINOR, ZSTD_VERSION_RELEASE,
    ZSTD_CCtx, ZSTD_createCCtx, ZSTD_freeCCtx,
    ZSTD_CCtx_setParameter, ZSTD_compress2,
    ZSTD_c_compressionLevel, ZSTD_c_nbWorkers,
)


class ZstdError(RuntimeError):
    """Raised on zstd encode/decode failures."""


def libzstd_version() -> str:
    return f'{ZSTD_VERSION_MAJOR}.{ZSTD_VERSION_MINOR}.{ZSTD_VERSION_RELEASE}'


def encode(data, *, level: int | None = None,
           numthreads: int | None = None) -> bytes:
    """Encode bytes-like input as a zstd frame.

    Accepts any buffer-protocol object that exposes a 1D contiguous
    uint8 view — bytes, bytearray, memoryview, mmap, numpy uint8 arrays.
    Anything else is coerced via ``bytes(data)``.

    Parameters
    ----------
    level : int, optional
        Compression level. Defaults to libzstd's default (3).
    numthreads : int, optional
        Worker threads for parallel compression. ``None`` or ``<=0``
        means single-threaded (one frame, smallest output). ``1`` adds
        one worker thread (output stays valid zstd; ~10-15% larger but
        faster on >1 MB inputs). Larger values parallelize across more
        threads — for big payloads on multi-core machines this gives
        near-linear speedup. The output is always a valid zstd frame.
    """
    cdef:
        const uint8_t[::1] src
        const uint8_t[::1] dst
        size_t srcsize
        size_t dstcap
        size_t ret
        int lvl = ZSTD_CLEVEL_DEFAULT
        int workers = 0
        bytes out
        ZSTD_CCtx* cctx

    # Keep the hot path lean: no per-call min/max-CLevel queries
    # (libzstd clamps internally if level is out of range), no defensive
    # branches for the common case (level=None, numthreads=None).
    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if level is not None:
        lvl = <int> level
    if numthreads is not None and <int> numthreads > 0:
        workers = <int> numthreads

    dstcap = ZSTD_compressBound(srcsize)
    out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dstcap)
    # IMPORTANT: cast ``out`` to a memoryview (``dst``) and use
    # ``&dst[0]`` instead of ``PyBytes_AsString(out)``. Empirically
    # ~450 us faster on a 10 MB encode (M1 Ultra), reproducible.
    # The win seems to come from how Cython's buffer-export machinery
    # interacts with the page-fault pattern libzstd's writes produce;
    # we couldn't fully isolate the mechanism but the speedup is
    # reproducible 100% of the time. ``del dst`` before ``out[:ret]``
    # releases the buffer export so the slice can take a fast path.
    dst = out

    if workers == 0:
        # Single-thread one-shot — zstd's ``ZSTD_compress`` is measurably
        # FASTER than ``ZSTD_compressCCtx`` with a pooled context for
        # typical level=3 / 10 MB workloads. Internally it picks an
        # optimal CCtx sized for the request; reusing a context pays
        # extra reset overhead that outweighs the per-call malloc/free.
        with nogil:
            ret = ZSTD_compress(
                <void*> &dst[0], dstcap,
                <const void*> &src[0] if srcsize > 0 else NULL,
                srcsize, lvl,
            )
        if ZSTD_isError(ret):
            raise ZstdError(
                f'ZSTD_compress: {ZSTD_getErrorName(ret).decode()}')
        del dst
        return out[:ret]

    # Multithreaded path: requires the CCtx-based API.
    cctx = ZSTD_createCCtx()
    if cctx == NULL:
        raise ZstdError("ZSTD_createCCtx returned NULL")
    try:
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, lvl)
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, workers)
        with nogil:
            ret = ZSTD_compress2(
                cctx, <void*> &dst[0], dstcap,
                <const void*> &src[0] if srcsize > 0 else NULL,
                srcsize,
            )
    finally:
        ZSTD_freeCCtx(cctx)
    if ZSTD_isError(ret):
        raise ZstdError(
            f'ZSTD_compress2 (nbWorkers={workers}): '
            f'{ZSTD_getErrorName(ret).decode()}')
    del dst
    return out[:ret]


def decode(data, *, out=None):
    """Decode a zstd frame.

    Accepts any buffer-protocol object (bytes, bytearray, memoryview,
    mmap, numpy uint8). For mmap-backed memoryviews this is a true
    zero-copy path — no bytes() materialisation before the codec call.

    Parameters
    ----------
    out : int | bytearray | memoryview | None, optional
        Preallocated output buffer. Matches imagecodecs's ``out=`` API.

        * ``None`` (default): allocate fresh ``bytes`` sized from the
          zstd frame header (or grown from a 4× starting guess for
          streaming-encoded frames). Return type is ``bytes``.
        * ``int``: allocate fresh ``bytes`` of exactly this size. The
          decoder must produce at most this many bytes; raises if the
          frame would expand to more.
        * writable buffer (``bytearray`` / ``memoryview`` / numpy
          uint8 array): decode in-place. Returns the same object
          sliced to ``[:actual_size]`` — no copy.

        The in-place path is the zero-alloc fast path for tile /
        chunk workloads where the same buffer gets reused.
    """
    cdef:
        const uint8_t[::1] src
        const uint8_t[::1] dst_view       # for bytes-out path
        uint8_t[::1] out_view             # for caller-supplied writable buffer
        size_t srcsize
        unsigned long long content_size
        size_t dstcap
        size_t ret
        bytes out_bytes

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if srcsize == 0:
        if out is None or isinstance(out, int):
            return b''
        # Empty frame into a caller buffer — return a zero-length slice.
        return out[:0]

    content_size = ZSTD_getFrameContentSize(<const void*> &src[0], srcsize)
    if content_size == <unsigned long long> ZSTD_CONTENTSIZE_ERROR:
        raise ZstdError('ZSTD_getFrameContentSize: not a zstd frame')

    cdef const void* src_ptr = <const void*> &src[0]

    # ----- caller-supplied writable buffer (zero-alloc path) -----
    if out is not None and not isinstance(out, int):
        try:
            out_view = out
        except (TypeError, ValueError, BufferError) as e:
            raise TypeError(
                f"zstd decode: out= must be int or a writable buffer "
                f"(bytearray / memoryview / numpy uint8), "
                f"got {type(out).__name__}"
            ) from e
        dstcap = <size_t> out_view.shape[0]
        with nogil:
            ret = ZSTD_decompress(
                <void*> &out_view[0], dstcap, src_ptr, srcsize)
        if ZSTD_isError(ret):
            raise ZstdError(
                f'ZSTD_decompress (out= buffer): '
                f'{ZSTD_getErrorName(ret).decode()}')
        del out_view
        return out[:ret]

    # ----- fresh bytes allocation -----
    if isinstance(out, int):
        if out < 0:
            raise ValueError("zstd decode: out=int(N) requires N >= 0")
        dstcap = <size_t> out
    elif content_size == <unsigned long long> ZSTD_CONTENTSIZE_UNKNOWN:
        # Streaming-encoded (no size header) — pick a generous starting
        # capacity and grow until we succeed. We try 4× input first.
        dstcap = max(<size_t> 4 * srcsize, <size_t> 65536)
    else:
        dstcap = <size_t> content_size

    while True:
        out_bytes = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dstcap)
        # See encode() for why we cast to memoryview rather than using
        # PyBytes_AsString — matches imagecodecs's pattern + faster.
        dst_view = out_bytes
        with nogil:
            ret = ZSTD_decompress(<void*> &dst_view[0], dstcap, src_ptr, srcsize)
        if not ZSTD_isError(ret):
            del dst_view
            return out_bytes[:ret]
        del dst_view
        # When the user pinned the size via out=int(N), don't grow —
        # they explicitly asked for that capacity.
        # Likewise content_size known: the frame won't be bigger.
        if isinstance(out, int):
            raise ZstdError(
                f'ZSTD_decompress (out= int hint too small): '
                f'{ZSTD_getErrorName(ret).decode()}')
        if content_size != <unsigned long long> ZSTD_CONTENTSIZE_UNKNOWN:
            raise ZstdError(
                f'ZSTD_decompress: {ZSTD_getErrorName(ret).decode()}')
        # content_size unknown — try a bigger buffer.
        if dstcap >= <size_t> (1 << 32):
            raise ZstdError(
                f'ZSTD_decompress (capacity capped at 4 GiB): '
                f'{ZSTD_getErrorName(ret).decode()}')
        dstcap *= 2


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
