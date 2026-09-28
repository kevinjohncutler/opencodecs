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
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, memset

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
    return _encode_impl(data, level, numthreads, False)


def encode_buffer(data, *, level: int | None = None,
                  numthreads: int | None = None):
    """Return a readonly view owning the encoded allocation without slicing it.

    Large results retain the compression-bound allocation until released;
    unused tail bytes are cleared before exposing that owner. Small results
    are compacted. This is for bounded pipelines consuming a buffer directly.
    Public encode continues to return an exact-sized bytes object.
    """
    return _encode_impl(data, level, numthreads, True)


cdef object _encode_impl(object data, object level, object numthreads,
                         bint buffer_result):
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
    # ~450 us faster on a 10 MB encode (Apple silicon), reproducible.
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
        return _owned_encoded_result(out, ret) if buffer_result else out[:ret]

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
    return _owned_encoded_result(out, ret) if buffer_result else out[:ret]


cdef object _owned_encoded_result(bytes output, size_t size):
    cdef size_t capacity = len(output)
    cdef char* data
    if size < capacity // 2:
        # Copying a small result releases a disproportionately large allocation.
        return memoryview(output[:size])
    # A view's .obj can expose the entire backing bytes object. Clear the
    # unwritten tail so no uninitialized allocation contents are observable.
    data = PyBytes_AsString(output)
    memset(data + size, 0, capacity - size)
    return memoryview(output)[:size]


def decode(data, *, out=None):
    """Decode a zstd frame.

    Accepts any buffer-protocol object (bytes, bytearray, memoryview,
    mmap, numpy uint8). For mmap-backed memoryviews this is a true
    zero-copy path — no bytes() materialization before the codec call.

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


from cpython.bytes cimport PyBytes_AsString
from zstd cimport (ZSTD_DCtx, ZSTD_createDCtx, ZSTD_freeDCtx,
                   ZSTD_inBuffer, ZSTD_outBuffer, ZSTD_decompressStream,
                   ZSTD_decompressDCtx)


cdef extern from "byteplanes.h" nogil:
    void oc_unshuffle(uint8_t* dst, const uint8_t* src, size_t plane,
                      size_t count, size_t k)


cdef class DecodeContext:
    """A reusable decompression context, for one thread at a time.

    ``ZSTD_decompress`` builds and frees a context on every call. Tile
    readers decode thousands of small frames per request, so each worker
    keeps one of these and hands it to :func:`decode_unshuffle_into`.
    Concurrent use of one context is unsupported.
    """
    cdef ZSTD_DCtx* _dctx

    def __cinit__(self):
        self._dctx = ZSTD_createDCtx()
        if self._dctx == NULL:
            raise MemoryError("ZSTD_createDCtx returned NULL")

    def __dealloc__(self):
        if self._dctx != NULL:
            ZSTD_freeDCtx(self._dctx)
            self._dctx = NULL


def decode_unshuffle_into(data, scratch, out, Py_ssize_t itemsize,
                          DecodeContext context=None):
    """Decompress one frame, then undo a byte-plane shuffle, in one call.

    The frame decodes into ``scratch`` (a writable buffer at least as long
    as ``out``); when ``itemsize > 1`` its ``itemsize`` byte planes are then
    interleaved into ``out``, and with ``itemsize == 1`` it decodes straight
    into ``out`` and ``scratch`` is unused. Both steps run under a single
    GIL release. As two separate calls, eight threads decoding 128 KB tiles
    queued on the GIL between them and scaled 1.8x instead of 2.6x.

    Returns the decompressed size, which the caller must check against
    ``len(out)``: a frame of any other size is malformed for this
    destination and, with ``itemsize > 1``, is left unshuffled in scratch.
    ``context`` reuses a :class:`DecodeContext`.
    """
    cdef:
        const uint8_t[::1] src = data
        uint8_t[::1] tmp
        uint8_t[::1] dst = out
        uint8_t* dp
        uint8_t* target
        size_t total = <size_t> dst.shape[0]
        size_t cap
        size_t ret
        Py_ssize_t n, k = itemsize
        ZSTD_DCtx* dctx = NULL

    if k < 1:
        raise ValueError(f"itemsize must be >= 1, got {itemsize}")
    if total % <size_t> k:
        raise ValueError(f"output length {total} is not a multiple of itemsize {k}")
    if src.shape[0] == 0 or total == 0:
        raise ZstdError("decode_unshuffle_into: empty frame or destination")
    if k == 1:
        target = &dst[0]
        cap = total
    else:
        tmp = scratch
        if <size_t> tmp.shape[0] < total:
            raise ValueError(f"scratch holds {tmp.shape[0]} bytes, need {total}")
        target = &tmp[0]
        cap = total
    if context is not None:
        dctx = context._dctx
    n = <Py_ssize_t> (total // <size_t> k)
    dp = &dst[0]
    with nogil:
        if dctx != NULL:
            ret = ZSTD_decompressDCtx(dctx, <void*> target, cap,
                                      <const void*> &src[0], <size_t> src.shape[0])
        else:
            ret = ZSTD_decompress(<void*> target, cap,
                                  <const void*> &src[0], <size_t> src.shape[0])
        if not ZSTD_isError(ret) and ret == total and k > 1:
            oc_unshuffle(dp, <const uint8_t*> target, <size_t> n, <size_t> n, <size_t> k)
    if ZSTD_isError(ret):
        raise ZstdError(f'ZSTD_decompress: {ZSTD_getErrorName(ret).decode()}')
    return ret



def decode_unshuffle_windows(data, scratch, out, Py_ssize_t itemsize,
                             Py_ssize_t samples, Py_ssize_t tile_h,
                             Py_ssize_t tile_w, windows,
                             DecodeContext context=None):
    """Decompress one tile once and write windows of it straight into outputs.

    The frame holds a ``tile_h`` x ``tile_w`` tile of ``samples`` elements
    per pixel, each ``itemsize`` bytes, stored as ``itemsize`` byte planes
    (a plain copy when ``itemsize == 1``). It decodes into ``scratch``; then
    for every ``(sy0, sy1, sx0, sx1, dy, dx)`` in ``windows``, tile rows
    ``sy0:sy1`` and columns ``sx0:sx1`` are interleaved into ``out``, a
    C-contiguous 2D byte view of the destination image (one row per image
    row), at image row ``dy``, pixel column ``dx``. ``out`` may also be a
    list of such views, and a window a 7-tuple whose last element picks
    one, so a tile shared by several requested regions decodes once.
    Decompression, unshuffle and placement share one GIL release, and no
    tile-sized array is allocated, which is what a region read of many
    small tiles otherwise spends its time on.

    Returns the decompressed size; outputs are written only when it equals
    the tile's size, and the caller must reject any other value.
    """
    cdef:
        const uint8_t[::1] src = data
        uint8_t[::1] tmp = scratch
        uint8_t[:, ::1] dst
        const uint8_t* sp
        uint8_t* drow
        size_t ret
        Py_ssize_t k = itemsize, s = samples
        Py_ssize_t n, row_elems, cnt, base, r, w_i, m, o_i, n_out
        Py_ssize_t sy0, sy1, sx0, sx1, dy, dx, target
        Py_ssize_t total, pixel_bytes
        Py_ssize_t* spec = NULL
        uint8_t** origins = NULL
        Py_ssize_t* strides = NULL
        ZSTD_DCtx* dctx = NULL

    if k < 1 or s < 1 or tile_h < 0 or tile_w < 0:
        raise ValueError("itemsize and samples must be >= 1, tile extents >= 0")
    pixel_bytes = s * k
    n = tile_h * tile_w * s
    total = n * k
    if total == 0 or src.shape[0] == 0:
        raise ZstdError("decode_unshuffle_windows: empty frame or tile")
    if tmp.shape[0] < total:
        raise ValueError(f"scratch holds {tmp.shape[0]} bytes, need {total}")
    outs = list(out) if isinstance(out, (list, tuple)) else [out]
    n_out = len(outs)
    windows = list(windows)
    m = len(windows)
    spec = <Py_ssize_t*> malloc(<size_t> (7 * m + 1) * sizeof(Py_ssize_t))
    origins = <uint8_t**> malloc(<size_t> (n_out + 1) * sizeof(uint8_t*))
    strides = <Py_ssize_t*> malloc(<size_t> (2 * n_out + 1) * sizeof(Py_ssize_t))
    if spec == NULL or origins == NULL or strides == NULL:
        free(spec)
        free(origins)
        free(strides)
        raise MemoryError()
    held = []
    try:
        for o_i in range(n_out):
            dst = outs[o_i]
            held.append(dst)
            origins[o_i] = &dst[0, 0] if dst.shape[0] > 0 and dst.shape[1] > 0 else NULL
            strides[2 * o_i] = dst.shape[0]
            strides[2 * o_i + 1] = dst.shape[1]
        for w_i in range(m):
            window = windows[w_i]
            if len(window) == 7:
                sy0, sy1, sx0, sx1, dy, dx, target = window
            else:
                sy0, sy1, sx0, sx1, dy, dx = window
                target = 0
            if not 0 <= target < n_out:
                raise ValueError("window names an output that was not given")
            if not (0 <= sy0 <= sy1 <= tile_h and 0 <= sx0 <= sx1 <= tile_w):
                raise ValueError("window lies outside the tile")
            if (dy < 0 or dx < 0 or dy + (sy1 - sy0) > strides[2 * target]
                    or (dx + (sx1 - sx0)) * pixel_bytes > strides[2 * target + 1]):
                raise ValueError("window lies outside the destination")
            spec[7 * w_i] = sy0
            spec[7 * w_i + 1] = sy1
            spec[7 * w_i + 2] = sx0
            spec[7 * w_i + 3] = sx1
            spec[7 * w_i + 4] = dy
            spec[7 * w_i + 5] = dx
            spec[7 * w_i + 6] = target
        if context is not None:
            dctx = context._dctx
        row_elems = tile_w * s
        sp = <const uint8_t*> &tmp[0]
        with nogil:
            if dctx != NULL:
                ret = ZSTD_decompressDCtx(dctx, <void*> &tmp[0], <size_t> total,
                                          <const void*> &src[0], <size_t> src.shape[0])
            else:
                ret = ZSTD_decompress(<void*> &tmp[0], <size_t> total,
                                      <const void*> &src[0], <size_t> src.shape[0])
            if not ZSTD_isError(ret) and ret == <size_t> total:
                for w_i in range(m):
                    sy0 = spec[7 * w_i]
                    sy1 = spec[7 * w_i + 1]
                    sx0 = spec[7 * w_i + 2]
                    sx1 = spec[7 * w_i + 3]
                    dy = spec[7 * w_i + 4]
                    dx = spec[7 * w_i + 5]
                    target = spec[7 * w_i + 6]
                    cnt = (sx1 - sx0) * s
                    if cnt <= 0:
                        continue
                    for r in range(sy0, sy1):
                        drow = (origins[target] + (dy + r - sy0) * strides[2 * target + 1]
                                + dx * pixel_bytes)
                        base = r * row_elems + sx0 * s
                        oc_unshuffle(drow, sp + base, <size_t> n, <size_t> cnt, <size_t> k)
    finally:
        free(spec)
        free(origins)
        free(strides)
    if ZSTD_isError(ret):
        raise ZstdError(f'ZSTD_decompress: {ZSTD_getErrorName(ret).decode()}')
    return ret

cdef class StreamDecoder:
    """One bounded Zstandard decode step with persistent native state."""
    cdef ZSTD_DCtx* _state
    cdef bint _finished

    def __cinit__(self):
        self._state = ZSTD_createDCtx()
        if self._state == NULL:
            raise MemoryError()

    def close(self):
        if self._state != NULL:
            ZSTD_freeDCtx(self._state)
            self._state = NULL

    def __dealloc__(self):
        if self._state != NULL:
            ZSTD_freeDCtx(self._state)

    def process(self, const uint8_t[::1] data, Py_ssize_t output_size):
        cdef ZSTD_inBuffer source
        cdef ZSTD_outBuffer destination
        cdef size_t status
        cdef bytes result
        if self._state == NULL or self._finished:
            raise ValueError("zstd stream is closed or finished")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        result = PyBytes_FromStringAndSize(NULL, output_size)
        source.src = &data[0] if data.shape[0] else NULL
        source.size = data.shape[0]
        source.pos = 0
        destination.dst = <void*> PyBytes_AsString(result)
        destination.size = output_size
        destination.pos = 0
        with nogil:
            status = ZSTD_decompressStream(self._state, &destination, &source)
        if ZSTD_isError(status):
            raise ZstdError(ZSTD_getErrorName(status).decode())
        self._finished = status == 0
        return source.pos, result[:destination.pos], bool(self._finished)


from zstd cimport ZSTD_compressStream2, ZSTD_EndDirective, ZSTD_e_continue, ZSTD_e_end


cdef class StreamEncoder:
    """Incremental Zstandard encoder with a caller-bounded output step."""
    cdef ZSTD_CCtx* _state

    def __cinit__(self, level=None):
        cdef size_t status
        cdef int quality = 3 if level is None else int(level)
        self._state = ZSTD_createCCtx()
        if self._state == NULL:
            raise MemoryError()
        status = ZSTD_CCtx_setParameter(self._state, ZSTD_c_compressionLevel, quality)
        if ZSTD_isError(status):
            raise ZstdError(ZSTD_getErrorName(status).decode())

    def close(self):
        if self._state != NULL:
            ZSTD_freeCCtx(self._state)
            self._state = NULL

    def __dealloc__(self):
        if self._state != NULL:
            ZSTD_freeCCtx(self._state)

    def process(self, const uint8_t[::1] data, Py_ssize_t output_size, bint finish=False):
        cdef ZSTD_inBuffer source
        cdef ZSTD_outBuffer destination
        cdef size_t status
        cdef bytes result
        cdef ZSTD_EndDirective operation = ZSTD_e_end if finish else ZSTD_e_continue
        if self._state == NULL:
            raise ValueError("zstd stream is closed")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        result = PyBytes_FromStringAndSize(NULL, output_size)
        source.src = &data[0] if data.shape[0] else NULL
        source.size = data.shape[0]
        source.pos = 0
        destination.dst = <void*> PyBytes_AsString(result)
        destination.size = output_size
        destination.pos = 0
        with nogil:
            status = ZSTD_compressStream2(self._state, &destination, &source, operation)
        if ZSTD_isError(status):
            raise ZstdError(ZSTD_getErrorName(status).decode())
        return source.pos, result[:destination.pos], bool(finish and status == 0)
