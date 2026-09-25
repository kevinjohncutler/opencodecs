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
        const uint8_t[::1] dst   # memoryview into output bytes
        size_t srcsize
        size_t dstcap
        size_t ret
        LZ4F_preferences_t prefs
        bytes out
        const void* src_ptr = NULL

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <size_t> src.shape[0]

    memset(<void*> &prefs, 0, sizeof(LZ4F_preferences_t))
    prefs.compressionLevel = 0 if level is None else int(level)
    # Write the uncompressed size into the frame header so the decoder
    # can pre-allocate the output buffer in one shot instead of falling
    # back to the chunked grow-buffer path. ~3-6x decode speedup on
    # large payloads, no impact on encode throughput, no change to wire
    # compatibility (LZ4F_getFrameInfo is the standard way to read it).
    prefs.frameInfo.contentSize = <unsigned long long> srcsize

    if srcsize > 0:
        src_ptr = <const void*> &src[0]

    dstcap = LZ4F_compressFrameBound(srcsize, &prefs)
    out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dstcap)
    # See _zstd.encode for why we use a memoryview cast (`dst = out`)
    # + ``del dst`` + ``out[:ret]`` slice rather than the PyBytes_AsString
    # + _PyBytes_Resize pattern — measurably faster on multi-MB outputs.
    dst = out
    with nogil:
        ret = LZ4F_compressFrame(
            <void*> &dst[0], dstcap, src_ptr, srcsize, &prefs,
        )
    if LZ4F_isError(ret):
        raise Lz4Error(
            f'LZ4F_compressFrame: {LZ4F_getErrorName(ret).decode()}')
    del dst
    return out[:ret]


def decode(data, *, out=None):
    """Decode an LZ4 frame.

    Parameters
    ----------
    out : int | bytearray | memoryview | None, optional
        See ``_zstd.decode`` for the full ``out=`` contract. ``int``
        pre-sizes the output bytes; a writable buffer enables the
        zero-alloc fast path (returns the same object sliced).
    """
    cdef:
        const uint8_t[::1] src
        const uint8_t[::1] dst_mv     # memoryview cast for the output bytes
        uint8_t[::1] out_view         # writable view of caller buffer
        size_t srcsize, src_consumed, src_remaining
        size_t dst_size, dst_used
        size_t ret
        LZ4F_dctx* dctx = NULL
        LZ4F_frameInfo_t info
        const void* src_ptr
        void* dst_ptr
        bytes out_bytes
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
        if out is None or isinstance(out, int):
            return b''
        return out[:0]

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

        src_pos = src_consumed

        # ----- caller-supplied writable buffer (zero-alloc path) -----
        if out is not None and not isinstance(out, int):
            try:
                out_view = out
            except (TypeError, ValueError, BufferError) as e:
                raise TypeError(
                    f"lz4 decode: out= must be int or writable buffer "
                    f"(bytearray / memoryview / numpy uint8), "
                    f"got {type(out).__name__}"
                ) from e
            dst_used = <size_t> out_view.shape[0]
            src_remaining = srcsize - src_consumed
            with nogil:
                ret = LZ4F_decompress(
                    dctx, <void*> &out_view[0], &dst_used,
                    <const void*> (&src[0] + src_consumed), &src_remaining,
                    NULL)
            if LZ4F_isError(ret):
                raise Lz4Error(
                    f'LZ4F_decompress (out= buffer): '
                    f'{LZ4F_getErrorName(ret).decode()}')
            if ret != 0:
                raise Lz4Error(
                    'LZ4F_decompress: output buffer too small '
                    '(decoder wants more space)')
            del out_view
            return out[:dst_used]

        # ----- pre-sized bytes (out=int or content-size in header) -----
        if isinstance(out, int):
            if out < 0:
                raise ValueError("lz4 decode: out=int(N) requires N >= 0")
            dst_size = <size_t> out
        elif info.contentSize > 0:
            dst_size = <size_t> info.contentSize
        else:
            dst_size = 0  # unknown; fall through to chunked path

        if dst_size > 0:
            out_bytes = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dst_size)
            # Memoryview cast for the output pointer — same pattern as
            # encode (see _zstd.pyx for the empirical rationale).
            dst_mv = out_bytes
            dst_used = dst_size
            src_remaining = srcsize - src_consumed
            with nogil:
                ret = LZ4F_decompress(
                    dctx, <void*> &dst_mv[0], &dst_used,
                    <const void*> (&src[0] + src_consumed), &src_remaining,
                    NULL)
            if LZ4F_isError(ret):
                raise Lz4Error(
                    f'LZ4F_decompress: {LZ4F_getErrorName(ret).decode()}')
            if ret != 0:
                raise Lz4Error(
                    'LZ4F_decompress: pre-sized buffer was too small '
                    '(decoder requests more output space)')
            del dst_mv
            return out_bytes[:dst_used]

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


cdef class StreamDecoder:
    """One bounded LZ4 frame decode step with persistent native state."""
    cdef LZ4F_dctx* _state
    cdef bint _finished

    def __cinit__(self):
        cdef size_t status = LZ4F_createDecompressionContext(&self._state, LZ4F_VERSION)
        if LZ4F_isError(status):
            raise Lz4Error(LZ4F_getErrorName(status).decode())

    def close(self):
        if self._state != NULL:
            LZ4F_freeDecompressionContext(self._state)
            self._state = NULL

    def __dealloc__(self):
        if self._state != NULL:
            LZ4F_freeDecompressionContext(self._state)

    def process(self, const uint8_t[::1] data, Py_ssize_t output_size):
        cdef size_t consumed = data.shape[0]
        cdef size_t produced
        cdef size_t status
        cdef const void* source = &data[0] if consumed else NULL
        cdef void* destination
        cdef bytes result
        if self._state == NULL or self._finished:
            raise ValueError("lz4 stream is closed or finished")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        result = PyBytes_FromStringAndSize(NULL, output_size)
        destination = <void*> PyBytes_AsString(result)
        produced = output_size
        with nogil:
            status = LZ4F_decompress(self._state, destination, &produced,
                                     source, &consumed, NULL)
        if LZ4F_isError(status):
            raise Lz4Error(LZ4F_getErrorName(status).decode())
        self._finished = status == 0
        return consumed, result[:produced], bool(self._finished)


from lz4 cimport (LZ4F_cctx, LZ4F_createCompressionContext,
    LZ4F_freeCompressionContext, LZ4F_compressBound, LZ4F_compressBegin,
    LZ4F_compressUpdate, LZ4F_compressEnd, LZ4F_max64KB,
    LZ4F_contentChecksumEnabled)


cdef class StreamEncoder:
    """Incremental LZ4 frame writer with at most one 64 KiB input block."""
    cdef LZ4F_cctx* _state
    cdef LZ4F_preferences_t _prefs
    cdef bytes _pending
    cdef bint _ended

    def __cinit__(self, level=None):
        cdef size_t status
        cdef bytes header
        memset(&self._prefs, 0, sizeof(LZ4F_preferences_t))
        self._prefs.compressionLevel = 0 if level is None else int(level)
        self._prefs.frameInfo.blockSizeID = LZ4F_max64KB
        self._prefs.frameInfo.contentChecksumFlag = LZ4F_contentChecksumEnabled
        self._prefs.autoFlush = 1
        status = LZ4F_createCompressionContext(&self._state, LZ4F_VERSION)
        if LZ4F_isError(status):
            raise Lz4Error(LZ4F_getErrorName(status).decode())
        header = PyBytes_FromStringAndSize(NULL, 19)
        status = LZ4F_compressBegin(self._state, <void*> PyBytes_AsString(header),
                                    19, &self._prefs)
        if LZ4F_isError(status):
            raise Lz4Error(LZ4F_getErrorName(status).decode())
        self._pending = header[:status]

    def close(self):
        if self._state != NULL:
            LZ4F_freeCompressionContext(self._state)
            self._state = NULL
        self._pending = b""

    def __dealloc__(self):
        if self._state != NULL:
            LZ4F_freeCompressionContext(self._state)

    def process(self, const uint8_t[::1] data, Py_ssize_t output_size, bint finish=False):
        cdef size_t consumed = 0
        cdef size_t capacity
        cdef size_t status
        cdef bytes result
        cdef void* destination
        if self._state == NULL:
            raise ValueError("lz4 stream is closed")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        if not self._pending and not self._ended:
            consumed = min(<size_t> data.shape[0], <size_t> 65536)
            if finish and consumed:
                raise ValueError("finish requires empty input")
            capacity = LZ4F_compressBound(consumed, &self._prefs)
            result = PyBytes_FromStringAndSize(NULL, capacity)
            destination = <void*> PyBytes_AsString(result)
            if finish:
                with nogil:
                    status = LZ4F_compressEnd(self._state, destination, capacity, NULL)
                self._ended = True
            elif consumed:
                with nogil:
                    status = LZ4F_compressUpdate(self._state, destination, capacity,
                                                 &data[0], consumed, NULL)
            else:
                status = 0
            if LZ4F_isError(status):
                raise Lz4Error(LZ4F_getErrorName(status).decode())
            self._pending = result[:status]
        result = self._pending[:output_size]
        self._pending = self._pending[output_size:]
        return consumed, result, bool(self._ended and not self._pending)
