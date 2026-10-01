# opencodecs/codecs/_lz4.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native LZ4 codec — frame format (.lz4 file format)."""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from cpython.bytearray cimport PyByteArray_Resize, PyByteArray_AS_STRING
from libc.string cimport memset
from libc.stdint cimport uint8_t

from lz4 cimport (
    LZ4F_VERSION, LZ4F_blockSizeID_t,
    LZ4F_contentChecksumEnabled, LZ4F_blockChecksumEnabled,
    LZ4F_preferences_t, LZ4F_frameInfo_t,
    LZ4F_compressFrame, LZ4F_compressFrameBound,
    LZ4F_dctx, LZ4F_createDecompressionContext, LZ4F_freeDecompressionContext,
    LZ4F_decompress, LZ4F_getFrameInfo,
    LZ4F_decompressOptions_t,
    LZ4F_isError, LZ4F_getErrorName,
)


class Lz4Error(RuntimeError):
    """Raised on LZ4 encode/decode failures."""


def _frame_level(level):
    """Return the LZ4 frame compression level imagecodecs writes.

    imagecodecs.lz4f_encode clamps ``level`` to -1 through 12
    (LZ4HC_CLEVEL_MAX), so every negative level writes level -1's bytes.
    None is liblz4's default, 0.
    """
    if level is None:
        return 0
    return max(-1, min(int(level), 12))


def encode(data, *, level: int | None = None, blocksizeid=None,
           contentchecksum=None, blockchecksum=None) -> bytes:
    """Encode bytes-like input as an LZ4 frame.

    The keywords are imagecodecs.lz4f_encode's. ``blocksizeid`` is the
    frame descriptor's Block Maximum Size code (LZ4 Frame Format, "BD
    byte"): 4, 5, 6 or 7 for 64 KB, 256 KB, 1 MB or 4 MB, and None or 0
    for liblz4's default (4). The specification reserves 0 to 3 in the
    header, so any other value raises ``ValueError`` rather than being
    written. ``contentchecksum`` and ``blockchecksum`` set the frame's
    content and block checksum flags (None means off). liblz4 lowers
    the block size, and marks blocks independent, when the whole input
    fits in one block. ``level`` is clamped to -1 through 12, as
    imagecodecs does, so a level below -1 writes level -1's bytes.
    """
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
    prefs.compressionLevel = _frame_level(level)
    # Write the uncompressed size into the frame header so the decoder
    # can pre-allocate the output buffer in one shot instead of falling
    # back to the chunked grow-buffer path. ~3-6x decode speedup on
    # large payloads, no impact on encode throughput, no change to wire
    # compatibility (LZ4F_getFrameInfo is the standard way to read it).
    prefs.frameInfo.contentSize = <unsigned long long> srcsize
    if blocksizeid is not None and int(blocksizeid) != 0:
        if int(blocksizeid) not in (4, 5, 6, 7):
            raise ValueError(
                f"lz4 encode: blocksizeid={blocksizeid!r} is not a block "
                f"maximum size code; the LZ4 frame format defines 4, 5, 6 "
                f"and 7 (0 or None for the default)")
        prefs.frameInfo.blockSizeID = <LZ4F_blockSizeID_t> (<int> int(blocksizeid))
    if contentchecksum:
        prefs.frameInfo.contentChecksumFlag = LZ4F_contentChecksumEnabled
    if blockchecksum:
        prefs.frameInfo.blockChecksumFlag = LZ4F_blockChecksumEnabled

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
    """Decode LZ4 frame-format data: one frame, or several.

    The LZ4 Frame Format specification (section "General Structure of
    LZ4 Frame format") lets frames be concatenated and defines skippable
    frames, which carry no content; the lz4 CLI decodes both. So does
    this function: concatenated frames decode to the concatenation of
    their contents, skippable frames are skipped, and bytes after the
    last frame that are not a frame raise :class:`Lz4Error` instead of
    being dropped.

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
        size_t srcsize, src_consumed
        size_t dst_size
        size_t ret = 0
        size_t pos, dst_pos, this_dst, this_src
        LZ4F_dctx* dctx = NULL
        LZ4F_frameInfo_t info
        bytes out_bytes
        Py_ssize_t chunk_cap
        bytearray out_buf
        void* dst_ptr

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
        ret = LZ4F_getFrameInfo(dctx, &info, <const void*> &src[0],
                                &src_consumed)
        if LZ4F_isError(ret):
            raise Lz4Error(
                f'LZ4F_getFrameInfo: {LZ4F_getErrorName(ret).decode()}')
        pos = src_consumed

        # ----- fixed capacity: caller buffer, or out=int -----
        # Every frame is decoded into the one buffer; running out of room
        # with frame data left is an error, never a silent truncation.
        if out is not None:
            if isinstance(out, int):
                if out < 0:
                    raise ValueError("lz4 decode: out=int(N) requires N >= 0")
                out_bytes = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> out)
                dst_ptr = <void*> PyBytes_AsString(out_bytes)
                dst_size = <size_t> out
            else:
                try:
                    out_view = out
                except (TypeError, ValueError, BufferError) as e:
                    raise TypeError(
                        f"lz4 decode: out= must be int or writable buffer "
                        f"(bytearray / memoryview / numpy uint8), "
                        f"got {type(out).__name__}"
                    ) from e
                dst_size = <size_t> out_view.shape[0]
                dst_ptr = <void*> &out_view[0] if dst_size else NULL
            dst_pos = 0
            while True:
                this_dst = dst_size - dst_pos
                this_src = srcsize - pos
                with nogil:
                    ret = LZ4F_decompress(
                        dctx, <void*> (<uint8_t*> dst_ptr + dst_pos), &this_dst,
                        <const void*> (&src[0] + pos), &this_src, NULL)
                if LZ4F_isError(ret):
                    raise Lz4Error(
                        f'LZ4F_decompress: {LZ4F_getErrorName(ret).decode()}')
                dst_pos += this_dst
                pos += this_src
                if ret == 0 and pos == srcsize:
                    break
                if this_dst == 0 and this_src == 0 or (
                        ret != 0 and pos == srcsize and dst_pos == dst_size):
                    raise Lz4Error(
                        'LZ4F_decompress: output buffer too small '
                        '(decoder wants more space)')
                if ret != 0 and pos == srcsize:
                    raise Lz4Error('LZ4F_decompress: truncated frame')
            if isinstance(out, int):
                return out_bytes[:dst_pos]
            del out_view
            return out[:dst_pos]

        # ----- one frame with its size recorded: decode straight into
        # an exact-size bytes object -----
        if info.contentSize > 0:
            dst_size = <size_t> info.contentSize
            out_bytes = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dst_size)
            # Memoryview cast for the output pointer — same pattern as
            # encode (see _zstd.pyx for the empirical rationale).
            dst_mv = out_bytes
            this_dst = dst_size
            this_src = srcsize - pos
            with nogil:
                ret = LZ4F_decompress(
                    dctx, <void*> &dst_mv[0], &this_dst,
                    <const void*> (&src[0] + pos), &this_src, NULL)
            del dst_mv
            if LZ4F_isError(ret):
                raise Lz4Error(
                    f'LZ4F_decompress: {LZ4F_getErrorName(ret).decode()}')
            if ret != 0:
                raise Lz4Error(
                    'LZ4F_decompress: frame is truncated or larger than '
                    'its recorded content size')
            pos += this_src
            if pos == srcsize:
                return out_bytes[:this_dst]
            # More frames follow. They go through the chunked loop
            # below, which handles any number of frames in one pass,
            # after this frame's bytes. (Recursing once per frame here
            # copied the whole result again for every frame, quadratic
            # in the frame count, and overflowed the C stack at about
            # ten thousand frames.)
            out_buf = bytearray(out_bytes[:this_dst])
            dst_pos = this_dst
        else:
            out_buf = bytearray()
            dst_pos = 0

        # No content-size hint, or frames after the first: decode
        # straight into the tail of a growing bytearray, at least 1 MB
        # of room per call. Growth is geometric, so any number of
        # frames and any expansion ratio cost linear time, without
        # pre-sizing.
        chunk_cap = 1 << 20
        while True:
            if <size_t> len(out_buf) - dst_pos < <size_t> chunk_cap:
                PyByteArray_Resize(
                    out_buf, max(len(out_buf) + len(out_buf) // 2,
                                 <Py_ssize_t> dst_pos + chunk_cap))
            dst_size = <size_t> len(out_buf) - dst_pos
            this_dst = dst_size
            this_src = srcsize - pos
            dst_ptr = <void*> (PyByteArray_AS_STRING(out_buf) + dst_pos)
            with nogil:
                ret = LZ4F_decompress(
                    dctx, dst_ptr, &this_dst,
                    <const void*> (&src[0] + pos), &this_src,
                    NULL)
            if LZ4F_isError(ret):
                raise Lz4Error(
                    f'LZ4F_decompress: {LZ4F_getErrorName(ret).decode()}')
            dst_pos += this_dst
            pos += this_src
            if ret == 0 and pos == srcsize:
                # Last frame complete.
                break
            if ret != 0 and pos == srcsize and this_dst < dst_size:
                raise Lz4Error('LZ4F_decompress: truncated frame')
            if this_src == 0 and this_dst == 0:
                # Decoder consumed nothing and produced nothing: stuck.
                raise Lz4Error('LZ4F_decompress: stalled (no progress)')
        return PyBytes_FromStringAndSize(
            PyByteArray_AS_STRING(out_buf), <Py_ssize_t> dst_pos)
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
        self._prefs.compressionLevel = _frame_level(level)
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


# ---------------------------------------------------------------------------
# Bare LZ4 blocks, and lz4-java's LZ4Block stream (the N5 "lz4" format)
# ---------------------------------------------------------------------------

from libc.string cimport memcpy
from libc.stdint cimport uint32_t, int32_t
from lz4 cimport LZ4_decompress_safe


def block_decode(data, Py_ssize_t size):
    """Decode one bare LZ4 block (no frame) of known decoded ``size``.

    The block format (lz4 "LZ4 Block Format Description") does not record
    the decoded length, so the caller supplies it; the block must decode
    to exactly that many bytes.
    """
    cdef const uint8_t[::1] src
    cdef bytes out
    cdef char* dst
    cdef int ret
    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    if size < 0 or size > 0x7E000000 or src.shape[0] > 0x7FFFFFFF:
        raise Lz4Error(f'lz4 block: unsupported size {size}')
    out = PyBytes_FromStringAndSize(NULL, size)
    if src.shape[0] == 0:
        if size == 0:
            return out
        raise Lz4Error('lz4 block: empty input')
    dst = PyBytes_AsString(out)
    with nogil:
        ret = LZ4_decompress_safe(<const char*> &src[0], dst,
                                  <int> src.shape[0], <int> size)
    if ret != size:
        raise Lz4Error(f'LZ4_decompress_safe returned {ret}, expected {size}')
    return out


cdef extern from *:
    """
    #define OC_XXH_P1 0x9E3779B1U
    #define OC_XXH_P2 0x85EBCA77U
    #define OC_XXH_P3 0xC2B2AE3DU
    #define OC_XXH_P4 0x27D4EB2FU
    #define OC_XXH_P5 0x165667B1U
    #define OC_LZ4BLOCK_SEED 0x9747B28CU
    """
    const uint32_t _XXH_P1 "OC_XXH_P1"
    const uint32_t _XXH_P2 "OC_XXH_P2"
    const uint32_t _XXH_P3 "OC_XXH_P3"
    const uint32_t _XXH_P4 "OC_XXH_P4"
    const uint32_t _XXH_P5 "OC_XXH_P5"
    const uint32_t _LZ4BLOCK_SEED "OC_LZ4BLOCK_SEED"


cdef inline uint32_t _rotl32(uint32_t x, int r) noexcept nogil:
    return (x << r) | (x >> (32 - r))


cdef inline uint32_t _le32(const uint8_t* p) noexcept nogil:
    return (<uint32_t> p[0] | (<uint32_t> p[1] << 8)
            | (<uint32_t> p[2] << 16) | (<uint32_t> p[3] << 24))


cdef uint32_t _xxh32(const uint8_t* p, size_t n, uint32_t seed) noexcept nogil:
    """XXH32 as the xxHash specification (doc/xxhash_spec.md) defines it."""
    cdef const uint8_t* end = p + n
    cdef uint32_t v1, v2, v3, v4, h
    if n >= 16:
        v1 = seed + <uint32_t> _XXH_P1 + <uint32_t> _XXH_P2
        v2 = seed + <uint32_t> _XXH_P2
        v3 = seed
        v4 = seed - <uint32_t> _XXH_P1
        while <size_t> (end - p) >= 16:
            v1 = _rotl32(v1 + _le32(p) * <uint32_t> _XXH_P2, 13) * <uint32_t> _XXH_P1
            v2 = _rotl32(v2 + _le32(p + 4) * <uint32_t> _XXH_P2, 13) * <uint32_t> _XXH_P1
            v3 = _rotl32(v3 + _le32(p + 8) * <uint32_t> _XXH_P2, 13) * <uint32_t> _XXH_P1
            v4 = _rotl32(v4 + _le32(p + 12) * <uint32_t> _XXH_P2, 13) * <uint32_t> _XXH_P1
            p += 16
        h = _rotl32(v1, 1) + _rotl32(v2, 7) + _rotl32(v3, 12) + _rotl32(v4, 18)
    else:
        h = seed + <uint32_t> _XXH_P5
    h += <uint32_t> n
    while <size_t> (end - p) >= 4:
        h = _rotl32(h + _le32(p) * <uint32_t> _XXH_P3, 17) * <uint32_t> _XXH_P4
        p += 4
    while p < end:
        h = _rotl32(h + <uint32_t> p[0] * <uint32_t> _XXH_P5, 11) * <uint32_t> _XXH_P1
        p += 1
    h ^= h >> 15
    h *= <uint32_t> _XXH_P2
    h ^= h >> 13
    h *= <uint32_t> _XXH_P3
    h ^= h >> 16
    return h


def xxh32(data, uint32_t seed=0) -> int:
    """The 32-bit xxHash of ``data`` (exposed for tests)."""
    cdef const uint8_t[::1] src
    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    if src.shape[0] == 0:
        return _xxh32(NULL, 0, seed)
    return _xxh32(&src[0], <size_t> src.shape[0], seed)


# lz4-java net.jpountz.lz4.LZ4BlockOutputStream, which the N5 reference
# implementation (saalfeldlab/n5 Lz4Compression) writes for "lz4" blocks.
LZ4BLOCK_MAGIC = b'LZ4Block'
cdef enum:
    _LZ4BLOCK_HEADER = 21          # magic 8 + token 1 + three int32 LE
    _LZ4BLOCK_RAW = 0x10
    _LZ4BLOCK_LZ4 = 0x20


cdef inline int32_t _le32s(const uint8_t* p) noexcept nogil:
    return <int32_t> _le32(p)


def lz4block_decode(data, *, bint verify=True):
    """Decode an lz4-java ``LZ4BlockOutputStream`` stream.

    That format has no published specification; it is defined by
    lz4-java (``LZ4BlockOutputStream`` / ``LZ4BlockInputStream``), and it
    is what the N5 reference implementation writes for ``"lz4"``
    compression. Each block is a 21-byte header (the magic
    ``LZ4Block``, a token whose high nibble is the method, 0x10 stored
    or 0x20 LZ4, and whose low nibble is log2(block size) - 10, then
    the compressed length, decoded length and checksum as little-endian
    int32) followed by the payload, a bare LZ4 block for method 0x20.
    The checksum is XXH32 of the decoded block with seed 0x9747B28C,
    masked to its low 28 bits as lz4-java stores it. A block with both
    lengths zero ends the stream.

    The same validation LZ4BlockInputStream applies is applied here, and
    a stream with no end block raises, as it does there. Streams written
    one after another decode to their concatenation; any other bytes
    after an end block raise rather than being dropped.
    """
    cdef:
        const uint8_t[::1] src
        const uint8_t* p
        size_t n, pos, total = 0
        Py_ssize_t written = 0
        size_t bstart
        int token, method, level
        int32_t clen, dlen, check
        uint32_t digest
        bytes out
        uint8_t* dst
        list blocks = []
        bint ended
        int ret
    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    n = <size_t> src.shape[0]
    if n == 0:
        raise Lz4Error('LZ4Block: empty input')
    p = &src[0]
    pos = 0
    # Pass 1: walk and validate the headers, and size the output.
    while pos < n:
        if n - pos < _LZ4BLOCK_HEADER or bytes(src[pos:pos + 8]) != LZ4BLOCK_MAGIC:
            if pos == 0:
                raise Lz4Error('LZ4Block: missing LZ4Block magic')
            raise Lz4Error(f'LZ4Block: no block header at offset {pos}')
        ended = False
        while True:
            if n - pos < _LZ4BLOCK_HEADER or bytes(src[pos:pos + 8]) != LZ4BLOCK_MAGIC:
                raise Lz4Error(
                    f'LZ4Block: stream ended prematurely at offset {pos} '
                    f'(no end block)')
            token = p[pos + 8]
            method = token & 0xF0
            level = 10 + (token & 0x0F)
            clen = _le32s(p + pos + 9)
            dlen = _le32s(p + pos + 13)
            check = _le32s(p + pos + 17)
            if method != _LZ4BLOCK_RAW and method != _LZ4BLOCK_LZ4:
                raise Lz4Error(f'LZ4Block: unknown method 0x{method:02x} at offset {pos}')
            if (dlen < 0 or clen < 0 or dlen > (1 << level)
                    or (dlen == 0) != (clen == 0)
                    or (method == _LZ4BLOCK_RAW and dlen != clen)):
                raise Lz4Error(f'LZ4Block: corrupted block header at offset {pos}')
            pos += _LZ4BLOCK_HEADER
            if dlen == 0:
                if check != 0:
                    raise Lz4Error(f'LZ4Block: corrupted end block at offset {pos - _LZ4BLOCK_HEADER}')
                break
            if <size_t> clen > n - pos:
                raise Lz4Error(f'LZ4Block: block at offset {pos - _LZ4BLOCK_HEADER} is truncated')
            blocks.append((pos, method, clen, dlen, check))
            total += <size_t> dlen
            pos += <size_t> clen
    # Pass 2: decode every block into one exact-size output.
    out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> total)
    dst = <uint8_t*> PyBytes_AsString(out)
    for bpos, method, clen, dlen, check in blocks:
        bstart = <size_t> bpos
        if method == _LZ4BLOCK_RAW:
            memcpy(dst + written, p + bstart, <size_t> dlen)
        else:
            with nogil:
                ret = LZ4_decompress_safe(<const char*> (p + bstart),
                                          <char*> (dst + written), clen, dlen)
            if ret != dlen:
                raise Lz4Error(
                    f'LZ4Block: block at offset {bpos - _LZ4BLOCK_HEADER} '
                    f'failed to decode ({ret})')
        if verify:
            with nogil:
                digest = _xxh32(dst + written, <size_t> dlen,
                                _LZ4BLOCK_SEED) & 0x0FFFFFFF
            if <int32_t> digest != check:
                raise Lz4Error(
                    f'LZ4Block: checksum mismatch in block at offset '
                    f'{bpos - _LZ4BLOCK_HEADER}')
        written += dlen
    return out


def lz4block_check_signature(data) -> bool:
    """True if ``data`` starts with lz4-java's ``LZ4Block`` magic."""
    try:
        return bytes(data[:8]) == LZ4BLOCK_MAGIC
    except Exception:
        return False
