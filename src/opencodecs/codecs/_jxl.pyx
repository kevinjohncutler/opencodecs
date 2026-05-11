# opencodecs/codecs/_jxl.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Streaming JPEG XL codec for opencodecs.

Provides:
    JxlReader  - cdef class. Parses header eagerly, yields decoded frames
                 one at a time via iter_frames(). Releases the GIL during
                 each JxlDecoderProcessInput call.
    JxlWriter  - cdef class. Streaming encoder; write_frame() drains
                 compressed bytes to the destination after each frame
                 (FlushInput is called between frames so the output is
                 available without waiting for the full input).

Color: first-class Display P3 / BT.2100 PQ / HLG / linear via the unified
ColorSpec from opencodecs.core.color.

Module-level convenience functions encode() / decode() / iter_frames() /
open() are thin wrappers over the cdef classes.
"""

import io
import os

import numpy as np
cimport numpy as cnp

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memset, memcpy, memmove
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString

from libjxl cimport *

cnp.import_array()


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

_DEC_STATUS_NAMES = {
    0: 'JXL_DEC_SUCCESS',
    1: 'JXL_DEC_ERROR',
    2: 'JXL_DEC_NEED_MORE_INPUT',
    3: 'JXL_DEC_NEED_PREVIEW_OUT_BUFFER',
    5: 'JXL_DEC_NEED_IMAGE_OUT_BUFFER',
    6: 'JXL_DEC_JPEG_NEED_MORE_OUTPUT',
    0x40: 'JXL_DEC_BASIC_INFO',
    0x100: 'JXL_DEC_COLOR_ENCODING',
    0x200: 'JXL_DEC_PREVIEW_IMAGE',
    0x400: 'JXL_DEC_FRAME',
    0x1000: 'JXL_DEC_FULL_IMAGE',
    0x2000: 'JXL_DEC_JPEG_RECONSTRUCTION',
    0x4000: 'JXL_DEC_BOX',
    0x8000: 'JXL_DEC_FRAME_PROGRESSION',
    0x10000: 'JXL_DEC_BOX_COMPLETE',
}


_ENC_STATUS_NAMES = {
    0: 'JXL_ENC_SUCCESS',
    1: 'JXL_ENC_ERROR',
    2: 'JXL_ENC_NEED_MORE_OUTPUT',
}


_ENC_ERR_NAMES = {
    0: 'JXL_ENC_ERR_OK',
    1: 'JXL_ENC_ERR_GENERIC',
    2: 'JXL_ENC_ERR_OOM',
    3: 'JXL_ENC_ERR_JBRD',
    4: 'JXL_ENC_ERR_BAD_INPUT',
    0x80: 'JXL_ENC_ERR_NOT_SUPPORTED',
    0x81: 'JXL_ENC_ERR_API_USAGE',
}


from opencodecs.core.errors import JxlError


cdef _raise_dec(str func, int status):
    name = _DEC_STATUS_NAMES.get(status, f'unknown {status}')
    raise JxlError(f'{func} returned {name}')


cdef _raise_enc(str func, int status, int err_code=-1):
    name = _ENC_STATUS_NAMES.get(status, f'unknown {status}')
    if err_code >= 0:
        ename = _ENC_ERR_NAMES.get(err_code, f'unknown {err_code}')
        raise JxlError(f'{func} returned {name} ({ename})')
    raise JxlError(f'{func} returned {name}')


# ---------------------------------------------------------------------------
# libjxl version
# ---------------------------------------------------------------------------

def libjxl_version() -> str:
    cdef uint32_t v = JxlDecoderVersion()
    return f'{v // 1000000}.{(v // 1000) % 1000}.{v % 1000}'


def check_signature(data) -> bool:
    """Return True if `data` looks like a JPEG XL stream."""
    cdef:
        const uint8_t[::1] buf = data
        JxlSignature sig
        size_t n = min(<size_t> buf.shape[0], <size_t> 16)
    if n == 0:
        return False
    sig = JxlSignatureCheck(&buf[0], n)
    return sig == JXL_SIG_CODESTREAM or sig == JXL_SIG_CONTAINER


# ---------------------------------------------------------------------------
# dtype <-> JxlPixelFormat helpers
# ---------------------------------------------------------------------------

cdef int _dtype_to_jxl(object dtype, JxlDataType* out, uint32_t* bps,
                      uint32_t* exp_bps) except -1:
    """Set JxlDataType + bits-per-sample from a numpy dtype."""
    cdef object d = np.dtype(dtype)
    if d == np.uint8:
        out[0] = JXL_TYPE_UINT8
        bps[0] = 8
        exp_bps[0] = 0
    elif d == np.uint16:
        out[0] = JXL_TYPE_UINT16
        bps[0] = 16
        exp_bps[0] = 0
    elif d == np.float32:
        out[0] = JXL_TYPE_FLOAT
        bps[0] = 32
        exp_bps[0] = 8
    elif d == np.float16:
        out[0] = JXL_TYPE_FLOAT16
        bps[0] = 16
        exp_bps[0] = 5
    else:
        raise ValueError(
            f'jxl: unsupported dtype {d!r} '
            '(want uint8, uint16, float16, or float32)'
        )
    return 0


cdef object _basic_info_to_dtype(const JxlBasicInfo* info):
    """Map decoded JxlBasicInfo back to the numpy dtype we'll output."""
    if info.exponent_bits_per_sample > 0:
        if info.bits_per_sample == 32:
            return np.float32
        if info.bits_per_sample == 16:
            return np.float16
        raise ValueError(
            f'jxl: unsupported float bits_per_sample={info.bits_per_sample}'
        )
    if info.bits_per_sample <= 8:
        return np.uint8
    if info.bits_per_sample <= 16:
        return np.uint16
    return np.float32


cdef _shape_from_basic_info(const JxlBasicInfo* info):
    """Return (frame_shape, samples) for a single decoded frame.

    v0.1 supports L (1ch), RGB, RGBA. Other configurations
    (planar grayscale + extras) raise NotImplementedError.
    """
    cdef:
        size_t color_ch = info.num_color_channels
        size_t extra_ch = info.num_extra_channels
        size_t samples
        bint has_alpha = info.alpha_bits > 0

    if color_ch == 1 and extra_ch == 0:
        return (int(info.ysize), int(info.xsize)), 1
    if color_ch == 3 and extra_ch == 0:
        return (int(info.ysize), int(info.xsize), 3), 3
    if color_ch == 3 and has_alpha and extra_ch == 1:
        return (int(info.ysize), int(info.xsize), 4), 4
    if color_ch == 1 and has_alpha and extra_ch == 1:
        return (int(info.ysize), int(info.xsize), 2), 2
    raise NotImplementedError(
        f'jxl: unsupported channel layout '
        f'(color={color_ch}, extra={extra_ch}, alpha={has_alpha}); '
        'v0.1 supports L, LA, RGB, RGBA only'
    )


# ---------------------------------------------------------------------------
# Per-frame decode helper — runs the whole event loop in nogil so we don't
# pay GIL ping-pong per JxlDecoderProcessInput call. Matches imagecodecs's
# pattern. On Linux x86_64 this is the difference between 0.44x and ~1x
# vs imagecodecs.
# ---------------------------------------------------------------------------

cdef int _RC_FRAME = 0
cdef int _RC_EOF = 1
cdef int _RC_NEED_INPUT = 2   # streaming: caller refills, then re-enters
# negative codes are errors
cdef int _RC_ERR_PROCESS = -1
cdef int _RC_ERR_NEED_INPUT_FATAL = -2  # bytes-in: truncated stream
cdef int _RC_ERR_BUF_SIZE_QUERY = -3
cdef int _RC_ERR_BUF_SIZE_MISMATCH = -4
cdef int _RC_ERR_SET_BUFFER = -5
cdef int _RC_ERR_SET_BIT_DEPTH = -6
cdef int _RC_ERR_DOUBLE_ALLOC = -7


cdef int _decode_one_frame_nogil(
    JxlDecoder* dec,
    JxlPixelFormat* pf,
    JxlBitDepth* bd,
    void* buf,
    size_t buf_size,
    bint* buffer_set_io,
    bint streaming,
) noexcept nogil:
    """Drive the decoder until one frame's pixels have been written into
    ``buf`` (size ``buf_size``), or the stream ends, or — in streaming
    mode — the decoder reports JXL_DEC_NEED_MORE_INPUT.

    ``buffer_set_io`` carries the "have we already called SetImageOutBuffer
    for this frame" state across nogil/refill round-trips so we don't
    call it twice.

    Returns:
      _RC_FRAME (0)         — one frame was decoded into ``buf``
      _RC_EOF   (1)         — JXL_DEC_SUCCESS, no more frames
      _RC_NEED_INPUT (2)    — streaming mode, caller must refill
      <0                    — error code (see _RC_ERR_*)
    """
    cdef:
        JxlDecoderStatus status
        size_t out_size
        bint buffer_set = buffer_set_io[0]
        bint set_bit_depth = bd.dtype == JXL_BIT_DEPTH_FROM_CODESTREAM

    while True:
        status = JxlDecoderProcessInput(dec)

        if status == JXL_DEC_FULL_IMAGE:
            buffer_set_io[0] = buffer_set
            return _RC_FRAME
        if status == JXL_DEC_SUCCESS:
            buffer_set_io[0] = buffer_set
            return _RC_EOF
        if status == JXL_DEC_NEED_MORE_INPUT:
            buffer_set_io[0] = buffer_set
            if streaming:
                return _RC_NEED_INPUT
            return _RC_ERR_NEED_INPUT_FATAL
        if status == JXL_DEC_NEED_IMAGE_OUT_BUFFER:
            if buffer_set:
                return _RC_ERR_DOUBLE_ALLOC
            if JxlDecoderImageOutBufferSize(
                dec, pf, &out_size
            ) != JXL_DEC_SUCCESS:
                return _RC_ERR_BUF_SIZE_QUERY
            if out_size != buf_size:
                return _RC_ERR_BUF_SIZE_MISMATCH
            if JxlDecoderSetImageOutBuffer(
                dec, pf, buf, buf_size
            ) != JXL_DEC_SUCCESS:
                return _RC_ERR_SET_BUFFER
            if set_bit_depth:
                if JxlDecoderSetImageOutBitDepth(
                    dec, bd
                ) != JXL_DEC_SUCCESS:
                    return _RC_ERR_SET_BIT_DEPTH
            buffer_set = True
            continue
        if status == JXL_DEC_ERROR:
            return _RC_ERR_PROCESS
        # JXL_DEC_BASIC_INFO / FRAME / COLOR_ENCODING — informational, ignore.


# ---------------------------------------------------------------------------
# Per-frame encode helper — runs AddImageFrame + (optional) CloseInput +
# the full ProcessOutput drain in ONE nogil block. imagecodecs's pattern.
# Required to avoid GIL ping-pong during encode of big images, which
# otherwise costs 15-30% wall time on Linux x86_64 even when output bytes
# are byte-identical to imagecodecs's.
# ---------------------------------------------------------------------------


cdef int _ENC_OK = 0
cdef int _ENC_ERR_ADD_FRAME = -1
cdef int _ENC_ERR_PROCESS = -2
cdef int _ENC_ERR_REALLOC = -3


cdef int _add_frame_and_drain_nogil(
    JxlEncoder* enc,
    JxlEncoderFrameSettings* fs,
    JxlPixelFormat* pf,
    const void* pixels,
    size_t pixels_size,
    bint close_input,
    uint8_t** outbuf_io,
    size_t* outbuf_capacity_io,
    size_t* outbuf_used_io,
    JxlEncoderStatus* last_status_out,
) noexcept nogil:
    """One nogil block: add the frame, optionally close input, drain output.

    The drain grows ``outbuf`` (doubling, capped at +32 MiB per growth) on
    JXL_ENC_NEED_MORE_OUTPUT. ``outbuf_io`` is updated in place if a realloc
    happened. ``last_status_out`` returns the final libjxl status for
    diagnostics on error.

    Returns _ENC_OK on success or one of the negative _ENC_ERR_* codes.
    """
    cdef:
        JxlEncoderStatus status
        uint8_t* next_out
        size_t avail_out
        uint8_t* outbuf = outbuf_io[0]
        size_t cap = outbuf_capacity_io[0]
        size_t used = outbuf_used_io[0]
        size_t new_capacity
        uint8_t* new_buf

    status = JxlEncoderAddImageFrame(fs, pf, pixels, pixels_size)
    if status != JXL_ENC_SUCCESS:
        last_status_out[0] = status
        return _ENC_ERR_ADD_FRAME

    if close_input:
        JxlEncoderCloseInput(enc)

    # Drain
    while True:
        next_out = outbuf + used
        avail_out = cap - used
        status = JxlEncoderProcessOutput(enc, &next_out, &avail_out)
        used = cap - avail_out
        if status == JXL_ENC_SUCCESS:
            outbuf_io[0] = outbuf
            outbuf_capacity_io[0] = cap
            outbuf_used_io[0] = used
            last_status_out[0] = status
            return _ENC_OK
        if status == JXL_ENC_NEED_MORE_OUTPUT:
            new_capacity = cap * 2
            if new_capacity > cap + 33554432:  # +32 MiB cap per growth
                new_capacity = cap + 33554432
            new_buf = <uint8_t*> realloc(<void*> outbuf, new_capacity)
            if new_buf == NULL:
                last_status_out[0] = status
                return _ENC_ERR_REALLOC
            outbuf = new_buf
            cap = new_capacity
            continue
        # JXL_ENC_ERROR or other
        outbuf_io[0] = outbuf
        outbuf_capacity_io[0] = cap
        outbuf_used_io[0] = used
        last_status_out[0] = status
        return _ENC_ERR_PROCESS


# ---------------------------------------------------------------------------
# JxlReader
# ---------------------------------------------------------------------------

DEF _STREAM_CHUNK_BYTES = 4194304   # 4 MiB
DEF _STREAM_PREFETCH = 4             # 16 MiB ahead in queue
DEF _STREAM_THRESHOLD = 4194304      # files <= this size: skip bg thread
                                     # and slurp in __init__ (the bg-thread
                                     # spawn cost would dominate)


cdef class JxlReader:
    """Streaming JPEG XL decoder.

    Header (shape, dtype, color) is parsed eagerly in __init__ so it's
    available without decoding any pixels. Frames are decoded one at a
    time when iter_frames() / read() / __iter__ is consumed.

    For path / file-like inputs larger than ~4 MiB, a background thread
    reads the file in 4 MiB chunks while the main thread feeds bytes to
    libjxl via ``JxlDecoderSetInput`` / ``JxlDecoderReleaseInput``. The
    bg-thread file.read() runs concurrently with libjxl's decode workers
    (the read syscall releases the GIL; libjxl ProcessInput runs in
    nogil) — so on slow storage you get real wall-clock overlap of I/O
    with decode work, bounded by ``max(read_time, decode_time)`` per
    chunk instead of ``read_time + decode_time``.

    Smaller files are slurped directly (the bg thread overhead would
    dominate the actual I/O cost for files that fit in one chunk).

    Bytes / memoryview inputs use the buffer directly — there's no I/O
    to overlap with.
    """

    cdef:
        JxlDecoder* _decoder
        void* _runner
        size_t _num_threads
        # bytes-in mode: keep a ref so the buffer stays alive
        object _data_ref
        const uint8_t* _src_data
        size_t _src_size
        # path-input mode: keep file open so buffers stay valid
        object _file_handle
        bint _own_file_handle
        # streaming mode: bg-thread chunk reader + rolling input buffer
        object _chunk_reader
        bint _streaming
        uint8_t* _input_buf
        size_t _input_buf_capacity
        size_t _input_buf_used
        bint _input_eof
        bint _input_closed
        # decoded state
        JxlBasicInfo _basic_info
        JxlColorEncoding _color_encoding
        JxlPixelFormat _pixel_format
        JxlBitDepth _bit_depth
        bint _have_basic_info
        bint _have_color
        bint _parse_color
        bint _icc_fetched
        bint _coalesce
        bint _keep_orientation
        bint _closed
        bint _frames_started
        bint _exhausted
        bytes _icc_profile
        object _frame_dtype
        object _frame_shape
        size_t _samples
        size_t _frame_nbytes

    def __cinit__(self):
        self._decoder = NULL
        self._runner = NULL
        self._num_threads = 0
        self._data_ref = None
        self._src_data = NULL
        self._src_size = 0
        self._file_handle = None
        self._own_file_handle = False
        self._chunk_reader = None
        self._streaming = False
        self._input_buf = NULL
        self._input_buf_capacity = 0
        self._input_buf_used = 0
        self._input_eof = False
        self._input_closed = False
        self._have_basic_info = False
        self._have_color = False
        self._parse_color = False
        self._icc_fetched = False
        self._coalesce = True
        self._keep_orientation = False
        self._closed = False
        self._frames_started = False
        self._exhausted = False
        self._icc_profile = None
        self._frame_dtype = None
        self._frame_shape = None
        self._samples = 0
        self._frame_nbytes = 0
        memset(<void*> &self._basic_info, 0, sizeof(JxlBasicInfo))
        memset(<void*> &self._color_encoding, 0, sizeof(JxlColorEncoding))
        memset(<void*> &self._pixel_format, 0, sizeof(JxlPixelFormat))
        memset(<void*> &self._bit_depth, 0, sizeof(JxlBitDepth))

    def __init__(self, data, *, numthreads=None, keep_orientation=False,
                 coalesce=True, parse_color=True, streaming=False,
                 skip_frames=0):
        # parse_color=True (default) subscribes to JXL_DEC_COLOR_ENCODING and
        # populates self.color from the encoded color tags. parse_color=False
        # skips that subscription and goes straight from BASIC_INFO to FULL_IMAGE,
        # which on Linux can be ~2x faster on the decode path because libjxl
        # doesn't have to construct the encoded color profile representation.
        # The decode() / read() helpers default to parse_color=False to match
        # imagecodecs's deliberate choice (their _jpegxl.pyx has the same
        # subscription commented out for the same reason). The ICC profile is
        # fetched lazily on first .icc_profile access regardless.
        cdef:
            const uint8_t[::1] view
            JxlSignature sig
            JxlDecoderStatus status
            int events
            size_t nthreads

        # Dispatch on input type:
        #   path/file-like + streaming=True + size > threshold → bg-thread
        #     streaming. Off by default because on typical NAS+APFS setups
        #     the kernel's prefetch during a single big read() outperforms
        #     our chunked-read pipeline (a 121 MB cold-NAS slurp is ~53 ms,
        #     while bg-thread chunked reads of the same file take ~130 ms —
        #     each smaller read can't be pipelined as aggressively as one
        #     big read by the SMB driver). Useful for very-slow storage or
        #     files larger than RAM where slurp would OOM. Set streaming=True
        #     to opt in.
        #   anything else → slurp into bytes, SetInput once. The kernel
        #     handles read-ahead during slurp.
        cdef bint can_stream = False
        cdef object src_for_stream = None
        cdef size_t threshold = _STREAM_THRESHOLD

        if streaming and isinstance(data, (str, os.PathLike)):
            try:
                fsize = os.path.getsize(data)
            except OSError:
                fsize = -1
            if fsize > threshold:
                src_for_stream = data
                can_stream = True
            else:
                with open(data, 'rb') as f:
                    data = f.read()
        elif streaming and (hasattr(data, 'read')
                            and not isinstance(data, (bytes, bytearray, memoryview))):
            fsize = -1
            try:
                fd_no = data.fileno()
                fsize = os.fstat(fd_no).st_size
            except (AttributeError, OSError):
                try:
                    cur = data.tell()
                    data.seek(0, 2)
                    fsize = data.tell()
                    data.seek(cur)
                except (AttributeError, OSError):
                    pass
            if fsize > threshold:
                src_for_stream = data
                can_stream = True
            else:
                data = data.read()
        elif isinstance(data, (str, os.PathLike)):
            with open(data, 'rb') as f:
                data = f.read()
        elif (hasattr(data, 'read')
              and not isinstance(data, (bytes, bytearray, memoryview))):
            data = data.read()

        if can_stream:
            self._init_streaming(src_for_stream)
        else:
            view = data
            self._data_ref = data
            self._src_data = &view[0] if view.shape[0] > 0 else NULL
            self._src_size = <size_t> view.shape[0]

        if self._src_size == 0:
            raise ValueError('jxl: empty input')
        sig = JxlSignatureCheck(
            self._src_data, min(self._src_size, <size_t> 16))
        if sig != JXL_SIG_CODESTREAM and sig != JXL_SIG_CONTAINER:
            raise ValueError('jxl: not a JPEG XL stream')

        if numthreads is None:
            nthreads = JxlThreadParallelRunnerDefaultNumWorkerThreads()
        elif numthreads <= 0:
            nthreads = JxlThreadParallelRunnerDefaultNumWorkerThreads()
        else:
            nthreads = <size_t> numthreads
        self._num_threads = nthreads
        self._coalesce = bool(coalesce)
        self._keep_orientation = bool(keep_orientation)

        self._decoder = JxlDecoderCreate(NULL)
        if self._decoder == NULL:
            raise JxlError('JxlDecoderCreate returned NULL')

        if nthreads > 1:
            self._runner = JxlThreadParallelRunnerCreate(NULL, nthreads)
            if self._runner == NULL:
                raise JxlError('JxlThreadParallelRunnerCreate returned NULL')
            status = JxlDecoderSetParallelRunner(
                self._decoder, JxlThreadParallelRunner, self._runner)
            if status != JXL_DEC_SUCCESS:
                _raise_dec('JxlDecoderSetParallelRunner', status)

        if self._keep_orientation:
            status = JxlDecoderSetKeepOrientation(self._decoder, JXL_TRUE)
            if status != JXL_DEC_SUCCESS:
                _raise_dec('JxlDecoderSetKeepOrientation', status)

        # Default coalescing is JXL_TRUE; only set explicitly when we want
        # to disable it. (imagecodecs never calls SetCoalescing — match.)
        if not self._coalesce:
            status = JxlDecoderSetCoalescing(self._decoder, JXL_FALSE)
            if status != JXL_DEC_SUCCESS:
                _raise_dec('JxlDecoderSetCoalescing', status)

        cdef bint do_parse_color = bool(parse_color)
        self._parse_color = do_parse_color
        # Match imagecodecs: only subscribe to BASIC_INFO + FULL_IMAGE on the
        # fast path. Each subscribed event is a thread-pool sync point inside
        # libjxl; on Linux x86_64 with big images this is the difference
        # between ~0.5x and parity. When parse_color is requested we add
        # COLOR_ENCODING (and pay the small extra cost). FRAME is never
        # subscribed because we don't need per-frame headers — we already
        # know the canvas shape from BASIC_INFO when coalescing is on.
        if do_parse_color:
            events = JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING | \
                JXL_DEC_FULL_IMAGE
        else:
            events = JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE
        status = JxlDecoderSubscribeEvents(self._decoder, events)
        if status != JXL_DEC_SUCCESS:
            _raise_dec('JxlDecoderSubscribeEvents', status)

        # Initial SetInput. Whether the bytes came from slurp, streaming
        # buffer, or a bytes-in caller, libjxl just sees one big buffer.
        status = JxlDecoderSetInput(
            self._decoder, self._src_data, self._src_size)
        if status != JXL_DEC_SUCCESS:
            _raise_dec('JxlDecoderSetInput', status)
        # Deliberately NOT calling CloseInput — imagecodecs also skips it.
        # NEED_MORE_INPUT in this path means truncated and we treat it as
        # an error.

        # If the caller wants to start at a non-zero frame (parallel
        # multi-frame decode pattern), tell libjxl to fast-skip past the
        # earlier frames. SkipFrames is bitstream-parse-only — it doesn't
        # do pixel decode for the skipped frames, so per-worker overhead
        # is small.
        if skip_frames > 0:
            JxlDecoderSkipFrames(self._decoder, <size_t> int(skip_frames))

        # drive until we've seen both BASIC_INFO and COLOR_ENCODING
        self._parse_header()

    cdef _init_streaming(self, src):
        """Set up the bg-thread chunk reader and the rolling input buffer
        for path/file-like inputs that are large enough to benefit from
        I/O+decode overlap.

        We allocate the input buffer at 2x chunk_size: at any time it
        holds the unprocessed tail (small) plus one fresh chunk. When
        ProcessInput returns NEED_MORE_INPUT, _refill_input memmoves the
        tail to the front, blocks on chunk_reader.get(), and SetInputs
        the combined buffer.
        """
        from opencodecs.core.io import BackgroundChunkReader
        cdef bytes first_chunk
        cdef ssize_t n
        cdef JxlSignature sig

        self._chunk_reader = BackgroundChunkReader(
            src,
            chunk_size=_STREAM_CHUNK_BYTES,
            prefetch=_STREAM_PREFETCH,
        )
        self._streaming = True
        # 2x chunk so we always have headroom for tail + one full chunk.
        self._input_buf_capacity = 2 * <size_t> _STREAM_CHUNK_BYTES
        self._input_buf = <uint8_t*> malloc(self._input_buf_capacity)
        if self._input_buf == NULL:
            raise MemoryError('jxl: failed to allocate streaming input buffer')

        first_chunk = self._chunk_reader.get()
        if first_chunk is None or len(first_chunk) == 0:
            raise ValueError('jxl: empty input')
        n = len(first_chunk)
        memcpy(self._input_buf,
               <const uint8_t*> PyBytes_AsString(first_chunk),
               <size_t> n)
        self._input_buf_used = <size_t> n
        # Signature check on the first bytes
        sig = JxlSignatureCheck(
            self._input_buf, min(self._input_buf_used, <size_t> 16))
        if sig != JXL_SIG_CODESTREAM and sig != JXL_SIG_CONTAINER:
            raise ValueError('jxl: not a JPEG XL stream')
        # Streaming mode advertises src_data/size as a sentinel for the
        # downstream code; the actual buffer is _input_buf.
        self._src_data = self._input_buf
        self._src_size = self._input_buf_used

    cdef _refill_input(self):
        """Called when ProcessInput returns NEED_MORE_INPUT.

        Move the unprocessed tail to the front of the input buffer, pull
        the next chunk from the bg reader (which has been filling the
        queue while libjxl was decoding), and SetInput on the combined
        bytes. CloseInput once the file is fully drained.
        """
        cdef:
            size_t unprocessed
            size_t tail_offset
            JxlDecoderStatus status
            ssize_t chunk_len
            bytes chunk

        unprocessed = JxlDecoderReleaseInput(self._decoder)
        tail_offset = self._input_buf_used - unprocessed
        if unprocessed > 0 and tail_offset > 0:
            memmove(<void*> self._input_buf,
                    <const void*> (self._input_buf + tail_offset),
                    unprocessed)
        self._input_buf_used = unprocessed

        if self._input_eof:
            if unprocessed == 0:
                raise JxlError(
                    'jxl: NEED_MORE_INPUT after EOF (truncated stream?)')
            # libjxl may want one more SetInput pass on the leftover.
        else:
            chunk = self._chunk_reader.get()
            if chunk is None:
                self._input_eof = True
            else:
                chunk_len = len(chunk)
                if (self._input_buf_used + <size_t> chunk_len
                        > self._input_buf_capacity):
                    # Should not happen with prefetch chunks bounded by
                    # _STREAM_CHUNK_BYTES; if it does, grow the buffer.
                    self._grow_input_buf(self._input_buf_used
                                         + <size_t> chunk_len)
                memcpy(self._input_buf + self._input_buf_used,
                       <const uint8_t*> PyBytes_AsString(chunk),
                       <size_t> chunk_len)
                self._input_buf_used += <size_t> chunk_len

        if self._input_buf_used == 0:
            raise JxlError('jxl: empty refill (truncated?)')

        status = JxlDecoderSetInput(
            self._decoder, self._input_buf, self._input_buf_used)
        if status != JXL_DEC_SUCCESS:
            _raise_dec('JxlDecoderSetInput', status)
        if self._input_eof and not self._input_closed:
            JxlDecoderCloseInput(self._decoder)
            self._input_closed = True

    cdef _grow_input_buf(self, size_t needed):
        cdef:
            size_t new_cap = self._input_buf_capacity
            uint8_t* new_buf
        while new_cap < needed:
            new_cap *= 2
        new_buf = <uint8_t*> realloc(<void*> self._input_buf, new_cap)
        if new_buf == NULL:
            raise MemoryError('jxl: failed to grow streaming input buffer')
        self._input_buf = new_buf
        self._input_buf_capacity = new_cap

    cdef _parse_header(self):
        cdef:
            JxlDecoderStatus status
            size_t icc_size = 0
            cnp.ndarray icc_arr

        # If we didn't subscribe to COLOR_ENCODING, stop after BASIC_INFO.
        # Pretend "have_color" is satisfied so the loop terminates.
        if not self._parse_color:
            self._have_color = True

        while not (self._have_basic_info and self._have_color):
            with nogil:
                status = JxlDecoderProcessInput(self._decoder)

            if status == JXL_DEC_ERROR:
                _raise_dec('JxlDecoderProcessInput', status)
            if status == JXL_DEC_NEED_MORE_INPUT:
                if self._streaming:
                    self._refill_input()
                    continue
                raise JxlError(
                    'JxlDecoderProcessInput needs more input than was '
                    'provided (truncated stream?)')
            if status == JXL_DEC_BASIC_INFO:
                if JxlDecoderGetBasicInfo(
                    self._decoder, &self._basic_info
                ) != JXL_DEC_SUCCESS:
                    raise JxlError('JxlDecoderGetBasicInfo failed')
                self._have_basic_info = True
                self._frame_shape, samples = _shape_from_basic_info(
                    &self._basic_info)
                self._samples = <size_t> samples
                self._frame_dtype = _basic_info_to_dtype(&self._basic_info)
                self._configure_pixel_format()
            elif status == JXL_DEC_COLOR_ENCODING:
                # Cheap: read encoded profile (primaries, transfer, etc.).
                # We deliberately DO NOT call GetColorAsICCProfile here —
                # that forces libjxl to construct the ICC bytes, which is
                # expensive (especially on Linux x86_64). Lazy ICC fetch via
                # the .icc_profile property handles that on demand.
                JxlDecoderGetColorAsEncodedProfile(
                    self._decoder,
                    JXL_COLOR_PROFILE_TARGET_DATA,
                    &self._color_encoding,
                )
                self._have_color = True
            elif status == JXL_DEC_FRAME:
                # We hit a frame event before BASIC_INFO+COLOR were processed.
                # Some streams don't emit COLOR_ENCODING; treat that as OK.
                self._have_color = True
                # Stop here; iter_frames will pick up.
                # We can't "rewind" the FRAME event. Instead, mark we already
                # entered frames; the outer iter_frames loop will respect this.
                self._frames_started = True
                break
            elif status == JXL_DEC_SUCCESS:
                # Stream had no pixel data (header-only?); leave _exhausted.
                self._exhausted = True
                break

    cdef _configure_pixel_format(self):
        cdef JxlBasicInfo* info = &self._basic_info

        self._pixel_format.num_channels = <uint32_t> self._samples
        self._pixel_format.endianness = JXL_NATIVE_ENDIAN
        self._pixel_format.align = 0

        if info.exponent_bits_per_sample > 0:
            if info.bits_per_sample == 32:
                self._pixel_format.data_type = JXL_TYPE_FLOAT
            elif info.bits_per_sample == 16:
                self._pixel_format.data_type = JXL_TYPE_FLOAT16
            else:
                raise ValueError(
                    f'jxl: unsupported float bps={info.bits_per_sample}')
            self._bit_depth.dtype = JXL_BIT_DEPTH_FROM_PIXEL_FORMAT
            self._bit_depth.bits_per_sample = info.bits_per_sample
            self._bit_depth.exponent_bits_per_sample = (
                info.exponent_bits_per_sample)
        elif info.bits_per_sample <= 8:
            self._pixel_format.data_type = JXL_TYPE_UINT8
            self._bit_depth.dtype = JXL_BIT_DEPTH_FROM_CODESTREAM
            self._bit_depth.bits_per_sample = info.bits_per_sample
            self._bit_depth.exponent_bits_per_sample = 0
        elif info.bits_per_sample <= 16:
            self._pixel_format.data_type = JXL_TYPE_UINT16
            self._bit_depth.dtype = JXL_BIT_DEPTH_FROM_CODESTREAM
            self._bit_depth.bits_per_sample = info.bits_per_sample
            self._bit_depth.exponent_bits_per_sample = 0
        else:
            self._pixel_format.data_type = JXL_TYPE_FLOAT
            self._bit_depth.dtype = JXL_BIT_DEPTH_FROM_PIXEL_FORMAT
            self._bit_depth.bits_per_sample = 32
            self._bit_depth.exponent_bits_per_sample = 8

        cdef size_t itemsize
        if self._pixel_format.data_type == JXL_TYPE_FLOAT:
            itemsize = 4
        elif self._pixel_format.data_type == JXL_TYPE_FLOAT16:
            itemsize = 2
        elif self._pixel_format.data_type == JXL_TYPE_UINT16:
            itemsize = 2
        else:
            itemsize = 1
        self._frame_nbytes = (
            <size_t> self._basic_info.xsize
            * <size_t> self._basic_info.ysize
            * <size_t> self._samples
            * itemsize
        )

    # ------------------------------------------------------------------ props

    @property
    def xsize(self) -> int:
        return int(self._basic_info.xsize)

    @property
    def ysize(self) -> int:
        return int(self._basic_info.ysize)

    @property
    def samples(self) -> int:
        return int(self._samples)

    @property
    def frame_shape(self):
        return self._frame_shape

    @property
    def dtype(self):
        return self._frame_dtype

    @property
    def is_animation(self) -> bool:
        return bool(self._basic_info.have_animation)

    @property
    def n_frames(self):
        """Best-effort frame count (None if unknown until decoded)."""
        # libjxl does not expose a frame count directly without scanning.
        return None

    @property
    def color(self):
        """Decoded JxlColorEncoding as a dict.

        None if the reader was opened with ``parse_color=False`` (the default
        for the decode/read fast path). Open with ``parse_color=True`` (or
        use ``opencodecs.jxl.open(...)``) to populate this.
        """
        if not self._parse_color:
            return None
        return {
            'color_space': int(self._color_encoding.color_space),
            'white_point': int(self._color_encoding.white_point),
            'primaries': int(self._color_encoding.primaries),
            'transfer_function': int(self._color_encoding.transfer_function),
            'rendering_intent': int(self._color_encoding.rendering_intent),
        }

    @property
    def icc_profile(self) -> bytes | None:
        """ICC profile bytes (lazy).

        Fetched on first access via JxlDecoderGetColorAsICCProfile so we don't
        pay the cost on every decode. Returns None if the reader was opened
        with parse_color=False or is already closed, or if the stream has no
        retrievable ICC profile.
        """
        if self._icc_fetched:
            return self._icc_profile
        if not self._parse_color or self._closed or self._decoder == NULL:
            return None
        self._fetch_icc_profile()
        return self._icc_profile

    cdef _fetch_icc_profile(self):
        cdef:
            size_t icc_size = 0
            cnp.ndarray icc_arr
        self._icc_fetched = True
        if JxlDecoderGetICCProfileSize(
            self._decoder,
            JXL_COLOR_PROFILE_TARGET_DATA,
            &icc_size,
        ) != JXL_DEC_SUCCESS or icc_size == 0:
            return
        icc_arr = np.empty(icc_size, dtype=np.uint8)
        if JxlDecoderGetColorAsICCProfile(
            self._decoder,
            JXL_COLOR_PROFILE_TARGET_DATA,
            <uint8_t*> cnp.PyArray_DATA(icc_arr),
            icc_size,
        ) == JXL_DEC_SUCCESS:
            self._icc_profile = bytes(icc_arr)

    @property
    def basic_info(self) -> dict:
        cdef JxlBasicInfo* info = &self._basic_info
        return {
            'xsize': int(info.xsize),
            'ysize': int(info.ysize),
            'bits_per_sample': int(info.bits_per_sample),
            'exponent_bits_per_sample': int(info.exponent_bits_per_sample),
            'num_color_channels': int(info.num_color_channels),
            'num_extra_channels': int(info.num_extra_channels),
            'alpha_bits': int(info.alpha_bits),
            'have_animation': bool(info.have_animation),
            'orientation': int(info.orientation),
            'intensity_target': float(info.intensity_target),
            'min_nits': float(info.min_nits),
            'uses_original_profile': bool(info.uses_original_profile),
        }

    # ------------------------------------------------------------------ iter

    cdef _raise_decode_error(self, int rc):
        msg = {
            _RC_ERR_PROCESS: 'JxlDecoderProcessInput returned JXL_DEC_ERROR',
            _RC_ERR_NEED_INPUT_FATAL: 'JxlDecoderProcessInput needs more '
                                      'input (truncated stream?)',
            _RC_ERR_BUF_SIZE_QUERY: 'JxlDecoderImageOutBufferSize failed',
            _RC_ERR_BUF_SIZE_MISMATCH: 'jxl: out buffer size mismatch',
            _RC_ERR_SET_BUFFER: 'JxlDecoderSetImageOutBuffer failed',
            _RC_ERR_SET_BIT_DEPTH: 'JxlDecoderSetImageOutBitDepth failed',
            _RC_ERR_DOUBLE_ALLOC: 'jxl: double-allocation in decode loop',
        }.get(rc, f'jxl: unknown decode error code {rc}')
        raise JxlError(msg)

    def iter_frames(self):
        """Yield one decoded numpy array per JXL frame.

        The per-frame event loop runs inside ``with nogil:``. In streaming
        mode, when libjxl reports JXL_DEC_NEED_MORE_INPUT we drop back to
        Python, refill from the bg-thread chunk reader (bytes likely
        already waiting in the queue from prefetch), then re-enter the
        nogil loop. This is where I/O and decode overlap on the wall
        clock — the bg thread's file.read() ran in parallel with libjxl's
        worker threads chewing on the previous chunk.
        """
        cdef:
            cnp.ndarray frame
            void* buf_ptr
            size_t expected_size
            int rc
            bint buffer_set
            bint streaming = self._streaming

        if self._closed:
            raise RuntimeError('JxlReader is closed')

        while not self._exhausted:
            frame = np.empty(self._frame_shape, dtype=self._frame_dtype)
            buf_ptr = cnp.PyArray_DATA(frame)
            expected_size = self._frame_nbytes
            buffer_set = False

            while True:
                with nogil:
                    rc = _decode_one_frame_nogil(
                        self._decoder,
                        &self._pixel_format,
                        &self._bit_depth,
                        buf_ptr, expected_size,
                        &buffer_set,
                        streaming,
                    )
                if rc == _RC_NEED_INPUT:
                    self._refill_input()
                    continue  # re-enter nogil with the new bytes
                break

            if rc == _RC_FRAME:
                yield frame
                continue
            if rc == _RC_EOF:
                self._exhausted = True
                return
            self._raise_decode_error(rc)

    def __iter__(self):
        return self.iter_frames()

    def read(self):
        """Decode the entire stream.

        Returns a single numpy array. Single-frame streams return the frame
        directly (shape == frame_shape). Multi-frame streams stack along
        a new leading axis.
        """
        frames = list(self.iter_frames())
        if not frames:
            raise JxlError('jxl: no frames in stream')
        if len(frames) == 1:
            return frames[0]
        return np.stack(frames, axis=0)

    # ------------------------------------------------------------------ ctx

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        if self._closed:
            return
        # Decoder MUST be destroyed first so it isn't still pointing at
        # _input_buf or _data_ref when we free them.
        if self._decoder != NULL:
            JxlDecoderDestroy(self._decoder)
            self._decoder = NULL
        if self._runner != NULL:
            JxlThreadParallelRunnerDestroy(self._runner)
            self._runner = NULL
        if self._chunk_reader is not None:
            try:
                self._chunk_reader.close()
            except Exception:
                pass
            self._chunk_reader = None
        if self._input_buf != NULL:
            free(self._input_buf)
            self._input_buf = NULL
        self._src_data = NULL
        self._src_size = 0
        self._data_ref = None
        if self._file_handle is not None and self._own_file_handle:
            try:
                self._file_handle.close()
            except Exception:
                pass
        self._file_handle = None
        self._closed = True

    def __dealloc__(self):
        if self._decoder != NULL:
            JxlDecoderDestroy(self._decoder)
            self._decoder = NULL
        if self._runner != NULL:
            JxlThreadParallelRunnerDestroy(self._runner)
            self._runner = NULL
        if self._input_buf != NULL:
            free(self._input_buf)
            self._input_buf = NULL


# ---------------------------------------------------------------------------
# JxlWriter
# ---------------------------------------------------------------------------

DEF _CHUNK_BYTES = 1048576  # 1 MiB drain chunk — keeps per-write Python
                            # overhead amortized for large encoded streams.
                            # Smaller = lower latency to first byte (matters
                            # for true network streaming, which v0.1 doesn't
                            # do anyway since we use the sync ProcessOutput).


cdef class JxlWriter:
    """Streaming JPEG XL encoder.

    Construct with a destination (path / file-like / None for in-memory),
    call write_frame(arr) for each frame, then close(). After each frame
    `JxlEncoderFlushInput` is called so output bytes are made available
    to the destination immediately rather than buffered until close.

    Usage::

        with JxlWriter('out.jxl', color='display-p3', lossless=True) as w:
            w.write_frame(arr)

        # In-memory:
        w = JxlWriter()
        w.write_frame(arr)
        data = w.close()
    """

    cdef:
        JxlEncoder* _encoder
        JxlEncoderFrameSettings* _frame_settings
        void* _runner
        size_t _num_threads

        object _dest_obj
        bint _own_dest
        bint _is_buffer

        JxlBasicInfo _basic_info
        JxlPixelFormat _pixel_format
        JxlBitDepth _bit_depth
        JxlColorEncoding _color_encoding

        bint _config_done
        bint _closed
        bint _animation
        bint _container

        # frame settings
        bint _lossless
        float _distance
        int _effort
        int _decoding_speed
        object _color_spec  # ColorSpec or None
        float _intensity_target  # nits; 0.0 = leave libjxl default
        bytes _icc_profile  # ICC profile bytes; if set, used instead of enum

        # first-frame validation cache
        object _first_dtype
        size_t _first_xsize, _first_ysize, _first_samples
        size_t _frame_nbytes

        # Growing output buffer (output_t-style). One realloc-grow on
        # NEED_MORE_OUTPUT keeps libjxl's thread-pool-sync count down — small
        # fixed chunks force ProcessOutput to flush thread state on each call,
        # which is the dominant overhead at numthreads=8 for big files.
        uint8_t* _outbuf
        size_t _outbuf_capacity
        size_t _outbuf_used

        size_t _frame_count

    def __cinit__(self):
        self._encoder = NULL
        self._frame_settings = NULL
        self._runner = NULL
        self._num_threads = 0
        self._dest_obj = None
        self._own_dest = False
        self._is_buffer = False
        self._config_done = False
        self._closed = False
        self._animation = False
        self._container = False
        self._lossless = False
        self._distance = 1.0
        self._effort = 5
        self._decoding_speed = 0
        self._color_spec = None
        self._intensity_target = 0.0
        self._icc_profile = None
        self._first_dtype = None
        self._first_xsize = 0
        self._first_ysize = 0
        self._first_samples = 0
        self._frame_nbytes = 0
        self._outbuf = NULL
        self._outbuf_capacity = 0
        self._outbuf_used = 0
        self._frame_count = 0
        memset(<void*> &self._basic_info, 0, sizeof(JxlBasicInfo))
        memset(<void*> &self._pixel_format, 0, sizeof(JxlPixelFormat))
        memset(<void*> &self._bit_depth, 0, sizeof(JxlBitDepth))
        memset(<void*> &self._color_encoding, 0, sizeof(JxlColorEncoding))

    def __init__(self, dest=None, *, color=None, lossless=None,
                 quality=None, distance=None, effort=5, decoding_speed=0,
                 numthreads=None, animation=False, container=False,
                 intensity_target=None, icc_profile=None):
        from opencodecs.core.color import parse_color, ColorSpec, JXL_TF_GAMMA

        # Resolve destination. dest=None means we hold the encoded bytes
        # in our own _outbuf and return them at close() — skipping BytesIO
        # avoids ~3 memcpys of the output.
        if dest is None:
            self._dest_obj = None
            self._own_dest = False
            self._is_buffer = True
        elif isinstance(dest, (str, os.PathLike)):
            self._dest_obj = open(dest, 'wb')
            self._own_dest = True
            self._is_buffer = False
        else:
            if not hasattr(dest, 'write'):
                raise TypeError(
                    f'dest must be path, file-like, or None; got {type(dest)}')
            self._dest_obj = dest
            self._own_dest = False
            self._is_buffer = False

        # Color
        self._color_spec = parse_color(color)

        # Quality / distance / lossless
        cdef float dist
        if quality is not None and distance is not None:
            raise ValueError('jxl: pass quality or distance, not both')
        if quality is not None:
            dist = JxlEncoderDistanceFromQuality(<float> quality)
            self._distance = dist
            if lossless is None:
                self._lossless = (quality > 100)
        elif distance is not None:
            self._distance = float(distance)
            if lossless is None:
                self._lossless = self._distance == 0.0
        else:
            self._distance = 1.0
            self._lossless = bool(lossless) if lossless is not None else False

        if lossless is not None:
            self._lossless = bool(lossless)

        if self._lossless:
            self._distance = 0.0

        self._effort = int(effort) if effort is not None else 5
        if self._effort < 1:
            self._effort = 1
        if self._effort > 10:
            self._effort = 10
        self._decoding_speed = int(decoding_speed) if decoding_speed else 0
        if self._decoding_speed < 0:
            self._decoding_speed = 0
        if self._decoding_speed > 4:
            self._decoding_speed = 4

        if numthreads is None or numthreads <= 0:
            self._num_threads = JxlThreadParallelRunnerDefaultNumWorkerThreads()
        else:
            self._num_threads = <size_t> numthreads

        self._animation = bool(animation)
        self._container = bool(container)

        # Optional brightness anchor for HDR / linear-light files.
        # 0.0 = leave libjxl default (255 nits for SDR transfer; we set
        # 10000 below for PQ/HLG). For linear-tagged HDR set this to the
        # nit level corresponding to the brightest encoded value (e.g.
        # 1200 for a 12x-SDR-headroom file with reference SDR=100).
        if intensity_target is None:
            self._intensity_target = 0.0
        else:
            self._intensity_target = float(intensity_target)
            if self._intensity_target < 0.0:
                raise ValueError('intensity_target must be >= 0')

        if icc_profile is None:
            self._icc_profile = None
        else:
            if not isinstance(icc_profile, (bytes, bytearray, memoryview)):
                raise TypeError(
                    'icc_profile must be bytes-like; got '
                    f'{type(icc_profile).__name__}')
            self._icc_profile = bytes(icc_profile)
            if len(self._icc_profile) < 128:
                raise ValueError(
                    f'icc_profile too short ({len(self._icc_profile)} bytes); '
                    'a valid ICC profile is at least 128 bytes')

        # Output buffer is allocated lazily in _configure_from_first_frame
        # once we know the input size — mirrors imagecodecs's
        # max(32KB, srcsize / 4) heuristic so most encodes fit in the
        # initial allocation and avoid realloc-grow round-trips.
        self._outbuf = NULL
        self._outbuf_capacity = 0
        self._outbuf_used = 0

        # Encoder + runner
        self._encoder = JxlEncoderCreate(NULL)
        if self._encoder == NULL:
            raise JxlError('JxlEncoderCreate returned NULL')

        cdef JxlEncoderStatus status
        if self._num_threads > 1:
            self._runner = JxlThreadParallelRunnerCreate(
                NULL, self._num_threads)
            if self._runner == NULL:
                raise JxlError('JxlThreadParallelRunnerCreate returned NULL')
            status = JxlEncoderSetParallelRunner(
                self._encoder, JxlThreadParallelRunner, self._runner)
            if status != JXL_ENC_SUCCESS:
                _raise_enc('JxlEncoderSetParallelRunner', status,
                           JxlEncoderGetError(self._encoder))

    # ------------------------------------------------------------------ cfg

    cdef _configure_from_first_frame(self, cnp.ndarray arr):
        """Set up basic_info/pixel_format/color/frame_settings from first frame."""
        from opencodecs.core.color import JXL_TF_GAMMA

        cdef:
            int ndim = arr.ndim
            size_t xsize, ysize, samples
            JxlDataType jxl_dt
            uint32_t bps, exp_bps
            JxlEncoderStatus status
            int err
            float dist

        # Validate / parse shape
        if ndim == 2:
            ysize = <size_t> arr.shape[0]
            xsize = <size_t> arr.shape[1]
            samples = 1
        elif ndim == 3:
            ysize = <size_t> arr.shape[0]
            xsize = <size_t> arr.shape[1]
            samples = <size_t> arr.shape[2]
            if samples not in (1, 2, 3, 4):
                raise ValueError(
                    f'jxl: last dim must be 1/2/3/4 channels (got {samples})')
        else:
            raise ValueError(
                f'jxl: ndim={ndim}; expected 2 (Y,X) or 3 (Y,X,C)')

        # dtype
        _dtype_to_jxl(arr.dtype, &jxl_dt, &bps, &exp_bps)

        self._first_dtype = arr.dtype
        self._first_xsize = xsize
        self._first_ysize = ysize
        self._first_samples = samples
        self._frame_nbytes = arr.nbytes

        # basic_info
        JxlEncoderInitBasicInfo(&self._basic_info)
        self._basic_info.xsize = <uint32_t> xsize
        self._basic_info.ysize = <uint32_t> ysize
        self._basic_info.bits_per_sample = bps
        self._basic_info.exponent_bits_per_sample = exp_bps
        if samples == 1 or samples == 2:
            self._basic_info.num_color_channels = 1
            self._basic_info.num_extra_channels = 1 if samples == 2 else 0
        else:
            self._basic_info.num_color_channels = 3
            self._basic_info.num_extra_channels = 1 if samples == 4 else 0
        if samples in (2, 4):
            self._basic_info.alpha_bits = bps
            self._basic_info.alpha_exponent_bits = exp_bps
        if self._lossless:
            self._basic_info.uses_original_profile = JXL_TRUE
        if self._animation:
            self._basic_info.have_animation = JXL_TRUE
            self._basic_info.animation.tps_numerator = 10
            self._basic_info.animation.tps_denominator = 1
            self._basic_info.animation.num_loops = 0

        # If HDR transfer (PQ/HLG), force original-profile so the encoder
        # doesn't silently re-encode.
        if (self._color_spec is not None
                and getattr(self._color_spec, 'is_hdr', False)):
            self._basic_info.uses_original_profile = JXL_TRUE
            # A reasonable default intensity target for HDR
            if self._basic_info.intensity_target == 0:
                self._basic_info.intensity_target = 10000.0  # PQ peak

        # User-supplied intensity_target overrides the default. This is
        # the way to tag linear-light files as HDR (Apple EDR / scRGB
        # convention: file value 1.0 = SDR diffuse white, intensity_target
        # = nit level corresponding to peak encoded value, telling the OS
        # how much headroom the file extends past SDR).
        if self._intensity_target > 0.0:
            self._basic_info.intensity_target = self._intensity_target

        # pixel format
        self._pixel_format.num_channels = <uint32_t> samples
        self._pixel_format.endianness = JXL_NATIVE_ENDIAN
        self._pixel_format.align = 0
        self._pixel_format.data_type = jxl_dt

        memset(<void*> &self._bit_depth, 0, sizeof(JxlBitDepth))
        self._bit_depth.dtype = JXL_BIT_DEPTH_FROM_PIXEL_FORMAT
        self._bit_depth.bits_per_sample = bps
        self._bit_depth.exponent_bits_per_sample = exp_bps

        # Use container if HDR / requested
        cdef JXL_BOOL want_container = (
            JXL_TRUE if (self._container or self._animation) else JXL_FALSE
        )
        status = JxlEncoderUseContainer(self._encoder, want_container)
        if status != JXL_ENC_SUCCESS:
            _raise_enc('JxlEncoderUseContainer', status,
                       JxlEncoderGetError(self._encoder))

        status = JxlEncoderSetBasicInfo(self._encoder, &self._basic_info)
        if status != JXL_ENC_SUCCESS:
            _raise_enc('JxlEncoderSetBasicInfo', status,
                       JxlEncoderGetError(self._encoder))

        # Color encoding. If a raw ICC profile is supplied, embed it
        # directly via JxlEncoderSetICCProfile and skip the enum-based
        # SetColorEncoding (the two APIs are mutually exclusive).
        # Note: libjxl may canonicalize the ICC profile into a CICP-
        # equivalent enum on encode, so exotic profiles (e.g. Apple's
        # kCGColorSpaceExtendedLinearDisplayP3) lose information. For
        # standard profiles (sRGB, Display P3, BT.2020) the round-trip
        # is fine.
        cdef bint is_gray = (samples in (1, 2))
        cdef const uint8_t* icc_buf
        cdef size_t icc_len
        memset(<void*> &self._color_encoding, 0, sizeof(JxlColorEncoding))
        if self._icc_profile is not None:
            icc_buf = <const uint8_t*> (<bytes> self._icc_profile)
            icc_len = <size_t> len(self._icc_profile)
            status = JxlEncoderSetICCProfile(self._encoder, icc_buf, icc_len)
            if status != JXL_ENC_SUCCESS:
                _raise_enc('JxlEncoderSetICCProfile', status,
                           JxlEncoderGetError(self._encoder))
        else:
            if self._color_spec is None:
                if jxl_dt == JXL_TYPE_UINT8:
                    JxlColorEncodingSetToSRGB(
                        &self._color_encoding,
                        JXL_TRUE if is_gray else JXL_FALSE)
                else:
                    JxlColorEncodingSetToLinearSRGB(
                        &self._color_encoding,
                        JXL_TRUE if is_gray else JXL_FALSE)
            else:
                self._color_encoding.color_space = (
                    JXL_COLOR_SPACE_GRAY if is_gray else JXL_COLOR_SPACE_RGB)
                self._color_encoding.white_point = (
                    <JxlWhitePoint> self._color_spec.white_point)
                self._color_encoding.primaries = (
                    <JxlPrimaries> self._color_spec.primaries)
                self._color_encoding.transfer_function = (
                    <JxlTransferFunction> self._color_spec.transfer)
                self._color_encoding.rendering_intent = (
                    <JxlRenderingIntent> self._color_spec.rendering_intent)
                if self._color_spec.transfer == JXL_TF_GAMMA:
                    self._color_encoding.gamma = (
                        <double> self._color_spec.gamma)

            status = JxlEncoderSetColorEncoding(
                self._encoder, &self._color_encoding)
            if status != JXL_ENC_SUCCESS:
                _raise_enc('JxlEncoderSetColorEncoding', status,
                           JxlEncoderGetError(self._encoder))

        # Frame settings
        self._frame_settings = JxlEncoderFrameSettingsCreate(
            self._encoder, NULL)
        if self._frame_settings == NULL:
            raise JxlError('JxlEncoderFrameSettingsCreate returned NULL')

        if self._lossless:
            status = JxlEncoderSetFrameLossless(
                self._frame_settings, JXL_TRUE)
            if status != JXL_ENC_SUCCESS:
                _raise_enc('JxlEncoderSetFrameLossless', status,
                           JxlEncoderGetError(self._encoder))
        else:
            dist = self._distance
            status = JxlEncoderSetFrameDistance(self._frame_settings, dist)
            if status != JXL_ENC_SUCCESS:
                _raise_enc('JxlEncoderSetFrameDistance', status,
                           JxlEncoderGetError(self._encoder))

        if self._effort != 7:
            status = JxlEncoderFrameSettingsSetOption(
                self._frame_settings,
                JXL_ENC_FRAME_SETTING_EFFORT,
                <int64_t> self._effort,
            )
            if status != JXL_ENC_SUCCESS:
                _raise_enc(
                    'JxlEncoderFrameSettingsSetOption EFFORT', status,
                    JxlEncoderGetError(self._encoder))
        if self._decoding_speed != 0:
            status = JxlEncoderFrameSettingsSetOption(
                self._frame_settings,
                JXL_ENC_FRAME_SETTING_DECODING_SPEED,
                <int64_t> self._decoding_speed,
            )
            if status != JXL_ENC_SUCCESS:
                _raise_enc(
                    'JxlEncoderFrameSettingsSetOption DECODING_SPEED', status,
                    JxlEncoderGetError(self._encoder))

        # Only call SetFrameBitDepth when we want a non-default bit depth
        # (i.e., FROM_CODESTREAM with a custom bits_per_sample). The default
        # FROM_PIXEL_FORMAT path doesn't need a SetFrameBitDepth call —
        # libjxl uses pixel_format.data_type. imagecodecs has the same
        # conditional. Skipping it shaves a small amount of per-encode setup
        # time and matches the imagecodecs build's API call sequence.
        if self._bit_depth.dtype != JXL_BIT_DEPTH_FROM_PIXEL_FORMAT:
            status = JxlEncoderSetFrameBitDepth(
                self._frame_settings, &self._bit_depth)
            if status != JXL_ENC_SUCCESS:
                _raise_enc('JxlEncoderSetFrameBitDepth', status,
                           JxlEncoderGetError(self._encoder))

        # Right-size the output buffer for the first frame. imagecodecs
        # uses max(32KB, srcsize/4) for lossless, max(32KB, srcsize/16) for
        # lossy. This sizes the buffer so most encodes fit in the initial
        # allocation, avoiding 4-6 realloc-doublings that each memcpy the
        # partially-encoded stream (the cost scales with image size).
        cdef size_t initial_target
        if self._lossless:
            initial_target = self._frame_nbytes // 4
        else:
            initial_target = self._frame_nbytes // 16
        # Round up to 64 KiB
        initial_target = (initial_target + 65535) & ~<size_t> 65535
        if initial_target < 32768:
            initial_target = 32768
        # _outbuf is NULL until first frame; allocate exactly the target
        # size in one shot.
        self._outbuf = <uint8_t*> malloc(initial_target)
        if self._outbuf == NULL:
            raise MemoryError('jxl: failed to allocate output buffer')
        self._outbuf_capacity = initial_target

        self._config_done = True

    # ------------------------------------------------------------------ frame

    def write_frame(self, arr, *, is_last=False):
        """Encode a single frame and stream the resulting bytes to `dest`.

        After the call, all bytes that the encoder is willing to emit for
        frames seen so far have been written. If `is_last` is True (or this
        is the only frame in non-animation mode), CloseInput is called and
        the trailing bytes are flushed.

        Parameters
        ----------
        arr : ndarray
            Shape (Y, X), (Y, X, 1), (Y, X, 3), or (Y, X, 4) for L / RGB /
            RGBA. (Y, X, 2) is LA.
        is_last : bool
            If True, finalize the stream after writing this frame.
        """
        cdef:
            cnp.ndarray src
            JxlEncoderStatus status
            JxlEncoderStatus last_status = JXL_ENC_SUCCESS
            int rc
            size_t framesize_bytes
            bint close_input
            const void* pixels_ptr
            JxlFrameHeader frame_header

        if self._closed:
            raise RuntimeError('JxlWriter is closed')

        src = np.ascontiguousarray(arr)

        if not self._config_done:
            self._configure_from_first_frame(src)
        else:
            # Validate shape/dtype consistency
            if src.dtype != self._first_dtype:
                raise ValueError(
                    f'jxl: frame dtype {src.dtype!r} does not match first '
                    f'frame dtype {self._first_dtype!r}')
            if src.ndim == 2:
                if (<size_t> src.shape[0] != self._first_ysize
                        or <size_t> src.shape[1] != self._first_xsize
                        or self._first_samples != 1):
                    raise ValueError(
                        'jxl: frame shape does not match first frame')
            elif src.ndim == 3:
                if (<size_t> src.shape[0] != self._first_ysize
                        or <size_t> src.shape[1] != self._first_xsize
                        or <size_t> src.shape[2] != self._first_samples):
                    raise ValueError(
                        'jxl: frame shape does not match first frame')
            else:
                raise ValueError(f'jxl: invalid ndim {src.ndim}')

        if not self._animation and self._frame_count >= 1:
            raise RuntimeError(
                'jxl: writer is not in animation mode; '
                'set animation=True for multi-frame output')

        framesize_bytes = <size_t> src.nbytes
        pixels_ptr = <const void*> cnp.PyArray_DATA(src)
        close_input = is_last or not self._animation

        # Match imagecodecs: always Init+SetFrameHeader. duration=1 is a
        # no-op when have_animation=False but the call apparently triggers
        # libjxl internal state setup that's measurable for big frames.
        JxlEncoderInitFrameHeader(&frame_header)
        frame_header.duration = 1
        if self._animation:
            frame_header.is_last = JXL_TRUE if is_last else JXL_FALSE
        status = JxlEncoderSetFrameHeader(
            self._frame_settings, &frame_header)
        if status != JXL_ENC_SUCCESS:
            _raise_enc('JxlEncoderSetFrameHeader', status,
                       JxlEncoderGetError(self._encoder))

        # AddImageFrame + CloseInput + ProcessOutput drain in ONE nogil
        # block. Holding the GIL through these calls measurably slows the
        # encode of large images on Linux x86_64 (~15-30% wall time) even
        # though libjxl's worker threads are independent of Python's GIL.
        # Match imagecodecs's pattern.
        with nogil:
            rc = _add_frame_and_drain_nogil(
                self._encoder,
                self._frame_settings,
                &self._pixel_format,
                pixels_ptr,
                framesize_bytes,
                close_input,
                &self._outbuf,
                &self._outbuf_capacity,
                &self._outbuf_used,
                &last_status,
            )

        self._frame_count += 1

        if rc == _ENC_OK:
            return
        if rc == _ENC_ERR_REALLOC:
            raise MemoryError('jxl: failed to grow output buffer')
        if rc == _ENC_ERR_ADD_FRAME:
            _raise_enc('JxlEncoderAddImageFrame', last_status,
                       JxlEncoderGetError(self._encoder))
        if rc == _ENC_ERR_PROCESS:
            _raise_enc('JxlEncoderProcessOutput', last_status,
                       JxlEncoderGetError(self._encoder))
        raise JxlError(f'jxl: unknown encode error code {rc}')

    cdef _drain_all(self):
        """Drain encoder output into the growing _outbuf.

        Calls JxlEncoderProcessOutput in a tight nogil loop, doubling the
        buffer on NEED_MORE_OUTPUT. Bytes stay in our buffer until
        _flush_to_dest() is called from close(); this matches imagecodecs's
        single-buffer pattern and avoids the per-chunk thread-pool sync
        overhead that hurts at numthreads=8 for big files.
        """
        cdef:
            JxlEncoderStatus status
            uint8_t* next_out
            size_t avail_out
            size_t new_capacity
            uint8_t* new_buf

        while True:
            with nogil:
                next_out = self._outbuf + self._outbuf_used
                avail_out = self._outbuf_capacity - self._outbuf_used
                status = JxlEncoderProcessOutput(
                    self._encoder, &next_out, &avail_out)
                # avail_out now = remaining unused bytes in our buffer
                self._outbuf_used = self._outbuf_capacity - avail_out

            if status == JXL_ENC_SUCCESS:
                return
            if status == JXL_ENC_NEED_MORE_OUTPUT:
                # Double the buffer, capped at +32 MiB per growth.
                new_capacity = self._outbuf_capacity * 2
                if new_capacity > self._outbuf_capacity + 33554432:
                    new_capacity = self._outbuf_capacity + 33554432
                new_buf = <uint8_t*> realloc(
                    <void*> self._outbuf, new_capacity)
                if new_buf == NULL:
                    raise MemoryError('jxl: failed to grow output buffer')
                self._outbuf = new_buf
                self._outbuf_capacity = new_capacity
                continue
            _raise_enc('JxlEncoderProcessOutput', status,
                       JxlEncoderGetError(self._encoder))

    cdef _flush_to_dest(self):
        """Write the accumulated _outbuf to the destination, then reset it.

        Only relevant when dest is a path or file-like — in dest=None
        (in-memory) mode we skip this and let close() return bytes built
        directly from _outbuf in one memcpy. The BytesIO round-trip
        (outbuf -> bytes -> BytesIO -> getvalue) does 3 memcpys of the
        encoded payload; for a 50 MB encode that's ~5 ms of pure memcpy
        we'd otherwise eat.
        """
        if self._outbuf_used == 0:
            return
        if self._is_buffer:
            return  # close() will materialize bytes directly from _outbuf
        cdef bytes payload = (<char*> self._outbuf)[:self._outbuf_used]
        self._dest_obj.write(payload)
        self._outbuf_used = 0

    # ------------------------------------------------------------------ ctx

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            # On error, still release native resources.
            self._destroy_native()
        return False

    def close(self):
        """Finalize encoding and release resources.

        Returns the encoded bytes if dest was None (in-memory mode),
        otherwise returns None.
        """
        if self._closed:
            return None
        if self._encoder != NULL:
            # Make sure CloseInput was called and remaining output drained.
            JxlEncoderCloseInput(self._encoder)
            self._drain_all()
        result = None
        if self._is_buffer:
            # Single memcpy from _outbuf -> bytes. No BytesIO round-trip.
            if self._outbuf_used > 0:
                result = (<char*> self._outbuf)[:self._outbuf_used]
            else:
                result = b""
        else:
            self._flush_to_dest()
            if self._own_dest:
                try:
                    self._dest_obj.close()
                except Exception:
                    pass
        self._destroy_native()
        self._closed = True
        return result

    cdef _destroy_native(self):
        if self._encoder != NULL:
            JxlEncoderDestroy(self._encoder)
            self._encoder = NULL
            self._frame_settings = NULL
        if self._runner != NULL:
            JxlThreadParallelRunnerDestroy(self._runner)
            self._runner = NULL
        if self._outbuf != NULL:
            free(self._outbuf)
            self._outbuf = NULL
        self._outbuf_capacity = 0
        self._outbuf_used = 0

    def __dealloc__(self):
        if self._encoder != NULL:
            JxlEncoderDestroy(self._encoder)
            self._encoder = NULL
        if self._runner != NULL:
            JxlThreadParallelRunnerDestroy(self._runner)
            self._runner = NULL
        if self._outbuf != NULL:
            free(self._outbuf)
            self._outbuf = NULL


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def encode(arr, *, color=None, lossless=None, quality=None, distance=None,
           effort=5, decoding_speed=0, numthreads=None, animation=False,
           container=False, intensity_target=None, icc_profile=None,
           dest=None):
    """Encode a single ndarray (or animation stack) to JPEG XL bytes.

    For a single still image, pass a 2D / 3D array. For an animation stack
    pass a (T, Y, X[, C]) array and animation=True.

    `intensity_target` (nits) populates the JXL basic-info brightness anchor.
    Most useful for linear-light HDR: set to the peak nit level represented
    by the brightest encoded value so the decoder knows the file extends
    past SDR. Default (None / 0) leaves libjxl's standard fallback (255 for
    SDR transfer; 10000 for PQ/HLG).
    """
    cdef cnp.ndarray a = np.asarray(arr)

    if animation:
        if a.ndim < 3:
            raise ValueError('animation requires (T, Y, X[, C]) input')
        with JxlWriter(
            dest, color=color, lossless=lossless, quality=quality,
            distance=distance, effort=effort,
            decoding_speed=decoding_speed, numthreads=numthreads,
            animation=True, container=container,
            intensity_target=intensity_target,
            icc_profile=icc_profile,
        ) as w:
            for i in range(a.shape[0]):
                w.write_frame(
                    a[i],
                    is_last=(i == a.shape[0] - 1),
                )
            return w.close()

    with JxlWriter(
        dest, color=color, lossless=lossless, quality=quality,
        distance=distance, effort=effort, decoding_speed=decoding_speed,
        numthreads=numthreads, animation=False, container=container,
        intensity_target=intensity_target,
    ) as w:
        w.write_frame(a, is_last=True)
        return w.close()


def decode(data, *, numthreads=None, keep_orientation=False,
           coalesce=True, parse_color=False, streaming=False,
           index=None):
    """Decode a JPEG XL bytes/path into a numpy array (single ndarray).

    Defaults to ``parse_color=False`` for the fast decode path — matches
    imagecodecs.jpegxl_decode's deliberate non-subscription to
    JXL_DEC_COLOR_ENCODING. Pass ``parse_color=True`` to also populate
    color/icc info on the reader, at a ~2x decode-time cost on Linux.

    ``streaming=True`` enables the bg-thread chunked-read path for path /
    file-like inputs (off by default — see JxlReader docstring for when
    it pays off).

    ``index=N`` decodes only frame N of a multi-frame stream (libjxl
    skips past the earlier frames at bitstream-parse cost — much cheaper
    than pixel-decoding them). Combined with multiple threads pre-reading
    the bytes once, this is the substrate for parallel-multi-frame decode.
    """
    cdef int skip = int(index) if index is not None else 0
    with JxlReader(
        data, numthreads=numthreads,
        keep_orientation=keep_orientation, coalesce=coalesce,
        parse_color=parse_color, streaming=streaming,
        skip_frames=skip,
    ) as r:
        if index is None:
            return r.read()
        # Single-frame fetch
        for frame in r.iter_frames():
            return frame
        raise IndexError(
            f"jxl: frame {index} not found (stream too short?)")


def frame_count(data, *, numthreads=None):
    """Return the number of frames in a JPEG XL stream.

    Implementation: subscribe to JXL_DEC_FRAME events, drive the decoder
    through the bitstream, count fired events. Doesn't pixel-decode any
    frame, so cost is dominated by libjxl's bitstream parser. For a
    16-frame 1024x1024 u16 stack this is ~5-10 ms.
    """
    cdef:
        const uint8_t[::1] view
        const uint8_t* src_data
        size_t src_size
        JxlDecoder* dec = NULL
        JxlDecoderStatus status
        size_t count = 0
        bytes raw

    if isinstance(data, (str, os.PathLike)):
        with open(data, 'rb') as f:
            data = f.read()
    elif hasattr(data, 'read') and not isinstance(data, (bytes, bytearray, memoryview)):
        data = data.read()

    raw = bytes(data) if not isinstance(data, bytes) else data
    view = raw
    src_data = &view[0] if view.shape[0] > 0 else NULL
    src_size = <size_t> view.shape[0]
    if src_size == 0:
        return 0

    dec = JxlDecoderCreate(NULL)
    if dec == NULL:
        raise JxlError('JxlDecoderCreate returned NULL')
    try:
        status = JxlDecoderSubscribeEvents(dec, JXL_DEC_FRAME)
        if status != JXL_DEC_SUCCESS:
            _raise_dec('JxlDecoderSubscribeEvents', status)
        status = JxlDecoderSetInput(dec, src_data, src_size)
        if status != JXL_DEC_SUCCESS:
            _raise_dec('JxlDecoderSetInput', status)
        while True:
            with nogil:
                status = JxlDecoderProcessInput(dec)
            if status == JXL_DEC_FRAME:
                count += 1
                continue
            if status == JXL_DEC_SUCCESS:
                break
            if status == JXL_DEC_NEED_MORE_INPUT:
                raise JxlError('jxl: truncated stream while counting frames')
            if status == JXL_DEC_ERROR:
                _raise_dec('JxlDecoderProcessInput', status)
            # other events: ignore (BASIC_INFO etc may fire if subscribed
            # implicitly; we did NOT subscribe to them so this is unexpected)
    finally:
        JxlDecoderDestroy(dec)
    return int(count)
