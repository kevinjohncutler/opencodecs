# opencodecs/codecs/_jpeg2k.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native JPEG-2000 codec via OpenJPEG (memory streams, JP2 + raw J2K)."""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memcpy, strlen
from libc.math cimport log
from libc.stdint cimport (int8_t, int16_t, int32_t, uint8_t, uint16_t,
                          uint32_t)

import numpy as np
cimport numpy as cnp

from openjpeg cimport (
    opj_set_decode_area,
    opj_get_decoded_tile,
    OPJ_BOOL, OPJ_INT32, OPJ_UINT32, OPJ_SIZE_T, OPJ_OFF_T,
    CODEC_FORMAT, COLOR_SPACE,
    OPJ_CODEC_J2K, OPJ_CODEC_JP2,
    OPJ_CLRSPC_GRAY, OPJ_CLRSPC_SRGB, OPJ_CLRSPC_UNSPECIFIED,
    OPJ_CLRSPC_SYCC, OPJ_CLRSPC_EYCC, OPJ_CLRSPC_CMYK, OPJ_RPCL,
    opj_image_t, opj_image_create, opj_image_destroy, opj_image_cmptparm,
    opj_codec_t, opj_create_decompress, opj_create_compress,
    opj_destroy_codec, opj_set_default_decoder_parameters,
    opj_set_default_encoder_parameters, opj_setup_decoder,
    opj_setup_encoder,
    opj_dparameters_t, opj_cparameters_t,
    opj_stream_t, opj_stream_default_create, opj_stream_create, opj_stream_destroy,
    opj_stream_set_read_function, opj_stream_set_write_function,
    opj_stream_set_skip_function, opj_stream_set_seek_function,
    opj_stream_set_user_data, opj_stream_set_user_data_length,
    opj_read_header, opj_decode, opj_end_decompress,
    opj_start_compress, opj_encode, opj_end_compress,
    opj_has_thread_support, opj_get_num_cpus, opj_codec_set_threads,
    opj_set_info_handler, opj_set_warning_handler, opj_set_error_handler,
)

cnp.import_array()


class Jpeg2kError(RuntimeError):
    """Raised on JPEG-2000 encode/decode failures."""


# ----- Memory stream user-data + callbacks (no GIL) -----

cdef struct mem_buffer_read:
    const uint8_t* data
    OPJ_SIZE_T size
    OPJ_SIZE_T offset
    void* source


cdef struct mem_buffer_write:
    uint8_t* data
    OPJ_SIZE_T cap
    OPJ_SIZE_T size
    OPJ_SIZE_T offset
    void* destination


cdef OPJ_SIZE_T _read_cb(
    void* p_buffer, OPJ_SIZE_T p_nb_bytes, void* p_user_data
) noexcept nogil:
    cdef mem_buffer_read* buf = <mem_buffer_read*> p_user_data
    if buf.source != NULL:
        return _source_read_cb(p_buffer, p_nb_bytes, buf)
    cdef OPJ_SIZE_T remaining = buf.size - buf.offset
    if remaining == 0:
        return <OPJ_SIZE_T> -1
    cdef OPJ_SIZE_T n = p_nb_bytes if p_nb_bytes < remaining else remaining
    memcpy(p_buffer, buf.data + buf.offset, n)
    buf.offset += n
    return n


cdef OPJ_SIZE_T _source_read_cb(
    void* output, OPJ_SIZE_T size, mem_buffer_read* buf
) noexcept with gil:
    cdef const uint8_t[::1] data
    source = <object>buf.source
    try:
        if buf.offset >= buf.size:
            return <OPJ_SIZE_T>-1
        data = source.read_at(buf.offset, size)
        memcpy(output, &data[0], data.shape[0])
        buf.offset += data.shape[0]
        return data.shape[0]
    except BaseException as error:
        source.error = error
        return <OPJ_SIZE_T>-1


cdef OPJ_OFF_T _skip_read_cb(OPJ_OFF_T p_nb_bytes, void* p_user_data) noexcept nogil:
    cdef mem_buffer_read* buf = <mem_buffer_read*> p_user_data
    cdef OPJ_OFF_T target = <OPJ_OFF_T>buf.offset + p_nb_bytes
    cdef OPJ_OFF_T n
    if target < 0:
        return -1
    if <OPJ_SIZE_T>target > buf.size:
        target = <OPJ_OFF_T>buf.size
    n = target - <OPJ_OFF_T>buf.offset
    buf.offset = <OPJ_SIZE_T>target
    return n


cdef OPJ_BOOL _seek_read_cb(OPJ_OFF_T p_nb_bytes, void* p_user_data) noexcept nogil:
    cdef mem_buffer_read* buf = <mem_buffer_read*> p_user_data
    if <OPJ_SIZE_T> p_nb_bytes > buf.size:
        return 0
    buf.offset = <OPJ_SIZE_T> p_nb_bytes
    return 1


cdef OPJ_SIZE_T _write_cb(
    void* p_buffer, OPJ_SIZE_T p_nb_bytes, void* p_user_data
) noexcept nogil:
    cdef mem_buffer_write* buf = <mem_buffer_write*> p_user_data
    if buf.destination != NULL:
        return _destination_write_cb(p_buffer, p_nb_bytes, buf)
    cdef OPJ_SIZE_T new_cap
    cdef uint8_t* new_data
    if buf.offset + p_nb_bytes > buf.cap:
        new_cap = buf.cap * 2 if buf.cap else 65536
        while new_cap < buf.offset + p_nb_bytes:
            new_cap *= 2
        new_data = <uint8_t*> realloc(buf.data, new_cap)
        if new_data == NULL:
            return <OPJ_SIZE_T> -1
        buf.data = new_data
        buf.cap = new_cap
    memcpy(buf.data + buf.offset, p_buffer, p_nb_bytes)
    buf.offset += p_nb_bytes
    if buf.offset > buf.size:
        buf.size = buf.offset
    return p_nb_bytes


cdef OPJ_SIZE_T _destination_write_cb(
    void* data, OPJ_SIZE_T size, mem_buffer_write* buf
) noexcept with gil:
    destination = <object>buf.destination
    cdef OPJ_SIZE_T copied = 0
    cdef OPJ_SIZE_T count
    try:
        while copied < size:
            count = min(<OPJ_SIZE_T>65536, size - copied)
            destination.write_at(buf.offset, PyBytes_FromStringAndSize(<char*>data + copied, count))
            copied += count
            buf.offset += count
        if buf.offset > buf.size:
            buf.size = buf.offset
        return size
    except BaseException as error:
        destination.error = error
        return <OPJ_SIZE_T>-1


cdef OPJ_OFF_T _skip_write_cb(OPJ_OFF_T p_nb_bytes, void* p_user_data) noexcept nogil:
    cdef mem_buffer_write* buf = <mem_buffer_write*> p_user_data
    buf.offset += <OPJ_SIZE_T> p_nb_bytes
    if buf.offset > buf.size:
        buf.size = buf.offset
    return p_nb_bytes


cdef OPJ_BOOL _seek_write_cb(OPJ_OFF_T p_nb_bytes, void* p_user_data) noexcept nogil:
    cdef mem_buffer_write* buf = <mem_buffer_write*> p_user_data
    buf.offset = <OPJ_SIZE_T> p_nb_bytes
    return 1


# ----- OpenJPEG messages (verbose=) -----
#
# imagecodecs' verbose: above 0 OpenJPEG's errors are logged, above 1
# its warnings too, above 2 its info messages too. OpenJPEG can emit
# from its worker threads while the calling thread waits inside a
# native call, so the handlers only append to a C buffer (OpenJPEG
# serializes worker messages under its own mutex); the messages are
# logged, in order, once the call returns.

import logging as _logging

_logger = _logging.getLogger('opencodecs')


cdef struct msg_log:
    char* buf
    size_t size
    size_t cap


cdef void _msg_append(msg_log* log, char kind, const char* msg) noexcept nogil:
    cdef size_t n = strlen(msg)
    cdef size_t need = log.size + n + 2
    cdef size_t cap
    cdef char* grown
    if need > log.cap:
        cap = log.cap * 2 if log.cap * 2 > need else need + 256
        grown = <char*> realloc(log.buf, cap)
        if grown == NULL:
            return
        log.buf = grown
        log.cap = cap
    log.buf[log.size] = kind
    memcpy(log.buf + log.size + 1, msg, n)
    log.buf[log.size + 1 + n] = 0
    log.size += n + 2


cdef void _msg_error(const char* msg, void* client_data) noexcept nogil:
    _msg_append(<msg_log*> client_data, b'E', msg)


cdef void _msg_warning(const char* msg, void* client_data) noexcept nogil:
    _msg_append(<msg_log*> client_data, b'W', msg)


cdef void _msg_info(const char* msg, void* client_data) noexcept nogil:
    _msg_append(<msg_log*> client_data, b'I', msg)


def _verbosity(verbose):
    """imagecodecs' verbose as an int; None and False are 0."""
    return int(verbose) if verbose else 0


cdef void _msg_handlers(opj_codec_t* codec, msg_log* log, int v) noexcept:
    """Route OpenJPEG's messages for ``codec`` into ``log`` per verbose."""
    if v > 0:
        opj_set_error_handler(codec, _msg_error, <void*> log)
    if v > 1:
        opj_set_warning_handler(codec, _msg_warning, <void*> log)
    if v > 2:
        opj_set_info_handler(codec, _msg_info, <void*> log)


cdef void _msg_flush(msg_log* log):
    """Log the collected messages, as imagecodecs words them; free them."""
    cdef size_t pos = 0
    cdef size_t n
    cdef char kind
    try:
        while pos < log.size:
            kind = log.buf[pos]
            n = strlen(log.buf + pos + 1)
            text = (log.buf + pos + 1)[:n].decode('utf-8', 'replace')
            _logger.warning(
                'JPEG2K %s: %s',
                'error' if kind == b'E' else
                'warning' if kind == b'W' else 'info',
                text.strip())
            pos += n + 2
    finally:
        free(log.buf)
        log.buf = NULL
        log.size = 0
        log.cap = 0


# ----- Public API -----


def decode(data, *, numthreads: int | None = None, reduce: int = 0,
           out=None, planar=None, verbose=None) -> np.ndarray:
    """Decode JPEG-2000 (JP2 or J2K codestream) bytes to a numpy array.

    The dtype follows each component's precision and sign (Ssiz,
    ISO/IEC 15444-1 Annex A.5.1): uint8/uint16/uint32 for unsigned and
    int8/int16/int32 for signed components, holding the true sample
    values, as imagecodecs and tifffile return them.

    Parameters
    ----------
    planar : bool, optional
        ``True`` returns a multi-component image as ``(C, H, W)``;
        ``None`` or ``False`` as ``(H, W, C)``.
    verbose : int, optional
        As in imagecodecs: above 0 OpenJPEG's error messages are logged
        (``logging`` warnings on the ``"opencodecs"`` logger, worded
        ``"JPEG2K error: ..."``), above 1 its warnings too, above 2 its
        info messages too. ``None`` or 0 logs nothing.
    reduce : int, optional
        Skip this many of the finest wavelet resolutions, returning an
        image roughly ``2**reduce`` times smaller on each axis. This is
        not a resize of the decoded image: the discarded subbands are
        never entropy-decoded, so the work drops with the pixel count.
        It is the reason JPEG 2000 is used for large imagery, and it is
        what :class:`opencodecs.Jpeg2kPyramidReader` is built on.

        ``0`` (the default) decodes at full resolution. A value larger
        than the codestream's decomposition count raises Jpeg2kError
        rather than silently returning full resolution; use
        :func:`decode_info` to discover the usable range.
    numthreads : int, optional
        Worker threads for OpenJPEG's parallel decoder. ``None``
        defaults to ``opj_get_num_cpus() / 2`` (matches imagecodecs).
        ``0`` or ``1`` forces single-threaded. Typical 2-4× speedup on
        tiled / large-precinct JP2s.
    out : np.ndarray | None, optional
        Preallocated output ndarray. OpenJPEG decodes into its own
        opj_image_t component planes; out= just provides the
        destination of the final interleave/copy step. See
        ``_png.decode`` for the full contract.
    """
    cdef:
        const uint8_t[::1] src
        OPJ_SIZE_T srcsize
        opj_codec_t* codec = NULL
        opj_image_t* image = NULL
        opj_stream_t* stream = NULL
        opj_dparameters_t dparams
        mem_buffer_read rdbuf
        OPJ_BOOL ok
        int codec_format
        int _opj_n
        cnp.ndarray result
        msg_log mlog

    mlog.buf = NULL
    mlog.size = 0
    mlog.cap = 0
    cdef int vlevel = _verbosity(verbose)
    if reduce < 0:
        raise ValueError(f'reduce must be >= 0, got {reduce}')

    rdbuf.source = NULL
    if hasattr(data, "read_at"):
        data.reset()
        src = data.read_at(0, 12)
        srcsize = data.size
        rdbuf.source = <void*>data
    else:
        if isinstance(data, (bytes, bytearray)):
            src = data
        else:
            src = bytes(data)
        srcsize = <OPJ_SIZE_T>src.shape[0]
    if srcsize < 12:
        raise Jpeg2kError('input too short')

    # JP2 starts with 0x0000000C 'jP  '\r\n; raw J2K starts with 0xFF4FFF51.
    if (
        src[0] == 0xFF and src[1] == 0x4F and
        src[2] == 0xFF and src[3] == 0x51
    ):
        codec_format = OPJ_CODEC_J2K
    else:
        codec_format = OPJ_CODEC_JP2

    rdbuf.data = &src[0]
    rdbuf.size = srcsize
    rdbuf.offset = 0

    stream = (opj_stream_create(4096, 1) if rdbuf.source != NULL
              else opj_stream_default_create(1))  # is_input=1
    if stream == NULL:
        raise Jpeg2kError('opj_stream_default_create failed')
    try:
        opj_stream_set_user_data(stream, &rdbuf, NULL)
        opj_stream_set_user_data_length(stream, srcsize)
        opj_stream_set_read_function(stream, _read_cb)
        opj_stream_set_skip_function(stream, _skip_read_cb)
        opj_stream_set_seek_function(stream, _seek_read_cb)

        codec = opj_create_decompress(<CODEC_FORMAT> codec_format)
        if codec == NULL:
            raise Jpeg2kError('opj_create_decompress failed')
        _msg_handlers(codec, &mlog, vlevel)
        opj_set_default_decoder_parameters(&dparams)
        # cp_reduce has to be in place before setup_decoder: openjpeg
        # uses it while parsing the tile headers to decide which
        # subbands to skip, so setting it afterwards would decode
        # everything and then throw the fine detail away.
        dparams.cp_reduce = <OPJ_UINT32> reduce
        if not opj_setup_decoder(codec, &dparams):
            raise Jpeg2kError('opj_setup_decoder failed')

        # Enable multithreaded T1 decoding when supported. Match
        # imagecodecs's default: half the CPUs when numthreads is None.
        if opj_has_thread_support():
            if numthreads is None:
                _opj_n = opj_get_num_cpus() // 2
                if _opj_n < 1: _opj_n = 1
            else:
                _opj_n = int(numthreads)
            if _opj_n > 1:
                opj_codec_set_threads(codec, _opj_n)

        # Released for the whole entropy-decode. openjpeg touches
        # nothing Python here -- the stream callbacks it calls back
        # into are all `noexcept nogil` over a raw buffer -- and
        # holding the GIL through it makes every caller that decodes
        # frames on threads measure exactly 1.00x. DICOM is the one
        # that surfaced it: 24 independent J2K frames on 8 threads ran
        # at 0.94x serial, which is a thread pool paying overhead to
        # take turns.
        with nogil:
            ok = opj_read_header(stream, codec, &image)
        if not ok or image == NULL:
            raise Jpeg2kError('opj_read_header failed')

        with nogil:
            ok = opj_decode(codec, stream, image)
        if not ok:
            if reduce:
                raise Jpeg2kError(
                    f'opj_decode failed at reduce={reduce}; the '
                    f'codestream may have fewer decomposition levels '
                    f'than that (decode_info reports the usable range)')
            raise Jpeg2kError('opj_decode failed')
        with nogil:
            opj_end_decompress(codec, stream)

        result = _image_to_ndarray(image, out, bool(planar))
        return result
    finally:
        if image != NULL:
            opj_image_destroy(image)
        if codec != NULL:
            opj_destroy_codec(codec)
        opj_stream_destroy(stream)
        _msg_flush(&mlog)
        if rdbuf.source != NULL:
            data.raise_error()


def decode_region(data, y0: int, y1: int, x0: int, x1: int, *,
                  reduce: int = 0, numthreads: int | None = None):
    """Decode the rectangle ``[y0:y1, x0:x1]`` and nothing else.

    JPEG 2000 stores tiles and resolution levels precisely so a viewer
    can take a window out of a large image without expanding it, and
    ``opj_set_decode_area`` is how openjpeg exposes that. Combined with
    ``reduce`` this is the pair a tile server wants: the right region
    at the right zoom.

    Coordinates are in the FULL-resolution image, matching how
    ``read_region`` works elsewhere in this package, and stay that way
    when ``reduce`` is non-zero: openjpeg applies the reduction to
    them itself, so a 512-wide window at ``reduce=1`` returns 256
    pixels. A caller picks a window once and changes only the zoom.
    """
    cdef:
        const uint8_t[::1] src
        OPJ_SIZE_T srcsize
        opj_codec_t* codec = NULL
        opj_image_t* image = NULL
        opj_stream_t* stream = NULL
        opj_dparameters_t dparams
        mem_buffer_read rdbuf
        int codec_format
        int _opj_n
        OPJ_BOOL ok
        OPJ_INT32 rx0, ry0, rx1, ry1

    if reduce < 0:
        raise ValueError(f'reduce must be >= 0, got {reduce}')
    if y1 <= y0 or x1 <= x0:
        raise ValueError(
            f'empty region: y[{y0}:{y1}] x[{x0}:{x1}]')

    rdbuf.source = NULL
    if hasattr(data, "read_at"):
        data.reset()
        src = data.read_at(0, 12)
        srcsize = data.size
        rdbuf.source = <void*>data
    else:
        if isinstance(data, (bytes, bytearray)):
            src = data
        else:
            src = bytes(data)
        srcsize = <OPJ_SIZE_T>src.shape[0]
    if srcsize < 12:
        raise Jpeg2kError('input too short')

    if (src[0] == 0xFF and src[1] == 0x4F and
            src[2] == 0xFF and src[3] == 0x51):
        codec_format = OPJ_CODEC_J2K
    else:
        codec_format = OPJ_CODEC_JP2

    rdbuf.data = &src[0]
    rdbuf.size = srcsize
    rdbuf.offset = 0

    stream = (opj_stream_create(4096, 1) if rdbuf.source != NULL
              else opj_stream_default_create(1))
    if stream == NULL:
        raise Jpeg2kError('opj_stream_default_create failed')
    try:
        opj_stream_set_user_data(stream, &rdbuf, NULL)
        opj_stream_set_user_data_length(stream, srcsize)
        opj_stream_set_read_function(stream, _read_cb)
        opj_stream_set_skip_function(stream, _skip_read_cb)
        opj_stream_set_seek_function(stream, _seek_read_cb)

        codec = opj_create_decompress(<CODEC_FORMAT> codec_format)
        if codec == NULL:
            raise Jpeg2kError('opj_create_decompress failed')
        opj_set_default_decoder_parameters(&dparams)
        dparams.cp_reduce = <OPJ_UINT32> reduce
        if not opj_setup_decoder(codec, &dparams):
            raise Jpeg2kError('opj_setup_decoder failed')

        if opj_has_thread_support():
            if numthreads is None:
                _opj_n = opj_get_num_cpus() // 2
                if _opj_n < 1: _opj_n = 1
            else:
                _opj_n = int(numthreads)
            if _opj_n > 1:
                opj_codec_set_threads(codec, _opj_n)

        with nogil:
            ok = opj_read_header(stream, codec, &image)
        if not ok or image == NULL:
            raise Jpeg2kError('opj_read_header failed')

        # set_decode_area takes coordinates on the full-resolution
        # reference grid -- openjpeg's own words are "in image
        # coordinates" -- and applies cp_reduce to them itself. An
        # earlier version shifted them first, which halved the region
        # twice: asking for a 512-wide window at reduce=1 returned 128
        # pixels instead of 256. Pass them through.
        rx0 = <OPJ_INT32> x0
        ry0 = <OPJ_INT32> y0
        rx1 = <OPJ_INT32> x1
        ry1 = <OPJ_INT32> y1

        if not opj_set_decode_area(codec, image, rx0, ry0, rx1, ry1):
            raise Jpeg2kError(
                f'opj_set_decode_area failed for y[{y0}:{y1}] '
                f'x[{x0}:{x1}] at reduce={reduce}')
        with nogil:
            ok = opj_decode(codec, stream, image)
            if ok:
                opj_end_decompress(codec, stream)
        if not ok:
            raise Jpeg2kError('opj_decode failed for the requested region')
        return _image_to_ndarray(image, None)
    finally:
        if image != NULL:
            opj_image_destroy(image)
        if codec != NULL:
            opj_destroy_codec(codec)
        opj_stream_destroy(stream)
        if rdbuf.source != NULL:
            data.raise_error()


def decode_tile(data, tile_index: int, *, numthreads: int | None = None):
    """Decode one tile of a tiled codestream, by index.

    The other half of the same capability: ``decode_region`` is for a
    caller with a viewport, this is for one walking the tile grid.
    Tiles are numbered in raster order.
    """
    cdef:
        const uint8_t[::1] src
        OPJ_SIZE_T srcsize
        opj_codec_t* codec = NULL
        opj_image_t* image = NULL
        opj_stream_t* stream = NULL
        opj_dparameters_t dparams
        mem_buffer_read rdbuf
        int codec_format
        int _opj_n
        OPJ_BOOL ok
        OPJ_UINT32 _tile

    if tile_index < 0:
        raise ValueError(f'tile_index must be >= 0, got {tile_index}')

    rdbuf.source = NULL
    if hasattr(data, "read_at"):
        data.reset()
        src = data.read_at(0, 12)
        srcsize = data.size
        rdbuf.source = <void*>data
    else:
        if isinstance(data, (bytes, bytearray)):
            src = data
        else:
            src = bytes(data)
        srcsize = <OPJ_SIZE_T>src.shape[0]
    if srcsize < 12:
        raise Jpeg2kError('input too short')

    if (src[0] == 0xFF and src[1] == 0x4F and
            src[2] == 0xFF and src[3] == 0x51):
        codec_format = OPJ_CODEC_J2K
    else:
        codec_format = OPJ_CODEC_JP2

    rdbuf.data = &src[0]
    rdbuf.size = srcsize
    rdbuf.offset = 0

    stream = (opj_stream_create(4096, 1) if rdbuf.source != NULL
              else opj_stream_default_create(1))
    if stream == NULL:
        raise Jpeg2kError('opj_stream_default_create failed')
    try:
        opj_stream_set_user_data(stream, &rdbuf, NULL)
        opj_stream_set_user_data_length(stream, srcsize)
        opj_stream_set_read_function(stream, _read_cb)
        opj_stream_set_skip_function(stream, _skip_read_cb)
        opj_stream_set_seek_function(stream, _seek_read_cb)

        codec = opj_create_decompress(<CODEC_FORMAT> codec_format)
        if codec == NULL:
            raise Jpeg2kError('opj_create_decompress failed')
        opj_set_default_decoder_parameters(&dparams)
        if not opj_setup_decoder(codec, &dparams):
            raise Jpeg2kError('opj_setup_decoder failed')

        if opj_has_thread_support():
            if numthreads is None:
                _opj_n = opj_get_num_cpus() // 2
                if _opj_n < 1: _opj_n = 1
            else:
                _opj_n = int(numthreads)
            if _opj_n > 1:
                opj_codec_set_threads(codec, _opj_n)

        with nogil:
            ok = opj_read_header(stream, codec, &image)
        if not ok or image == NULL:
            raise Jpeg2kError('opj_read_header failed')
        # Coerced out here: inside nogil, casting a Python int is
        # "Coercion from Python not allowed without the GIL".
        _tile = <OPJ_UINT32> tile_index
        with nogil:
            ok = opj_get_decoded_tile(codec, stream, image, _tile)
        if not ok:
            raise Jpeg2kError(
                f'opj_get_decoded_tile({tile_index}) failed; the '
                f'codestream may have fewer tiles than that')
        return _image_to_ndarray(image, None)
    finally:
        if image != NULL:
            opj_image_destroy(image)
        if codec != NULL:
            opj_destroy_codec(codec)
        opj_stream_destroy(stream)
        if rdbuf.source != NULL:
            data.raise_error()


def decode_info(data, *, reduce: int = 0) -> dict:
    """Report an image's shape at a given reduction without decoding it.

    Reads the codestream headers only, which for JPEG 2000 is enough to
    know the reduced geometry: ``cp_reduce`` is applied while the
    headers are parsed, so openjpeg reports the dimensions the decode
    would produce. That makes enumerating a pyramid's levels cheap --
    no entropy decoding happens here.

    Returns a dict with ``shape``, ``dtype``, ``numcomps``,
    ``precision`` and ``signed``. Raises :class:`Jpeg2kError` when ``reduce`` exceeds
    what the codestream carries.
    """
    cdef:
        const uint8_t[::1] src
        OPJ_SIZE_T srcsize
        opj_codec_t* codec = NULL
        opj_image_t* image = NULL
        opj_stream_t* stream = NULL
        opj_dparameters_t dparams
        mem_buffer_read rdbuf
        int codec_format
        OPJ_UINT32 w, h, prec, numcomps, sgnd

    if reduce < 0:
        raise ValueError(f'reduce must be >= 0, got {reduce}')

    rdbuf.source = NULL
    if hasattr(data, "read_at"):
        data.reset()
        src = data.read_at(0, 12)
        srcsize = data.size
        rdbuf.source = <void*>data
    else:
        if isinstance(data, (bytes, bytearray)):
            src = data
        else:
            src = bytes(data)
        srcsize = <OPJ_SIZE_T>src.shape[0]
    if srcsize < 12:
        raise Jpeg2kError('input too short')

    if (
        src[0] == 0xFF and src[1] == 0x4F and
        src[2] == 0xFF and src[3] == 0x51
    ):
        codec_format = OPJ_CODEC_J2K
    else:
        codec_format = OPJ_CODEC_JP2

    rdbuf.data = &src[0]
    rdbuf.size = srcsize
    rdbuf.offset = 0

    stream = (opj_stream_create(4096, 1) if rdbuf.source != NULL
              else opj_stream_default_create(1))
    if stream == NULL:
        raise Jpeg2kError('opj_stream_default_create failed')
    try:
        opj_stream_set_user_data(stream, &rdbuf, NULL)
        opj_stream_set_user_data_length(stream, srcsize)
        opj_stream_set_read_function(stream, _read_cb)
        opj_stream_set_skip_function(stream, _skip_read_cb)
        opj_stream_set_seek_function(stream, _seek_read_cb)

        codec = opj_create_decompress(<CODEC_FORMAT> codec_format)
        if codec == NULL:
            raise Jpeg2kError('opj_create_decompress failed')
        opj_set_default_decoder_parameters(&dparams)
        dparams.cp_reduce = <OPJ_UINT32> reduce
        if not opj_setup_decoder(codec, &dparams):
            raise Jpeg2kError('opj_setup_decoder failed')

        if not opj_read_header(stream, codec, &image) or image == NULL:
            raise Jpeg2kError(
                f'opj_read_header failed at reduce={reduce}')
        if image.numcomps == 0:
            raise Jpeg2kError('image has 0 components')

        numcomps = image.numcomps
        w = image.comps[0].w
        h = image.comps[0].h
        prec = image.comps[0].prec
        sgnd = image.comps[0].sgnd
        if w == 0 or h == 0:
            raise Jpeg2kError(
                f'reduce={reduce} leaves a zero-sized image')

        return {
            'shape': (int(h), int(w)) if numcomps == 1
                     else (int(h), int(w), int(numcomps)),
            'dtype': _component_dtype(prec, sgnd).type,
            'signed': bool(sgnd),
            'numcomps': int(numcomps),
            'precision': int(prec),
        }
    finally:
        if image != NULL:
            opj_image_destroy(image)
        if codec != NULL:
            opj_destroy_codec(codec)
        opj_stream_destroy(stream)
        if rdbuf.source != NULL:
            data.raise_error()


ctypedef fused _sample_t:
    uint8_t
    int8_t
    uint16_t
    int16_t
    uint32_t
    int32_t


cdef void _copy_components(
    _sample_t* dst, opj_image_t* image, size_t n_pixels, size_t numcomps,
    bint planar, bint clamp, long long lo, long long hi,
) noexcept nogil:
    """Copy openjpeg's int32 component planes into dst, clamped to the
    component's nominal range (ISO/IEC 15444-1 Annex G.1.2: clipping is
    the usual treatment of quantization overshoot; OpenJPEG clips too).
    """
    cdef size_t c, i, base, step
    cdef long long v
    cdef OPJ_INT32* comp_data
    for c in range(numcomps):
        comp_data = image.comps[c].data
        if planar:
            base = c * n_pixels
            step = 1
        else:
            base = c
            step = numcomps
        if clamp:
            for i in range(n_pixels):
                v = comp_data[i]
                if v < lo:
                    v = lo
                elif v > hi:
                    v = hi
                dst[base + i * step] = <_sample_t> v
        else:
            for i in range(n_pixels):
                dst[base + i * step] = <_sample_t> comp_data[i]


cdef object _component_dtype(OPJ_UINT32 prec, OPJ_UINT32 sgnd):
    """dtype for a component of `prec` bits, signed when Ssiz says so.

    ISO/IEC 15444-1 Annex A.5.1: bit 7 of Ssiz marks a signed component,
    and Annex G.1 applies the DC level shift only to unsigned ones, so a
    signed component decodes to signed values. imagecodecs and tifffile
    return int8/int16/int32 for these, and so does opencodecs' HTJ2K.
    """
    if prec < 1 or prec > 32:
        raise Jpeg2kError(f'unsupported precision {prec} bits')
    itemsize = 1 if prec <= 8 else (2 if prec <= 16 else 4)
    return np.dtype(f"{'i' if sgnd else 'u'}{itemsize}")


cdef cnp.ndarray _image_to_ndarray(opj_image_t* image, object out,
                                    bint planar=False):
    """Copy openjpeg image planes into a numpy array.

    ``planar`` returns ``(C, H, W)`` instead of ``(H, W, C)``. ``out``
    is the caller's preallocated ndarray (or None); see the public
    ``decode`` for the full out= contract.
    """
    cdef:
        OPJ_UINT32 numcomps = image.numcomps
        OPJ_UINT32 width
        OPJ_UINT32 height
        OPJ_UINT32 prec
        OPJ_UINT32 sgnd
        OPJ_UINT32 c
        size_t n_pixels
        cnp.ndarray out_arr
        void* dst
        bint clamp
        long long lo, hi
        bint as_planar
        bint is_int
        int size

    if numcomps == 0:
        raise Jpeg2kError('image has 0 components')

    width = image.comps[0].w
    height = image.comps[0].h
    prec = image.comps[0].prec
    sgnd = image.comps[0].sgnd

    for c in range(numcomps):
        if image.comps[c].w != width or image.comps[c].h != height:
            raise Jpeg2kError(
                'JPEG-2000: components have differing sizes (subsampled?); '
                'not supported')
        if image.comps[c].prec != prec:
            raise Jpeg2kError(
                'JPEG-2000: components have differing precision; '
                'not supported')
        if image.comps[c].sgnd != sgnd:
            raise Jpeg2kError(
                'JPEG-2000: components differ in signedness; not supported')

    dtype = _component_dtype(prec, sgnd)
    as_planar = planar and numcomps > 1
    if numcomps == 1:
        expected_shape = (int(height), int(width))
    elif as_planar:
        expected_shape = (int(numcomps), int(height), int(width))
    else:
        expected_shape = (int(height), int(width), int(numcomps))

    if out is not None:
        if not isinstance(out, np.ndarray):
            raise Jpeg2kError(
                f"jpeg2k decode: out= must be an ndarray, "
                f"got {type(out).__name__}")
        if out.shape != expected_shape:
            raise Jpeg2kError(
                f"jpeg2k decode: out= shape {out.shape} does not match "
                f"expected {expected_shape}")
        if out.dtype != dtype:
            raise Jpeg2kError(
                f"jpeg2k decode: out= dtype {out.dtype} does not match "
                f"expected {dtype}")
        if not out.flags['C_CONTIGUOUS']:
            raise Jpeg2kError("jpeg2k decode: out= must be C-contiguous")
        out_arr = out
    else:
        out_arr = np.empty(expected_shape, dtype=dtype)

    n_pixels = <size_t> width * height
    if sgnd:
        lo = -((<long long> 1) << (prec - 1))
        hi = ((<long long> 1) << (prec - 1)) - 1
    else:
        lo = 0
        hi = ((<long long> 1) << prec) - 1
    # openjpeg holds samples as int32, so a full 32-bit unsigned
    # component is only meaningful as a bit pattern.
    clamp = prec < 32
    dst = cnp.PyArray_DATA(out_arr)
    is_int = dtype.kind == 'i'
    size = dtype.itemsize
    with nogil:
        if size == 1:
            if is_int:
                _copy_components(<int8_t*> dst, image, n_pixels, numcomps,
                                 as_planar, clamp, lo, hi)
            else:
                _copy_components(<uint8_t*> dst, image, n_pixels, numcomps,
                                 as_planar, clamp, lo, hi)
        elif size == 2:
            if is_int:
                _copy_components(<int16_t*> dst, image, n_pixels, numcomps,
                                 as_planar, clamp, lo, hi)
            else:
                _copy_components(<uint16_t*> dst, image, n_pixels, numcomps,
                                 as_planar, clamp, lo, hi)
        else:
            if is_int:
                _copy_components(<int32_t*> dst, image, n_pixels, numcomps,
                                 as_planar, clamp, lo, hi)
            else:
                _copy_components(<uint32_t*> dst, image, n_pixels, numcomps,
                                 as_planar, clamp, lo, hi)
    return out_arr


_COLORSPACES = {
    'UNSPECIFIED': OPJ_CLRSPC_UNSPECIFIED,
    'UNKNOWN': OPJ_CLRSPC_UNSPECIFIED,
    'SRGB': OPJ_CLRSPC_SRGB,
    'RGB': OPJ_CLRSPC_SRGB,
    'RGBA': OPJ_CLRSPC_SRGB,
    'GRAY': OPJ_CLRSPC_GRAY,
    'GRAYSCALE': OPJ_CLRSPC_GRAY,
    'MINISBLACK': OPJ_CLRSPC_GRAY,
    'MINISWHITE': OPJ_CLRSPC_GRAY,
    'SYCC': OPJ_CLRSPC_SYCC,
    'EYCC': OPJ_CLRSPC_EYCC,
    'CMYK': OPJ_CLRSPC_CMYK,
}


def _colorspace(colorspace, samples):
    """imagecodecs' colorspace rule: gray up to 2 samples, sRGB up to 4."""
    if colorspace is None:
        if samples <= 2:
            return OPJ_CLRSPC_GRAY
        if samples <= 4:
            return OPJ_CLRSPC_SRGB
        return OPJ_CLRSPC_UNSPECIFIED
    if isinstance(colorspace, str):
        try:
            return _COLORSPACES[colorspace.upper()]
        except KeyError:
            raise ValueError(
                f'unknown colorspace {colorspace!r}; expected one of '
                f'{sorted(_COLORSPACES)}') from None
    value = int(colorspace)
    if value not in (OPJ_CLRSPC_UNSPECIFIED, OPJ_CLRSPC_SRGB, OPJ_CLRSPC_GRAY,
                     OPJ_CLRSPC_SYCC, OPJ_CLRSPC_EYCC, OPJ_CLRSPC_CMYK):
        raise ValueError(f'unknown colorspace {colorspace!r}')
    return value


def _codec_format(codec, codecformat):
    """'jp2' or 'j2k' from opencodecs' codec= or imagecodecs' codecformat=."""
    names = {'jp2': OPJ_CODEC_JP2, 'j2k': OPJ_CODEC_J2K}
    chosen = []
    for value in (codec, codecformat):
        if value is None:
            continue
        if isinstance(value, str):
            key = value.lower()
            if key not in names:
                raise Jpeg2kError(
                    f"codec must be 'jp2' or 'j2k', got {value!r}")
            chosen.append(names[key])
        elif int(value) in (OPJ_CODEC_JP2, OPJ_CODEC_J2K):
            chosen.append(int(value))
        else:
            raise Jpeg2kError(f"codec must be 'jp2' or 'j2k', got {value!r}")
    if len(chosen) == 2 and chosen[0] != chosen[1]:
        raise ValueError(
            f'codec={codec!r} and codecformat={codecformat!r} disagree')
    return chosen[0] if chosen else OPJ_CODEC_JP2


def _numresolution(height, width, resolutions):
    """imagecodecs' resolution count: log2 of the smaller side minus 2,
    between 1 and `resolutions` (default 6).

    OpenJPEG refuses a tile side shorter than 2**(numresolution - 1),
    which is why a fixed 6 failed on every image under 32 pixels.
    """
    cdef int limit = 6 if resolutions is None else int(resolutions)
    limit = min(max(limit, 1), 33)
    cdef int n = <int> (log(<double> min(height, width)) / log(2.0) - 2)
    return min(max(n, 1), limit)


def encode(data, level=None, *, lossless=None, codec=None,
           codecformat=None, colorspace=None, planar=None, tile=None,
           bitspersample=None, resolutions=None, reversible=None,
           mct=True, ratio=None, verbose=None,
           numthreads: int | None = None, destination=None) -> bytes:
    """Encode a numpy array as JPEG-2000 (JP2 by default; ``codec='j2k'``
    for the raw codestream).

    The keywords and their meanings are ``imagecodecs.jpeg2k_encode``'s,
    plus ``lossless`` and ``ratio``.

    Parameters
    ----------
    data
        ``(H, W)``, ``(H, W, C)``, or with ``planar=True`` ``(C, H, W)``;
        uint8, int8, uint16 or int16. Signed arrays are written as
        signed components (Ssiz bit 7).
    level : float, optional
        Target quality as a PSNR in dB, from 1 to 1000, on a single
        quality layer (OpenJPEG's fixed-quality mode). ``None``, values
        below 1 and values above 1000 mean lossless, as in imagecodecs.
    lossless : bool, optional
        ``None`` (default) follows ``level``, ``ratio`` and
        ``reversible``: lossless unless one of them asks for loss.
        ``True`` refuses any of them that would lose data. ``False``
        with neither ``level`` nor ``ratio`` uses a 10:1 rate.
    ratio : float, optional
        Target compression ratio (OpenJPEG's ``tcp_rates``, what
        ``opj_compress -r`` takes) instead of a PSNR target; lossy.
        Before 0.4.1 opencodecs read ``level`` as ``100 / ratio``.
    codec, codecformat : {"jp2", "j2k"}
        Container. ``jp2`` is the boxed format; ``j2k`` is the raw
        codestream that DICOM transfer syntaxes use.
    colorspace : str or int, optional
        JP2 color space. ``None`` picks gray for 1-2 samples and sRGB
        for 3-4.
    planar : bool, optional
        ``True`` reads ``data`` as ``(C, H, W)``. ``None`` does so only
        when the last axis is longer than 4 and the first is 4 or
        shorter, imagecodecs' rule.
    bitspersample : int, optional
        Component precision, e.g. 12 for 12-bit data in uint16: 1 to 8
        for (u)int8 data and 9 to 16 for (u)int16 data, the bands in
        which imagecodecs uses it. Other values raise ValueError (where
        imagecodecs ignores them), as does a sample that does not fit.
    resolutions : int, optional
        Upper bound on the resolution count (default 6). The count is
        also capped by the image size, so small images encode.
    reversible : bool, optional
        5/3 (reversible) or 9/7 (irreversible) wavelet. ``None`` picks
        5/3 for lossless and 9/7 for lossy.
    mct : bool
        Component transform (RCT/ICT) for 3-sample sRGB images.
    tile
        Not implemented (as in imagecodecs); raises.
    verbose : int, optional
        As in :func:`decode`: OpenJPEG's errors (above 0), warnings
        (above 1) and info messages (above 2) go to the
        ``"opencodecs"`` logger.
    numthreads : int, optional
        Worker threads for OpenJPEG's parallel T1 encoder. ``None``
        defaults to ``opj_get_num_cpus() / 2``.
    """
    cdef:
        cnp.ndarray arr
        opj_image_cmptparm* cmptparms = NULL
        opj_image_t* image = NULL
        opj_codec_t* opj_codec = NULL
        opj_stream_t* stream = NULL
        opj_cparameters_t cparams
        mem_buffer_write wrbuf
        OPJ_UINT32 width, height, numcomps, prec, sgnd
        OPJ_UINT32 c
        size_t i, n_pixels, step, base
        OPJ_INT32* comp_data
        bytes out
        int cf
        int _opj_n
        int color_space
        int numres
        float quality = 0.0
        bint is_planar
        msg_log mlog

    if tile:
        raise NotImplementedError('jpeg2k encode: writing tiles is not '
                                  'implemented')
    cdef int vlevel = _verbosity(verbose)

    arr = np.ascontiguousarray(data)
    if arr.dtype.kind not in 'ui' or arr.dtype.itemsize not in (1, 2):
        raise Jpeg2kError(f'unsupported dtype {arr.dtype}')
    sgnd = 1 if arr.dtype.kind == 'i' else 0
    prec = 8 * arr.dtype.itemsize

    if arr.ndim == 2:
        is_planar = True
        numcomps = 1
        height = <OPJ_UINT32> arr.shape[0]
        width = <OPJ_UINT32> arr.shape[1]
    elif arr.ndim == 3:
        if planar is None:
            planar = arr.shape[2] > 4 and arr.shape[0] <= 4
        is_planar = bool(planar)
        if is_planar:
            numcomps = <OPJ_UINT32> arr.shape[0]
            height = <OPJ_UINT32> arr.shape[1]
            width = <OPJ_UINT32> arr.shape[2]
        else:
            height = <OPJ_UINT32> arr.shape[0]
            width = <OPJ_UINT32> arr.shape[1]
            numcomps = <OPJ_UINT32> arr.shape[2]
    else:
        raise Jpeg2kError(f'unsupported ndim {arr.ndim}')
    if numcomps < 1 or numcomps > 4095:
        raise Jpeg2kError(f'unsupported channel count {numcomps}')
    if height < 1 or width < 1:
        raise Jpeg2kError(f'empty image {(<object> arr).shape}')

    if bitspersample is not None:
        bps = int(bitspersample)
        # imagecodecs' bands: 1 to 8 bits for 8-bit data, 9 to 16 for
        # 16-bit data. imagecodecs silently writes the full width for a
        # value outside the band; here it raises, since a uint16 array
        # written as an 8-bit codestream would come back as uint8.
        lo_bps = 1 if prec == 8 else 9
        if bps < lo_bps or bps > prec:
            raise ValueError(
                f'bitspersample={bitspersample} is outside {lo_bps} to '
                f'{prec}, the range for {arr.dtype} data')
        if arr.size:
            lo_ok = -(1 << (bps - 1)) if sgnd else 0
            hi_ok = (1 << (bps - 1)) - 1 if sgnd else (1 << bps) - 1
            if int(arr.min()) < lo_ok or int(arr.max()) > hi_ok:
                raise ValueError(
                    f'bitspersample={bitspersample}: samples outside '
                    f'[{lo_ok}, {hi_ok}] would be lost')
        prec = <OPJ_UINT32> bps

    cf = _codec_format(codec, codecformat)
    color_space = _colorspace(colorspace, numcomps)

    # What `level`, `ratio`, `reversible` and `lossless` ask for.
    if level is not None and float(level) > 0:
        quality = <float> float(level)
    lossy_level = 1.0 <= quality <= 1000.0
    if not lossy_level:
        quality = 0.0
    if ratio is not None:
        ratio = float(ratio)
        if not ratio >= 1.0:
            raise ValueError(f'ratio must be >= 1, got {ratio}')
        if lossy_level:
            raise ValueError(
                'pass level= (a PSNR target in dB) or ratio= (a '
                'compression ratio), not both')
    if lossless is None:
        lossless = not (lossy_level or ratio is not None
                        or reversible is False)
    elif lossless:
        if lossy_level or ratio is not None:
            raise ValueError(
                f'lossless=True contradicts level={level!r} '
                f'ratio={ratio!r}; level from 1 to 1000 is a lossy PSNR '
                f'target in dB')
        if reversible is False:
            raise ValueError(
                'lossless=True needs the reversible 5/3 wavelet; '
                'drop reversible=False')
    else:
        if level is not None and not lossy_level:
            raise ValueError(
                f'lossless=False: level={level!r} is not a PSNR target '
                f'between 1 and 1000 dB')
        if not lossy_level and ratio is None:
            ratio = 10.0

    n_pixels = <size_t> width * height
    cmptparms = <opj_image_cmptparm*> malloc(
        numcomps * sizeof(opj_image_cmptparm))
    if cmptparms == NULL:
        raise MemoryError('failed to allocate cmptparms')
    try:
        for c in range(numcomps):
            cmptparms[c].dx = 1
            cmptparms[c].dy = 1
            cmptparms[c].w = width
            cmptparms[c].h = height
            cmptparms[c].x0 = 0
            cmptparms[c].y0 = 0
            cmptparms[c].prec = prec
            cmptparms[c].bpp = prec
            cmptparms[c].sgnd = sgnd
        image = opj_image_create(numcomps, cmptparms,
                                 <COLOR_SPACE> color_space)
    finally:
        free(cmptparms)
    if image == NULL:
        raise Jpeg2kError('opj_image_create failed')
    image.x0 = 0
    image.y0 = 0
    image.x1 = width
    image.y1 = height

    # Samples go in as their plain values; OpenJPEG applies the DC
    # level shift itself from each component's sgnd.
    cdef uint8_t* in_u8 = <uint8_t*> cnp.PyArray_DATA(arr)
    cdef int8_t* in_i8 = <int8_t*> cnp.PyArray_DATA(arr)
    cdef uint16_t* in_u16 = <uint16_t*> cnp.PyArray_DATA(arr)
    cdef int16_t* in_i16 = <int16_t*> cnp.PyArray_DATA(arr)
    cdef int kind_size = arr.dtype.itemsize * 2 + sgnd
    step = 1 if is_planar else numcomps
    with nogil:
        for c in range(numcomps):
            comp_data = image.comps[c].data
            base = c * n_pixels if is_planar else c
            if kind_size == 2:
                for i in range(n_pixels):
                    comp_data[i] = <OPJ_INT32> in_u8[base + i * step]
            elif kind_size == 3:
                for i in range(n_pixels):
                    comp_data[i] = <OPJ_INT32> in_i8[base + i * step]
            elif kind_size == 4:
                for i in range(n_pixels):
                    comp_data[i] = <OPJ_INT32> in_u16[base + i * step]
            else:
                for i in range(n_pixels):
                    comp_data[i] = <OPJ_INT32> in_i16[base + i * step]

    opj_codec = opj_create_compress(<CODEC_FORMAT> cf)
    if opj_codec == NULL:
        opj_image_destroy(image)
        raise Jpeg2kError('opj_create_compress failed')
    mlog.buf = NULL
    mlog.size = 0
    mlog.cap = 0
    _msg_handlers(opj_codec, &mlog, vlevel)
    opj_set_default_encoder_parameters(&cparams)
    cparams.tcp_numlayers = 1
    # imagecodecs' resolution count, which keeps OpenJPEG's rule that a
    # tile side be at least 2**(numresolution - 1) and makes the
    # codestream byte-identical to imagecodecs' at every size.
    numres = _numresolution(height, width, resolutions)
    cparams.numresolution = <OPJ_UINT32> numres
    # Component transform (RCT reversible / ICT irreversible) for sRGB,
    # as imagecodecs does; libopenjpeg picks which from `irreversible`.
    if numcomps == 3 and color_space == OPJ_CLRSPC_SRGB:
        cparams.tcp_mct = <char> (1 if mct else 0)
    if lossless:
        cparams.irreversible = 0
        cparams.cp_disto_alloc = 1
        cparams.tcp_rates[0] = 0
    elif lossy_level:
        # imagecodecs' fixed-quality settings: PSNR target in dB, RPCL
        # progression, 256 x 256 precincts.
        cparams.irreversible = 0 if reversible else 1
        cparams.prog_order = OPJ_RPCL
        cparams.cp_fixed_quality = 1
        cparams.tcp_distoratio[0] = quality
        cparams.res_spec = numres
        for i in range(<OPJ_UINT32> numres):
            cparams.prcw_init[i] = 256
            cparams.prch_init[i] = 256
        cparams.csty = 1
    elif ratio is not None:
        cparams.irreversible = 0 if reversible else 1
        cparams.cp_disto_alloc = 1
        cparams.tcp_rates[0] = <float> ratio
    else:
        # lossless=None with reversible=False: the 9/7 wavelet with every
        # bit kept, which is what imagecodecs writes for that request.
        cparams.irreversible = 1
        cparams.cp_disto_alloc = 1
        cparams.tcp_rates[0] = 0

    if not opj_setup_encoder(opj_codec, &cparams, image):
        opj_destroy_codec(opj_codec)
        opj_image_destroy(image)
        _msg_flush(&mlog)
        raise Jpeg2kError('opj_setup_encoder failed')

    # Enable multithreaded T1 encoding when libopenjp2 supports it.
    if opj_has_thread_support():
        if numthreads is None:
            _opj_n = opj_get_num_cpus() // 2
            if _opj_n < 1: _opj_n = 1
        else:
            _opj_n = int(numthreads)
        if _opj_n > 1:
            opj_codec_set_threads(opj_codec, _opj_n)

    wrbuf.data = NULL
    wrbuf.cap = 0
    wrbuf.size = 0
    wrbuf.offset = 0
    wrbuf.destination = NULL if destination is None else <void*>destination

    stream = opj_stream_default_create(0)  # is_input=0
    if stream == NULL:
        opj_destroy_codec(opj_codec)
        opj_image_destroy(image)
        _msg_flush(&mlog)
        raise Jpeg2kError('opj_stream_default_create failed')
    opj_stream_set_user_data(stream, &wrbuf, NULL)
    opj_stream_set_write_function(stream, _write_cb)
    opj_stream_set_skip_function(stream, _skip_write_cb)
    opj_stream_set_seek_function(stream, _seek_write_cb)

    try:
        if not opj_start_compress(opj_codec, image, stream):
            raise Jpeg2kError('opj_start_compress failed')
        if not opj_encode(opj_codec, stream):
            raise Jpeg2kError('opj_encode failed')
        if not opj_end_compress(opj_codec, stream):
            raise Jpeg2kError('opj_end_compress failed')
        if destination is not None:
            destination.finish(wrbuf.size)
            return None
        out = PyBytes_FromStringAndSize(<char*> wrbuf.data,
                                        <Py_ssize_t> wrbuf.size)
        return out
    finally:
        opj_stream_destroy(stream)
        opj_destroy_codec(opj_codec)
        opj_image_destroy(image)
        _msg_flush(&mlog)
        if wrbuf.data != NULL:
            free(wrbuf.data)
        if destination is not None and destination.error is not None:
            raise destination.error


def check_signature(data) -> bool:
    """True if `data` looks like JP2 (jP  ) or raw J2K codestream (FF4F)."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:12])
    else:
        try:
            head = bytes(data)[:12]
        except Exception:
            return False
    if len(head) >= 4 and head[0] == 0xFF and head[1] == 0x4F and head[2] == 0xFF and head[3] == 0x51:
        return True
    if len(head) >= 12 and head[4:8] == b'jP  ':
        return True
    return False
