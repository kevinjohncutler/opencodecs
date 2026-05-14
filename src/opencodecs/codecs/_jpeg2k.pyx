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
from libc.string cimport memcpy
from libc.stdint cimport int32_t, uint8_t

import numpy as np
cimport numpy as cnp

from openjpeg cimport (
    OPJ_BOOL, OPJ_INT32, OPJ_UINT32, OPJ_SIZE_T, OPJ_OFF_T,
    CODEC_FORMAT, COLOR_SPACE,
    OPJ_CODEC_J2K, OPJ_CODEC_JP2,
    OPJ_CLRSPC_GRAY, OPJ_CLRSPC_SRGB, OPJ_CLRSPC_UNSPECIFIED,
    opj_image_t, opj_image_create, opj_image_destroy, opj_image_cmptparm,
    opj_codec_t, opj_create_decompress, opj_create_compress,
    opj_destroy_codec, opj_set_default_decoder_parameters,
    opj_set_default_encoder_parameters, opj_setup_decoder,
    opj_setup_encoder,
    opj_dparameters_t, opj_cparameters_t,
    opj_stream_t, opj_stream_default_create, opj_stream_destroy,
    opj_stream_set_read_function, opj_stream_set_write_function,
    opj_stream_set_skip_function, opj_stream_set_seek_function,
    opj_stream_set_user_data, opj_stream_set_user_data_length,
    opj_read_header, opj_decode, opj_end_decompress,
    opj_start_compress, opj_encode, opj_end_compress,
    opj_has_thread_support, opj_get_num_cpus, opj_codec_set_threads,
)

cnp.import_array()


class Jpeg2kError(RuntimeError):
    """Raised on JPEG-2000 encode/decode failures."""


# ----- Memory stream user-data + callbacks (no GIL) -----

cdef struct mem_buffer_read:
    const uint8_t* data
    OPJ_SIZE_T size
    OPJ_SIZE_T offset


cdef struct mem_buffer_write:
    uint8_t* data
    OPJ_SIZE_T cap
    OPJ_SIZE_T size
    OPJ_SIZE_T offset


cdef OPJ_SIZE_T _read_cb(
    void* p_buffer, OPJ_SIZE_T p_nb_bytes, void* p_user_data
) noexcept nogil:
    cdef mem_buffer_read* buf = <mem_buffer_read*> p_user_data
    cdef OPJ_SIZE_T remaining = buf.size - buf.offset
    if remaining == 0:
        return <OPJ_SIZE_T> -1
    cdef OPJ_SIZE_T n = p_nb_bytes if p_nb_bytes < remaining else remaining
    memcpy(p_buffer, buf.data + buf.offset, n)
    buf.offset += n
    return n


cdef OPJ_OFF_T _skip_read_cb(OPJ_OFF_T p_nb_bytes, void* p_user_data) noexcept nogil:
    cdef mem_buffer_read* buf = <mem_buffer_read*> p_user_data
    cdef OPJ_SIZE_T remaining = buf.size - buf.offset
    cdef OPJ_OFF_T n = p_nb_bytes if <OPJ_SIZE_T> p_nb_bytes < remaining else <OPJ_OFF_T> remaining
    if n <= 0:
        return -1
    buf.offset += <OPJ_SIZE_T> n
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


# ----- Public API -----


def decode(data, *, numthreads: int | None = None) -> np.ndarray:
    """Decode JPEG-2000 (JP2 or J2K codestream) bytes to a numpy array.

    Parameters
    ----------
    numthreads : int, optional
        Worker threads for OpenJPEG's parallel decoder. ``None``
        defaults to ``opj_get_num_cpus() / 2`` (matches imagecodecs).
        ``0`` or ``1`` forces single-threaded. Typical 2-4× speedup on
        tiled / large-precinct JP2s.
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
        cnp.ndarray out

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <OPJ_SIZE_T> src.shape[0]
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

    stream = opj_stream_default_create(1)  # is_input=1
    if stream == NULL:
        raise Jpeg2kError('opj_stream_default_create failed')
    try:
        opj_stream_set_user_data(stream, &rdbuf, NULL)
        opj_stream_set_user_data_length(stream, <OPJ_UINT32> srcsize)
        opj_stream_set_read_function(stream, _read_cb)
        opj_stream_set_skip_function(stream, _skip_read_cb)
        opj_stream_set_seek_function(stream, _seek_read_cb)

        codec = opj_create_decompress(<CODEC_FORMAT> codec_format)
        if codec == NULL:
            raise Jpeg2kError('opj_create_decompress failed')
        opj_set_default_decoder_parameters(&dparams)
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

        ok = opj_read_header(stream, codec, &image)
        if not ok or image == NULL:
            raise Jpeg2kError('opj_read_header failed')

        ok = opj_decode(codec, stream, image)
        if not ok:
            raise Jpeg2kError('opj_decode failed')
        opj_end_decompress(codec, stream)

        out = _image_to_ndarray(image)
        return out
    finally:
        if image != NULL:
            opj_image_destroy(image)
        if codec != NULL:
            opj_destroy_codec(codec)
        opj_stream_destroy(stream)


cdef cnp.ndarray _image_to_ndarray(opj_image_t* image):
    """Copy openjpeg image planes into an interleaved numpy array."""
    cdef:
        OPJ_UINT32 numcomps = image.numcomps
        OPJ_UINT32 width
        OPJ_UINT32 height
        OPJ_UINT32 prec
        OPJ_UINT32 c, i, n_pixels
        OPJ_INT32* comp_data
        cnp.ndarray out
        cnp.npy_intp shape[3]
        int ndim
        int dtype_num

    if numcomps == 0:
        raise Jpeg2kError('image has 0 components')

    width = image.comps[0].w
    height = image.comps[0].h
    prec = image.comps[0].prec

    for c in range(numcomps):
        if image.comps[c].w != width or image.comps[c].h != height:
            raise Jpeg2kError(
                'JPEG-2000: components have differing sizes (subsampled?); '
                'not supported')
        if image.comps[c].prec != prec:
            raise Jpeg2kError(
                'JPEG-2000: components have differing precision; '
                'not supported')

    if prec <= 8:
        dtype_num = cnp.NPY_UINT8
    elif prec <= 16:
        dtype_num = cnp.NPY_UINT16
    else:
        raise Jpeg2kError(f'unsupported precision {prec} bits')

    if numcomps == 1:
        ndim = 2
    else:
        ndim = 3
        shape[2] = numcomps
    shape[0] = height
    shape[1] = width
    out = cnp.PyArray_EMPTY(ndim, shape, dtype_num, 0)

    n_pixels = width * height
    cdef uint8_t* out8 = <uint8_t*> cnp.PyArray_DATA(out)
    cdef unsigned short* out16 = <unsigned short*> cnp.PyArray_DATA(out)
    cdef OPJ_INT32 v
    cdef int sgnd
    cdef OPJ_UINT32 max_val = (<OPJ_UINT32> 1 << prec) - 1 if prec < 32 else 0xFFFFFFFF
    if dtype_num == cnp.NPY_UINT8:
        for c in range(numcomps):
            comp_data = image.comps[c].data
            sgnd = image.comps[c].sgnd
            for i in range(n_pixels):
                v = comp_data[i]
                if sgnd:
                    v += <OPJ_INT32>(1 << (prec - 1))
                if v < 0: v = 0
                if v > 255: v = 255
                out8[i * numcomps + c] = <uint8_t> v
    else:
        for c in range(numcomps):
            comp_data = image.comps[c].data
            sgnd = image.comps[c].sgnd
            for i in range(n_pixels):
                v = comp_data[i]
                if sgnd:
                    v += <OPJ_INT32>(1 << (prec - 1))
                if v < 0: v = 0
                if v > <OPJ_INT32> max_val: v = <OPJ_INT32> max_val
                out16[i * numcomps + c] = <unsigned short> v
    return out


def encode(data, *, level: int | None = None,
           lossless: bool = False, codec: str = 'jp2',
           numthreads: int | None = None) -> bytes:
    """Encode a numpy array as JPEG-2000 (JP2 by default; ``codec='j2k'`` for
    raw codestream).

    Parameters
    ----------
    level : int, optional
        Compression: lower = more compressed (default ~10 ratio).
        Ignored when ``lossless=True``.
    lossless : bool
        Use the lossless 5/3 integer wavelet.
    codec : {"jp2", "j2k"}
        Container format. ``jp2`` is the boxed format; ``j2k`` is the raw
        codestream that DICOM transfer syntaxes use.
    numthreads : int, optional
        Worker threads for OpenJPEG's parallel T1 encoder. ``None``
        defaults to ``opj_get_num_cpus() / 2``. Typical 2-3× speedup on
        tiled / large-precinct encodes.
    """
    cdef:
        cnp.ndarray arr
        opj_image_cmptparm cmptparms[4]
        opj_image_t* image = NULL
        opj_codec_t* opj_codec = NULL
        opj_stream_t* stream = NULL
        opj_cparameters_t cparams
        mem_buffer_write wrbuf
        OPJ_BOOL ok
        OPJ_UINT32 width, height, numcomps, prec
        OPJ_UINT32 c, i, n_pixels
        OPJ_INT32* comp_data
        bytes out
        int cf
        int _opj_n

    if not isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data)
    else:
        arr = np.ascontiguousarray(data)

    if arr.dtype == np.uint8:
        prec = 8
    elif arr.dtype == np.uint16:
        prec = 16
    else:
        raise Jpeg2kError(f'unsupported dtype {arr.dtype}')

    if arr.ndim == 2:
        numcomps = 1
        height = <OPJ_UINT32> arr.shape[0]
        width = <OPJ_UINT32> arr.shape[1]
    elif arr.ndim == 3:
        height = <OPJ_UINT32> arr.shape[0]
        width = <OPJ_UINT32> arr.shape[1]
        numcomps = <OPJ_UINT32> arr.shape[2]
        if numcomps not in (1, 2, 3, 4):
            raise Jpeg2kError(f'unsupported channel count {numcomps}')
    else:
        raise Jpeg2kError(f'unsupported ndim {arr.ndim}')

    if codec == 'jp2':
        cf = OPJ_CODEC_JP2
    elif codec == 'j2k':
        cf = OPJ_CODEC_J2K
    else:
        raise Jpeg2kError(f"codec must be 'jp2' or 'j2k', got {codec!r}")

    for c in range(numcomps):
        cmptparms[c].dx = 1
        cmptparms[c].dy = 1
        cmptparms[c].w = width
        cmptparms[c].h = height
        cmptparms[c].x0 = 0
        cmptparms[c].y0 = 0
        cmptparms[c].prec = prec
        cmptparms[c].bpp = prec
        cmptparms[c].sgnd = 0

    color_space = OPJ_CLRSPC_SRGB if numcomps >= 3 else (
        OPJ_CLRSPC_GRAY if numcomps == 1 else OPJ_CLRSPC_UNSPECIFIED)
    image = opj_image_create(numcomps, cmptparms, <COLOR_SPACE> color_space)
    if image == NULL:
        raise Jpeg2kError('opj_image_create failed')
    image.x0 = 0
    image.y0 = 0
    image.x1 = width
    image.y1 = height

    n_pixels = width * height
    cdef uint8_t* in8 = <uint8_t*> cnp.PyArray_DATA(arr)
    cdef unsigned short* in16 = <unsigned short*> cnp.PyArray_DATA(arr)
    if prec == 8:
        for c in range(numcomps):
            comp_data = image.comps[c].data
            for i in range(n_pixels):
                comp_data[i] = <OPJ_INT32> in8[i * numcomps + c]
    else:
        for c in range(numcomps):
            comp_data = image.comps[c].data
            for i in range(n_pixels):
                comp_data[i] = <OPJ_INT32> in16[i * numcomps + c]

    opj_codec = opj_create_compress(<CODEC_FORMAT> cf)
    if opj_codec == NULL:
        opj_image_destroy(image)
        raise Jpeg2kError('opj_create_compress failed')
    opj_set_default_encoder_parameters(&cparams)
    if lossless:
        cparams.tcp_numlayers = 1
        cparams.tcp_rates[0] = 0  # 0 = lossless
        cparams.cp_disto_alloc = 1
        cparams.irreversible = 0
    else:
        # Use a single quality layer; cp_fixed_quality + tcp_distoratio.
        # Simpler: use tcp_rates with a compression ratio derived from level.
        cparams.tcp_numlayers = 1
        # level 1..100 -> ratio (lower level = more aggressive). Default ~10.
        ratio = 10.0 if level is None else max(1.0, 100.0 / max(1, int(level)))
        cparams.tcp_rates[0] = <float> ratio
        cparams.cp_disto_alloc = 1
        cparams.irreversible = 1  # 9-7 wavelet

    if not opj_setup_encoder(opj_codec, &cparams, image):
        opj_destroy_codec(opj_codec)
        opj_image_destroy(image)
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

    stream = opj_stream_default_create(0)  # is_input=0
    if stream == NULL:
        opj_destroy_codec(opj_codec)
        opj_image_destroy(image)
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
        out = PyBytes_FromStringAndSize(<char*> wrbuf.data,
                                        <Py_ssize_t> wrbuf.size)
        return out
    finally:
        opj_stream_destroy(stream)
        opj_destroy_codec(opj_codec)
        opj_image_destroy(image)
        if wrbuf.data != NULL:
            free(wrbuf.data)


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
