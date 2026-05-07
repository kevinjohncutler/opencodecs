# opencodecs/codecs/_heif.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native HEIF / HEIC codec via libheif (system; depends on libde265 / x265)."""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdlib cimport realloc, free
from libc.string cimport memcpy
from libc.stdint cimport uint8_t

import numpy as np
cimport numpy as cnp

from heif cimport (
    heif_init, heif_context, heif_context_alloc, heif_context_free,
    heif_error_code,
    heif_context_read_from_memory_without_copy,
    heif_context_get_primary_image_handle,
    heif_image_handle, heif_image_handle_release,
    heif_image_handle_get_width, heif_image_handle_get_height,
    heif_image_handle_has_alpha_channel,
    heif_decode_image, heif_image, heif_image_release,
    heif_image_get_plane_readonly, heif_image_get_plane,
    heif_image_create, heif_image_add_plane,
    heif_colorspace_RGB,
    heif_chroma_interleaved_RGB, heif_chroma_interleaved_RGBA,
    heif_channel_interleaved,
    heif_compression_HEVC,
    heif_context_get_encoder_for_format,
    heif_encoder, heif_encoder_release,
    heif_encoder_set_lossy_quality, heif_encoder_set_lossless,
    heif_context_encode_image,
    heif_writer, heif_context_write,
    heif_error,
)

cnp.import_array()


class HeifError(RuntimeError):
    """Raised on HEIF/HEIC encode/decode failures."""


cdef bint _heif_initialized = False


cdef _ensure_init():
    global _heif_initialized
    if not _heif_initialized:
        heif_init(NULL)
        _heif_initialized = True


def decode(data) -> np.ndarray:
    """Decode HEIF/HEIC bytes to a uint8 numpy array."""
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        heif_context* ctx = NULL
        heif_image_handle* handle = NULL
        heif_image* img = NULL
        heif_error err
        int width, height, channels, has_alpha, stride
        const uint8_t* plane
        cnp.ndarray out
        cnp.npy_intp shape[3]
        int y

    _ensure_init()

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <size_t> src.shape[0]

    ctx = heif_context_alloc()
    if ctx == NULL:
        raise HeifError('heif_context_alloc failed')
    try:
        err = heif_context_read_from_memory_without_copy(
            ctx, &src[0], srcsize, NULL)
        if err.code != 0:
            raise HeifError(
                f'heif_context_read_from_memory: {err.message.decode()}')

        err = heif_context_get_primary_image_handle(ctx, &handle)
        if err.code != 0:
            raise HeifError(
                f'get_primary_image_handle: {err.message.decode()}')

        width = heif_image_handle_get_width(handle)
        height = heif_image_handle_get_height(handle)
        has_alpha = heif_image_handle_has_alpha_channel(handle)
        channels = 4 if has_alpha else 3

        with nogil:
            err = heif_decode_image(
                handle, &img, heif_colorspace_RGB,
                heif_chroma_interleaved_RGBA if has_alpha
                else heif_chroma_interleaved_RGB,
                NULL,
            )
        if err.code != 0:
            raise HeifError(f'heif_decode_image: {err.message.decode()}')

        plane = heif_image_get_plane_readonly(
            img, heif_channel_interleaved, &stride)
        if plane == NULL:
            raise HeifError('heif_image_get_plane_readonly returned NULL')

        shape[0] = height
        shape[1] = width
        shape[2] = channels
        out = cnp.PyArray_EMPTY(3, shape, cnp.NPY_UINT8, 0)
        for y in range(height):
            memcpy(<uint8_t*> cnp.PyArray_DATA(out) + y * width * channels,
                   plane + y * stride,
                   <size_t>(width * channels))
        return out
    finally:
        if img != NULL:
            heif_image_release(img)
        if handle != NULL:
            heif_image_handle_release(handle)
        heif_context_free(ctx)


# ---------------------------------------------------------------------------
# Encode (HEVC/HEIC). Captures the writer output via a small bytestream
# accumulator passed in via heif_writer.
# ---------------------------------------------------------------------------


cdef struct write_buffer:
    uint8_t* data
    size_t cap
    size_t size


cdef heif_error _writer_cb(
    heif_context* ctx, const void* data, size_t size, void* userdata,
) noexcept nogil:
    cdef write_buffer* buf = <write_buffer*> userdata
    cdef heif_error err
    err.code = <heif_error_code> 0
    err.subcode = 0
    err.message = NULL
    cdef size_t new_cap
    cdef uint8_t* new_data
    if buf.size + size > buf.cap:
        new_cap = buf.cap * 2 if buf.cap else 65536
        while new_cap < buf.size + size:
            new_cap *= 2
        new_data = <uint8_t*> realloc(buf.data, new_cap)
        if new_data == NULL:
            err.code = <heif_error_code> 1  # libheif treats nonzero as failure
            return err
        buf.data = new_data
        buf.cap = new_cap
    memcpy(buf.data + buf.size, data, size)
    buf.size += size
    return err


def encode(data, *, level: int | None = None,
           lossless: bool = False) -> bytes:
    """Encode an array as HEIC.

    ``level`` is quality 0-100 (default 50). ``lossless=True`` uses the
    lossless mode where supported by the HEVC encoder.
    """
    cdef:
        cnp.ndarray arr
        heif_context* ctx = NULL
        heif_encoder* enc = NULL
        heif_image* img = NULL
        heif_image_handle* handle = NULL
        heif_error err
        write_buffer wbuf
        heif_writer wr
        int width, height
        int has_alpha, channels
        uint8_t* plane
        int stride
        bytes out
        unsigned int y

    _ensure_init()

    if not isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data, dtype=np.uint8)
    else:
        if data.dtype != np.uint8:
            raise HeifError(f'HEIF: only uint8 supported, got {data.dtype}')
        arr = np.ascontiguousarray(data)

    if arr.ndim == 2:
        arr = np.ascontiguousarray(np.stack([arr] * 3, axis=-1))
        has_alpha = 0
    elif arr.ndim == 3 and arr.shape[2] == 3:
        has_alpha = 0
    elif arr.ndim == 3 and arr.shape[2] == 4:
        has_alpha = 1
    else:
        raise HeifError(f'HEIF encode: unsupported shape ndim={arr.ndim}')

    height = <int> arr.shape[0]
    width = <int> arr.shape[1]
    channels = 4 if has_alpha else 3

    ctx = heif_context_alloc()
    if ctx == NULL:
        raise HeifError('heif_context_alloc failed')
    try:
        err = heif_context_get_encoder_for_format(
            ctx, heif_compression_HEVC, &enc)
        if err.code != 0:
            raise HeifError(
                f'get_encoder_for_format(HEVC): {err.message.decode()}')

        if lossless:
            heif_encoder_set_lossless(enc, 1)
        else:
            heif_encoder_set_lossy_quality(enc, 50 if level is None else int(level))

        err = heif_image_create(
            width, height, heif_colorspace_RGB,
            heif_chroma_interleaved_RGBA if has_alpha
            else heif_chroma_interleaved_RGB, &img)
        if err.code != 0:
            raise HeifError(f'heif_image_create: {err.message.decode()}')

        err = heif_image_add_plane(
            img, heif_channel_interleaved, width, height, 8)
        if err.code != 0:
            raise HeifError(f'heif_image_add_plane: {err.message.decode()}')

        plane = heif_image_get_plane(img, heif_channel_interleaved, &stride)
        if plane == NULL:
            raise HeifError('heif_image_get_plane returned NULL')
        for y in range(height):
            memcpy(plane + y * stride,
                   <const uint8_t*> cnp.PyArray_DATA(arr) + y * width * channels,
                   <size_t>(width * channels))

        with nogil:
            err = heif_context_encode_image(ctx, img, enc, NULL, &handle)
        if err.code != 0:
            raise HeifError(
                f'heif_context_encode_image: {err.message.decode()}')

        wbuf.data = NULL
        wbuf.cap = 0
        wbuf.size = 0
        wr.writer_api_version = 1
        wr.write = _writer_cb

        err = heif_context_write(ctx, &wr, &wbuf)
        if err.code != 0:
            free(wbuf.data)
            raise HeifError(f'heif_context_write: {err.message.decode()}')

        try:
            out = PyBytes_FromStringAndSize(<char*> wbuf.data,
                                            <Py_ssize_t> wbuf.size)
            return out
        finally:
            free(wbuf.data)
    finally:
        if handle != NULL:
            heif_image_handle_release(handle)
        if img != NULL:
            heif_image_release(img)
        if enc != NULL:
            heif_encoder_release(enc)
        heif_context_free(ctx)


def check_signature(data) -> bool:
    """True if `data` looks like HEIF (ftyp box with heic/heix/mif1/msf1)."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:32])
    else:
        try:
            head = bytes(data)[:32]
        except Exception:
            return False
    if len(head) < 12 or head[4:8] != b'ftyp':
        return False
    brands = head[8:32]
    for b in (b'heic', b'heix', b'heim', b'heis', b'hevc',
              b'mif1', b'msf1'):
        if b in brands:
            return True
    return False
