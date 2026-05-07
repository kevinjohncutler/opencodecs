# opencodecs/codecs/_png.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native PNG codec via libspng.

Decode preserves PNG color/depth where practical:
  - 8-bit grayscale       -> (H, W) uint8
  - 8-bit grayscale+alpha -> (H, W, 2) uint8
  - 8-bit RGB             -> (H, W, 3) uint8
  - 8-bit RGBA            -> (H, W, 4) uint8
  - 8-bit indexed         -> (H, W, 4) uint8 (palette expanded to RGBA)
  - 16-bit grayscale      -> (H, W) uint16 (host-endian)
  - 16-bit gray+alpha     -> (H, W, 2) uint16
  - 16-bit RGB            -> (H, W, 3) uint16
  - 16-bit RGBA           -> (H, W, 4) uint16
  - 1/2/4-bit             -> upscaled to 8-bit grayscale or RGBA

Encode picks color type / bit depth from numpy shape and dtype:
  - 2D uint8 / uint16        -> grayscale
  - (H, W, 2) uint8/uint16   -> grayscale+alpha
  - (H, W, 3) uint8/uint16   -> RGB
  - (H, W, 4) uint8/uint16   -> RGBA
"""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.string cimport memcpy
from libc.stdint cimport uint8_t, uint32_t
from libc.stdlib cimport free
from libc.stddef cimport size_t

import numpy as np
cimport numpy as cnp

from spng cimport (
    spng_ctx, spng_ctx_new, spng_ctx_free,
    spng_set_png_buffer, spng_get_png_buffer,
    spng_set_image_limits, spng_set_chunk_limits, spng_set_option,
    spng_decoded_image_size, spng_decode_image, spng_encode_image,
    spng_ihdr, spng_get_ihdr, spng_set_ihdr,
    spng_strerror,
    SPNG_COLOR_TYPE_GRAYSCALE, SPNG_COLOR_TYPE_TRUECOLOR,
    SPNG_COLOR_TYPE_INDEXED, SPNG_COLOR_TYPE_GRAYSCALE_ALPHA,
    SPNG_COLOR_TYPE_TRUECOLOR_ALPHA,
    SPNG_FMT_RGBA8, SPNG_FMT_PNG,
    SPNG_CTX_ENCODER, SPNG_ENCODE_FINALIZE,
    SPNG_IMG_COMPRESSION_LEVEL, SPNG_ENCODE_TO_BUFFER,
)

cnp.import_array()


class PngError(RuntimeError):
    """Raised on PNG encode/decode failures."""


cdef inline _check(int rc, str where):
    if rc != 0:
        raise PngError(f'{where}: {spng_strerror(rc).decode()}')


def decode(data) -> np.ndarray:
    """Decode a PNG byte string to a numpy array."""
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        spng_ctx* ctx = NULL
        spng_ihdr ihdr
        int rc
        int fmt
        size_t out_size
        cnp.ndarray out
        cnp.npy_intp shape[3]
        int ndim
        object dtype

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if srcsize < 8:
        raise PngError('input too short to be a PNG')

    ctx = spng_ctx_new(0)
    if ctx == NULL:
        raise PngError('spng_ctx_new failed')
    try:
        # Generous limits; libspng default is conservative.
        spng_set_image_limits(ctx, 200000, 200000)
        spng_set_chunk_limits(ctx, 64 * 1024 * 1024, 64 * 1024 * 1024)
        rc = spng_set_png_buffer(ctx, <const void*> &src[0], srcsize)
        _check(rc, 'spng_set_png_buffer')
        rc = spng_get_ihdr(ctx, &ihdr)
        _check(rc, 'spng_get_ihdr')

        # Pick output fmt + numpy shape/dtype based on PNG color type/depth.
        # SPNG_FMT_PNG returns data in host byte order matching the PNG IHDR
        # (no scaling/conversion). For indexed and sub-byte depths, ask spng
        # to expand to RGBA8.
        if ihdr.color_type == SPNG_COLOR_TYPE_INDEXED or ihdr.bit_depth < 8:
            fmt = SPNG_FMT_RGBA8
            ndim = 3
            shape[2] = 4
            dtype = np.uint8
        elif ihdr.color_type == SPNG_COLOR_TYPE_GRAYSCALE:
            fmt = SPNG_FMT_PNG
            ndim = 2
            dtype = np.uint16 if ihdr.bit_depth == 16 else np.uint8
        elif ihdr.color_type == SPNG_COLOR_TYPE_GRAYSCALE_ALPHA:
            fmt = SPNG_FMT_PNG
            ndim = 3
            shape[2] = 2
            dtype = np.uint16 if ihdr.bit_depth == 16 else np.uint8
        elif ihdr.color_type == SPNG_COLOR_TYPE_TRUECOLOR:
            fmt = SPNG_FMT_PNG
            ndim = 3
            shape[2] = 3
            dtype = np.uint16 if ihdr.bit_depth == 16 else np.uint8
        elif ihdr.color_type == SPNG_COLOR_TYPE_TRUECOLOR_ALPHA:
            fmt = SPNG_FMT_PNG
            ndim = 3
            shape[2] = 4
            dtype = np.uint16 if ihdr.bit_depth == 16 else np.uint8
        else:
            raise PngError(f'unsupported PNG color type {ihdr.color_type}')

        rc = spng_decoded_image_size(ctx, fmt, &out_size)
        _check(rc, 'spng_decoded_image_size')

        shape[0] = ihdr.height
        shape[1] = ihdr.width
        out = cnp.PyArray_EMPTY(ndim, shape, cnp.NPY_UINT16 if dtype is np.uint16
                                else cnp.NPY_UINT8, 0)
        if out.nbytes != <Py_ssize_t> out_size:
            raise PngError(
                f'decoded image size mismatch: spng={out_size} '
                f'numpy={out.nbytes}')

        with nogil:
            rc = spng_decode_image(
                ctx, cnp.PyArray_DATA(out), out_size, fmt, 0,
            )
        _check(rc, 'spng_decode_image')

        return out
    finally:
        spng_ctx_free(ctx)


def encode(data, *, level: int | None = None) -> bytes:
    """Encode a numpy array as a PNG byte string."""
    cdef:
        spng_ctx* ctx = NULL
        spng_ihdr ihdr
        int rc
        int fmt
        size_t img_len
        size_t buf_len = 0
        int err = 0
        void* png_buf
        cnp.ndarray arr
        bytes out
        int compression
        bint need_byteswap = False

    if not isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data)
    else:
        arr = np.ascontiguousarray(data)

    if arr.dtype not in (np.uint8, np.uint16):
        raise PngError(f'PNG encode: unsupported dtype {arr.dtype}')

    # Determine color_type and channels.
    if arr.ndim == 2:
        color_type = SPNG_COLOR_TYPE_GRAYSCALE
        channels = 1
    elif arr.ndim == 3:
        c = arr.shape[2]
        if c == 1:
            color_type = SPNG_COLOR_TYPE_GRAYSCALE
            channels = 1
            arr = arr[:, :, 0]
        elif c == 2:
            color_type = SPNG_COLOR_TYPE_GRAYSCALE_ALPHA
            channels = 2
        elif c == 3:
            color_type = SPNG_COLOR_TYPE_TRUECOLOR
            channels = 3
        elif c == 4:
            color_type = SPNG_COLOR_TYPE_TRUECOLOR_ALPHA
            channels = 4
        else:
            raise PngError(
                f'PNG encode: unsupported number of channels {c}')
    else:
        raise PngError(
            f'PNG encode: unsupported array ndim {arr.ndim}')

    bit_depth = 16 if arr.dtype == np.uint16 else 8

    # spng_encode_image only accepts SPNG_FMT_PNG (host-endian, no
    # conversion) or SPNG_FMT_RAW (big-endian). Use SPNG_FMT_PNG; spng
    # converts to PNG file byte order (big-endian) internally.
    fmt = SPNG_FMT_PNG
    arr = np.ascontiguousarray(arr)

    ctx = spng_ctx_new(SPNG_CTX_ENCODER)
    if ctx == NULL:
        raise PngError('spng_ctx_new(ENCODER) failed')
    try:
        # Internal buffer mode (default; spng allocates and we read it back).
        ihdr.width = arr.shape[0] if arr.ndim == 1 else (
            <uint32_t> arr.shape[1])
        ihdr.height = <uint32_t> arr.shape[0]
        ihdr.bit_depth = <uint8_t> bit_depth
        ihdr.color_type = <uint8_t> color_type
        ihdr.compression_method = 0
        ihdr.filter_method = 0
        ihdr.interlace_method = 0
        rc = spng_set_ihdr(ctx, &ihdr)
        _check(rc, 'spng_set_ihdr')

        # Tell spng to allocate the output buffer internally; we fetch it
        # back via spng_get_png_buffer after encode.
        rc = spng_set_option(ctx, SPNG_ENCODE_TO_BUFFER, 1)
        _check(rc, 'spng_set_option(SPNG_ENCODE_TO_BUFFER)')

        if level is not None:
            compression = int(level)
            if compression < 0: compression = 0
            if compression > 9: compression = 9
            rc = spng_set_option(
                ctx, SPNG_IMG_COMPRESSION_LEVEL, compression)
            _check(rc, 'spng_set_option(compression_level)')

        img_len = <size_t> arr.nbytes
        with nogil:
            rc = spng_encode_image(
                ctx, cnp.PyArray_DATA(arr), img_len, fmt,
                SPNG_ENCODE_FINALIZE,
            )
        _check(rc, 'spng_encode_image')

        png_buf = spng_get_png_buffer(ctx, &buf_len, &err)
        if png_buf == NULL or err != 0:
            raise PngError(
                f'spng_get_png_buffer: {spng_strerror(err).decode()}')
        try:
            out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> buf_len)
            memcpy(<void*> PyBytes_AsString(out), png_buf, buf_len)
            return out
        finally:
            free(png_buf)
    finally:
        spng_ctx_free(ctx)


def check_signature(data) -> bool:
    """True if `data` starts with the 8-byte PNG signature."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:8])
    else:
        try:
            head = bytes(data)[:8]
        except Exception:
            return False
    return head == b'\x89PNG\r\n\x1a\n'
