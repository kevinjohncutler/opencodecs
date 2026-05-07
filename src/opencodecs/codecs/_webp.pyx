# opencodecs/codecs/_webp.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native WebP codec via libwebp.

Encode: 2D uint8 (gray, expanded to RGB), (H, W, 3) uint8 RGB,
        (H, W, 4) uint8 RGBA. Set ``lossless=True`` for lossless.
Decode: returns (H, W, 3) RGB or (H, W, 4) RGBA.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.string cimport memcpy
from libc.stdint cimport uint8_t

import numpy as np
cimport numpy as cnp

from webp cimport (
    WebPEncodeRGB, WebPEncodeRGBA,
    WebPEncodeLosslessRGB, WebPEncodeLosslessRGBA,
    WebPFree,
    WebPGetFeatures, WebPBitstreamFeatures,
    WebPDecodeRGB, WebPDecodeRGBA,
    VP8_STATUS_OK,
)

cnp.import_array()


class WebpError(RuntimeError):
    """Raised on WebP encode/decode failures."""


def encode(data, *, level: int | None = None,
           lossless: bool = False) -> bytes:
    """Encode a uint8 image as WebP.

    ``level`` is quality 0-100 (default 75); ignored when ``lossless=True``.
    """
    cdef:
        cnp.ndarray arr
        const uint8_t* src_ptr
        uint8_t* out_ptr = NULL
        size_t out_size
        int width, height, stride
        float quality
        bytes out

    if not isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data, dtype=np.uint8)
    else:
        if data.dtype != np.uint8:
            raise WebpError(f'WebP encode: unsupported dtype {data.dtype}')
        arr = np.ascontiguousarray(data)

    has_alpha = False
    if arr.ndim == 2:
        # Promote grayscale to RGB so libwebp accepts it.
        arr = np.stack([arr] * 3, axis=-1)
        arr = np.ascontiguousarray(arr)
    elif arr.ndim == 3:
        if arr.shape[2] == 3:
            has_alpha = False
        elif arr.shape[2] == 4:
            has_alpha = True
        else:
            raise WebpError(
                f'WebP encode: unsupported channel count {arr.shape[2]}')
    else:
        raise WebpError(f'WebP encode: unsupported ndim {arr.ndim}')

    height = <int> arr.shape[0]
    width = <int> arr.shape[1]
    stride = <int> arr.strides[0]
    src_ptr = <const uint8_t*> cnp.PyArray_DATA(arr)

    quality = 75.0 if level is None else float(level)
    if quality < 0: quality = 0
    if quality > 100: quality = 100

    if lossless:
        if has_alpha:
            with nogil:
                out_size = WebPEncodeLosslessRGBA(
                    src_ptr, width, height, stride, &out_ptr)
        else:
            with nogil:
                out_size = WebPEncodeLosslessRGB(
                    src_ptr, width, height, stride, &out_ptr)
    else:
        if has_alpha:
            with nogil:
                out_size = WebPEncodeRGBA(
                    src_ptr, width, height, stride, quality, &out_ptr)
        else:
            with nogil:
                out_size = WebPEncodeRGB(
                    src_ptr, width, height, stride, quality, &out_ptr)

    if out_ptr == NULL or out_size == 0:
        raise WebpError('WebP encode failed')
    try:
        out = PyBytes_FromStringAndSize(<char*> out_ptr, <Py_ssize_t> out_size)
        return out
    finally:
        WebPFree(out_ptr)


def decode(data) -> np.ndarray:
    """Decode WebP bytes into a uint8 array."""
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        WebPBitstreamFeatures features
        int rc
        int width = 0, height = 0
        uint8_t* dec_ptr
        cnp.ndarray out
        cnp.npy_intp shape[3]
        int channels

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if srcsize < 12:
        raise WebpError('input too short to be WebP')

    rc = WebPGetFeatures(&src[0], srcsize, &features)
    if rc != VP8_STATUS_OK:
        raise WebpError(f'WebPGetFeatures failed: status {rc}')

    has_alpha = bool(features.has_alpha)
    channels = 4 if has_alpha else 3

    if has_alpha:
        with nogil:
            dec_ptr = WebPDecodeRGBA(&src[0], srcsize, &width, &height)
    else:
        with nogil:
            dec_ptr = WebPDecodeRGB(&src[0], srcsize, &width, &height)
    if dec_ptr == NULL:
        raise WebpError('WebP decode failed')
    try:
        shape[0] = height
        shape[1] = width
        shape[2] = channels
        out = cnp.PyArray_EMPTY(3, shape, cnp.NPY_UINT8, 0)
        memcpy(cnp.PyArray_DATA(out), dec_ptr,
               <size_t>(width * height * channels))
        return out
    finally:
        WebPFree(dec_ptr)


def check_signature(data) -> bool:
    """True if `data` is a RIFF/WEBP container."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:12])
    else:
        try:
            head = bytes(data)[:12]
        except Exception:
            return False
    return len(head) >= 12 and head[:4] == b'RIFF' and head[8:12] == b'WEBP'
