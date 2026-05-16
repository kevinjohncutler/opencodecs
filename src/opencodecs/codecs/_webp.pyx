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
from libc.stdint cimport uint8_t

import numpy as np
cimport numpy as cnp

from webp cimport (
    WebPEncodeRGB, WebPEncodeRGBA,
    WebPEncodeLosslessRGB, WebPEncodeLosslessRGBA,
    oc_webp_encode, oc_webp_free,
    WebPFree,
    WebPGetFeatures, WebPBitstreamFeatures,
    WebPDecodeRGBInto, WebPDecodeRGBAInto,
    VP8_STATUS_OK,
)

cnp.import_array()


class WebpError(RuntimeError):
    """Raised on WebP encode/decode failures."""


def encode(data, *, level: int | None = None,
           lossless: bool = False,
           numthreads: int | None = None,
           method: int = -1) -> bytes:
    """Encode a uint8 image as WebP.

    Parameters
    ----------
    level : int, optional
        Quality 0-100 (default 75); ignored when ``lossless=True``.
    lossless : bool
        Use the near-lossless preset (preset level 6).
    numthreads : int, optional
        ``None`` or ``<=0`` uses the libwebp default (single-threaded).
        Any positive value enables libwebp's worker thread for entropy
        coding (``WebPConfig.thread_level=1``). libwebp's threading model
        is a binary on/off, not an N-way pool — additional workers don't
        help. Typical speedup: 1.3-1.8× on lossy RGB encode.
    method : int
        libwebp speed/quality tradeoff 0..6. ``-1`` (default) leaves
        libwebp's own default (4).
    """
    cdef:
        cnp.ndarray arr
        const uint8_t* src_ptr
        uint8_t* out_ptr = NULL
        uint8_t* shim_ptr = NULL
        size_t out_size = 0
        int width, height, stride
        float quality
        int thread_level
        int has_alpha_c = 0
        int lossless_c = 1 if lossless else 0
        int method_c = int(method)
        int rc
        bytes out

    if not isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data, dtype=np.uint8)
    else:
        if data.dtype != np.uint8:
            raise WebpError(f'WebP encode: unsupported dtype {data.dtype}')
        arr = np.ascontiguousarray(data)

    if arr.ndim == 2:
        # Promote grayscale to RGB so libwebp accepts it.
        arr = np.stack([arr] * 3, axis=-1)
        arr = np.ascontiguousarray(arr)
    elif arr.ndim == 3:
        if arr.shape[2] == 3:
            has_alpha_c = 0
        elif arr.shape[2] == 4:
            has_alpha_c = 1
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

    if numthreads is None or int(numthreads) <= 0:
        thread_level = 0
    else:
        thread_level = 1

    if thread_level or method_c >= 0:
        # Advanced API via the shim — needed to set thread_level / method.
        with nogil:
            rc = oc_webp_encode(
                src_ptr, width, height, stride,
                has_alpha_c, lossless_c, quality,
                thread_level, method_c,
                &shim_ptr, &out_size,
            )
        if rc != 0 or shim_ptr == NULL or out_size == 0:
            if shim_ptr != NULL:
                oc_webp_free(shim_ptr)
            raise WebpError(f'WebP encode failed (rc={rc})')
        try:
            out = PyBytes_FromStringAndSize(
                <char*> shim_ptr, <Py_ssize_t> out_size)
            return out
        finally:
            oc_webp_free(shim_ptr)

    # Simple API — minimal overhead, no advanced config available.
    if lossless:
        if has_alpha_c:
            with nogil:
                out_size = WebPEncodeLosslessRGBA(
                    src_ptr, width, height, stride, &out_ptr)
        else:
            with nogil:
                out_size = WebPEncodeLosslessRGB(
                    src_ptr, width, height, stride, &out_ptr)
    else:
        if has_alpha_c:
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


def decode(data, *, out=None) -> np.ndarray:
    """Decode WebP bytes into a uint8 array.

    ``out=`` is a preallocated ``(H, W, 3) | (H, W, 4) uint8`` ndarray
    matching the WebP file's geometry. See ``_png.decode`` for the full
    contract. WebPDecode{RGB,RGBA}Into write directly into the buffer
    so this is a true zero-alloc fast path.
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        WebPBitstreamFeatures features
        int rc
        int width = 0, height = 0
        uint8_t* dec_ptr
        cnp.ndarray out_arr
        cnp.npy_intp shape[3]
        int channels
        size_t out_size
        int out_stride
        tuple expected_shape

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

    width = features.width
    height = features.height
    has_alpha = bool(features.has_alpha)
    channels = 4 if has_alpha else 3
    shape[0] = height
    shape[1] = width
    shape[2] = channels
    expected_shape = (height, width, channels)

    if out is not None:
        if not isinstance(out, np.ndarray):
            raise TypeError(
                f"webp decode: out= must be an ndarray, "
                f"got {type(out).__name__}")
        if out.shape != expected_shape:
            raise ValueError(
                f"webp decode: out= shape {out.shape} does not match "
                f"expected {expected_shape}")
        if out.dtype != np.uint8:
            raise ValueError(
                f"webp decode: out= dtype must be uint8, got {out.dtype}")
        if not out.flags['C_CONTIGUOUS']:
            raise ValueError("webp decode: out= must be C-contiguous")
        out_arr = out
    else:
        out_arr = cnp.PyArray_EMPTY(3, shape, cnp.NPY_UINT8, 0)
    out_stride = width * channels
    out_size = <size_t> (out_stride * height)

    # Decode straight into the numpy array's buffer — skips the
    # malloc+memcpy step the WebPDecode{RGB,RGBA} variants would do.
    if has_alpha:
        with nogil:
            dec_ptr = WebPDecodeRGBAInto(
                &src[0], srcsize,
                <uint8_t*> cnp.PyArray_DATA(out_arr), out_size, out_stride,
            )
    else:
        with nogil:
            dec_ptr = WebPDecodeRGBInto(
                &src[0], srcsize,
                <uint8_t*> cnp.PyArray_DATA(out_arr), out_size, out_stride,
            )
    if dec_ptr == NULL:
        raise WebpError('WebP decode failed')
    return out_arr


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
