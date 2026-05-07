# opencodecs/codecs/_avif.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native AVIF codec via libavif (linked against system aom)."""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.string cimport memcpy
from libc.stdint cimport uint8_t

import numpy as np
cimport numpy as cnp

from avif cimport (
    AVIF_QUALITY_LOSSLESS, AVIF_RESULT_OK,
    AVIF_PIXEL_FORMAT_YUV444, AVIF_PIXEL_FORMAT_YUV420,
    AVIF_RGB_FORMAT_RGB, AVIF_RGB_FORMAT_RGBA,
    avifPixelFormat,
    avifImage, avifImageCreate, avifImageCreateEmpty, avifImageDestroy,
    avifRGBImage, avifRGBImageSetDefaults,
    avifRGBImageAllocatePixels, avifRGBImageFreePixels,
    avifImageRGBToYUV, avifImageYUVToRGB,
    avifEncoder, avifEncoderCreate, avifEncoderDestroy, avifEncoderWrite,
    avifDecoder, avifDecoderCreate, avifDecoderDestroy, avifDecoderReadMemory,
    avifRWData, avifRWDataFree,
    avifResultToString,
)

cnp.import_array()


class AvifError(RuntimeError):
    """Raised on AVIF encode/decode failures."""


def encode(data, *, level: int | None = None,
           lossless: bool = False, speed: int = 6) -> bytes:
    """Encode an array as AVIF.

    ``level`` is quality 0-100 (default 60); ignored if ``lossless=True``.
    ``speed`` 0-10 (lower = better quality but slower; default 6).
    """
    cdef:
        cnp.ndarray arr
        avifImage* image = NULL
        avifRGBImage rgb
        avifEncoder* encoder = NULL
        avifRWData out_data
        int rc
        bytes out
        int has_alpha
        int quality
        avifPixelFormat yuv_format
        int channels
        size_t row_bytes_in
        unsigned int y

    if not isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data, dtype=np.uint8)
    else:
        if data.dtype != np.uint8:
            raise AvifError(f'AVIF: only uint8 supported, got {data.dtype}')
        arr = np.ascontiguousarray(data)

    if arr.ndim == 2:
        # Promote grayscale to RGB.
        arr = np.ascontiguousarray(np.stack([arr] * 3, axis=-1))
        has_alpha = 0
    elif arr.ndim == 3 and arr.shape[2] == 3:
        has_alpha = 0
    elif arr.ndim == 3 and arr.shape[2] == 4:
        has_alpha = 1
    else:
        raise AvifError(f'AVIF encode: unsupported shape ndim={arr.ndim}')

    quality = AVIF_QUALITY_LOSSLESS if lossless else (60 if level is None else int(level))
    if quality < 0: quality = 0
    if quality > 100: quality = 100

    # Lossless requires YUV444; lossy default to YUV420 (smaller).
    yuv_format = AVIF_PIXEL_FORMAT_YUV444 if lossless else AVIF_PIXEL_FORMAT_YUV420

    image = avifImageCreate(<unsigned int> arr.shape[1],
                            <unsigned int> arr.shape[0],
                            8, yuv_format)
    if image == NULL:
        raise AvifError('avifImageCreate failed')
    if lossless:
        # Identity matrix means YUV planes ARE the RGB planes verbatim,
        # so YUV444 + identity = byte-perfect lossless.
        image.matrixCoefficients = 0  # AVIF_MATRIX_COEFFICIENTS_IDENTITY
    encoder = avifEncoderCreate()
    if encoder == NULL:
        avifImageDestroy(image)
        raise AvifError('avifEncoderCreate failed')

    out_data.data = NULL
    out_data.size = 0

    try:
        avifRGBImageSetDefaults(&rgb, image)
        rgb.format = AVIF_RGB_FORMAT_RGBA if has_alpha else AVIF_RGB_FORMAT_RGB
        rc = avifRGBImageAllocatePixels(&rgb)
        if rc != AVIF_RESULT_OK:
            raise AvifError(
                f'avifRGBImageAllocatePixels: '
                f'{avifResultToString(rc).decode()}')
        try:
            # Copy our row-major numpy data into the rgb buffer (which may
            # have a stride). Both should match for contiguous uint8 input.
            channels = 4 if has_alpha else 3
            row_bytes_in = <size_t>(<int> arr.shape[1] * channels)
            for y in range(<int> arr.shape[0]):
                memcpy(rgb.pixels + y * rgb.rowBytes,
                       <const uint8_t*> cnp.PyArray_DATA(arr) + y * row_bytes_in,
                       row_bytes_in)
            rc = avifImageRGBToYUV(image, &rgb)
            if rc != AVIF_RESULT_OK:
                raise AvifError(
                    f'avifImageRGBToYUV: '
                    f'{avifResultToString(rc).decode()}')
        finally:
            avifRGBImageFreePixels(&rgb)

        encoder.quality = quality
        encoder.qualityAlpha = quality
        if speed >= 0 and speed <= 10:
            encoder.speed = speed
        encoder.maxThreads = 4

        with nogil:
            rc = avifEncoderWrite(encoder, image, &out_data)
        if rc != AVIF_RESULT_OK:
            raise AvifError(
                f'avifEncoderWrite: {avifResultToString(rc).decode()}')

        out = PyBytes_FromStringAndSize(<char*> out_data.data,
                                        <Py_ssize_t> out_data.size)
        return out
    finally:
        avifRWDataFree(&out_data)
        avifEncoderDestroy(encoder)
        avifImageDestroy(image)


def decode(data) -> np.ndarray:
    """Decode AVIF bytes to a uint8 numpy array."""
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        avifDecoder* decoder = NULL
        avifImage* image = NULL
        avifRGBImage rgb
        int rc
        cnp.ndarray out
        cnp.npy_intp shape[3]
        int has_alpha
        int channels
        unsigned int y
        size_t row_bytes_out

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <size_t> src.shape[0]

    decoder = avifDecoderCreate()
    if decoder == NULL:
        raise AvifError('avifDecoderCreate failed')
    image = avifImageCreateEmpty()
    if image == NULL:
        avifDecoderDestroy(decoder)
        raise AvifError('avifImageCreateEmpty failed')
    decoder.maxThreads = 4

    try:
        with nogil:
            rc = avifDecoderReadMemory(decoder, image, &src[0], srcsize)
        if rc != AVIF_RESULT_OK:
            raise AvifError(
                f'avifDecoderReadMemory: {avifResultToString(rc).decode()}')

        avifRGBImageSetDefaults(&rgb, image)
        # Detect alpha by checking image's alpha plane pointer.
        has_alpha = 1 if image.alphaPlane != NULL else 0
        channels = 4 if has_alpha else 3
        rgb.format = AVIF_RGB_FORMAT_RGBA if has_alpha else AVIF_RGB_FORMAT_RGB

        rc = avifRGBImageAllocatePixels(&rgb)
        if rc != AVIF_RESULT_OK:
            raise AvifError(
                f'avifRGBImageAllocatePixels: '
                f'{avifResultToString(rc).decode()}')
        try:
            with nogil:
                rc = avifImageYUVToRGB(image, &rgb)
            if rc != AVIF_RESULT_OK:
                raise AvifError(
                    f'avifImageYUVToRGB: '
                    f'{avifResultToString(rc).decode()}')

            shape[0] = image.height
            shape[1] = image.width
            shape[2] = channels
            out = cnp.PyArray_EMPTY(3, shape, cnp.NPY_UINT8, 0)
            row_bytes_out = <size_t>(image.width * channels)
            for y in range(image.height):
                memcpy(<uint8_t*> cnp.PyArray_DATA(out) + y * row_bytes_out,
                       rgb.pixels + y * rgb.rowBytes,
                       row_bytes_out)
            return out
        finally:
            avifRGBImageFreePixels(&rgb)
    finally:
        avifImageDestroy(image)
        avifDecoderDestroy(decoder)


def check_signature(data) -> bool:
    """True if data is an AVIF (ftyp box with 'avif' brand)."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:32])
    else:
        try:
            head = bytes(data)[:32]
        except Exception:
            return False
    if len(head) < 12:
        return False
    # 'ftyp' box major brand 'avif' or compatible brand 'avif' / 'avis'.
    if head[4:8] != b'ftyp':
        return False
    return b'avif' in head[8:32] or b'avis' in head[8:32]
