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
from libc.stdint cimport uint8_t, uint16_t

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


# CICP (Coding-Independent Code Points) values used by libavif. These match
# JxlPrimaries / JxlTransferFunction so the same ColorSpec works for both
# codecs.
cdef:
    int AVIF_COLOR_PRIMARIES_BT709 = 1
    int AVIF_COLOR_PRIMARIES_UNSPECIFIED = 2
    int AVIF_COLOR_PRIMARIES_BT2020 = 9
    int AVIF_COLOR_PRIMARIES_DCI_P3 = 12  # NOT Display P3 — use SMPTE_RP_431_2 (=11)
    int AVIF_COLOR_PRIMARIES_SMPTE_RP_431_2 = 11  # = Display P3 (D65)

    int AVIF_TRANSFER_CHARACTERISTICS_BT709 = 1
    int AVIF_TRANSFER_CHARACTERISTICS_UNSPECIFIED = 2
    int AVIF_TRANSFER_CHARACTERISTICS_LINEAR = 8
    int AVIF_TRANSFER_CHARACTERISTICS_SRGB = 13
    int AVIF_TRANSFER_CHARACTERISTICS_SMPTE2084 = 16  # PQ
    int AVIF_TRANSFER_CHARACTERISTICS_HLG = 18

    int AVIF_MATRIX_COEFFICIENTS_IDENTITY = 0  # used for lossless
    int AVIF_MATRIX_COEFFICIENTS_BT709 = 1
    int AVIF_MATRIX_COEFFICIENTS_UNSPECIFIED = 2
    int AVIF_MATRIX_COEFFICIENTS_BT2020_NCL = 9

cnp.import_array()


class AvifError(RuntimeError):
    """Raised on AVIF encode/decode failures."""


def encode(data, *, level: int | None = None,
           lossless: bool = False, speed: int = 6,
           color=None, bit_depth: int | None = None,
           numthreads: int | None = None) -> bytes:
    """Encode an array as AVIF.

    Parameters
    ----------
    data : ndarray
        2-D grayscale, 3-D HxWx3 (RGB), or 3-D HxWx4 (RGBA). uint8 or uint16.
    level : int, optional
        Quality 0-100 (default 60); ignored if ``lossless=True``.
    lossless : bool, default False
        If True, encode in mathematically lossless mode (YUV444 + identity
        matrix). Required for fidelity-critical use; ~2-4x larger than lossy.
    speed : int, default 6
        Encoder speed 0-10 (lower = slower / smaller files).
    color : str or ColorSpec, optional
        Color-encoding spec. Same vocabulary as the JXL codec accepts:
        'srgb', 'display-p3', 'rec2020-pq', 'rec2020-hlg', etc. If None,
        libavif's defaults are used (typically BT.709 sRGB).
    bit_depth : int, optional
        Override bit depth (8, 10, 12). Default: 8 for uint8 input, 10 for
        uint16 input. uint16 with bit_depth=10 means values 0..1023 are
        stored in the low bits; values >1023 are clamped.
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
        int dtype_bytes  # 1 for uint8, 2 for uint16
        int actual_bit_depth
        size_t row_bytes_in
        unsigned int y

    # Accept uint8 or uint16 input.
    if not isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data, dtype=np.uint8)
    else:
        if data.dtype == np.uint8:
            arr = np.ascontiguousarray(data)
        elif data.dtype == np.uint16:
            arr = np.ascontiguousarray(data)
        else:
            raise AvifError(
                f'AVIF: uint8 or uint16 input supported, got {data.dtype}')

    dtype_bytes = 1 if arr.dtype == np.uint8 else 2

    # Default bit_depth from dtype.
    if bit_depth is None:
        actual_bit_depth = 8 if dtype_bytes == 1 else 10
    else:
        actual_bit_depth = int(bit_depth)
    if actual_bit_depth not in (8, 10, 12):
        raise AvifError(
            f'AVIF: bit_depth must be 8, 10, or 12 (got {actual_bit_depth})')
    if dtype_bytes == 1 and actual_bit_depth != 8:
        raise AvifError(
            f'AVIF: uint8 input requires bit_depth=8 (got {actual_bit_depth})')

    if arr.ndim == 2:
        arr = np.ascontiguousarray(np.stack([arr] * 3, axis=-1))
        has_alpha = 0
    elif arr.ndim == 3 and arr.shape[2] == 3:
        has_alpha = 0
    elif arr.ndim == 3 and arr.shape[2] == 4:
        has_alpha = 1
    else:
        raise AvifError(f'AVIF encode: unsupported shape ndim={arr.ndim}')

    # Resolve color spec to CICP values.
    cdef int cp = AVIF_COLOR_PRIMARIES_UNSPECIFIED
    cdef int tc = AVIF_TRANSFER_CHARACTERISTICS_UNSPECIFIED
    cdef int mc = AVIF_MATRIX_COEFFICIENTS_UNSPECIFIED
    if color is not None:
        from opencodecs.core.color import parse_color
        spec = parse_color(color)
        # JxlPrimaries / JxlTransferFunction enums are CICP-aligned.
        # JXL primary 11 = P3 -> AVIF SMPTE_RP_431_2 (also 11).
        cp = int(spec.primaries)
        tc = int(spec.transfer)

    quality = AVIF_QUALITY_LOSSLESS if lossless else (60 if level is None else int(level))
    if quality < 0: quality = 0
    if quality > 100: quality = 100

    yuv_format = AVIF_PIXEL_FORMAT_YUV444 if lossless else AVIF_PIXEL_FORMAT_YUV420

    image = avifImageCreate(<unsigned int> arr.shape[1],
                            <unsigned int> arr.shape[0],
                            <unsigned int> actual_bit_depth, yuv_format)
    if image == NULL:
        raise AvifError('avifImageCreate failed')

    # Set color encoding. Identity matrix is REQUIRED for byte-perfect
    # lossless (YUV planes equal RGB planes); the colorPrimaries and
    # transferCharacteristics still tag the colorimetry of those values.
    if lossless:
        mc = AVIF_MATRIX_COEFFICIENTS_IDENTITY
    elif cp == AVIF_COLOR_PRIMARIES_BT2020:
        mc = AVIF_MATRIX_COEFFICIENTS_BT2020_NCL
    image.colorPrimaries = <unsigned int> cp
    image.transferCharacteristics = <unsigned int> tc
    image.matrixCoefficients = <unsigned int> mc

    encoder = avifEncoderCreate()
    if encoder == NULL:
        avifImageDestroy(image)
        raise AvifError('avifEncoderCreate failed')

    out_data.data = NULL
    out_data.size = 0

    try:
        avifRGBImageSetDefaults(&rgb, image)
        rgb.format = AVIF_RGB_FORMAT_RGBA if has_alpha else AVIF_RGB_FORMAT_RGB
        rgb.depth = <unsigned int> actual_bit_depth

        rc = avifRGBImageAllocatePixels(&rgb)
        if rc != AVIF_RESULT_OK:
            raise AvifError(
                f'avifRGBImageAllocatePixels: '
                f'{avifResultToString(rc).decode()}')
        try:
            channels = 4 if has_alpha else 3
            # uint16 input = 2 bytes per sample; uint8 = 1 byte.
            row_bytes_in = <size_t>(<int> arr.shape[1] * channels * dtype_bytes)
            # Sanity: for uint16 input, libavif expects values left-aligned to
            # bit_depth's range (e.g. for 10-bit, values 0..1023). We DON'T
            # auto-shift here — caller is responsible for ensuring values are
            # in [0, 2^bit_depth - 1].
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
        if numthreads is None or numthreads <= 0:
            import os as _os
            encoder.maxThreads = _os.cpu_count() or 4
        else:
            encoder.maxThreads = int(numthreads)

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


def decode(data, *, numthreads: int | None = None, out=None) -> np.ndarray:
    """Decode AVIF bytes to a numpy array.

    Returns uint8 for 8-bit AVIFs, uint16 for 10/12-bit AVIFs (values
    left-aligned to bit_depth — i.e. for 10-bit the array contains values
    0..1023, not shifted into the upper bits).

    ``out=`` is a preallocated ``(H, W, 3) | (H, W, 4)`` ndarray of the
    right dtype (uint8 / uint16). libavif allocates its own RGB buffer
    internally; out= skips the second allocation that the default path
    does (we still pay the libavif internal one). See ``_png.decode``
    for the full contract.
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        avifDecoder* decoder = NULL
        avifImage* image = NULL
        avifRGBImage rgb
        int rc
        cnp.ndarray out_arr
        cnp.npy_intp shape[3]
        int has_alpha
        int channels
        int dtype_bytes
        int img_depth
        unsigned int y
        size_t row_bytes_out
        tuple expected_shape
        object expected_dtype

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
    if numthreads is None or numthreads <= 0:
        import os as _os
        decoder.maxThreads = _os.cpu_count() or 4
    else:
        decoder.maxThreads = int(numthreads)

    try:
        with nogil:
            rc = avifDecoderReadMemory(decoder, image, &src[0], srcsize)
        if rc != AVIF_RESULT_OK:
            raise AvifError(
                f'avifDecoderReadMemory: {avifResultToString(rc).decode()}')

        img_depth = <int> image.depth
        dtype_bytes = 1 if img_depth <= 8 else 2

        avifRGBImageSetDefaults(&rgb, image)
        has_alpha = 1 if image.alphaPlane != NULL else 0
        channels = 4 if has_alpha else 3
        rgb.format = AVIF_RGB_FORMAT_RGBA if has_alpha else AVIF_RGB_FORMAT_RGB
        rgb.depth = <unsigned int> img_depth

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
            expected_shape = (int(image.height), int(image.width), channels)
            expected_dtype = np.uint8 if dtype_bytes == 1 else np.uint16

            if out is not None:
                if not isinstance(out, np.ndarray):
                    raise TypeError(
                        f"avif decode: out= must be an ndarray, "
                        f"got {type(out).__name__}")
                if out.shape != expected_shape:
                    raise ValueError(
                        f"avif decode: out= shape {out.shape} does not "
                        f"match expected {expected_shape}")
                if out.dtype != expected_dtype:
                    raise ValueError(
                        f"avif decode: out= dtype {out.dtype} does not "
                        f"match expected {np.dtype(expected_dtype)}")
                if not out.flags['C_CONTIGUOUS']:
                    raise ValueError("avif decode: out= must be C-contiguous")
                out_arr = out
            elif dtype_bytes == 1:
                out_arr = cnp.PyArray_EMPTY(3, shape, cnp.NPY_UINT8, 0)
            else:
                out_arr = cnp.PyArray_EMPTY(3, shape, cnp.NPY_UINT16, 0)
            row_bytes_out = <size_t>(image.width * channels * dtype_bytes)
            for y in range(image.height):
                memcpy(<uint8_t*> cnp.PyArray_DATA(out_arr) + y * row_bytes_out,
                       rgb.pixels + y * rgb.rowBytes,
                       row_bytes_out)
            return out_arr
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
