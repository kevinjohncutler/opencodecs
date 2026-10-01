# opencodecs/codecs/_avif.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native AVIF codec via libavif (linked against system aom)."""

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.bytearray cimport PyByteArray_AS_STRING
from libc.string cimport memcpy, memset
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t

import numpy as np
cimport numpy as cnp

from avif cimport (
    AVIF_QUALITY_LOSSLESS, AVIF_QUALITY_DEFAULT, AVIF_RESULT_OK,
    AVIF_RESULT_IO_ERROR,
    avifIO, avifROData, avifResult, avifDecoderSetIO,
    AVIF_PIXEL_FORMAT_YUV444, AVIF_PIXEL_FORMAT_YUV422,
    AVIF_PIXEL_FORMAT_YUV420, AVIF_PIXEL_FORMAT_YUV400,
    AVIF_RGB_FORMAT_RGB, AVIF_RGB_FORMAT_RGBA,
    AVIF_RGB_FORMAT_GRAY, AVIF_RGB_FORMAT_GRAYA,
    AVIF_CODEC_CHOICE_AUTO, AVIF_CODEC_CHOICE_AOM, AVIF_CODEC_CHOICE_SVT,
    avifPixelFormat, avifCodecChoice,
    avifImage, avifImageCreate, avifImageCreateEmpty, avifImageDestroy,
    avifRGBImage, avifRGBImageSetDefaults,
    avifRGBImageAllocatePixels, avifRGBImageFreePixels,
    avifImageRGBToYUV, avifImageYUVToRGB,
    avifEncoder, avifEncoderCreate, avifEncoderDestroy, avifEncoderWrite,
    avifEncoderSetCodecSpecificOption,
    avifDecoder, avifDecoderCreate, avifDecoderDestroy, avifDecoderReadMemory,
    avifDecoderSetIOMemory, avifDecoderParse,
    avifDecoderNextImage, avifDecoderNthImage,
    avifRWData, avifRWDataFree,
    avifResultToString,
    avifImageSetProfileICC,
)
from cpython.bytes cimport PyBytes_FromStringAndSize


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
    int AVIF_MATRIX_COEFFICIENTS_BT601 = 6
    int AVIF_MATRIX_COEFFICIENTS_BT2020_NCL = 9

cnp.import_array()


class AvifError(RuntimeError):
    """Raised on AVIF encode/decode failures."""


_YUV_FORMATS = {
    '444': AVIF_PIXEL_FORMAT_YUV444, 'yuv444': AVIF_PIXEL_FORMAT_YUV444,
    '422': AVIF_PIXEL_FORMAT_YUV422, 'yuv422': AVIF_PIXEL_FORMAT_YUV422,
    '420': AVIF_PIXEL_FORMAT_YUV420, 'yuv420': AVIF_PIXEL_FORMAT_YUV420,
    '400': AVIF_PIXEL_FORMAT_YUV400, 'yuv400': AVIF_PIXEL_FORMAT_YUV400,
}


def encode(data, *, level: int | None = None,
           lossless: bool | None = None, speed: int | None = None,
           color=None, bit_depth: int | None = None,
           numthreads: int | None = None,
           iccprofile: bytes | None = None,
           codec: str | None = None,
           tile_cols_log2: int | None = None,
           tile_rows_log2: int | None = None,
           auto_tiling: bool = False,
           yuv_format: str | None = None,
           codec_options: dict | None = None,
           primaries: int | None = None,
           transfer: int | None = None,
           matrix: int | None = None) -> bytes:
    """Encode an array as AVIF.

    Parameters
    ----------
    data : ndarray
        (H, W) or (H, W, 1) gray, (H, W, 2) gray plus alpha, (H, W, 3)
        RGB or (H, W, 4) RGBA; uint8 or uint16. Gray is coded as AV1
        monochrome (4:0:0, ``mono_chrome=1``), which is what the format
        provides for it, and decodes to (H, W) or (H, W, 2), so
        (H, W, 1) input comes back as (H, W).
    level : int, optional
        Quality 0-100, imagecodecs' meaning: with ``lossless`` left at
        None, no level or a level of 100 is lossless and anything lower
        is lossy at that quality. A level of -1 or lower is lossy at
        libavif's own default quality (AVIF_QUALITY_DEFAULT), as in
        imagecodecs. With ``lossless=False`` and no level the quality is
        60. Unlike imagecodecs, which codes gray (1 or 2 sample) input
        lossless whatever the level, the level applies to gray too.
    lossless : bool, optional
        None (default) follows ``level`` as above. True forces lossless
        (YUV 4:4:4 with the identity matrix for color, 4:0:0 for gray)
        and raises if ``level`` asks for less. False forces lossy.
        ``level`` sets the quality of the color planes only; an alpha
        channel is always coded lossless, as imagecodecs does.
    speed : int, optional
        Encoder speed 0-10 (lower = slower / smaller files). None
        (default) leaves libavif's own default, as imagecodecs does;
        values outside 0-10 are clamped to that range, also as
        imagecodecs does.
    color : str or ColorSpec, optional
        Color-encoding spec. Same vocabulary as the JXL codec accepts:
        'srgb', 'display-p3', 'rec2020-pq', 'rec2020-hlg', etc. If None,
        the primaries and transfer stay unspecified (CICP 2).
    bit_depth : int, optional
        Coded bit depth, 8 for uint8 and 10 or 12 for uint16. For uint16
        with no ``bit_depth`` the smallest of 10 and 12 that holds the
        data's largest value is used. uint16 values are stored as they
        are, in the low bits (10-bit means 0..1023). Data that does not
        fit raises AvifError rather than being clamped: AV1 stores at
        most 12 bits, so full-range uint16 cannot be written.
    codec : {'aom', 'svt', 'auto', None}, optional
        AV1 encoder backend. ``'aom'`` = libaom (the reference encoder
        and the right default). ``'svt'`` = SVT-AV1 (Intel/Netflix).
        ``None``/``'auto'`` lets libavif pick (currently prefers aom).
        SVT is only available if libavif was built with
        ``AVIF_CODEC_SVT`` enabled; it raises ``AvifError`` otherwise.

        SVT is exposed as an escape hatch, not as a faster path. On
        Apple Silicon (M-series, libavif 1.3.0 + SVT-AV1 3.1.2) it
        measured ~7x SLOWER than aom and produced much larger files
        (speed=10, 2048x2048, 4:2:0: 262 ms / 659 KiB vs aom's 34 ms /
        78 KiB). libavif drives SVT in its video configuration rather
        than single-image mode, which is the likely cause. Benchmark
        on your own hardware before selecting it.
    tile_cols_log2, tile_rows_log2 : int, optional
        Log2 of the number of tiles per axis. ``2`` = 4 tiles per axis,
        so ``tile_cols_log2=2, tile_rows_log2=2`` splits the frame into
        16 tiles encoded by independent threads. Default: ``2`` for
        images >= 1024 px on the long axis, ``0`` otherwise (small
        images don't benefit and pay the per-tile header overhead).
        Pass an explicit integer to override. imagecodecs does not tile
        by default, so for images of 1024 px or more pass 0 for both to
        write the bytes it writes.

        Measured on a 2048x2048 RGB frame (level=80, speed=6, 4:2:0
        chroma, Apple Silicon): 186 ms untiled, 109 ms at 2x2, 92 ms at 4x4, 90 ms
        at 8x8; so 4x4 is ~2x faster than untiled and 8x8 buys nothing
        further. Size cost is +1.2% at 2x2, +4.4% at 4x4 and +8.7% at
        8x8, with PSNR unchanged, which is why 4x4 is the ceiling
        used for the default.
    auto_tiling : bool, default False
        If True, let libavif pick tile counts based on image dimensions.
        Overrides ``tile_cols_log2`` / ``tile_rows_log2``.
    yuv_format : {'420', '422', '444', '400'}, optional
        Chroma layout for lossy color input; default '444', imagecodecs'
        default, which keeps chroma at full resolution. Lossless color
        is always 4:4:4 and gray always 4:0:0, so any other value there
        raises instead of being ignored.
    primaries, transfer, matrix : int, optional
        CICP code points (ITU-T H.273) written to the file, overriding
        ``color``. ``matrix`` cannot be combined with lossless color
        input, which needs the identity matrix.

    Lossy color output is tagged with matrix coefficients 6 (BT.601),
    the matrix libavif converts with; an unspecified matrix (2) would
    leave a reader outside libavif to guess. BT.2020 primaries get the
    BT.2020 non-constant-luminance matrix (9). Gray (4:0:0) has no
    chroma, so no matrix applies to it and it is tagged unspecified
    (2), as imagecodecs tags it.
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
        int monochrome
        int quality
        int channels
        int dtype_bytes  # 1 for uint8, 2 for uint16
        int actual_bit_depth
        size_t row_bytes_in
        unsigned int y

    from opencodecs._avif_heif_params import (
        resolve_bit_depth, resolve_lossless, gray_layout)

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    if data.dtype != np.uint8 and data.dtype != np.uint16:
        raise AvifError(
            f'AVIF: uint8 or uint16 input supported, got {data.dtype}')
    layout = gray_layout(data)
    if layout is None:
        raise AvifError(
            f'AVIF encode: unsupported shape {data.shape}; expected '
            f'(H, W), (H, W, 1), (H, W, 2), (H, W, 3) or (H, W, 4)')
    channels, has_alpha = layout
    monochrome = channels <= 2
    arr = np.ascontiguousarray(data)
    dtype_bytes = 1 if arr.dtype == np.uint8 else 2

    actual_bit_depth = resolve_bit_depth(
        arr, bit_depth, name='AVIF', error=AvifError)
    lossless, quality = resolve_lossless(
        level, lossless, name='avif', lossless_from=AVIF_QUALITY_LOSSLESS,
        default_quality=60, library_default_at=AVIF_QUALITY_DEFAULT)

    # Tile defaults: ``log2=2`` (4×4 = 16 tiles) auto-enables for images
    # with the long axis >= 1024 px. Measured 3-4× wall-clock speedup
    # with negligible (<1%) size cost. Going higher (8×8, 16×16) gives
    # no further speedup on <=16-P-core machines and grows the file
    # 0.5-2% per doubling. Small images skip tiling — overhead would
    # dominate.
    if tile_cols_log2 is None or tile_rows_log2 is None:
        long_axis = max(int(arr.shape[0]), int(arr.shape[1]))
        _auto = 2 if long_axis >= 1024 else 0
        if tile_cols_log2 is None:
            tile_cols_log2 = _auto
        if tile_rows_log2 is None:
            tile_rows_log2 = _auto

    # Resolve color spec to CICP values.
    cdef int cp = AVIF_COLOR_PRIMARIES_UNSPECIFIED
    cdef int tc = AVIF_TRANSFER_CHARACTERISTICS_UNSPECIFIED
    cdef int mc = AVIF_MATRIX_COEFFICIENTS_BT601
    if color is not None:
        from opencodecs.core.color import parse_color
        spec = parse_color(color)
        # JxlPrimaries / JxlTransferFunction enums are CICP-aligned.
        # JXL primary 11 = P3 -> AVIF SMPTE_RP_431_2 (also 11).
        cp = int(spec.primaries)
        tc = int(spec.transfer)
    if primaries is not None:
        cp = int(primaries)
    if transfer is not None:
        tc = int(transfer)

    # Chroma layout. Gray is always 4:0:0 (AV1 mono_chrome); there is
    # no chroma to lay out. Color is 4:4:4 unless the caller asks for
    # subsampling: always for lossless (subsampling is lossy by
    # definition), and by default for lossy too, imagecodecs' default.
    # 4:4:4 keeps chroma at full luma resolution, which matters for
    # sharp edges and sparse-bright content (fluorescence dye spots,
    # scientific plots) where 4:2:0 visibly bleeds.
    cdef avifPixelFormat _yuv
    if yuv_format is not None and str(yuv_format).lower() not in _YUV_FORMATS:
        raise AvifError(
            f"yuv_format must be '420'/'422'/'444'/'400' (got {yuv_format!r})")
    if monochrome:
        if yuv_format is not None and _YUV_FORMATS[str(yuv_format).lower()] != AVIF_PIXEL_FORMAT_YUV400:
            raise AvifError(
                f"AVIF encode: gray input is coded as 4:0:0; "
                f"yuv_format={yuv_format!r} applies to color input only")
        _yuv = AVIF_PIXEL_FORMAT_YUV400
    elif lossless:
        if yuv_format is not None and _YUV_FORMATS[str(yuv_format).lower()] != AVIF_PIXEL_FORMAT_YUV444:
            raise AvifError(
                f"AVIF encode: lossless color is coded as 4:4:4; "
                f"yuv_format={yuv_format!r} would subsample it")
        _yuv = AVIF_PIXEL_FORMAT_YUV444
    elif yuv_format is None:
        _yuv = AVIF_PIXEL_FORMAT_YUV444
    else:
        _yuv = _YUV_FORMATS[str(yuv_format).lower()]

    # Matrix. Identity is REQUIRED for byte-perfect lossless color (YUV
    # planes equal RGB planes); the primaries and transfer still tag the
    # colorimetry of those values. AV1 allows the identity matrix only
    # with 4:4:4, so gray (4:0:0) is tagged unspecified: its single
    # plane has no chroma for a matrix to act on, and imagecodecs tags
    # it the same way. Lossy color is tagged with the matrix libavif
    # converts with.
    if monochrome and matrix is None:
        mc = AVIF_MATRIX_COEFFICIENTS_UNSPECIFIED
    elif lossless and not monochrome:
        if matrix is not None and int(matrix) != AVIF_MATRIX_COEFFICIENTS_IDENTITY:
            raise AvifError(
                f"AVIF encode: lossless color needs the identity matrix "
                f"(0); matrix={matrix} would make it lossy")
        mc = AVIF_MATRIX_COEFFICIENTS_IDENTITY
    elif matrix is not None:
        mc = int(matrix)
    elif cp == AVIF_COLOR_PRIMARIES_BT2020:
        mc = AVIF_MATRIX_COEFFICIENTS_BT2020_NCL

    image = avifImageCreate(<unsigned int> arr.shape[1],
                            <unsigned int> arr.shape[0],
                            <unsigned int> actual_bit_depth, _yuv)
    if image == NULL:
        raise AvifError('avifImageCreate failed')

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
        if monochrome:
            rgb.format = AVIF_RGB_FORMAT_GRAYA if has_alpha else AVIF_RGB_FORMAT_GRAY
        else:
            rgb.format = AVIF_RGB_FORMAT_RGBA if has_alpha else AVIF_RGB_FORMAT_RGB
        rgb.depth = <unsigned int> actual_bit_depth

        rc = avifRGBImageAllocatePixels(&rgb)
        if rc != AVIF_RESULT_OK:
            raise AvifError(
                f'avifRGBImageAllocatePixels: '
                f'{avifResultToString(rc).decode()}')
        try:
            # uint16 input = 2 bytes per sample; uint8 = 1 byte. Values
            # sit in the low bits of each sample (10-bit is 0..1023),
            # which resolve_bit_depth has already checked.
            row_bytes_in = <size_t>(<int> arr.shape[1] * channels * dtype_bytes)
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
        # Alpha is coded lossless whatever the color quality, as
        # imagecodecs codes it: level trades color fidelity for size,
        # and a mask or transparency channel is not color.
        encoder.qualityAlpha = AVIF_QUALITY_LOSSLESS
        if speed is not None:
            # imagecodecs clamps an out-of-range speed rather than
            # dropping it; None keeps libavif's default.
            encoder.speed = max(0, min(10, int(speed)))
        if numthreads is None or numthreads <= 0:
            import os as _os
            encoder.maxThreads = _os.cpu_count() or 4
        else:
            encoder.maxThreads = int(numthreads)
        # Backend selection — must be set BEFORE avifEncoderWrite. libavif
        # falls back to AVIF_CODEC_CHOICE_AUTO (= libaom for encode) if
        # the requested backend wasn't compiled in.
        if codec is None or codec == 'auto':
            encoder.codecChoice = AVIF_CODEC_CHOICE_AUTO
        elif codec == 'aom':
            encoder.codecChoice = AVIF_CODEC_CHOICE_AOM
        elif codec == 'svt':
            encoder.codecChoice = AVIF_CODEC_CHOICE_SVT
        else:
            raise AvifError(
                f"unknown AVIF codec: {codec!r} "
                f"(expected 'aom', 'svt', 'auto', or None)")
        if auto_tiling:
            encoder.autoTiling = 1
        else:
            encoder.autoTiling = 0
            if tile_cols_log2 > 0:
                encoder.tileColsLog2 = int(tile_cols_log2)
            if tile_rows_log2 > 0:
                encoder.tileRowsLog2 = int(tile_rows_log2)

        # Backend-specific options, passed straight through to the
        # encoder (libaom: 'enable-cdef', 'enable-restoration', 'row-mt',
        # 'aq-mode', 'tune', 'sharpness', 'enable-tpl-model',
        # 'deltaq-mode', ...). Names must match the backend exactly; an
        # unknown key raises AvifError. Note 'cdef' is NOT a libaom
        # option name, the toggle is 'enable-cdef'.
        #
        # This is an escape hatch, not a tuning recipe. Disabling the
        # post-process filters measured 0.83x (slower) at speed=10 on a
        # 2048x2048 frame with identical output size, so do not reach
        # for it without benchmarking your own workload.
        if codec_options is not None:
            for _k, _v in codec_options.items():
                _kb = str(_k).encode()
                _vb = str(_v).encode()
                rc = avifEncoderSetCodecSpecificOption(
                    encoder, <const char*> _kb, <const char*> _vb)
                if rc != AVIF_RESULT_OK:
                    raise AvifError(
                        f"avifEncoderSetCodecSpecificOption({_k!r}={_v!r}): "
                        f"{avifResultToString(rc).decode()}")

        if iccprofile is not None and len(iccprofile) > 0:
            _icc_bytes = bytes(iccprofile)
            avifImageSetProfileICC(
                image,
                <const uint8_t*> <const char*> _icc_bytes,
                <size_t> len(_icc_bytes),
            )

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


#: Most threads an AVIF decode uses when the caller leaves it to us. AV1
#: decodes in parallel across tiles, so threads only help a tiled image,
#: and there they stop helping around 8. Measured on a 64-core x86-64 host,
#: 2048 x 2048: untiled 189 ms on 1 thread, 189 on 8 and 199 on one thread
#: per hardware thread; tiled 129 ms on 1, 75 on 8 and 83 on all of them.
_AVIF_DECODE_MAX_THREADS = 8


def _decode_threads(numthreads):
    """numthreads None: up to 8, within this call's fair share; <= 0: every core."""
    if numthreads is not None and numthreads <= 0:
        import os as _os
        return _os.cpu_count() or 4
    from opencodecs.core.parallel import auto_threads
    return auto_threads(numthreads, max_threads=_AVIF_DECODE_MAX_THREADS)


cdef int _rgb_layout(avifImage* image, avifRGBImage* rgb) noexcept:
    """Pick the output format for ``image``; return its sample count.

    A 4:0:0 image (AV1 ``mono_chrome=1``) has a single plane, and it is
    returned as gray, (H, W), or gray plus alpha, (H, W, 2), not as three
    copies of the same plane. libavif's GRAY formats apply the file's
    range, so a limited-range gray file still decodes to full range.
    """
    cdef bint alpha = image.alphaPlane != NULL
    if image.yuvFormat == AVIF_PIXEL_FORMAT_YUV400:
        rgb.format = AVIF_RGB_FORMAT_GRAYA if alpha else AVIF_RGB_FORMAT_GRAY
        return 2 if alpha else 1
    rgb.format = AVIF_RGB_FORMAT_RGBA if alpha else AVIF_RGB_FORMAT_RGB
    return 4 if alpha else 3


def decode(data, *, numthreads: int | None = None, out=None) -> np.ndarray:
    """Decode AVIF bytes to a numpy array.

    Returns uint8 for 8-bit AVIFs, uint16 for 10/12-bit AVIFs (values in
    the low bits: for 10-bit the array contains values 0..1023, not
    shifted into the upper bits).

    The shape is (H, W, 3) for RGB, (H, W, 4) for RGBA, and for a
    monochrome (4:0:0) file (H, W), or (H, W, 2) with alpha, as
    imagecodecs returns them.

    Lossy files can decode a level or two apart from imagecodecs on the
    same bytes. The AV1 decode is bit-exact; the difference is the YUV
    to RGB step, where this build uses libavif's built-in converter and
    the imagecodecs build uses libyuv. ITU-T H.273 gives that step as
    real-valued equations, and libavif's converter matches their
    rounded result. Lossless files decode identically.

    ``out=`` is a preallocated ndarray of the decoded shape and dtype
    (uint8 / uint16). libavif allocates its own RGB buffer internally;
    out= skips the second allocation that the default path does (we
    still pay the libavif internal one). See ``_png.decode`` for the
    full contract.
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
    decoder.maxThreads = _decode_threads(numthreads)

    from opencodecs.core.parallel import parallel_call
    try:
        with parallel_call():
            with nogil:
                rc = avifDecoderReadMemory(decoder, image, &src[0], srcsize)
        if rc != AVIF_RESULT_OK:
            raise AvifError(
                f'avifDecoderReadMemory: {avifResultToString(rc).decode()}')

        img_depth = <int> image.depth
        dtype_bytes = 1 if img_depth <= 8 else 2

        avifRGBImageSetDefaults(&rgb, image)
        channels = _rgb_layout(image, &rgb)
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
            if channels == 1:
                expected_shape = (int(image.height), int(image.width))
            else:
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
            else:
                out_arr = cnp.PyArray_EMPTY(
                    2 if channels == 1 else 3, shape,
                    cnp.NPY_UINT8 if dtype_bytes == 1 else cnp.NPY_UINT16, 0)
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


cdef avifResult _source_read(avifIO* io, uint32_t flags, uint64_t offset,
                            size_t size, avifROData* output) noexcept with gil:
    source = <object>io.data
    cdef const uint8_t[::1] piece
    cdef char* buffer
    cdef size_t copied = 0
    try:
        if flags or offset > source.size:
            raise ValueError("invalid AVIF source range")
        size = min(size, source.size - offset)
        source._avif_buffer = bytearray(size)
        buffer = PyByteArray_AS_STRING(source._avif_buffer)
        while copied < size:
            piece = source.read_at(offset + copied, size - copied)
            memcpy(buffer + copied, &piece[0], piece.shape[0])
            copied += piece.shape[0]
        output.data = <const uint8_t*>buffer
        output.size = size
        return AVIF_RESULT_OK
    except BaseException as error:
        source.error = error
        return AVIF_RESULT_IO_ERROR


cdef class AvifSequence:
    """An open AVIF decoder, for files that hold more than one image.

    AVIF carries image sequences (the same machinery as AV1 video) and
    progressive layers, both reached through avifDecoderNthImage. We
    only ever decoded the primary image, so an animated AVIF read back
    as its first frame with no indication that the rest existed.

    The decoder is kept open across frames deliberately. Re-reading the
    file per frame would re-parse the container every time and turn
    reading N frames into N parses; here the parse happens once and a
    frame is a seek in the already-built sample table.

    Two ownership rules the C API imposes, both load-bearing:
    avifDecoderSetIOMemory does not copy, so ``_buf`` holds the input
    alive for the decoder's life; and ``decoder.image`` is owned by the
    decoder and its contents are replaced by the next NthImage call, so
    every frame is copied out before returning.
    """
    cdef avifDecoder* _decoder
    cdef object _buf
    cdef object _source
    cdef avifIO _io
    cdef int _default_threads
    cdef readonly int n_frames
    cdef readonly int width
    cdef readonly int height
    cdef readonly int depth
    cdef readonly bint has_alpha
    cdef readonly bint monochrome
    cdef readonly double duration

    def __cinit__(self, data, numthreads: int | None = None):
        cdef const uint8_t[::1] src
        cdef size_t srcsize
        cdef int rc

        self._decoder = NULL
        # Held for the decoder's lifetime: SetIOMemory borrows it.
        if hasattr(data, "read_at"):
            self._source = data
            self._source.reset()
        else:
            self._buf = data if isinstance(data, bytes) else bytes(data)
            src = self._buf
            srcsize = <size_t> src.shape[0]

        self._decoder = avifDecoderCreate()
        if self._decoder == NULL:
            raise AvifError('avifDecoderCreate failed')
        self._decoder.maxThreads = _decode_threads(numthreads)

        self._default_threads = self._decoder.maxThreads
        if self._source is not None:
            memset(&self._io, 0, sizeof(avifIO))
            self._io.read = _source_read
            self._io.sizeHint = self._source.size
            self._io.persistent = 0
            self._io.data = <void*>self._source
            avifDecoderSetIO(self._decoder, &self._io)
            rc = AVIF_RESULT_OK
        else:
            rc = avifDecoderSetIOMemory(self._decoder, &src[0], srcsize)
        if rc != AVIF_RESULT_OK:
            raise AvifError(
                f'avifDecoderSetIOMemory: {avifResultToString(rc).decode()}')
        with nogil:
            rc = avifDecoderParse(self._decoder)
        if self._source is not None:
            self._source.raise_error()
        if rc != AVIF_RESULT_OK:
            raise AvifError(
                f'avifDecoderParse: {avifResultToString(rc).decode()}')

        self.n_frames = self._decoder.imageCount
        self.width = <int> self._decoder.image.width
        self.height = <int> self._decoder.image.height
        self.depth = <int> self._decoder.image.depth
        # alphaPresent is the field to read before any frame is
        # decoded; image.alphaPlane does not exist yet at this point.
        self.has_alpha = self._decoder.alphaPresent != 0
        # The parse fills yuvFormat from the av1C box, so a monochrome
        # file is known before any frame is decoded.
        self.monochrome = (
            self._decoder.image.yuvFormat == AVIF_PIXEL_FORMAT_YUV400)
        self.duration = (
            <double> self._decoder.durationInTimescales
            / <double> self._decoder.timescale
        ) if self._decoder.timescale else 0.0

    def __dealloc__(self):
        if self._decoder != NULL:
            avifDecoderDestroy(self._decoder)
            self._decoder = NULL

    def frame(self, int index, *, numthreads=None):
        """Decode frame ``index`` and copy it out as an ndarray."""
        cdef int rc
        cdef unsigned int idx
        cdef avifRGBImage rgb
        cdef cnp.ndarray out_arr
        cdef cnp.npy_intp shape[3]
        cdef int channels, dtype_bytes, y
        cdef size_t row_bytes_out
        cdef avifImage* image

        if self._decoder == NULL:
            raise ValueError("AVIF decoder is closed")
        self._decoder.maxThreads = self._default_threads if numthreads is None else max(1, int(numthreads))
        if index < 0:
            index += self.n_frames
        if not 0 <= index < self.n_frames:
            raise IndexError(
                f'avif: frame {index} out of range for {self.n_frames}')
        idx = <unsigned int> index
        with nogil:
            rc = avifDecoderNthImage(self._decoder, idx)
        if self._source is not None:
            self._source.raise_error()
        if rc != AVIF_RESULT_OK:
            raise AvifError(
                f'avifDecoderNthImage({index}): '
                f'{avifResultToString(rc).decode()}')

        image = self._decoder.image
        dtype_bytes = 1 if image.depth <= 8 else 2
        avifRGBImageSetDefaults(&rgb, image)
        channels = _rgb_layout(image, &rgb)
        rgb.depth = <unsigned int> image.depth
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
                    f'avifImageYUVToRGB: {avifResultToString(rc).decode()}')
            shape[0] = image.height
            shape[1] = image.width
            shape[2] = channels
            out_arr = cnp.PyArray_EMPTY(
                2 if channels == 1 else 3, shape,
                cnp.NPY_UINT8 if dtype_bytes == 1 else cnp.NPY_UINT16, 0)
            row_bytes_out = <size_t>(image.width * channels * dtype_bytes)
            for y in range(image.height):
                memcpy(
                    <uint8_t*> cnp.PyArray_DATA(out_arr) + y * row_bytes_out,
                    rgb.pixels + y * rgb.rowBytes, row_bytes_out)
            return out_arr
        finally:
            avifRGBImageFreePixels(&rgb)


def frame_count(data) -> int:
    """How many images an AVIF holds; 1 for a plain still.

    Parses the container and nothing else, so this does not pay for a
    frame decode just to answer the question.
    """
    return AvifSequence(data, numthreads=1).n_frames


def read_icc_profile(data) -> bytes | None:
    """Return the embedded ICC profile bytes from an AVIF, or ``None``."""
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        avifDecoder* decoder = NULL
        avifImage* image = NULL
        int rc
        bytes out

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if srcsize < 12:
        return None
    decoder = avifDecoderCreate()
    if decoder == NULL:
        raise AvifError('avifDecoderCreate failed')
    image = avifImageCreateEmpty()
    if image == NULL:
        avifDecoderDestroy(decoder)
        raise AvifError('avifImageCreateEmpty failed')
    try:
        rc = avifDecoderReadMemory(decoder, image, &src[0], srcsize)
        if rc != AVIF_RESULT_OK:
            return None
        if image.icc.data == NULL or image.icc.size == 0:
            return None
        out = PyBytes_FromStringAndSize(
            <char*> image.icc.data, <Py_ssize_t> image.icc.size)
        return out
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
