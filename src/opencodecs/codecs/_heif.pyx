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
from libc.stdint cimport uint8_t, uint16_t

import numpy as np
cimport numpy as cnp

from heif cimport (
    heif_init, heif_context, heif_context_alloc, heif_context_free,
    heif_error_code, heif_chroma,
    heif_context_read_from_memory_without_copy,
    heif_context_get_primary_image_handle,
    heif_image_handle, heif_image_handle_release,
    heif_image_handle_get_width, heif_image_handle_get_height,
    heif_image_handle_has_alpha_channel,
    heif_image_handle_get_luma_bits_per_pixel,
    heif_decode_image, heif_image, heif_image_release,
    heif_image_get_plane_readonly, heif_image_get_plane,
    heif_image_create, heif_image_add_plane,
    heif_colorspace_RGB,
    heif_chroma_interleaved_RGB, heif_chroma_interleaved_RGBA,
    heif_chroma_interleaved_RRGGBB_LE, heif_chroma_interleaved_RRGGBBAA_LE,
    heif_channel_interleaved,
    heif_compression_HEVC,
    heif_context_get_encoder_for_format,
    heif_encoder, heif_encoder_release,
    heif_encoder_set_lossy_quality, heif_encoder_set_lossless,
    heif_encoder_set_parameter_string,
    heif_encoder_set_parameter_integer,
    heif_context_set_max_decoding_threads,
    heif_context_encode_image,
    heif_writer, heif_context_write,
    heif_error,
    heif_color_profile_nclx,
    heif_nclx_color_profile_alloc, heif_nclx_color_profile_free,
    heif_nclx_color_profile_set_color_primaries,
    heif_nclx_color_profile_set_transfer_characteristics,
    heif_nclx_color_profile_set_matrix_coefficients,
    heif_image_set_nclx_color_profile,
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


def decode(data, *, numthreads: int | None = None) -> np.ndarray:
    """Decode HEIF/HEIC bytes to a numpy array.

    Returns uint8 for 8-bit HEIFs, uint16 for 10/12-bit HEIFs (values
    left-aligned to the source bit_depth — i.e. for 10-bit the array
    contains values 0..1023, not shifted into the upper bits).

    Parameters
    ----------
    numthreads : int, optional
        Max worker threads for the HEVC decoder. ``None`` (default)
        leaves libheif's compile-time default (typically 4). ``0`` or
        ``1`` forces single-threaded. Larger values give near-linear
        speedup on 4K+ images.
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        heif_context* ctx = NULL
        heif_image_handle* handle = NULL
        heif_image* img = NULL
        heif_error err
        int width, height, channels, has_alpha, stride
        int img_depth
        int dtype_bytes
        heif_chroma chroma
        const uint8_t* plane
        cnp.ndarray out
        cnp.npy_intp shape[3]
        int y
        int _heif_n
        size_t row_bytes_out

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
        if numthreads is not None:
            _heif_n = int(numthreads)
            if _heif_n < 1: _heif_n = 1
            heif_context_set_max_decoding_threads(ctx, _heif_n)
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

        # Probe luma bit depth (libheif 1.4+).
        img_depth = heif_image_handle_get_luma_bits_per_pixel(handle)
        if img_depth <= 0:
            img_depth = 8
        dtype_bytes = 1 if img_depth <= 8 else 2

        if dtype_bytes == 1:
            chroma = (heif_chroma_interleaved_RGBA if has_alpha
                      else heif_chroma_interleaved_RGB)
        else:
            chroma = (heif_chroma_interleaved_RRGGBBAA_LE if has_alpha
                      else heif_chroma_interleaved_RRGGBB_LE)

        with nogil:
            err = heif_decode_image(
                handle, &img, heif_colorspace_RGB, chroma, NULL,
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
        if dtype_bytes == 1:
            out = cnp.PyArray_EMPTY(3, shape, cnp.NPY_UINT8, 0)
        else:
            out = cnp.PyArray_EMPTY(3, shape, cnp.NPY_UINT16, 0)
        row_bytes_out = <size_t>(width * channels * dtype_bytes)
        for y in range(height):
            memcpy(<uint8_t*> cnp.PyArray_DATA(out) + y * row_bytes_out,
                   plane + y * stride,
                   row_bytes_out)
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
           lossless: bool = False, color=None,
           bit_depth: int | None = None,
           numthreads: int | None = None) -> bytes:
    """Encode an array as HEIC.

    Parameters
    ----------
    data : ndarray
        2-D grayscale, 3-D HxWx3 (RGB), or 3-D HxWx4 (RGBA). uint8 or uint16.
    level : int, optional
        Quality 0-100 (default 50); ignored if ``lossless=True``.
    lossless : bool, default False
        If True, encode in lossless mode where the HEVC encoder supports it.
    color : str or ColorSpec, optional
        Color-encoding spec. Same vocabulary as the JXL/AVIF codecs accept:
        'srgb', 'display-p3', 'rec2020-pq', 'rec2020-hlg', etc. Writes an
        NCLX colr box. If None, no NCLX is written (Apple typically defaults
        to sRGB).
    bit_depth : int, optional
        Override bit depth (8, 10, 12). Default: 8 for uint8 input, 10 for
        uint16 input. uint16 values must be left-aligned within bit_depth's
        range (e.g. for 10-bit: values 0..1023, not shifted to upper bits).
    numthreads : int, optional
        Worker threads for the HEVC encoder (``threads`` parameter on the
        x265 / kvazaar plugin). ``None`` (default) leaves the encoder
        plugin's own default. Typical 2-6× speedup on 4K+ encodes.
    """
    cdef:
        cnp.ndarray arr
        heif_context* ctx = NULL
        heif_encoder* enc = NULL
        heif_image* img = NULL
        heif_image_handle* handle = NULL
        heif_color_profile_nclx* nclx = NULL
        heif_error err
        write_buffer wbuf
        heif_writer wr
        int width, height
        int has_alpha, channels
        int dtype_bytes
        int actual_bit_depth
        uint8_t* plane
        int stride
        bytes out
        unsigned int y
        size_t row_bytes_in
        heif_chroma chroma

    _ensure_init()

    if not isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data, dtype=np.uint8)
    else:
        if data.dtype == np.uint8:
            arr = np.ascontiguousarray(data)
        elif data.dtype == np.uint16:
            arr = np.ascontiguousarray(data)
        else:
            raise HeifError(
                f'HEIF: uint8 or uint16 input supported, got {data.dtype}')

    dtype_bytes = 1 if arr.dtype == np.uint8 else 2

    if bit_depth is None:
        actual_bit_depth = 8 if dtype_bytes == 1 else 10
    else:
        actual_bit_depth = int(bit_depth)
    if actual_bit_depth not in (8, 10, 12):
        raise HeifError(
            f'HEIF: bit_depth must be 8, 10, or 12 (got {actual_bit_depth})')
    if dtype_bytes == 1 and actual_bit_depth != 8:
        raise HeifError(
            f'HEIF: uint8 input requires bit_depth=8 (got {actual_bit_depth})')

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

    # Chroma layout: 8-bit interleaved, or 16-bit little-endian for HDR.
    if dtype_bytes == 1:
        chroma = (heif_chroma_interleaved_RGBA if has_alpha
                  else heif_chroma_interleaved_RGB)
    else:
        chroma = (heif_chroma_interleaved_RRGGBBAA_LE if has_alpha
                  else heif_chroma_interleaved_RRGGBB_LE)

    # Resolve color spec to CICP values for NCLX.
    cdef int cp = -1
    cdef int tc = -1
    cdef int mc = -1
    if color is not None:
        from opencodecs.core.color import parse_color
        spec = parse_color(color)
        # ColorSpec primaries/transfer enums are CICP-aligned.
        cp = int(spec.primaries)
        tc = int(spec.transfer)
        # Use BT.2020 NCL matrix for BT.2020 primaries; BT.709 for others.
        if cp == 9:
            mc = 9
        else:
            mc = 1
    if lossless and cp < 0:
        # In lossless mode without an explicit color spec, force NCLX
        # with matrix_coefficients=0 (identity / "GBR"). Without this,
        # libheif's default BT.709 matrix triggers an RGB→YUV→RGB
        # transform whose integer rounding introduces ±1 LSB errors
        # in 30%+ of pixels even with chroma=4:4:4. Identity matrix
        # stores R/G/B directly into the YUV planes — true lossless.
        cp = 1     # sRGB primaries (any value works; identity matrix
                   # bypasses chromaticity transforms)
        tc = 13    # sRGB transfer (likewise — purely a tag)
        mc = 0     # IDENTITY — the bit that actually makes it lossless

    ctx = heif_context_alloc()
    if ctx == NULL:
        raise HeifError('heif_context_alloc failed')
    try:
        err = heif_context_get_encoder_for_format(
            ctx, heif_compression_HEVC, &enc)
        if err.code != 0:
            raise HeifError(
                f'get_encoder_for_format(HEVC): {err.message.decode()}')

        if numthreads is not None and int(numthreads) > 0:
            # x265 / kvazaar plugins accept a `threads` int parameter.
            # The set_parameter call returns an error if the plugin
            # doesn't expose this knob — ignore it (default behavior wins).
            heif_encoder_set_parameter_integer(
                enc, b'threads', int(numthreads))

        if lossless:
            heif_encoder_set_lossless(enc, 1)
            # x265 defaults to 4:2:0 chroma subsampling even in lossless
            # mode — force 4:4:4 to preserve every channel byte-exactly.
            heif_encoder_set_parameter_string(enc, b'chroma', b'444')
        else:
            heif_encoder_set_lossy_quality(enc, 50 if level is None else int(level))

        err = heif_image_create(
            width, height, heif_colorspace_RGB, chroma, &img)
        if err.code != 0:
            raise HeifError(f'heif_image_create: {err.message.decode()}')

        err = heif_image_add_plane(
            img, heif_channel_interleaved, width, height, actual_bit_depth)
        if err.code != 0:
            raise HeifError(f'heif_image_add_plane: {err.message.decode()}')

        # Attach NCLX color profile (writes a colr box to the HEIF container).
        if cp >= 0:
            nclx = heif_nclx_color_profile_alloc()
            if nclx == NULL:
                raise HeifError('heif_nclx_color_profile_alloc failed')
            err = heif_nclx_color_profile_set_color_primaries(
                nclx, <uint16_t> cp)
            if err.code != 0:
                raise HeifError(
                    f'nclx set_color_primaries({cp}): {err.message.decode()}')
            err = heif_nclx_color_profile_set_transfer_characteristics(
                nclx, <uint16_t> tc)
            if err.code != 0:
                raise HeifError(
                    f'nclx set_transfer({tc}): {err.message.decode()}')
            err = heif_nclx_color_profile_set_matrix_coefficients(
                nclx, <uint16_t> mc)
            if err.code != 0:
                raise HeifError(
                    f'nclx set_matrix({mc}): {err.message.decode()}')
            nclx.full_range_flag = 1
            err = heif_image_set_nclx_color_profile(img, nclx)
            if err.code != 0:
                raise HeifError(
                    f'heif_image_set_nclx_color_profile: {err.message.decode()}')

        plane = heif_image_get_plane(img, heif_channel_interleaved, &stride)
        if plane == NULL:
            raise HeifError('heif_image_get_plane returned NULL')
        row_bytes_in = <size_t>(width * channels * dtype_bytes)
        for y in range(height):
            memcpy(plane + y * stride,
                   <const uint8_t*> cnp.PyArray_DATA(arr) + y * row_bytes_in,
                   row_bytes_in)

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
        if nclx != NULL:
            heif_nclx_color_profile_free(nclx)
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
