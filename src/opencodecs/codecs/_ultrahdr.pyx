# opencodecs/codecs/_ultrahdr.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Ultra HDR (gainmap JPEG) — Cython binding to Google's ``libultrahdr``.

Ultra HDR (ISO 21496) is a backwards-compatible HDR image format:
a regular SDR JPEG with a small "gainmap" JPEG embedded as MPF
metadata. SDR viewers ignore the gainmap and see a normal JPEG; HDR
viewers multiply by the gainmap to recover the HDR image. This is
the format Android Camera writes by default since Android 14 and
that iOS 18+ reads natively.

The opencodecs Pareto defaults are:

* **Encode**: input is ``(H, W, 4) float16`` linear-light RGBA in BT.2100
  primaries (full-range). Quality = 95 (matches our ``_jpeg``
  default and the libultrahdr ``BEST_QUALITY`` preset). Gainmap is
  full resolution (``scale=1``). Returns gainmap-JPEG bytes.

* **Decode**: returns ``(H, W, 4) float16`` linear-light RGBA by
  default — the HDR pixels reconstructed from base + gainmap. Pass
  ``dtype=np.uint8`` to get the SDR tonemapped fallback instead.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t, uint16_t, uint32_t
from libc.string cimport memset, memcpy

import numpy as np
cimport numpy as cnp

from libultrahdr cimport (
    UHDR_LIB_VERSION_STR,
    uhdr_codec_err_t, UHDR_CODEC_OK,
    uhdr_img_fmt_t,
    UHDR_IMG_FMT_64bppRGBAHalfFloat, UHDR_IMG_FMT_32bppRGBA8888,
    UHDR_IMG_FMT_32bppRGBA1010102,
    uhdr_color_gamut_t,
    UHDR_CG_BT_709, UHDR_CG_BT_2100, UHDR_CG_DISPLAY_P3,
    uhdr_color_transfer_t,
    UHDR_CT_LINEAR, UHDR_CT_HLG, UHDR_CT_SRGB,
    uhdr_color_range_t, UHDR_CR_FULL_RANGE,
    uhdr_img_label_t, UHDR_HDR_IMG, UHDR_BASE_IMG, UHDR_GAIN_MAP_IMG,
    uhdr_enc_preset_t, UHDR_USAGE_BEST_QUALITY, UHDR_USAGE_REALTIME,
    UHDR_PLANE_PACKED,
    uhdr_raw_image_t, uhdr_compressed_image_t,
    uhdr_codec_private_t, uhdr_error_info_t,
    uhdr_create_encoder, uhdr_release_encoder,
    uhdr_create_decoder, uhdr_release_decoder,
    uhdr_enc_set_raw_image, uhdr_enc_set_preset, uhdr_enc_set_quality,
    uhdr_enc_set_gainmap_scale_factor,
    uhdr_encode, uhdr_get_encoded_stream,
    uhdr_dec_set_image, uhdr_dec_set_out_img_format,
    uhdr_dec_set_out_color_transfer, uhdr_dec_set_out_max_display_boost,
    uhdr_dec_probe, uhdr_decode,
    uhdr_dec_get_image_width, uhdr_dec_get_image_height,
    uhdr_get_decoded_image,
)


cnp.import_array()


class UltrahdrError(RuntimeError):
    """Raised on libultrahdr encode/decode failures."""


cdef inline str _err_detail(uhdr_error_info_t err):
    if err.has_detail:
        return (<bytes> err.detail).decode("ascii", "replace")
    return ""


cdef inline void _check(uhdr_error_info_t err, str func):
    if err.error_code != UHDR_CODEC_OK:
        raise UltrahdrError(
            f"{func}: code={int(err.error_code)} {_err_detail(err)}"
        )


def version() -> str:
    """Return the libultrahdr runtime version string."""
    return (<bytes> UHDR_LIB_VERSION_STR).decode("ascii")


def encode(arr, *, level=None, scale=None, fast=False) -> bytes:
    """Encode an HDR float16 RGBA image as a gainmap JPEG.

    Parameters
    ----------
    arr : np.ndarray
        ``(H, W, 4)`` float16, linear-light RGBA in BT.2100 primaries.
    level : int, optional
        JPEG quality 0-100 for both base and gainmap. Default 95
        (matches the libultrahdr ``BEST_QUALITY`` preset and our
        ``_jpeg`` default).
    scale : int, optional
        Gainmap subsampling factor (1, 2, 4, 8, ... up to 128).
        Default 1 (full-resolution gainmap, best HDR reconstruction).
    fast : bool, optional
        Use the ``REALTIME`` preset (smaller gainmap, faster
        encode, ~10-20% quality loss in highlights). Default False.
    """
    cdef:
        cnp.ndarray contig
        uhdr_codec_private_t* encoder = NULL
        uhdr_raw_image_t raw_image
        uhdr_compressed_image_t* compressed
        uhdr_error_info_t err
        int quality, scale_factor, height, width
        uhdr_enc_preset_t preset

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    contig = np.ascontiguousarray(arr)
    if (
        contig.ndim != 3 or contig.shape[2] != 4
        or contig.dtype != np.float16
    ):
        raise ValueError(
            f"ultrahdr encode: expected (H, W, 4) float16, "
            f"got shape {np.shape(contig)} dtype {contig.dtype}"
        )

    height = <int> contig.shape[0]
    width = <int> contig.shape[1]
    if height <= 0 or width <= 0 or height > 65535 or width > 65535:
        raise ValueError(
            f"ultrahdr encode: bad dimensions {height}x{width}"
        )

    quality = 95 if level is None else int(level)
    if quality < 0:
        quality = 0
    if quality > 100:
        quality = 100
    scale_factor = 1 if scale is None else int(scale)
    if scale_factor < 1:
        scale_factor = 1
    preset = UHDR_USAGE_REALTIME if fast else UHDR_USAGE_BEST_QUALITY

    memset(<void*> &raw_image, 0, sizeof(uhdr_raw_image_t))
    raw_image.fmt = UHDR_IMG_FMT_64bppRGBAHalfFloat
    raw_image.cg = UHDR_CG_BT_2100
    raw_image.ct = UHDR_CT_LINEAR
    raw_image.range = UHDR_CR_FULL_RANGE
    raw_image.w = <unsigned int> width
    raw_image.h = <unsigned int> height
    raw_image.planes[UHDR_PLANE_PACKED] = <void*> contig.data
    raw_image.stride[UHDR_PLANE_PACKED] = <unsigned int> width

    try:
        encoder = uhdr_create_encoder()
        if encoder == NULL:
            raise UltrahdrError("uhdr_create_encoder returned NULL")

        with nogil:
            err = uhdr_enc_set_preset(encoder, preset)
        _check(err, "uhdr_enc_set_preset")

        with nogil:
            err = uhdr_enc_set_raw_image(encoder, &raw_image, UHDR_HDR_IMG)
        _check(err, "uhdr_enc_set_raw_image")

        with nogil:
            err = uhdr_enc_set_quality(encoder, quality, UHDR_BASE_IMG)
        _check(err, "uhdr_enc_set_quality(base)")

        with nogil:
            err = uhdr_enc_set_quality(encoder, quality, UHDR_GAIN_MAP_IMG)
        _check(err, "uhdr_enc_set_quality(gainmap)")

        with nogil:
            err = uhdr_enc_set_gainmap_scale_factor(encoder, scale_factor)
        _check(err, "uhdr_enc_set_gainmap_scale_factor")

        with nogil:
            err = uhdr_encode(encoder)
        _check(err, "uhdr_encode")

        compressed = uhdr_get_encoded_stream(encoder)
        if compressed == NULL:
            raise UltrahdrError("uhdr_get_encoded_stream returned NULL")

        return PyBytes_FromStringAndSize(
            <const char*> compressed.data, <Py_ssize_t> compressed.data_sz
        )
    finally:
        if encoder != NULL:
            uhdr_release_encoder(encoder)


def decode(data, *, dtype=None, boost=None) -> "np.ndarray":
    """Decode a gainmap JPEG into an HDR or SDR RGBA array.

    Parameters
    ----------
    data : bytes-like
        Gainmap JPEG bytes.
    dtype : numpy dtype, optional
        Output dtype. Choices:

        * ``np.float16`` (default) — linear-light HDR RGBA,
          shape ``(H, W, 4)``.
        * ``np.uint8`` — sRGB-gamut SDR RGBA (the base JPEG without
          gainmap applied), shape ``(H, W, 4)``.
        * ``np.uint16`` — HLG-encoded RGBA1010102 unpacked to 10-bit
          channels (R, G, B in [0, 1023]; A in [0, 3]),
          shape ``(H, W, 4)``.
    boost : float, optional
        Max display brightness boost relative to SDR. Default lets
        libultrahdr choose based on the gainmap metadata.
    """
    cdef:
        const uint8_t[::1] src
        uhdr_codec_private_t* decoder = NULL
        uhdr_compressed_image_t compressed_image
        uhdr_raw_image_t* raw_image
        uhdr_error_info_t err
        uhdr_img_fmt_t fmt
        uhdr_color_transfer_t out_ct
        int width, height, channels = 4
        ssize_t bpp
        cnp.ndarray out_arr
        float disp_boost

    dtype_obj = np.float16 if dtype is None else np.dtype(dtype)
    if dtype_obj == np.float16:
        fmt = UHDR_IMG_FMT_64bppRGBAHalfFloat
        out_ct = UHDR_CT_LINEAR
        bpp = 8
    elif dtype_obj == np.uint8:
        fmt = UHDR_IMG_FMT_32bppRGBA8888
        out_ct = UHDR_CT_SRGB
        bpp = 4
    elif dtype_obj == np.uint16:
        # libultrahdr returns RGBA1010102 packed into uint32; we
        # unpack to four uint16 channels per pixel below.
        fmt = UHDR_IMG_FMT_32bppRGBA1010102
        out_ct = UHDR_CT_HLG
        bpp = 8
    else:
        raise ValueError(
            f"ultrahdr decode: dtype must be float16/uint8/uint16, "
            f"got {dtype_obj}"
        )

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    if src.shape[0] < 8:
        raise UltrahdrError("ultrahdr decode: stream too short")

    memset(<void*> &compressed_image, 0, sizeof(uhdr_compressed_image_t))
    compressed_image.data = <void*> &src[0]
    compressed_image.data_sz = <size_t> src.shape[0]
    compressed_image.capacity = <size_t> src.shape[0]

    try:
        decoder = uhdr_create_decoder()
        if decoder == NULL:
            raise UltrahdrError("uhdr_create_decoder returned NULL")

        with nogil:
            err = uhdr_dec_set_image(decoder, &compressed_image)
        _check(err, "uhdr_dec_set_image")

        with nogil:
            err = uhdr_dec_set_out_img_format(decoder, fmt)
        _check(err, "uhdr_dec_set_out_img_format")

        with nogil:
            err = uhdr_dec_set_out_color_transfer(decoder, out_ct)
        _check(err, "uhdr_dec_set_out_color_transfer")

        if boost is not None:
            disp_boost = float(boost)
            if disp_boost >= 1.0:
                with nogil:
                    err = uhdr_dec_set_out_max_display_boost(
                        decoder, disp_boost
                    )
                _check(err, "uhdr_dec_set_out_max_display_boost")

        with nogil:
            err = uhdr_dec_probe(decoder)
        _check(err, "uhdr_dec_probe")

        width = uhdr_dec_get_image_width(decoder)
        height = uhdr_dec_get_image_height(decoder)
        if width <= 0 or height <= 0:
            raise UltrahdrError(
                f"ultrahdr decode: bad probed dimensions {height}x{width}"
            )

        with nogil:
            err = uhdr_decode(decoder)
        _check(err, "uhdr_decode")

        raw_image = uhdr_get_decoded_image(decoder)
        if raw_image == NULL:
            raise UltrahdrError("uhdr_get_decoded_image returned NULL")
        if raw_image.h != height or raw_image.w != width:
            raise UltrahdrError(
                f"ultrahdr decode: probe vs decode shape mismatch "
                f"({raw_image.h}x{raw_image.w} != {height}x{width})"
            )

        out_arr = np.empty((height, width, channels), dtype=dtype_obj)

        if fmt == UHDR_IMG_FMT_32bppRGBA1010102:
            # Unpack 10:10:10:2 packed uint32 → four uint16 channels.
            _unpack_rgba1010102(
                <uint16_t*> out_arr.data,
                <uint32_t*> raw_image.planes[UHDR_PLANE_PACKED],
                <ssize_t> raw_image.stride[UHDR_PLANE_PACKED],
                height, width,
            )
        else:
            _copy_planes(
                <char*> out_arr.data,
                <char*> raw_image.planes[UHDR_PLANE_PACKED],
                <ssize_t> (width * bpp),
                <ssize_t> raw_image.stride[UHDR_PLANE_PACKED] * bpp,
                height,
            )
    finally:
        if decoder != NULL:
            uhdr_release_decoder(decoder)

    return out_arr


cdef inline void _copy_planes(
    char* dst, char* src,
    ssize_t dst_stride, ssize_t src_stride, ssize_t height,
) noexcept nogil:
    cdef ssize_t i
    for i in range(height):
        memcpy(<void*> (dst + i * dst_stride),
               <const void*> (src + i * src_stride),
               <size_t> dst_stride)


cdef inline void _unpack_rgba1010102(
    uint16_t* dst, uint32_t* src,
    ssize_t stride, ssize_t height, ssize_t width,
) noexcept nogil:
    cdef:
        ssize_t i, j, k = 0
        uint32_t rgba
    for j in range(height):
        for i in range(width):
            rgba = src[i]
            dst[k] = <uint16_t> (rgba & 0x3ff); k += 1
            dst[k] = <uint16_t> ((rgba >> 10) & 0x3ff); k += 1
            dst[k] = <uint16_t> ((rgba >> 20) & 0x3ff); k += 1
            dst[k] = <uint16_t> ((rgba >> 30) & 0x3); k += 1
        src += stride


def check_signature(data) -> bool:
    """Return True if ``data`` looks like an MPF JPEG (gainmap JPEG).

    Ultra HDR files are JPEGs whose APP marker chain includes the
    ISO 21496 MPF extension; the full check requires parsing the
    JPEG segment headers. Here we do a cheap magic-byte check
    (``FFD8FF``) — a false positive simply means the libultrahdr
    decoder will reject the stream on probe.
    """
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:3])
    else:
        try:
            head = bytes(data)[:3]
        except Exception:
            return False
    return len(head) >= 3 and head[0] == 0xFF and head[1] == 0xD8 and head[2] == 0xFF
