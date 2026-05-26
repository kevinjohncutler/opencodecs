# opencodecs/codecs/_uhdr.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Cython binding for Google's libultrahdr (libuhdr) 1.4.x.

Encodes a single HDR raster (linear-light Display-P3 or Rec.2020, fp16)
into an ISO 21496-1 Ultra-HDR JPEG: an SDR JPEG base + a per-pixel gain
map that HDR-aware decoders (Chrome 116+, Safari 26+, libuhdr, Apple
Photos, etc.) composite to display-headroom. Browsers that don't know
about the gain map just decode the SDR base -- which is exactly the
"correctly tone-mapped, not clipped" fallback we want.

v1 surface is intentionally minimal:

    encode(hdr_fp16, gamut='display-p3', transfer='pq', quality=95,
           container='jpg') -> bytes
    decode(data) -> dict { 'hdr_fp16', 'sdr_u8', 'gainmap_u8',
                            'gainmap_metadata', 'width', 'height' }
    is_uhdr(data) -> bool
    libuhdr_version() -> str

Everything else (multi-channel gain maps, scale-factor tuning,
target-display-peak control, AVIF/HEIF containers, GPU acceleration,
in-place effect ops) can be exposed later by widening the Python API
on top of this binding.
"""

import io
import os

import numpy as np
cimport numpy as cnp

from libc.stdint cimport uint8_t, uint16_t, uint32_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from libc.math cimport log2f, powf
from cpython.bytes cimport PyBytes_FromStringAndSize

from libuhdr cimport *

cnp.import_array()


# ---------------------------------------------------------------------------
# Version + signature probes
# ---------------------------------------------------------------------------

cdef extern from "ultrahdr_api.h":
    int UHDR_LIB_VER_MAJOR
    int UHDR_LIB_VER_MINOR
    int UHDR_LIB_VER_PATCH


def libuhdr_version() -> str:
    """Return the compile-time libuhdr version as ``major.minor.patch``."""
    return f"{UHDR_LIB_VER_MAJOR}.{UHDR_LIB_VER_MINOR}.{UHDR_LIB_VER_PATCH}"


def is_uhdr(data) -> bool:
    """Return True if ``data`` looks like an ISO 21496-1 / Ultra-HDR
    container (JPEG / HEIF / AVIF with the gain-map metadata box)."""
    cdef const unsigned char[::1] view = _coerce_bytes_view(data)
    if view.shape[0] == 0:
        return False
    return is_uhdr_image(<void*>&view[0], <int>view.shape[0]) != 0


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

from opencodecs.core.errors import OpenCodecsError


class UhdrError(OpenCodecsError):
    """Raised when libuhdr returns a non-OK status."""


cdef _check(uhdr_error_info_t info, str func):
    if info.error_code == UHDR_CODEC_OK:
        return
    detail = ""
    if info.has_detail:
        detail = ": " + bytes(info.detail).decode("utf-8", "replace").rstrip("\x00")
    raise UhdrError(f"{func} -> {_err_name(info.error_code)}{detail}")


cdef str _err_name(uhdr_codec_err_t code):
    if code == UHDR_CODEC_OK: return "UHDR_CODEC_OK"
    if code == UHDR_CODEC_ERROR: return "UHDR_CODEC_ERROR"
    if code == UHDR_CODEC_UNKNOWN_ERROR: return "UHDR_CODEC_UNKNOWN_ERROR"
    if code == UHDR_CODEC_INVALID_PARAM: return "UHDR_CODEC_INVALID_PARAM"
    if code == UHDR_CODEC_MEM_ERROR: return "UHDR_CODEC_MEM_ERROR"
    if code == UHDR_CODEC_INVALID_OPERATION: return "UHDR_CODEC_INVALID_OPERATION"
    if code == UHDR_CODEC_UNSUPPORTED_FEATURE: return "UHDR_CODEC_UNSUPPORTED_FEATURE"
    return f"UNKNOWN({code})"


# ---------------------------------------------------------------------------
# Enum mapping
# ---------------------------------------------------------------------------

_GAMUT_MAP = {
    "bt709":      UHDR_CG_BT_709,
    "srgb":       UHDR_CG_BT_709,
    "display-p3": UHDR_CG_DISPLAY_P3,
    "p3":         UHDR_CG_DISPLAY_P3,
    "rec2020":    UHDR_CG_BT_2100,
    "bt2020":     UHDR_CG_BT_2100,
    "bt2100":     UHDR_CG_BT_2100,
    "rec2100":    UHDR_CG_BT_2100,
}

_TRANSFER_MAP = {
    "linear":   UHDR_CT_LINEAR,
    "hlg":      UHDR_CT_HLG,
    "pq":       UHDR_CT_PQ,
    "srgb":     UHDR_CT_SRGB,
    "gamma22":  UHDR_CT_SRGB,
}

_CODEC_MAP = {
    "jpg": UHDR_CODEC_JPG,
    "jpeg": UHDR_CODEC_JPG,
    "heif": UHDR_CODEC_HEIF,
    "heic": UHDR_CODEC_HEIF,
    "avif": UHDR_CODEC_AVIF,
}

_PRESET_MAP = {
    "realtime": UHDR_USAGE_REALTIME,
    "fast":     UHDR_USAGE_REALTIME,
    "best":     UHDR_USAGE_BEST_QUALITY,
    "quality":  UHDR_USAGE_BEST_QUALITY,
}


cdef uhdr_color_gamut_t _resolve_gamut(name) except *:
    cdef int v
    if isinstance(name, int):
        v = <int>name
        return <uhdr_color_gamut_t>v
    key = str(name).lower()
    if key not in _GAMUT_MAP:
        raise ValueError(f"unknown gamut {name!r}; expected one of "
                         f"{sorted(_GAMUT_MAP.keys())}")
    v = <int>_GAMUT_MAP[key]
    return <uhdr_color_gamut_t>v


cdef uhdr_color_transfer_t _resolve_transfer(name) except *:
    cdef int v
    if isinstance(name, int):
        v = <int>name
        return <uhdr_color_transfer_t>v
    key = str(name).lower()
    if key not in _TRANSFER_MAP:
        raise ValueError(f"unknown transfer {name!r}; expected one of "
                         f"{sorted(_TRANSFER_MAP.keys())}")
    v = <int>_TRANSFER_MAP[key]
    return <uhdr_color_transfer_t>v


cdef uhdr_codec_t _resolve_codec(name) except *:
    cdef int v
    if isinstance(name, int):
        v = <int>name
        return <uhdr_codec_t>v
    key = str(name).lower()
    if key not in _CODEC_MAP:
        raise ValueError(f"unknown container {name!r}; expected one of "
                         f"{sorted(_CODEC_MAP.keys())}")
    v = <int>_CODEC_MAP[key]
    return <uhdr_codec_t>v


cdef uhdr_enc_preset_t _resolve_preset(name) except *:
    key = str(name).lower()
    if key not in _PRESET_MAP:
        raise ValueError(f"unknown preset {name!r}; expected one of "
                         f"{sorted(_PRESET_MAP.keys())}")
    return _PRESET_MAP[key]


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

cdef const unsigned char[::1] _coerce_bytes_view(data):
    """Return a 1-D contiguous read-only view of ``data``."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        return memoryview(data).cast("B")
    arr = np.ascontiguousarray(data, dtype=np.uint8)
    return arr.reshape(-1)


import cython


# ---------------------------------------------------------------------------
# (Earlier revisions of this binding shipped a "fast path" that did the
# gain-map + SDR-base computation in a fused Cython kernel and handed
# pre-encoded JPEG layers to libuhdr just for container assembly. That
# wrapped libuhdr's own encoder pipeline, which is a maintenance hazard
# — libuhdr controls the gain-map formula and any tweak upstream would
# silently diverge from our reimplementation. We removed it; this
# binding now uses libuhdr's encode() exclusively. If anyone needs a
# from-scratch gain-map computation, build it on top of the public
# libuhdr API rather than re-implementing inside this binding.)
# ---------------------------------------------------------------------------


cdef void _rgb_to_rgba_pack_kernel(const uint8_t* src, uint8_t* dst,
                                     Py_ssize_t n_pixels) noexcept nogil:
    """Pack n_pixels of RGB888 into RGBA8888 with alpha=255.
    Tight C loop -- ~3-5x faster than numpy's stride-mismatched
    ``rgba[:, :, 0:3] = arr`` which can't memcpy because the
    destination stride on the inner axis is 4, source is 3."""
    cdef Py_ssize_t i
    for i in range(n_pixels):
        dst[i * 4 + 0] = src[i * 3 + 0]
        dst[i * 4 + 1] = src[i * 3 + 1]
        dst[i * 4 + 2] = src[i * 3 + 2]
        dst[i * 4 + 3] = 255


@cython.boundscheck(False)
@cython.wraparound(False)
def _sdr_to_rgba_u8(sdr):
    """Coerce ``sdr`` (HxWx3 or HxWx4 uint8 / float [0,1]) to a
    HxWx4 contiguous RGBA uint8 buffer. RGB inputs are packed into
    RGBA via a tight C loop (~3-5x faster than numpy's stride-
    mismatched broadcast copy) which dominated our preprocessing
    time before this change.

    libuhdr requires UHDR_IMG_FMT_32bppRGBA8888 for the SDR intent
    (it rejects 24bppRGB888 -- only YUV420 and RGBA8888 are accepted
    for SDR), so we can't avoid the pack."""
    arr = np.asarray(sdr)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(
            f"SDR input must be HxWx3 or HxWx4, got shape {arr.shape}")
    cdef Py_ssize_t H = int(arr.shape[0])
    cdef Py_ssize_t W = int(arr.shape[1])
    cdef int C = int(arr.shape[2])
    if arr.dtype != np.uint8:
        a = np.clip(arr.astype(np.float32, copy=False), 0.0, 1.0)
        arr = (a * 255.0 + 0.5).astype(np.uint8)
    cdef cnp.ndarray arr_arr = np.ascontiguousarray(arr)
    cdef cnp.ndarray[cnp.uint8_t, ndim=3, mode='c'] rgba_arr = np.empty(
        (H, W, 4), dtype=np.uint8)
    cdef const uint8_t* src_ptr
    cdef uint8_t* dst_ptr = <uint8_t*>cnp.PyArray_DATA(rgba_arr)
    if C == 3:
        src_ptr = <const uint8_t*>cnp.PyArray_DATA(arr_arr)
        with nogil:
            _rgb_to_rgba_pack_kernel(src_ptr, dst_ptr, H * W)
    else:
        # Already RGBA -- straight memcpy.
        memcpy(dst_ptr, cnp.PyArray_DATA(arr_arr),
               <size_t>(H * W * 4))
    return rgba_arr


@cython.boundscheck(True)
@cython.wraparound(True)
def _hdr_to_rgba_fp16(hdr, alpha=None):
    """Coerce ``hdr`` (HxWx3 or HxWx4 float, any dtype) to an HxWx4
    fp16 contiguous RGBA array. ``alpha``, if supplied, overrides the
    last channel; if neither source has alpha, default opacity = 1.0.

    libuhdr's ``UHDR_IMG_FMT_64bppRGBAHalfFloat`` expects values in
    the nominal range [0, 10000/203] -- i.e. PQ/HLG linear-light with
    1.0 = SDR-reference white (203 nits per BT.2100). Callers are
    responsible for arranging their input to match that convention.

    Decorated with boundscheck + wraparound back on because the
    file-level directives disable them, but numpy's ``arr[..., :3]``
    slicing semantics can misbehave when Cython compiles it under
    wraparound=False.
    """
    arr = np.asarray(hdr)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(
            f"HDR input must be HxWx3 or HxWx4, got shape {arr.shape}")
    H = int(arr.shape[0])
    W = int(arr.shape[1])
    C = int(arr.shape[2])
    if arr.dtype != np.float16:
        arr = arr.astype(np.float16, copy=False)
    rgba = np.empty((H, W, 4), dtype=np.float16)
    # Use plain numpy slicing instead of ellipsis -- avoids any
    # Cython directive interaction.
    rgba[:, :, 0:3] = arr[:, :, 0:3]
    if C == 4 and alpha is None:
        rgba[:, :, 3] = arr[:, :, 3]
    elif alpha is not None:
        a = np.asarray(alpha)
        if a.shape != (H, W):
            raise ValueError(
                f"alpha shape {a.shape} doesn't match HDR HxW {(H, W)}")
        rgba[:, :, 3] = a.astype(np.float16, copy=False)
    else:
        rgba[:, :, 3] = np.float16(1.0)
    if not rgba.flags.c_contiguous:
        rgba = np.ascontiguousarray(rgba)
    return rgba


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def encode_assembled(*, base_jpeg, gainmap_jpeg, metadata, gamut="display-p3"):
    """Native fast-path: assemble an Ultra-HDR JPEG from pre-encoded
    base + gain map + caller-computed metadata.

    libuhdr's "api - 4" entry point: when both the SDR base and the
    gain map are already encoded as JPEGs and the gain-map metadata
    is supplied, libuhdr just does container assembly (~5 ms) --
    skipping its internal gain-map computation + two libjpeg-turbo
    encodes entirely.

    The high-level helper :func:`opencodecs.uhdr.encode_native`
    computes the gain map + parallel-encodes both layers + calls
    this. Use that unless you need finer control.

    Parameters
    ----------
    base_jpeg : bytes-like
        SDR base layer as a complete JPEG bitstream.
    gainmap_jpeg : bytes-like
        Gain map sub-image as a complete JPEG bitstream. Dimensions
        must match the gain-map size implied by the base.
    metadata : dict
        Gain-map metadata. Required keys (floats unless noted):
            max_content_boost : float or [r, g, b]   -- linear scale, > 1
            min_content_boost : float or [r, g, b]   -- linear scale, >= 0
            gamma             : float or [r, g, b]   -- usually 1.0
            offset_sdr        : float or [r, g, b]   -- usually 0.0
            offset_hdr        : float or [r, g, b]   -- usually 0.0
            hdr_capacity_min  : float                -- usually 1.0
            hdr_capacity_max  : float                -- equals max_content_boost
            use_base_cg       : bool                 -- True = gain map in base colorspace

    Returns
    -------
    bytes  -- ISO 21496-1 Ultra-HDR JPEG.
    """
    cdef const unsigned char[::1] base_view = _coerce_bytes_view(base_jpeg)
    cdef const unsigned char[::1] gm_view = _coerce_bytes_view(gainmap_jpeg)
    if base_view.shape[0] < 2 or gm_view.shape[0] < 2:
        raise ValueError("both base_jpeg and gainmap_jpeg must be non-empty bytes")

    cdef uhdr_codec_private_t* enc = uhdr_create_encoder()
    if enc == NULL:
        raise UhdrError("uhdr_create_encoder returned NULL (OOM)")

    cdef uhdr_compressed_image_t base_img
    cdef uhdr_compressed_image_t gm_img
    cdef uhdr_gainmap_metadata_t meta
    cdef uhdr_error_info_t info
    cdef uhdr_compressed_image_t* out_stream
    cdef bytes out_bytes
    cdef int i

    # Fan out scalar entries to per-channel arrays.
    def _as3(v, name):
        if hasattr(v, '__len__') and not isinstance(v, (str, bytes)):
            if len(v) != 3:
                raise ValueError(f"metadata['{name}'] must be scalar or length-3")
            return [float(x) for x in v]
        return [float(v)] * 3

    mb = _as3(metadata['max_content_boost'], 'max_content_boost')
    mi = _as3(metadata['min_content_boost'], 'min_content_boost')
    gm = _as3(metadata['gamma'], 'gamma')
    osr = _as3(metadata['offset_sdr'], 'offset_sdr')
    ohr = _as3(metadata['offset_hdr'], 'offset_hdr')
    capmin = float(metadata['hdr_capacity_min'])
    capmax = float(metadata['hdr_capacity_max'])
    use_base_cg = bool(metadata.get('use_base_cg', True))

    try:
        memset(&base_img, 0, sizeof(base_img))
        base_img.data = <void*>&base_view[0]
        base_img.data_sz = <size_t>base_view.shape[0]
        base_img.capacity = <size_t>base_view.shape[0]
        base_img.cg = _resolve_gamut(gamut)
        base_img.ct = UHDR_CT_SRGB
        base_img.range = UHDR_CR_FULL_RANGE

        memset(&gm_img, 0, sizeof(gm_img))
        gm_img.data = <void*>&gm_view[0]
        gm_img.data_sz = <size_t>gm_view.shape[0]
        gm_img.capacity = <size_t>gm_view.shape[0]
        gm_img.cg = UHDR_CG_UNSPECIFIED
        gm_img.ct = UHDR_CT_UNSPECIFIED
        gm_img.range = UHDR_CR_FULL_RANGE

        memset(&meta, 0, sizeof(meta))
        for i in range(3):
            meta.max_content_boost[i] = mb[i]
            meta.min_content_boost[i] = mi[i]
            meta.gamma[i] = gm[i]
            meta.offset_sdr[i] = osr[i]
            meta.offset_hdr[i] = ohr[i]
        meta.hdr_capacity_min = capmin
        meta.hdr_capacity_max = capmax
        meta.use_base_cg = 1 if use_base_cg else 0

        info = uhdr_enc_set_compressed_image(enc, &base_img, UHDR_BASE_IMG)
        _check(info, "uhdr_enc_set_compressed_image(BASE)")

        info = uhdr_enc_set_gainmap_image(enc, &gm_img, &meta)
        _check(info, "uhdr_enc_set_gainmap_image")

        with nogil:
            info = uhdr_encode(enc)
        _check(info, "uhdr_encode(assembled)")

        out_stream = uhdr_get_encoded_stream(enc)
        if out_stream == NULL or out_stream.data == NULL:
            raise UhdrError("uhdr_get_encoded_stream returned NULL")
        out_bytes = PyBytes_FromStringAndSize(
            <char*>out_stream.data, <Py_ssize_t>out_stream.data_sz)
    finally:
        uhdr_release_encoder(enc)

    return out_bytes


def encode(hdr,
           *,
           sdr=None,
           sdr_base_compressed=None,
           gamut="display-p3",
           transfer="linear",
           sdr_white_nits=203.0,
           quality=95,
           container="jpg",
           gainmap_scale_factor=1,
           gainmap_gamma=1.0,
           multi_channel_gainmap=True,
           preset="best",
           target_display_peak_nits=0.0,
           min_content_boost=0.0,
           max_content_boost=0.0):
    """Encode a single HDR raster as ISO 21496-1 / Ultra-HDR.

    Parameters
    ----------
    hdr : ndarray (H, W, 3) or (H, W, 4)
        Float HDR pixels. Cast to float16 internally. Values are
        linear-light; the meaning of ``1.0`` is set by ``sdr_white_nits``
        below.
    gamut : str or int
        Source colour gamut: ``'display-p3'`` (default), ``'rec2020'``,
        or ``'bt709'``. Accepts the integer enum directly too.
    transfer : str or int
        Source transfer function. With float input pass ``'linear'``
        (the encoder applies the appropriate OETF internally). ``'pq'``
        / ``'hlg'`` / ``'srgb'`` are also valid for already-encoded
        inputs.
    sdr_white_nits : float
        Brightness in nits that the caller's input ``1.0`` represents.
        libuhdr's internal convention is ``1.0 == 203 nits``
        (BT.2100 reference SDR-white); pipelines that use a different
        scale (a downstream imaging pipeline uses ``1.0 == 1600 nits`` = Apple XDR HDR peak)
        must set this so the encoder sees the correct HDR headroom.
        Encoding silently rescales the input by
        ``sdr_white_nits / 203`` before handing it off to libuhdr.
        If your input peak is, say, ``2.5`` and ``sdr_white_nits=1600``,
        libuhdr sees ``2.5 * 1600 / 203 = 19.7`` which it understands
        as ~4 kilonits -- 20x SDR-white, so the gain map is built
        with that much headroom. Defaults to 203 (no rescale).
    quality : int 0..100
        JPEG quality for the SDR base image. The gain-map image
        inherits the same quality unless we add a separate knob later.
    container : str
        One of ``'jpg'`` (default), ``'heif'``, ``'avif'``.
    gainmap_scale_factor : int 1..128
        Gain map resolution = base / scale_factor. 1 = full-res
        (largest, most accurate), 4 = quarter (typical default in
        libultrahdr's CLI), 128 = smallest.
    gainmap_gamma : float
        Encoding gamma of the gain map. 1.0 = linear (default).
    multi_channel_gainmap : bool
        If True (default) emit a per-RGB gain map; if False, a single
        luminance-only gain map (smaller file, less colour fidelity).
    preset : str
        ``'best'`` (default, quality-tuned) or ``'realtime'`` (faster).
    target_display_peak_nits : float
        If > 0, claim this peak brightness in the metadata. Set to
        the brightest value the encoded scene actually contains, in
        nits. 0 (default) means leave it at the encoder default.
    min_content_boost, max_content_boost : float
        Override gain-map content-boost limits (linear scale). 0
        means use encoder defaults.

    Returns
    -------
    bytes
        Encoded ISO 21496-1 stream.
    """
    # Rescale to libuhdr's "1.0 == 203 nits" convention before
    # casting to fp16. This is the single most impactful knob:
    # without it, content authored to a larger SDR-white reference
    # (e.g. a downstream imaging pipeline's 1.0 == 1600 nits) collapses into libuhdr's
    # SDR range and the gain map encodes "no HDR headroom in use"
    # -- on display you get an SDR-looking image even on HDR
    # monitors. Default 203 means no rescale (matches libuhdr's
    # native convention).
    if sdr_white_nits != 203.0:
        hdr = np.asarray(hdr, dtype=np.float32) * np.float32(
            sdr_white_nits / 203.0)
    rgba = _hdr_to_rgba_fp16(hdr)
    cdef cnp.ndarray rgba_arr = rgba
    cdef unsigned int H = rgba_arr.shape[0]
    cdef unsigned int W = rgba_arr.shape[1]

    # Optional caller-provided SDR base. libuhdr requires
    # 32bppRGBA8888 for the SDR intent (it rejects 24bppRGB888),
    # so we always pack to RGBA -- but via a Cython kernel that's
    # ~5x faster than numpy's stride-mismatched broadcast copy.
    cdef cnp.ndarray sdr_arr_obj = None
    if sdr is not None:
        sdr_arr_obj = _sdr_to_rgba_u8(sdr)
        if sdr_arr_obj.shape[0] != H or sdr_arr_obj.shape[1] != W:
            raise ValueError(
                f"SDR base shape {(sdr_arr_obj.shape[0], sdr_arr_obj.shape[1])} "
                f"doesn't match HDR shape {(H, W)}")

    # Optional caller-provided pre-encoded SDR JPEG. When set,
    # libuhdr uses these bytes verbatim as the output base layer
    # and SKIPS its own internal libjpeg-turbo encode of the SDR
    # rendition. The raw ``sdr`` array is still used for gain-map
    # computation. Recommended workflow:
    #   sdr_u8 = your_naive_sdr_function(rgb_lin_p3)
    #   sdr_jpeg = imagecodecs.jpeg_encode(sdr_u8, level=quality)
    #   data = uhdr.encode(hdr, sdr=sdr_u8, sdr_base_compressed=sdr_jpeg, ...)
    # This saves the ~13 ms libuhdr otherwise spends encoding the
    # base layer itself.
    cdef const unsigned char[::1] sdr_jpeg_view
    cdef bint have_sdr_jpeg = False
    if sdr_base_compressed is not None:
        sdr_jpeg_view = _coerce_bytes_view(sdr_base_compressed)
        if sdr_jpeg_view.shape[0] < 2:
            raise ValueError("sdr_base_compressed must be a non-empty bytes-like JPEG")
        have_sdr_jpeg = True

    cdef uhdr_codec_private_t* enc = uhdr_create_encoder()
    if enc == NULL:
        raise UhdrError("uhdr_create_encoder returned NULL (OOM)")

    cdef uhdr_raw_image_t img
    cdef uhdr_raw_image_t sdr_img
    cdef uhdr_compressed_image_t sdr_jpeg_img
    cdef uhdr_error_info_t info
    cdef uhdr_compressed_image_t* out_stream
    cdef bytes out_bytes

    try:
        memset(&img, 0, sizeof(img))
        img.fmt = UHDR_IMG_FMT_64bppRGBAHalfFloat
        img.cg = _resolve_gamut(gamut)
        img.ct = _resolve_transfer(transfer)
        img.range = UHDR_CR_FULL_RANGE
        img.w = W
        img.h = H
        img.planes[0] = cnp.PyArray_DATA(rgba_arr)
        img.stride[0] = W

        info = uhdr_enc_set_raw_image(enc, &img, UHDR_HDR_IMG)
        _check(info, "uhdr_enc_set_raw_image(HDR)")

        # Optional: pass through caller's pre-tonemapped SDR base.
        # When set, libuhdr stops auto-generating the SDR (which
        # uses a fixed internal tonemap) and uses this image as the
        # base, computing the gain map needed to reach the HDR input.
        # Same gamut + dimensions as the HDR image are required.
        if sdr_arr_obj is not None:
            memset(&sdr_img, 0, sizeof(sdr_img))
            sdr_img.fmt = UHDR_IMG_FMT_32bppRGBA8888
            sdr_img.cg = _resolve_gamut(gamut)
            sdr_img.ct = UHDR_CT_SRGB
            sdr_img.range = UHDR_CR_FULL_RANGE
            sdr_img.w = W
            sdr_img.h = H
            sdr_img.planes[0] = cnp.PyArray_DATA(sdr_arr_obj)
            sdr_img.stride[0] = W
            info = uhdr_enc_set_raw_image(enc, &sdr_img, UHDR_SDR_IMG)
            _check(info, "uhdr_enc_set_raw_image(SDR)")

        # Pre-encoded SDR JPEG path: libuhdr can take the SDR base
        # as already-compressed JPEG bytes (set_compressed_image with
        # intent UHDR_SDR_IMG) instead of re-encoding raw SDR pixels
        # itself. This saves ~13 ms per encode (one libjpeg-turbo
        # call) when the caller has the bytes ready.
        if have_sdr_jpeg:
            memset(&sdr_jpeg_img, 0, sizeof(sdr_jpeg_img))
            sdr_jpeg_img.data = <void*>&sdr_jpeg_view[0]
            sdr_jpeg_img.data_sz = <size_t>sdr_jpeg_view.shape[0]
            sdr_jpeg_img.capacity = <size_t>sdr_jpeg_view.shape[0]
            sdr_jpeg_img.cg = _resolve_gamut(gamut)
            sdr_jpeg_img.ct = UHDR_CT_SRGB
            sdr_jpeg_img.range = UHDR_CR_FULL_RANGE
            info = uhdr_enc_set_compressed_image(
                enc, &sdr_jpeg_img, UHDR_SDR_IMG)
            _check(info, "uhdr_enc_set_compressed_image(SDR)")

        info = uhdr_enc_set_quality(enc, int(quality), UHDR_BASE_IMG)
        _check(info, "uhdr_enc_set_quality(BASE)")

        info = uhdr_enc_set_using_multi_channel_gainmap(
            enc, 1 if multi_channel_gainmap else 0)
        _check(info, "uhdr_enc_set_using_multi_channel_gainmap")

        info = uhdr_enc_set_gainmap_scale_factor(enc, int(gainmap_scale_factor))
        _check(info, "uhdr_enc_set_gainmap_scale_factor")

        info = uhdr_enc_set_gainmap_gamma(enc, float(gainmap_gamma))
        _check(info, "uhdr_enc_set_gainmap_gamma")

        if max_content_boost > 0.0:
            info = uhdr_enc_set_min_max_content_boost(
                enc, float(min_content_boost), float(max_content_boost))
            _check(info, "uhdr_enc_set_min_max_content_boost")

        if target_display_peak_nits > 0.0:
            info = uhdr_enc_set_target_display_peak_brightness(
                enc, float(target_display_peak_nits))
            _check(info, "uhdr_enc_set_target_display_peak_brightness")

        info = uhdr_enc_set_preset(enc, _resolve_preset(preset))
        _check(info, "uhdr_enc_set_preset")

        info = uhdr_enc_set_output_format(enc, _resolve_codec(container))
        _check(info, "uhdr_enc_set_output_format")

        # Release the GIL around the long-running encode call so
        # batch encoders (e.g. ThreadPoolExecutor over many scenes)
        # actually parallelize across cores. Without this the GIL
        # serialises ~150 ms encodes on a 20-core machine, capping
        # speedup to ~1.3x. ``uhdr_encode`` is declared ``nogil``
        # in the pxd (libuhdr is C, does no Python callbacks) so
        # this is safe.
        with nogil:
            info = uhdr_encode(enc)
        _check(info, "uhdr_encode")

        out_stream = uhdr_get_encoded_stream(enc)
        if out_stream == NULL or out_stream.data == NULL:
            raise UhdrError("uhdr_get_encoded_stream returned NULL")
        out_bytes = PyBytes_FromStringAndSize(
            <char*>out_stream.data, <Py_ssize_t>out_stream.data_sz)
    finally:
        uhdr_release_encoder(enc)

    return out_bytes


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decode(data, *, want_hdr=True, want_gainmap=False,
           want_base=False, display_boost=1.0):
    """Decode an Ultra-HDR stream and return a dict of components.

    Parameters
    ----------
    data : bytes-like
        Encoded ISO 21496-1 stream.
    want_hdr : bool
        Return the composited HDR image (fp16 RGBA). Default True.
    want_gainmap : bool
        Return the raw gain-map image + metadata (uint8 RGBA + a
        Python dict of the metadata fields).
    want_base : bool
        Return the **raw compressed SDR base layer** as a stand-alone
        bytes blob (key ``base_compressed``). For a JPEG-container
        Ultra-HDR file this is a valid standalone SDR JPEG that any
        viewer can render -- exactly what an HDR-unaware decoder
        sees when it strips the gain-map metadata. Useful for A/B
        testing the SDR fallback against the gain-map composite.
    display_boost : float
        Display headroom in linear scale that the decoder will tone
        toward when reconstructing HDR. 1.0 = no boost (SDR-only
        display), 2.0..10.0 typical for HDR monitors.

    Returns
    -------
    dict with optional keys ``hdr_fp16``, ``gainmap_u8``,
    ``gainmap_metadata``, ``base_compressed``, plus always
    ``width``, ``height``.
    """
    cdef const unsigned char[::1] view = _coerce_bytes_view(data)
    cdef uhdr_codec_private_t* dec
    cdef uhdr_compressed_image_t in_img
    cdef uhdr_error_info_t info
    cdef uhdr_raw_image_t* raw_img
    cdef uhdr_raw_image_t* raw_gainmap
    cdef uhdr_gainmap_metadata_t* gm_meta
    cdef uhdr_mem_block_t* base_blk
    cdef int width, height
    cdef Py_ssize_t row_bytes
    cdef Py_ssize_t r
    cdef cnp.ndarray[cnp.uint16_t, ndim=3, mode='c'] hdr_arr
    cdef cnp.ndarray[cnp.uint8_t,  ndim=3, mode='c'] gm_arr
    cdef int gw, gh
    cdef Py_ssize_t row_bytes_g

    if view.shape[0] == 0:
        raise ValueError("empty input")

    dec = uhdr_create_decoder()
    if dec == NULL:
        raise UhdrError("uhdr_create_decoder returned NULL (OOM)")

    result = {}
    try:
        memset(&in_img, 0, sizeof(in_img))
        in_img.data = <void*>&view[0]
        in_img.data_sz = <size_t>view.shape[0]
        in_img.capacity = <size_t>view.shape[0]
        in_img.cg = UHDR_CG_UNSPECIFIED
        in_img.ct = UHDR_CT_UNSPECIFIED
        in_img.range = UHDR_CR_UNSPECIFIED

        info = uhdr_dec_set_image(dec, &in_img)
        _check(info, "uhdr_dec_set_image")

        info = uhdr_dec_set_out_img_format(
            dec, UHDR_IMG_FMT_64bppRGBAHalfFloat)
        _check(info, "uhdr_dec_set_out_img_format")

        info = uhdr_dec_set_out_max_display_boost(dec, float(display_boost))
        _check(info, "uhdr_dec_set_out_max_display_boost")

        info = uhdr_dec_probe(dec)
        _check(info, "uhdr_dec_probe")

        width = uhdr_dec_get_image_width(dec)
        height = uhdr_dec_get_image_height(dec)
        result["width"] = int(width)
        result["height"] = int(height)

        # libuhdr lazily decodes pixels — uhdr_get_decoded_image and
        # uhdr_get_decoded_gainmap_image both return NULL until
        # uhdr_decode runs once. So fire the decode whenever the
        # caller wants any of the decoded raster outputs.
        if want_hdr or want_gainmap:
            info = uhdr_decode(dec)
            _check(info, "uhdr_decode")

        if want_hdr:
            raw_img = uhdr_get_decoded_image(dec)
            if raw_img == NULL or raw_img.planes[0] == NULL:
                raise UhdrError("uhdr_get_decoded_image returned NULL")
            hdr_arr = np.empty((raw_img.h, raw_img.w, 4),
                                dtype=np.uint16)  # fp16 view layout
            row_bytes = raw_img.w * 4 * 2
            for r in range(raw_img.h):
                memcpy(
                    <void*>(<uint8_t*>cnp.PyArray_DATA(hdr_arr)
                            + r * row_bytes),
                    <const void*>(<uint8_t*>raw_img.planes[0]
                                   + r * raw_img.stride[0] * 8),
                    row_bytes,
                )
            result["hdr_fp16"] = hdr_arr.view(np.float16)

        if want_base:
            base_blk = uhdr_dec_get_base_image(dec)
            if base_blk != NULL and base_blk.data != NULL and base_blk.data_sz > 0:
                result["base_compressed"] = PyBytes_FromStringAndSize(
                    <char*>base_blk.data, <Py_ssize_t>base_blk.data_sz)

        if want_gainmap:
            gm_meta = uhdr_dec_get_gainmap_metadata(dec)
            if gm_meta != NULL:
                result["gainmap_metadata"] = {
                    "max_content_boost": [float(gm_meta.max_content_boost[i])
                                           for i in range(3)],
                    "min_content_boost": [float(gm_meta.min_content_boost[i])
                                           for i in range(3)],
                    "gamma": [float(gm_meta.gamma[i]) for i in range(3)],
                    "offset_sdr": [float(gm_meta.offset_sdr[i]) for i in range(3)],
                    "offset_hdr": [float(gm_meta.offset_hdr[i]) for i in range(3)],
                    "hdr_capacity_min": float(gm_meta.hdr_capacity_min),
                    "hdr_capacity_max": float(gm_meta.hdr_capacity_max),
                    "use_base_cg": int(gm_meta.use_base_cg),
                }
            raw_gainmap = uhdr_get_decoded_gainmap_image(dec)
            if raw_gainmap != NULL and raw_gainmap.planes[0] != NULL:
                gw = raw_gainmap.w
                gh = raw_gainmap.h
                gm_arr = np.empty((gh, gw, 4), dtype=np.uint8)
                row_bytes_g = gw * 4
                for r in range(gh):
                    memcpy(
                        <void*>(<uint8_t*>cnp.PyArray_DATA(gm_arr)
                                + r * row_bytes_g),
                        <const void*>(<uint8_t*>raw_gainmap.planes[0]
                                       + r * raw_gainmap.stride[0] * 4),
                        row_bytes_g,
                    )
                result["gainmap_u8"] = gm_arr
    finally:
        uhdr_release_decoder(dec)

    return result
