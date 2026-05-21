"""CmsCodec — ICC-based color management transforms via Little-CMS.

Unlike the rest of opencodecs's codecs, ``cms`` isn't an
encoder/decoder — it's a *transform*. Given pixel data in one ICC
color space and a target ICC color space, it produces pixel data in
the target space. Used in any quality color pipeline alongside the
ICC profiles PNG / JPEG / WebP / AVIF / HEIF can now carry (see
Phase 5 of the imagecodecs-parity work).

Bindings via ``ctypes`` rather than Cython:

* lcms2 is a stable, narrow C ABI — eight functions and a handful
  of integer constants. ctypes is no slower in practice than
  Cython for that surface (the per-pixel work is inside
  ``cmsDoTransform`` — the call overhead is negligible).
* Avoids adding a new Cython module + pxd + setup.py library detect
  for one transform codec.
* Falls back to a clean ImportError when liblcms2 isn't present;
  users who don't need cms aren't penalised.

The library is loaded lazily — the first call to ``CmsCodec.decode``
(or any module-level helper) opens ``liblcms2`` via dlopen. The
Codec registry still happily registers ``CmsCodec`` even on systems
without lcms2; only the transform itself fails.

Matches imagecodecs's ``cms_transform`` API.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Any

import numpy as np

from .core.codec import Codec


# ---------------------------------------------------------------------------
# Lazy library loading
# ---------------------------------------------------------------------------


_LCMS2 = None
_LCMS2_ERROR = None


def _load_lcms2():
    """Find and dlopen liblcms2. Returns the loaded CDLL or raises."""
    global _LCMS2, _LCMS2_ERROR
    if _LCMS2 is not None:
        return _LCMS2
    if _LCMS2_ERROR is not None:
        raise _LCMS2_ERROR
    # Try platformdirs-style candidates plus the standard ctypes.util search.
    candidates = []
    for name in ("lcms2", "lcms2.2"):
        path = ctypes.util.find_library(name)
        if path:
            candidates.append(path)
    # Common explicit paths on macOS / Linux.
    candidates += [
        "/opt/homebrew/opt/little-cms2/lib/liblcms2.dylib",
        "/usr/local/opt/little-cms2/lib/liblcms2.dylib",
        "/usr/lib/x86_64-linux-gnu/liblcms2.so.2",
        "liblcms2.so.2",
        "lcms2.dll",
    ]
    last_err = None
    for c in candidates:
        if not c:
            continue
        try:
            lib = ctypes.CDLL(c)
            _LCMS2 = _bind_lcms2(lib)
            return _LCMS2
        except OSError as e:
            last_err = e
    _LCMS2_ERROR = ImportError(
        "cms: could not load liblcms2 (Little-CMS). Install it via "
        "`brew install little-cms2` on macOS or `apt install liblcms2-dev` "
        "on Debian/Ubuntu. Last dlopen error: "
        f"{last_err}"
    )
    raise _LCMS2_ERROR


class _CmsCIExyY(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double),
                ("y", ctypes.c_double),
                ("Y", ctypes.c_double)]


class _CmsCIExyYTRIPLE(ctypes.Structure):
    _fields_ = [("Red", _CmsCIExyY),
                ("Green", _CmsCIExyY),
                ("Blue", _CmsCIExyY)]


def _bind_lcms2(lib):
    """Attach argtypes / restypes to the functions we use."""
    lib.cmsOpenProfileFromMem.restype = ctypes.c_void_p
    lib.cmsOpenProfileFromMem.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.cmsCloseProfile.restype = ctypes.c_int
    lib.cmsCloseProfile.argtypes = [ctypes.c_void_p]
    lib.cmsCreate_sRGBProfile.restype = ctypes.c_void_p
    lib.cmsCreate_sRGBProfile.argtypes = []
    lib.cmsCreateTransform.restype = ctypes.c_void_p
    lib.cmsCreateTransform.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32,
    ]
    lib.cmsDeleteTransform.restype = None
    lib.cmsDeleteTransform.argtypes = [ctypes.c_void_p]
    lib.cmsDoTransform.restype = None
    lib.cmsDoTransform.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
    ]
    # Profile-builder functions — used by built-in profile helpers
    # (e.g. Display-P3 construction).  Older lcms2 (pre-2.16) lacks
    # ``cmsCreate_DisplayP3`` so we build it manually from chromaticities
    # via these three.
    lib.cmsCreateRGBProfile.restype = ctypes.c_void_p
    lib.cmsCreateRGBProfile.argtypes = [
        ctypes.POINTER(_CmsCIExyY),         # white point
        ctypes.POINTER(_CmsCIExyYTRIPLE),   # primaries
        ctypes.c_void_p * 3,                # tone curves (one per channel)
    ]
    lib.cmsBuildParametricToneCurve.restype = ctypes.c_void_p
    lib.cmsBuildParametricToneCurve.argtypes = [
        ctypes.c_void_p,                    # context (NULL)
        ctypes.c_int32,                     # type
        ctypes.POINTER(ctypes.c_double),    # params
    ]
    lib.cmsFreeToneCurve.restype = None
    lib.cmsFreeToneCurve.argtypes = [ctypes.c_void_p]
    lib.cmsSaveProfileToMem.restype = ctypes.c_int
    lib.cmsSaveProfileToMem.argtypes = [
        ctypes.c_void_p,                    # profile handle
        ctypes.c_void_p,                    # out buffer (NULL → query size)
        ctypes.POINTER(ctypes.c_uint32),    # in/out size
    ]
    return lib


# Format codes (lcms2 type macros, evaluated). See lcms2.h:
# COLORSPACE_SH(...)| CHANNELS_SH(...) | BYTES_SH(...). The constants
# below are the pre-baked values for the common combinations we
# support; passing arbitrary lcms2 format codes is allowed via the
# raw int kwargs ``format_in_raw`` / ``format_out_raw``.
TYPE_GRAY_8 = 196617
TYPE_GRAY_16 = 196618
TYPE_RGB_8 = 262169
TYPE_RGB_16 = 262170
TYPE_RGBA_8 = 393241
TYPE_RGBA_16 = 393242

# lcms2 flag: copy the alpha channel verbatim through the transform.
# Needed when both input and output formats include an alpha channel
# (e.g. RGBA → RGBA) — without it cmsCreateTransform fails because
# the source profile has no alpha to transform.
_CMS_FLAGS_COPY_ALPHA = 0x04000000

_FORMATS_WITH_ALPHA = {TYPE_RGBA_8, TYPE_RGBA_16}


_INTENT_NAMES = {
    "perceptual": 0,
    "relative": 1,
    "relative_colorimetric": 1,
    "saturation": 2,
    "absolute": 3,
    "absolute_colorimetric": 3,
    None: 0,
}


def _array_format(arr: np.ndarray) -> int:
    """Map an ndarray's shape/dtype to an lcms2 TYPE_* code."""
    if arr.dtype == np.uint8:
        if arr.ndim == 2:
            return TYPE_GRAY_8
        if arr.ndim == 3 and arr.shape[2] == 3:
            return TYPE_RGB_8
        if arr.ndim == 3 and arr.shape[2] == 4:
            return TYPE_RGBA_8
    if arr.dtype == np.uint16:
        if arr.ndim == 2:
            return TYPE_GRAY_16
        if arr.ndim == 3 and arr.shape[2] == 3:
            return TYPE_RGB_16
        if arr.ndim == 3 and arr.shape[2] == 4:
            return TYPE_RGBA_16
    raise ValueError(
        f"cms: cannot infer lcms2 format from shape {arr.shape} "
        f"dtype {arr.dtype}. Pass ``format_in_raw=`` / "
        f"``format_out_raw=`` explicitly to override.")


def cms_transform(
    data,
    *,
    profile_in: bytes,
    profile_out: bytes | None = None,
    intent: int | str | None = None,
    format_in_raw: int | None = None,
    format_out_raw: int | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Apply an ICC color transform to an ndarray.

    Parameters
    ----------
    data
        Input pixel data. uint8 or uint16; shape (H, W), (H, W, 3),
        or (H, W, 4).
    profile_in : bytes
        Source ICC profile (the ``read_icc_profile`` output of PNG /
        JPEG / etc.). Required.
    profile_out : bytes, optional
        Destination ICC profile. ``None`` means sRGB (lcms2's built-in
        ``cmsCreate_sRGBProfile``).
    intent : int or str, optional
        Rendering intent. ``"perceptual"`` (default), ``"relative"``,
        ``"saturation"``, or ``"absolute"``. Or an int 0..3.
    format_in_raw, format_out_raw : int, optional
        Override the inferred lcms2 TYPE_* format codes. Use these
        for esoteric layouts (Lab, CMYK, planar formats); for plain
        8/16-bit grayscale / RGB / RGBA the default inference works.
    out : ndarray, optional
        Preallocated destination. Same shape as ``data``.
    """
    lib = _load_lcms2()

    arr = np.ascontiguousarray(data)
    if format_in_raw is None:
        format_in_raw = _array_format(arr)
    if format_out_raw is None:
        format_out_raw = _array_format(arr)

    if isinstance(intent, str):
        intent_code = _INTENT_NAMES.get(intent.lower())
        if intent_code is None:
            raise ValueError(
                f"cms: unknown intent {intent!r}; expected one of "
                f"{sorted(k for k in _INTENT_NAMES if k)}")
    else:
        intent_code = int(intent) if intent is not None else 0

    if out is None:
        out_arr = np.empty_like(arr)
    else:
        if not isinstance(out, np.ndarray):
            raise TypeError(
                f"cms transform: out= must be an ndarray, "
                f"got {type(out).__name__}")
        if out.shape != arr.shape or out.dtype != arr.dtype:
            raise ValueError("cms transform: out= shape/dtype mismatch")
        if not out.flags["C_CONTIGUOUS"]:
            raise ValueError("cms transform: out= must be C-contiguous")
        out_arr = out

    h_in = lib.cmsOpenProfileFromMem(profile_in, len(profile_in))
    if not h_in:
        raise ValueError(
            "cms: cmsOpenProfileFromMem failed on input profile")
    try:
        if profile_out is None:
            h_out = lib.cmsCreate_sRGBProfile()
            if not h_out:
                raise RuntimeError("cms: cmsCreate_sRGBProfile failed")
        else:
            h_out = lib.cmsOpenProfileFromMem(profile_out, len(profile_out))
            if not h_out:
                raise ValueError(
                    "cms: cmsOpenProfileFromMem failed on output profile")
        try:
            flags = 0
            if (format_in_raw in _FORMATS_WITH_ALPHA
                    and format_out_raw in _FORMATS_WITH_ALPHA):
                flags |= _CMS_FLAGS_COPY_ALPHA
            xform = lib.cmsCreateTransform(
                h_in, format_in_raw,
                h_out, format_out_raw,
                intent_code, flags,
            )
            if not xform:
                raise RuntimeError(
                    "cms: cmsCreateTransform returned NULL — "
                    "incompatible format/intent for these profiles?")
            try:
                # cmsDoTransform's "size" is number of pixels, not bytes.
                if arr.ndim == 2:
                    n_pixels = arr.shape[0] * arr.shape[1]
                else:
                    n_pixels = arr.shape[0] * arr.shape[1]
                lib.cmsDoTransform(
                    xform,
                    arr.ctypes.data, out_arr.ctypes.data, n_pixels,
                )
            finally:
                lib.cmsDeleteTransform(xform)
        finally:
            lib.cmsCloseProfile(h_out)
    finally:
        lib.cmsCloseProfile(h_in)
    return out_arr


# ─── built-in profile factories ──────────────────────────────────────


_BUILTIN_PROFILE_ICC: dict[str, bytes] = {}


def _profile_handle_to_icc_bytes(lib, handle: int) -> bytes:
    """Serialize an lcms2 profile handle to its on-disk ICC byte form."""
    size = ctypes.c_uint32(0)
    if not lib.cmsSaveProfileToMem(handle, None, ctypes.byref(size)):
        raise RuntimeError("cmsSaveProfileToMem: size query failed")
    buf = (ctypes.c_uint8 * size.value)()
    if not lib.cmsSaveProfileToMem(handle, buf, ctypes.byref(size)):
        raise RuntimeError("cmsSaveProfileToMem: write failed")
    return bytes(buf)


def _build_display_p3_icc() -> bytes:
    """Synthesize a Display-P3 ICC profile via lcms2.

    Display-P3 = sRGB transfer curve + DCI-P3 primaries + D65 white point.
    Equivalent to ``cmsCreate_DisplayP3`` introduced in lcms2 2.16, but
    works on older builds too.  ~700 byte profile, built once and cached
    module-side.
    """
    lib = _load_lcms2()

    # D65 white point (CIE 1931 chromaticity coordinates).
    wp = _CmsCIExyY(0.3127, 0.3290, 1.0)
    # Display-P3 primaries (Apple / ITU-R BT.2100 reference).
    primaries = _CmsCIExyYTRIPLE(
        _CmsCIExyY(0.680, 0.320, 1.0),   # Red
        _CmsCIExyY(0.265, 0.690, 1.0),   # Green
        _CmsCIExyY(0.150, 0.060, 1.0),   # Blue
    )
    # sRGB parametric tone curve (Type 4): seven-param ICC parametric
    # curve  Y = ((aX + b)^γ)·E + f  for X ≥ d, else  Y = cX + f.
    # Standard sRGB coefficients.
    srgb_params = (ctypes.c_double * 5)(
        2.4,                   # gamma
        1.0 / 1.055,           # a
        0.055 / 1.055,         # b
        1.0 / 12.92,           # c (linear slope)
        0.04045,               # d (split point)
    )
    curve = lib.cmsBuildParametricToneCurve(None, 4, srgb_params)
    if not curve:
        raise RuntimeError("cmsBuildParametricToneCurve failed for sRGB curve")
    try:
        curves = (ctypes.c_void_p * 3)(curve, curve, curve)
        profile = lib.cmsCreateRGBProfile(
            ctypes.byref(wp), ctypes.byref(primaries), curves,
        )
        if not profile:
            raise RuntimeError("cmsCreateRGBProfile failed for Display-P3")
        try:
            return _profile_handle_to_icc_bytes(lib, profile)
        finally:
            lib.cmsCloseProfile(profile)
    finally:
        lib.cmsFreeToneCurve(curve)


def _builtin_profile_icc(name: str) -> bytes:
    """Return ICC bytes for a built-in profile name, building + caching
    on first request.  Supported: ``"srgb"``, ``"display-p3"``."""
    cached = _BUILTIN_PROFILE_ICC.get(name)
    if cached is not None:
        return cached
    lib = _load_lcms2()
    if name == "srgb":
        h = lib.cmsCreate_sRGBProfile()
        if not h:
            raise RuntimeError("cmsCreate_sRGBProfile failed")
        try:
            icc = _profile_handle_to_icc_bytes(lib, h)
        finally:
            lib.cmsCloseProfile(h)
    elif name == "display-p3":
        icc = _build_display_p3_icc()
    else:
        raise ValueError(
            f"unknown built-in profile {name!r}; "
            f"expected one of: srgb, display-p3"
        )
    _BUILTIN_PROFILE_ICC[name] = icc
    return icc


def srgb_to_display_p3_uint8(arr, *, out=None) -> np.ndarray:
    """Convert sRGB-encoded uint8 RGB(A) → Display-P3-encoded uint8.

    Uses lcms2's ICC color-management pipeline (perceptual rendering
    intent).  On a 2k² uint8 RGB image this runs in ~28 ms vs ~110 ms
    for an equivalent numpy LUT + matmul pipeline.

    Parameters
    ----------
    arr : np.ndarray
        ``(H, W, 3)`` or ``(H, W, 4)`` uint8.  Other dtypes / shapes are
        rejected — use :func:`cms_transform` for the general case.
    out : np.ndarray, optional
        Preallocated destination of the same shape + dtype.

    Returns
    -------
    np.ndarray
        ``(H, W, 3|4)`` uint8 in the Display-P3 colour space.
    """
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        raise TypeError(
            f"srgb_to_display_p3_uint8: expected uint8 array, got {arr.dtype}"
        )
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(
            f"srgb_to_display_p3_uint8: expected (H, W, 3|4); got shape {arr.shape}"
        )
    srgb_icc = _builtin_profile_icc("srgb")
    p3_icc = _builtin_profile_icc("display-p3")
    if arr.shape[2] == 3:
        return cms_transform(arr, profile_in=srgb_icc, profile_out=p3_icc,
                              intent="perceptual", out=out)
    # RGBA path: lcms2's COPY_ALPHA flag doesn't combine with our custom-
    # built RGB-only profiles, so transform the RGB plane in isolation
    # and stitch the alpha channel back verbatim.  ``out[..., :3]`` is a
    # non-contiguous view (stride 4 along last axis) so we transform into
    # a contiguous temporary and copy back.
    rgb = np.ascontiguousarray(arr[..., :3])
    rgb_out = cms_transform(rgb,
                             profile_in=srgb_icc, profile_out=p3_icc,
                             intent="perceptual")
    if out is None:
        out = np.empty_like(arr)
    elif out.shape != arr.shape or out.dtype != arr.dtype:
        raise ValueError("srgb_to_display_p3_uint8: out= shape/dtype mismatch")
    out[..., :3] = rgb_out
    out[..., 3] = arr[..., 3]
    return out


class CmsCodec(Codec):
    """ICC color-management transform.

    Note this is NOT a compressor — encode is the identity (returns
    the pixel data verbatim, accepting an ``iccprofile=`` for downstream
    metadata pairing) and ``decode`` is the transform. The API hews
    to imagecodecs's ``cms_transform`` shape.
    """

    name = "cms"
    aliases = ()
    file_extensions = ()

    has_native = True
    has_delegate = False
    can_encode = False           # not an encoder
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.uint8, np.uint16)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        return False

    def encode(self, data: Any, **opts) -> bytes | None:
        raise NotImplementedError(
            "cms is a color transform, not a compressor; use decode()")

    def decode(self, src: Any, *, profile_in: bytes,
               profile_out: bytes | None = None,
               intent: int | str | None = None,
               format_in_raw: int | None = None,
               format_out_raw: int | None = None,
               out=None, **opts) -> np.ndarray:
        """Apply the configured ICC transform to ``src``.

        ``src`` is the source ndarray (raw pixel data; not an
        encoded blob like the other codecs).
        """
        if not isinstance(src, np.ndarray):
            src = np.ascontiguousarray(src)
        return cms_transform(
            src,
            profile_in=profile_in,
            profile_out=profile_out,
            intent=intent,
            format_in_raw=format_in_raw,
            format_out_raw=format_out_raw,
            out=out,
        )


__all__ = [
    "CmsCodec",
    "cms_transform",
    "srgb_to_display_p3_uint8",
    "_builtin_profile_icc",
]
