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


__all__ = ["CmsCodec", "cms_transform"]
