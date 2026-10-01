"""Write a multi-image HEIF with libheif itself, through ctypes.

Nothing available here writes one otherwise: imagecodecs ships no
heif_encode, and this package's own encoder takes a single image. That
matters beyond convenience -- a fixture produced by the code under test
and read back by the same code would pass even if both agreed on
something wrong. Driving libheif directly keeps the reference outside
the code being tested.

Returns None when libheif cannot be found or has no HEVC encoder
compiled in, so callers skip rather than fail.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob

import numpy as np

# libheif enum values used below, from heif.h.
_COLORSPACE_RGB = 1
_CHROMA_INTERLEAVED_RGB = 10
_COMPRESSION_HEVC = 1          # not 2; 2 is AVC
_CHANNEL_INTERLEAVED = 10


class _Error(ctypes.Structure):
    _fields_ = [("code", ctypes.c_int),
                ("subcode", ctypes.c_int),
                ("message", ctypes.c_char_p)]


def _load():
    candidates = glob.glob("/opt/homebrew/opt/libheif/lib/libheif*.dylib")
    candidates += glob.glob("/usr/local/opt/libheif/lib/libheif*.dylib")
    found = ctypes.util.find_library("heif")
    if found:
        candidates.append(found)
    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        lib.heif_context_alloc.restype = ctypes.c_void_p
        for fn in ("heif_image_create", "heif_image_add_plane",
                   "heif_context_get_encoder_for_format",
                   "heif_context_encode_image", "heif_context_write_to_file",
                   "heif_context_read_from_file"):
            getattr(lib, fn).restype = _Error
        lib.heif_image_get_plane.restype = ctypes.POINTER(ctypes.c_uint8)
        lib.heif_context_get_number_of_top_level_images.restype = ctypes.c_int
        return lib
    return None


def write_multi_image_heif(path, frames) -> int | None:
    """Write ``frames`` as N top-level images. Returns N, or None.

    ``frames`` is a sequence of (H, W, 3) uint8 arrays; they need not
    share a shape, which is the case a HEIF with a depth map has.
    """
    lib = _load()
    if lib is None:
        return None
    ctx = ctypes.c_void_p(lib.heif_context_alloc())
    enc = ctypes.c_void_p()
    err = lib.heif_context_get_encoder_for_format(
        ctx, _COMPRESSION_HEVC, ctypes.byref(enc))
    if err.code != 0:
        return None                      # no HEVC encoder in this build
    for f in frames:
        f = np.ascontiguousarray(f, dtype=np.uint8)
        h, w = f.shape[:2]
        img = ctypes.c_void_p()
        err = lib.heif_image_create(w, h, _COLORSPACE_RGB,
                                    _CHROMA_INTERLEAVED_RGB,
                                    ctypes.byref(img))
        if err.code != 0:
            return None
        lib.heif_image_add_plane(img, _CHANNEL_INTERLEAVED, w, h, 8)
        stride = ctypes.c_int()
        plane = lib.heif_image_get_plane(img, _CHANNEL_INTERLEAVED,
                                         ctypes.byref(stride))
        base = ctypes.addressof(plane.contents)
        for y in range(h):
            ctypes.memmove(base + y * stride.value, f[y].ctypes.data, w * 3)
        err = lib.heif_context_encode_image(ctx, img, enc, None, None)
        if err.code != 0:
            return None
    err = lib.heif_context_write_to_file(ctx, str(path).encode())
    if err.code != 0:
        return None
    return count_top_level_images(path)


def count_top_level_images(path) -> int | None:
    """Ask libheif how many top-level images a file has.

    The independent check that the fixture really is what it claims,
    and separately that our own count agrees with libheif's.
    """
    lib = _load()
    if lib is None:
        return None
    ctx = ctypes.c_void_p(lib.heif_context_alloc())
    err = lib.heif_context_read_from_file(ctx, str(path).encode(), None)
    if err.code != 0:
        return None
    return int(lib.heif_context_get_number_of_top_level_images(ctx))


_COLORSPACE_MONOCHROME = 2
_CHROMA_MONOCHROME = 0
_CHANNEL_Y = 0


def write_monochrome_heif(path, gray) -> bool | None:
    """Write a 2-D uint8 array as a lossless monochrome HEIF with libheif.

    Returns True on success, or None when libheif or its HEVC encoder is
    unavailable. The image is created in libheif's monochrome colorspace
    with a single Y plane, which is how libheif and the HEVC encoder code
    a gray image (chroma_format_idc 0).
    """
    lib = _load()
    if lib is None:
        return None
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    h, w = gray.shape
    ctx = ctypes.c_void_p(lib.heif_context_alloc())
    enc = ctypes.c_void_p()
    err = lib.heif_context_get_encoder_for_format(
        ctx, _COMPRESSION_HEVC, ctypes.byref(enc))
    if err.code != 0:
        return None
    lib.heif_encoder_set_lossless.restype = _Error
    if lib.heif_encoder_set_lossless(enc, 1).code != 0:
        return None
    img = ctypes.c_void_p()
    err = lib.heif_image_create(w, h, _COLORSPACE_MONOCHROME,
                                _CHROMA_MONOCHROME, ctypes.byref(img))
    if err.code != 0:
        return None
    if lib.heif_image_add_plane(img, _CHANNEL_Y, w, h, 8).code != 0:
        return None
    stride = ctypes.c_int()
    plane = lib.heif_image_get_plane(img, _CHANNEL_Y, ctypes.byref(stride))
    base = ctypes.addressof(plane.contents)
    for y in range(h):
        ctypes.memmove(base + y * stride.value, gray[y].ctypes.data, w)
    if lib.heif_context_encode_image(ctx, img, enc, None, None).code != 0:
        return None
    if lib.heif_context_write_to_file(ctx, str(path).encode()).code != 0:
        return None
    return True


def primary_image_layout(path) -> tuple | None:
    """libheif's view of the primary image: (monochrome, has_alpha, bits).

    ``monochrome`` comes from heif_image_handle_get_preferred_decoding_colorspace,
    the coded configuration rather than anything our decoder decides.
    """
    lib = _load()
    if lib is None:
        return None
    ctx = ctypes.c_void_p(lib.heif_context_alloc())
    err = lib.heif_context_read_from_file(ctx, str(path).encode(), None)
    if err.code != 0:
        return None
    lib.heif_context_get_primary_image_handle.restype = _Error
    lib.heif_image_handle_get_preferred_decoding_colorspace.restype = _Error
    handle = ctypes.c_void_p()
    if lib.heif_context_get_primary_image_handle(
            ctx, ctypes.byref(handle)).code != 0:
        return None
    colorspace, chroma = ctypes.c_int(), ctypes.c_int()
    if lib.heif_image_handle_get_preferred_decoding_colorspace(
            handle, ctypes.byref(colorspace), ctypes.byref(chroma)).code != 0:
        return None
    alpha = bool(lib.heif_image_handle_has_alpha_channel(handle))
    bits = int(lib.heif_image_handle_get_luma_bits_per_pixel(handle))
    return colorspace.value == _COLORSPACE_MONOCHROME, alpha, bits


_COLORSPACE_YCBCR = 0
_CHROMA_444 = 3
CHANNEL_Y, CHANNEL_CB, CHANNEL_CR, CHANNEL_ALPHA = 0, 1, 2, 6


def write_planes_heif(path, planes, *, color=False) -> bool | None:
    """Write a lossless HEIF from separate planes, each at its own depth.

    ``planes`` is a list of ``(channel, array, bits)``: ``CHANNEL_Y`` and
    optionally ``CHANNEL_ALPHA`` for a monochrome image, or Y, Cb and Cr
    (4:4:4) plus optional alpha with ``color=True``. HEIF codes alpha as
    a separate image, so its depth may differ from the main image's;
    libheif writes such a file, which our own encoder never does.
    Returns True, or None when libheif or its HEVC encoder is missing.
    """
    lib = _load()
    if lib is None:
        return None
    lib.heif_encoder_set_lossless.restype = _Error
    h, w = planes[0][1].shape
    ctx = ctypes.c_void_p(lib.heif_context_alloc())
    enc = ctypes.c_void_p()
    if lib.heif_context_get_encoder_for_format(
            ctx, _COMPRESSION_HEVC, ctypes.byref(enc)).code != 0:
        return None
    if lib.heif_encoder_set_lossless(enc, 1).code != 0:
        return None
    img = ctypes.c_void_p()
    if color:
        cs, chroma = _COLORSPACE_YCBCR, _CHROMA_444
    else:
        cs, chroma = _COLORSPACE_MONOCHROME, _CHROMA_MONOCHROME
    if lib.heif_image_create(w, h, cs, chroma, ctypes.byref(img)).code != 0:
        return None
    for channel, arr, bits in planes:
        dtype = np.uint8 if bits <= 8 else np.uint16
        arr = np.ascontiguousarray(arr, dtype=dtype)
        if lib.heif_image_add_plane(img, channel, w, h, bits).code != 0:
            return None
        stride = ctypes.c_int()
        plane = lib.heif_image_get_plane(img, channel, ctypes.byref(stride))
        base = ctypes.addressof(plane.contents)
        for y in range(h):
            ctypes.memmove(base + y * stride.value, arr[y].ctypes.data,
                           arr[y].nbytes)
    if lib.heif_context_encode_image(ctx, img, enc, None, None).code != 0:
        return None
    if lib.heif_context_write_to_file(ctx, str(path).encode()).code != 0:
        return None
    return True


def encoder_name_for_format(compression: int) -> str | None:
    """Name of the encoder libheif picks for a heif_compression_format value.

    An independent reading of what an integer ``compression`` means: it
    asks the linked libheif, not our own constants. None when libheif or
    an encoder for that format is unavailable.
    """
    lib = _load()
    if lib is None:
        return None
    lib.heif_encoder_get_name.restype = ctypes.c_char_p
    ctx = ctypes.c_void_p(lib.heif_context_alloc())
    enc = ctypes.c_void_p()
    err = lib.heif_context_get_encoder_for_format(
        ctx, int(compression), ctypes.byref(enc))
    if err.code != 0 or not enc.value:
        return None
    name = lib.heif_encoder_get_name(enc)
    lib.heif_encoder_release(enc)
    return name.decode() if name else None
