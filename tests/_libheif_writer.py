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
