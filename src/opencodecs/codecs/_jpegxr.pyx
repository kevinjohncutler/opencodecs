# opencodecs/codecs/_jpegxr.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: language_level = 3

"""JPEG XR (ITU-T T.832) decode via jxrlib.

Zeiss CZI stores sub-blocks with compression 4 as JPEG XR; slide scanners
use it for most whole-slide acquisitions. Decode only: images come back in
their stored layout, (height, width) or (height, width, samples), with the
dtype the pixel format declares.
"""

import numpy as np
cimport numpy as cnp

from libc.stdint cimport uint8_t

cnp.import_array()


cdef extern from "jpegxr_shim.h" nogil:
    ctypedef struct oc_jxr:
        pass

    ctypedef struct oc_jxr_info:
        int width
        int height
        int channels
        int bitdepth
        int bits_per_pixel
        int has_alpha
        int bgr

    int oc_jxr_open(const void* data, size_t size, oc_jxr** handle,
                    oc_jxr_info* info)
    int oc_jxr_copy(oc_jxr* handle, void* out, size_t stride)
    void oc_jxr_close(oc_jxr* handle)


class JpegXrError(RuntimeError):
    """Raised when jxrlib cannot parse or decode an image."""


# jxrlib BITDEPTH_BITS -> numpy dtype. BD_1 (bilevel) and the packed
# 5/10/565-bit formats have no plain array layout and are refused.
_DTYPES = {1: "u1", 2: "u2", 3: "i2", 4: "f2", 5: "u4", 6: "i4", 7: "f4"}


cdef object _layout(oc_jxr_info* info):
    name = _DTYPES.get(info.bitdepth)
    if name is None:
        raise JpegXrError(
            f"JPEG XR pixel format with bit depth code {info.bitdepth} "
            f"has no array layout")
    dtype = np.dtype(name)
    unit = dtype.itemsize * 8
    if info.bits_per_pixel <= 0 or info.bits_per_pixel % unit:
        raise JpegXrError(
            f"JPEG XR pixel of {info.bits_per_pixel} bits is not a whole "
            f"number of {dtype} samples")
    samples = info.bits_per_pixel // unit
    shape = ((info.height, info.width) if samples == 1
             else (info.height, info.width, samples))
    return shape, dtype, samples


def decode(data, *, out=None, bgr=None):
    """Decode one JPEG XR image.

    ``out`` is an optional writable C-contiguous array (or buffer) of
    exactly the image's size to decode into; it is returned, reshaped.
    ``bgr=True`` or ``False`` asks for the first three channels in that
    order and swaps them when the stream stores the other one; ``None``
    keeps the stored order. Returns ``(height, width)`` or
    ``(height, width, samples)``.
    """
    cdef:
        const uint8_t[::1] src = data
        oc_jxr* handle = NULL
        oc_jxr_info info
        int err
        size_t stride
        void* dst
        size_t size = <size_t> src.shape[0]
        const void* ptr

    if size == 0:
        raise JpegXrError("empty JPEG XR stream")
    ptr = <const void*> &src[0]
    with nogil:
        err = oc_jxr_open(ptr, size, &handle, &info)
    if err != 0:
        raise JpegXrError(f"JPEG XR header parse failed (jxrlib error {err})")
    try:
        shape, dtype, samples = _layout(&info)
        if out is None:
            result = np.empty(shape, dtype=dtype)
        else:
            if not isinstance(out, np.ndarray):
                out = np.frombuffer(out, dtype=np.uint8)
            if not out.flags.c_contiguous or not out.flags.writeable:
                raise ValueError("out must be a writable C-contiguous array")
            nbytes = int(np.prod(shape)) * dtype.itemsize
            if out.nbytes != nbytes:
                raise ValueError(
                    f"out holds {out.nbytes} bytes; the image needs {nbytes}")
            result = out.reshape(-1).view(dtype).reshape(shape)
        stride = <size_t> (info.width * (info.bits_per_pixel // 8))
        dst = <void*> cnp.PyArray_DATA(result)
        with nogil:
            err = oc_jxr_copy(handle, dst, stride)
        if err != 0:
            raise JpegXrError(f"JPEG XR decode failed (jxrlib error {err})")
    finally:
        oc_jxr_close(handle)
    if bgr is not None and samples >= 3 and bool(bgr) != bool(info.bgr):
        result[..., :3] = result[..., 2::-1].copy()
    return result


def info(data):
    """Header fields of a JPEG XR stream, without decoding pixels."""
    cdef:
        const uint8_t[::1] src = data
        oc_jxr* handle = NULL
        oc_jxr_info i
        int err
    if src.shape[0] == 0:
        raise JpegXrError("empty JPEG XR stream")
    err = oc_jxr_open(<const void*> &src[0], <size_t> src.shape[0], &handle, &i)
    if err != 0:
        raise JpegXrError(f"JPEG XR header parse failed (jxrlib error {err})")
    oc_jxr_close(handle)
    return {"width": i.width, "height": i.height, "channels": i.channels,
            "bitdepth": i.bitdepth, "bits_per_pixel": i.bits_per_pixel,
            "has_alpha": bool(i.has_alpha), "bgr": bool(i.bgr)}
