# opencodecs/codecs/_charls.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""JPEG-LS codec via CharLS (libcharls).

JPEG-LS is the predictive JPEG variant standardized in ISO/IEC 14495-1
(2003) — lossless or "near-lossless" compression (bounded-error
quantization). Used heavily in medical DICOM transfer syntaxes and
remote sensing.

Bit depths
==========

Supports 2-16 bits per sample, 1 / 3 / 4 components. Pixels are
exposed as uint8 (bit_depth ≤ 8) or uint16 (bit_depth ≤ 16).
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t, uint16_t, uint32_t
from libc.string cimport memcpy

import numpy as np
cimport numpy as cnp

from charls cimport (
    charls_jpegls_encoder, charls_jpegls_decoder,
    charls_frame_info,
    charls_jpegls_encoder_create, charls_jpegls_encoder_destroy,
    charls_jpegls_encoder_set_frame_info,
    charls_jpegls_encoder_set_near_lossless,
    charls_jpegls_encoder_set_interleave_mode,
    charls_jpegls_encoder_get_estimated_destination_size,
    charls_jpegls_encoder_set_destination_buffer,
    charls_jpegls_encoder_encode_from_buffer,
    charls_jpegls_encoder_get_bytes_written,
    charls_jpegls_decoder_create, charls_jpegls_decoder_destroy,
    charls_jpegls_decoder_set_source_buffer,
    charls_jpegls_decoder_read_header,
    charls_jpegls_decoder_get_frame_info,
    charls_jpegls_decoder_get_destination_size,
    charls_jpegls_decoder_decode_to_buffer,
    charls_get_error_message,
    CHARLS_INTERLEAVE_MODE_SAMPLE,
)

cnp.import_array()


class CharlsError(RuntimeError):
    """Raised on CharLS encode/decode failures."""


cdef _check(int errc, str where):
    if errc != 0:
        msg = charls_get_error_message(errc).decode("ascii", errors="replace")
        raise CharlsError(f"{where}: {msg} (errc={errc})")


def encode(data, *, near_lossless: int = 0) -> bytes:
    """Encode an ndarray as JPEG-LS.

    Parameters
    ----------
    data
        2D (H, W) or 3D (H, W, C) array of uint8 or uint16.
    near_lossless : int
        0 (default) = mathematically lossless. Positive integers up to
        9 enable JPEG-LS's bounded-error mode: each decoded sample is
        within ``near_lossless`` of the source. Larger = smaller files,
        more error.
    """
    cdef:
        cnp.ndarray arr
        charls_jpegls_encoder* enc = NULL
        charls_frame_info info
        size_t dst_size = 0
        size_t written = 0
        bytes out
        int rc
        uint32_t stride
        int bps
        int component_count

    if not isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data)
    else:
        arr = np.ascontiguousarray(data)

    if arr.dtype == np.uint8:
        bps = 8
    elif arr.dtype == np.uint16:
        bps = 16
    else:
        raise CharlsError(
            f"CharLS encode: unsupported dtype {arr.dtype}; "
            f"expected uint8 or uint16"
        )

    if arr.ndim == 2:
        component_count = 1
        info.width = <uint32_t> arr.shape[1]
        info.height = <uint32_t> arr.shape[0]
        stride = <uint32_t> (arr.shape[1] * arr.dtype.itemsize)
    elif arr.ndim == 3:
        component_count = arr.shape[2]
        if component_count not in (1, 3, 4):
            raise CharlsError(
                f"CharLS encode: component count must be 1, 3, or 4 "
                f"(got {component_count})"
            )
        info.width = <uint32_t> arr.shape[1]
        info.height = <uint32_t> arr.shape[0]
        stride = <uint32_t> (arr.shape[1] * component_count * arr.dtype.itemsize)
    else:
        raise CharlsError(
            f"CharLS encode: unsupported ndim {arr.ndim}"
        )
    info.bits_per_sample = bps
    info.component_count = component_count

    enc = charls_jpegls_encoder_create()
    if enc == NULL:
        raise CharlsError("charls_jpegls_encoder_create failed")
    try:
        rc = charls_jpegls_encoder_set_frame_info(enc, &info)
        _check(rc, "set_frame_info")
        if near_lossless > 0:
            rc = charls_jpegls_encoder_set_near_lossless(enc, near_lossless)
            _check(rc, "set_near_lossless")
        # Interleaved sample layout (RGBRGB) for multi-component frames.
        # 1-component frames default to mode=NONE which is correct.
        if component_count > 1:
            rc = charls_jpegls_encoder_set_interleave_mode(
                enc, CHARLS_INTERLEAVE_MODE_SAMPLE)
            _check(rc, "set_interleave_mode")
        rc = charls_jpegls_encoder_get_estimated_destination_size(enc, &dst_size)
        _check(rc, "get_estimated_destination_size")
        # CharLS's estimate assumes compressible content; on random
        # pixels the encoded output can exceed it. Pad to at least
        # ``raw_size + 64KB`` so we never bounce off "destination too
        # small". Wasted bytes get trimmed at the end via out[:written].
        if dst_size < <size_t> arr.nbytes + 65536:
            dst_size = <size_t> arr.nbytes + 65536
        out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dst_size)
        rc = charls_jpegls_encoder_set_destination_buffer(
            enc, <void*> <const char*> out, dst_size,
        )
        _check(rc, "set_destination_buffer")
        rc = charls_jpegls_encoder_encode_from_buffer(
            enc, <const void*> cnp.PyArray_DATA(arr), arr.nbytes, stride,
        )
        _check(rc, "encode_from_buffer")
        rc = charls_jpegls_encoder_get_bytes_written(enc, &written)
        _check(rc, "get_bytes_written")
        # Truncate to the actual encoded size.
        return out[:written]
    finally:
        charls_jpegls_encoder_destroy(enc)


def decode(data, *, out=None) -> np.ndarray:
    """Decode JPEG-LS bytes to an ndarray.

    ``out=`` is a preallocated ndarray. charls's
    charls_jpegls_decoder_decode_to_buffer writes directly into the
    caller's buffer — true zero-alloc. See ``_png.decode`` for the
    full contract.
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        charls_jpegls_decoder* dec = NULL
        charls_frame_info info
        size_t dst_size = 0
        uint32_t stride
        int rc
        cnp.ndarray out_arr
        cnp.npy_intp shape[3]
        int ndim
        int bps
        tuple expected_shape
        object expected_dtype

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <size_t> src.shape[0]

    dec = charls_jpegls_decoder_create()
    if dec == NULL:
        raise CharlsError("charls_jpegls_decoder_create failed")
    try:
        rc = charls_jpegls_decoder_set_source_buffer(dec, &src[0], srcsize)
        _check(rc, "set_source_buffer")
        rc = charls_jpegls_decoder_read_header(dec)
        _check(rc, "read_header")
        rc = charls_jpegls_decoder_get_frame_info(dec, &info)
        _check(rc, "get_frame_info")
        bps = info.bits_per_sample
        if info.component_count == 1:
            ndim = 2
            shape[0] = info.height
            shape[1] = info.width
            stride = info.width * (1 if bps <= 8 else 2)
            expected_shape = (int(info.height), int(info.width))
        else:
            ndim = 3
            shape[0] = info.height
            shape[1] = info.width
            shape[2] = info.component_count
            stride = (info.width * info.component_count *
                      (1 if bps <= 8 else 2))
            expected_shape = (int(info.height), int(info.width),
                              int(info.component_count))
        expected_dtype = np.uint8 if bps <= 8 else np.uint16

        if out is not None:
            if not isinstance(out, np.ndarray):
                raise CharlsError(
                    f"jpegls decode: out= must be an ndarray, "
                    f"got {type(out).__name__}")
            if out.shape != expected_shape:
                raise CharlsError(
                    f"jpegls decode: out= shape {out.shape} does not match "
                    f"expected {expected_shape}")
            if out.dtype != expected_dtype:
                raise CharlsError(
                    f"jpegls decode: out= dtype {out.dtype} does not match "
                    f"expected {np.dtype(expected_dtype)}")
            if not out.flags['C_CONTIGUOUS']:
                raise CharlsError("jpegls decode: out= must be C-contiguous")
            out_arr = out
        else:
            out_arr = cnp.PyArray_EMPTY(
                ndim, shape,
                cnp.NPY_UINT8 if bps <= 8 else cnp.NPY_UINT16,
                0,
            )
        rc = charls_jpegls_decoder_get_destination_size(dec, stride, &dst_size)
        _check(rc, "get_destination_size")
        if <size_t> out_arr.nbytes < dst_size:
            raise CharlsError(
                f"output buffer too small ({out_arr.nbytes} < {dst_size})"
            )
        rc = charls_jpegls_decoder_decode_to_buffer(
            dec, <void*> cnp.PyArray_DATA(out_arr), out_arr.nbytes, stride,
        )
        _check(rc, "decode_to_buffer")
        return out_arr
    finally:
        charls_jpegls_decoder_destroy(dec)
