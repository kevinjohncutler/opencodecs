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
exposed as uint8 (bit_depth <= 8) or uint16 (bit_depth <= 16).

Stream layout
=============

The encoder writes a bare JPEG-LS codestream (SOI, SOF55, [LSE], SOS,
EOI) with no SPIFF header, and sample interleave (ILV=2) for every
multi-component image. Both are conforming choices under ITU-T T.87 |
ISO/IEC 14495-1; SPIFF (ITU-T T.84 Annex F) is optional, and a bare
codestream is what DICOM and TIFF embed. imagecodecs writes a SPIFF
header with a fixed 300 dpi resolution and uses line interleave for
four components, so the bytes differ by those segments while the
decoded pixels are identical in both directions.

The decoder reads all three interleave modes, including ILV=0 (one
scan per component), and always returns (H, W, C) for a
multi-component frame.
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
    charls_jpegls_decoder_get_interleave_mode,
    charls_interleave_mode,
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


def _resolve_near(near_lossless, level):
    """The JPEG-LS NEAR parameter from ``near_lossless`` and ``level``.

    ``level`` is imagecodecs' name for the same thing
    (``jpegls_encode(data, level=N)`` sets NEAR=N, None means 0), so it
    is accepted as an alias. Passing both with different values is a
    contradiction and raises rather than silently picking one.
    """
    near = 0 if near_lossless is None else int(near_lossless)
    if level is not None:
        lvl = max(0, int(level))
        if near_lossless is not None and near != lvl:
            raise ValueError(
                f"jpegls encode: level={level} and near_lossless="
                f"{near_lossless} disagree; pass one of them")
        near = lvl
    if near < 0:
        raise ValueError(
            f"jpegls encode: near_lossless must be >= 0 (got {near})")
    return near


def encode(data, *, near_lossless: int | None = None,
           level: int | None = None) -> bytes:
    """Encode an ndarray as JPEG-LS.

    Parameters
    ----------
    data
        2D (H, W) or 3D (H, W, C) array of uint8 or uint16.
    near_lossless : int, optional
        The JPEG-LS NEAR parameter (ITU-T T.87). 0 (the default) is
        mathematically lossless. A positive value bounds the error:
        each decoded sample is within ``near_lossless`` of the source.
        Larger means smaller files and more error.
    level : int, optional
        Alias of ``near_lossless`` under imagecodecs' name, so
        ``encode(a, level=2)`` writes the same stream as
        ``imagecodecs.jpegls_encode(a, level=2)`` apart from the SPIFF
        header imagecodecs adds.
    """
    cdef int near = _resolve_near(near_lossless, level)
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
        if near > 0:
            rc = charls_jpegls_encoder_set_near_lossless(enc, near)
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

    A multi-component frame is returned as (H, W, C) whatever
    interleave mode the stream uses. ILV=1 (line) and ILV=2 (sample)
    decode straight into the output; ILV=0 (one scan per component,
    ITU-T T.87 Annex C.2.3) decodes into component planes, which are
    then interleaved into the output. imagecodecs returns the same
    (H, W, C) layout for all three.

    ``out=`` is a preallocated C-contiguous ndarray of the decoded shape
    and dtype. For a single component or an interleaved stream CharLS
    writes straight into it; an ILV=0 stream goes through one planar
    scratch buffer first.
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        charls_jpegls_decoder* dec = NULL
        charls_frame_info info
        charls_interleave_mode ilv_mode
        size_t dst_size = 0
        uint32_t stride
        int rc
        int itemsize
        bint planar
        cnp.ndarray out_arr
        cnp.ndarray dst_arr
        void* dst_ptr
        size_t dst_bytes
        tuple expected_shape
        object expected_dtype

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if srcsize == 0:
        raise CharlsError("jpegls decode: empty input")

    dec = charls_jpegls_decoder_create()
    if dec == NULL:
        raise CharlsError("charls_jpegls_decoder_create failed")
    try:
        rc = charls_jpegls_decoder_set_source_buffer(dec, &src[0], srcsize)
        _check(rc, "set_source_buffer")
        with nogil:
            rc = charls_jpegls_decoder_read_header(dec)
        _check(rc, "read_header")
        rc = charls_jpegls_decoder_get_frame_info(dec, &info)
        _check(rc, "get_frame_info")
        rc = charls_jpegls_decoder_get_interleave_mode(dec, &ilv_mode)
        _check(rc, "get_interleave_mode")
        itemsize = 1 if info.bits_per_sample <= 8 else 2
        expected_dtype = np.uint8 if itemsize == 1 else np.uint16
        planar = (info.component_count > 1
                  and <int> ilv_mode == 0)  # CHARLS_INTERLEAVE_MODE_NONE
        if info.component_count == 1:
            expected_shape = (int(info.height), int(info.width))
        else:
            expected_shape = (int(info.height), int(info.width),
                              int(info.component_count))

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
            out_arr = np.empty(expected_shape, dtype=expected_dtype)

        if planar:
            # ILV=0: CharLS writes component planes one after another,
            # each row ``width`` samples long, so the stride is a plane
            # row and the destination is (C, H, W).
            dst_arr = np.empty(
                (int(info.component_count), int(info.height),
                 int(info.width)), dtype=expected_dtype)
            stride = <uint32_t> (info.width * itemsize)
        else:
            dst_arr = out_arr
            stride = <uint32_t> (info.width * info.component_count * itemsize)

        rc = charls_jpegls_decoder_get_destination_size(dec, stride, &dst_size)
        _check(rc, "get_destination_size")
        if <size_t> dst_arr.nbytes < dst_size:
            raise CharlsError(
                f"output buffer too small ({dst_arr.nbytes} < {dst_size})"
            )
        # The entropy decode is the expensive half and charls touches
        # nothing Python in it: the destination is a raw pointer into
        # an array this function owns, and the source buffer was handed
        # over before. Holding the GIL through it made every caller
        # decoding JPEG-LS frames on threads measure 0.99x, which is a
        # thread pool paying overhead to take turns.
        dst_ptr = <void*> cnp.PyArray_DATA(dst_arr)
        dst_bytes = <size_t> dst_arr.nbytes
        with nogil:
            rc = charls_jpegls_decoder_decode_to_buffer(
                dec, dst_ptr, dst_bytes, stride)
        _check(rc, "decode_to_buffer")
        if planar:
            out_arr[...] = np.moveaxis(dst_arr, 0, -1)
        return out_arr
    finally:
        charls_jpegls_decoder_destroy(dec)
