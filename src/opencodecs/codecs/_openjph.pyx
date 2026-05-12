# opencodecs/codecs/_openjph.pyx
# distutils: language = c++
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""HTJ2K codec via OpenJPH (libopenjph).

High-Throughput JPEG-2000 (ISO/IEC 15444-15) — a block-coder
replacement for the Part-1 EBCOT entropy coder. Same wavelet front
end as classic JPEG-2000 but ~10-20x faster encode/decode.

Built on top of OpenJPH's C++ ``ojph::codestream`` API via a thin
C shim (``openjph_shim.cpp``). All I/O goes through OpenJPH's
``mem_infile`` / ``mem_outfile`` so there are no temp files.

Supported pixels
================

  * 1, 3, or 4 components
  * 1-16 bit_depth (uint8 if bit_depth <= 8 else uint16)
  * 2-D ``(H, W)`` or 3-D ``(H, W, C)`` ndarrays

Output codestream is always raw HTJ2K (.j2c-style) — no JP2 box wrapping.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t, uint16_t
from libc.string cimport memcpy

import numpy as np
cimport numpy as cnp

from openjph cimport (
    opencodecs_htj2k_encode,
    opencodecs_htj2k_decode,
    opencodecs_htj2k_decode_info,
    opencodecs_htj2k_free,
    opencodecs_htj2k_last_error,
)

cnp.import_array()


class OpenJphError(RuntimeError):
    """Raised on OpenJPH (HTJ2K) encode/decode failures."""


cdef _raise(int rc, str where):
    msg = opencodecs_htj2k_last_error().decode("utf-8", errors="replace")
    raise OpenJphError(f"{where}: {msg} (rc={rc})")


def encode(
    data,
    *,
    level: float | None = None,
    num_decomp: int = 5,
) -> bytes:
    """Encode an ndarray as an HTJ2K codestream.

    Parameters
    ----------
    data
        2-D ``(H, W)`` grayscale or 3-D ``(H, W, C)`` multi-component
        array of uint8 / uint16 (or int8 / int16). C must be 1, 3, or 4.
    level : float, optional
        ``None`` (default) selects reversible (mathematically lossless)
        HTJ2K. A float in roughly ``(0, 1]`` selects irreversible (lossy)
        with the value used as the quantization base step delta.
        Smaller numbers -> closer to lossless, larger files; bigger
        numbers -> smaller files, more loss.
    num_decomp : int
        DWT decomposition levels (default 5). Reduce on tiny images.

    Returns
    -------
    bytes
        Raw HTJ2K codestream (.j2c).
    """
    cdef:
        cnp.ndarray arr
        int width, height, components, bit_depth, is_signed_in
        int bytes_per_sample, reversible
        int num_decomp_c = <int> num_decomp
        float irrev_delta
        void* out_buf = NULL
        size_t out_size = 0
        int rc

    if not isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data)
    else:
        arr = np.ascontiguousarray(data)

    if arr.dtype == np.uint8:
        bit_depth = 8
        bytes_per_sample = 1
        is_signed_in = 0
    elif arr.dtype == np.int8:
        bit_depth = 8
        bytes_per_sample = 1
        is_signed_in = 1
    elif arr.dtype == np.uint16:
        bit_depth = 16
        bytes_per_sample = 2
        is_signed_in = 0
    elif arr.dtype == np.int16:
        bit_depth = 16
        bytes_per_sample = 2
        is_signed_in = 1
    else:
        raise OpenJphError(
            f"HTJ2K encode: unsupported dtype {arr.dtype}; "
            f"expected uint8/int8/uint16/int16"
        )

    if arr.ndim == 2:
        components = 1
        height = arr.shape[0]
        width = arr.shape[1]
        planar = arr
    elif arr.ndim == 3:
        components = arr.shape[2]
        if components not in (1, 3, 4):
            raise OpenJphError(
                f"HTJ2K encode: component count must be 1, 3, or 4 "
                f"(got {components})"
            )
        height = arr.shape[0]
        width = arr.shape[1]
        # Shim expects planar layout: comp 0 plane, then comp 1, ...
        planar = np.ascontiguousarray(
            np.transpose(arr, (2, 0, 1))
        )
    else:
        raise OpenJphError(
            f"HTJ2K encode: unsupported ndim {arr.ndim}"
        )

    if level is None:
        reversible = 1
        irrev_delta = 0.0
    else:
        reversible = 0
        irrev_delta = <float> float(level)
        if not (irrev_delta > 0.0):
            raise OpenJphError(
                f"HTJ2K encode: lossy level must be > 0 (got {level})"
            )

    rc = opencodecs_htj2k_encode(
        <const void*> cnp.PyArray_DATA(planar),
        width, height, components,
        bit_depth, is_signed_in, bytes_per_sample,
        reversible, irrev_delta, num_decomp_c,
        &out_buf, &out_size,
    )
    if rc != 0:
        _raise(rc, "encode")
    try:
        return PyBytes_FromStringAndSize(<const char*> out_buf,
                                         <Py_ssize_t> out_size)
    finally:
        opencodecs_htj2k_free(out_buf)


def decode_info(data) -> dict:
    """Read HTJ2K SIZ marker without decoding any samples."""
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        int width = 0, height = 0, components = 0
        int bit_depth = 0, is_signed_out = 0
        int rc

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <size_t> src.shape[0]

    rc = opencodecs_htj2k_decode_info(
        <const void*> &src[0], srcsize,
        &width, &height, &components, &bit_depth, &is_signed_out,
    )
    if rc != 0:
        _raise(rc, "decode_info")
    return {
        "width": width,
        "height": height,
        "components": components,
        "bit_depth": bit_depth,
        "signed": bool(is_signed_out),
    }


def decode(data) -> np.ndarray:
    """Decode an HTJ2K codestream to an ndarray."""
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        int width = 0, height = 0, components = 0
        int bit_depth = 0, is_signed_out = 0
        int bytes_per_sample, rc
        cnp.ndarray planar
        cnp.npy_intp shape[3]
        int ndim

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <size_t> src.shape[0]

    rc = opencodecs_htj2k_decode_info(
        <const void*> &src[0], srcsize,
        &width, &height, &components, &bit_depth, &is_signed_out,
    )
    if rc != 0:
        _raise(rc, "decode_info")

    bytes_per_sample = 1 if bit_depth <= 8 else 2

    if bit_depth <= 8:
        np_dtype = np.int8 if is_signed_out else np.uint8
        npy_type = cnp.NPY_INT8 if is_signed_out else cnp.NPY_UINT8
    else:
        np_dtype = np.int16 if is_signed_out else np.uint16
        npy_type = cnp.NPY_INT16 if is_signed_out else cnp.NPY_UINT16

    if components == 1:
        ndim = 2
        shape[0] = height
        shape[1] = width
    else:
        # Allocate planar (C, H, W) — shim writes that layout — then
        # transpose back to (H, W, C) for the caller.
        ndim = 3
        shape[0] = components
        shape[1] = height
        shape[2] = width

    planar = cnp.PyArray_EMPTY(ndim, shape, npy_type, 0)

    rc = opencodecs_htj2k_decode(
        <const void*> &src[0], srcsize,
        <void*> cnp.PyArray_DATA(planar),
        <size_t> planar.nbytes,
        bytes_per_sample,
    )
    if rc != 0:
        _raise(rc, "decode")

    if components == 1:
        return planar
    # Transpose (C, H, W) -> (H, W, C) and densify so callers can use
    # the result as a contiguous image without surprises.
    return np.ascontiguousarray(np.transpose(planar, (1, 2, 0)))
