# opencodecs/codecs/_eer.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""EER (Electron Event Representation) decoder.

EER is the raw output format of Thermo Fisher Falcon 4 / Selectris X
cryo-EM direct detectors. Each frame is a variable-length bitstream
of detected electron events; the decoder rasterises them into a
``(H, W)`` count image.

Storage container
=================

EER frames are wrapped in a TIFF file with a custom compression tag
(``compression = 65000 / 65001 / 65002``) and three private tags
giving the bit-field widths: ``skipbits`` (gap), ``horzbits``
(sub-pixel H), ``vertbits`` (sub-pixel V). Tifffile already parses
the wrapper; this module decodes the bitstream payload of one
strip / tile.

Output dtype
============

If ``out`` is a uint16 array, samples accumulate as uint16 (multiple
events on the same sub-pixel sum). Otherwise the decoder produces a
binary uint8 count clipped to [0, 255].

Implementation
==============

The decoder lives in ``3rdparty/oc_eer/`` and is opencodecs' own
code, written against the bitstream layout that RELION's
``renderEER.cpp`` documents and validated against genuine Falcon 4
output from EMPIAR-10568.
"""

from libc.stdint cimport uint8_t, uint16_t, uint32_t

import numpy as np
cimport numpy as cnp

from eer cimport (
    oc_eer_decode_u8,
    oc_eer_decode_u16,
)

cnp.import_array()


class EerError(RuntimeError):
    """Raised on EER bitstream decode failures."""


def decode(
    data,
    shape,
    int skipbits,
    int horzbits,
    int vertbits,
    *,
    int superres = 0,
    out = None,
):
    """Decode one EER strip / tile to a 2-D count image.

    Parameters
    ----------
    data : bytes-like
        EER bitstream for one frame / strip / tile.
    shape : (height, width)
        Output raster size in pixels. In super-resolution mode this
        is the post-upsampling size — width and height must be
        divisible by ``2**horzbits`` and ``2**vertbits`` respectively.
    skipbits : int
        Width (in bits) of the run-length skip field. 4..14.
    horzbits, vertbits : int
        Width (in bits) of the sub-pixel H / V offset fields. 1..4.
    superres : int, optional
        Upsampling factor in bits. 0 = no upsampling (one event per
        coarse pixel, binary image), >0 = use sub-pixel offsets.
    out : ndarray, optional
        Pre-allocated destination. If ``dtype.char == 'H'`` (uint16)
        events accumulate as uint16 counts; otherwise a fresh
        ``uint8`` array is allocated and written.

    Returns
    -------
    ndarray
        ``(H, W)`` array of event counts.
    """
    cdef:
        const uint8_t[::1] src
        ssize_t srcsize
        ssize_t ret = 0
        ssize_t height = shape[0]
        ssize_t width = shape[1]
        cnp.ndarray dst
        bint use_u2

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <ssize_t> src.shape[0]

    if data is out:
        raise EerError("cannot decode in-place")

    if not (1 < skipbits < 15 and 0 < horzbits < 5 and 0 < vertbits < 5
            and 8 < skipbits + horzbits + vertbits < 17):
        raise EerError(
            f"invalid skipbits/horzbits/vertbits combination: "
            f"({skipbits}, {horzbits}, {vertbits})"
        )

    use_u2 = (
        out is not None
        and isinstance(out, np.ndarray)
        and out.dtype.char == "H"
    )
    if out is None:
        dst = np.zeros((height, width), dtype=np.uint8)
    elif use_u2:
        if out.shape != (height, width):
            raise EerError(
                f"out shape {out.shape} != expected ({height}, {width})"
            )
        dst = out
        # uint16 accumulator must be zeroed; uint8 default also zeroed
        # by np.zeros above. Don't auto-zero a user buffer — caller may
        # want to add events into an existing frame.
    else:
        if not isinstance(out, np.ndarray):
            raise EerError("out must be a numpy ndarray")
        if out.dtype.char != "B":
            raise EerError(
                f"out dtype must be uint8 or uint16, got {out.dtype}"
            )
        if out.shape != (height, width):
            raise EerError(
                f"out shape {out.shape} != expected ({height}, {width})"
            )
        dst = out

    cdef uint8_t* p8 = <uint8_t*> cnp.PyArray_DATA(dst)
    cdef uint16_t* p16 = <uint16_t*> cnp.PyArray_DATA(dst)

    with nogil:
        if use_u2:
            ret = oc_eer_decode_u16(
                &src[0], <size_t> srcsize, p16,
                <size_t> height, <size_t> width,
                <unsigned> skipbits, <unsigned> horzbits,
                <unsigned> vertbits, <unsigned> superres,
            )
        else:
            ret = oc_eer_decode_u8(
                &src[0], <size_t> srcsize, p8,
                <size_t> height, <size_t> width,
                <unsigned> skipbits, <unsigned> horzbits,
                <unsigned> vertbits, <unsigned> superres,
            )

    if ret < 0:
        raise EerError(f"eer_decode returned error code {ret}")
    return dst
