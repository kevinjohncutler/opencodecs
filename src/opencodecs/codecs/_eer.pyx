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

With no ``out`` the frame decodes to a bool array. One frame holds at
most one event per output cell (the position advances past every
event), EER TIFFs declare these pages BitsPerSample=1, and imagecodecs'
eer_decode, tifffile and opencodecs' own TIFF/EER reader all return
bool for them. Counts above one only arise when frames are summed, so
that is what ``out=`` is for: a uint8 or uint16 array accumulates events
into the caller's buffer (uint8 saturates at 255, uint16 at 65535),
and a bool array has the frame's events OR-ed into it.

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
        Pre-allocated ``(H, W)`` destination, not cleared first. uint8
        or uint16 accumulates event counts; bool sets each cell an
        event lands on. Without it a new zeroed bool array is returned.

    Returns
    -------
    ndarray
        ``(H, W)`` bool event map, or ``out``.
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

    cdef object result
    cdef object bool_out = None
    if out is None:
        # A fresh zeroed frame: the kernel's saturating increment can
        # only take a cell from 0 to 1, since a frame holds at most one
        # event per cell, so it writes valid bool bytes directly.
        result = np.zeros((height, width), dtype=np.bool_)
        dst = result.view(np.uint8)
        use_u2 = False
    else:
        if not isinstance(out, np.ndarray):
            raise EerError("out must be a numpy ndarray")
        if out.shape != (height, width):
            raise EerError(
                f"out shape {out.shape} != expected ({height}, {width})"
            )
        if not out.flags["C_CONTIGUOUS"]:
            raise EerError("out must be C-contiguous")
        result = out
        # Don't auto-zero a user buffer: the caller may be adding this
        # frame's events into an existing image.
        if out.dtype.char == "H":
            dst = out
            use_u2 = True
        elif out.dtype.char == "B":
            dst = out
            use_u2 = False
        elif out.dtype == np.bool_:
            # A bool that already holds True must stay a valid bool, so
            # decode into scratch and OR it in rather than incrementing.
            bool_out = out
            dst = np.zeros((height, width), dtype=np.uint8)
            use_u2 = False
        else:
            raise EerError(
                f"out dtype must be bool, uint8 or uint16, got {out.dtype}"
            )

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
    if bool_out is not None:
        np.logical_or(bool_out, dst.view(np.bool_), out=bool_out)
    return result
