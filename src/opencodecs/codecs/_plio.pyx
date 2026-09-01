# opencodecs/codecs/_plio.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""PLIO_1 decode — Cython binding to cfitsio's ``pl_l2pi`` for use in
the FITS compressed-image reader.

PLIO (Pixel List I/O) is IRAF's run-length codec for image masks: the
input is a stream of 16-bit signed integers describing alternating
runs of zeros and constant values. It's the canonical encoding for
segmentation / catalog masks attached to images in IRAF pipelines and
appears as ``ZCMPTYPE = 'PLIO_1'`` in FITS BINTABLE compressed images.

The vendored cfitsio source is ``3rdparty/cfitsio/pliocomp.c`` (Doug
Tody, NRAO; translated from IRAF SPP). Public-domain U.S. Government
work — see ``3rdparty/cfitsio/License.txt``.
"""

from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as cnp


cnp.import_array()


cdef extern from *:
    """
    /* Forward-declare cfitsio's pl_l2pi so we don't need a header.
       cfitsio 4.7.0 added the srclen argument together with the bounds
       checks that stop a crafted line list reading past the payload;
       see the "Added checks for ... plio compression" commit. */
    int pl_l2pi(short *ll_src, size_t srclen, int xs, int *px_dst, int npix);
    """
    int pl_l2pi(short* ll_src, size_t srclen, int xs,
                int* px_dst, int npix) nogil


class PlioError(RuntimeError):
    """Raised on PLIO_1 decode failures."""


def decode_raw(data, *, int nelements):
    """Decode a raw PLIO_1 tile bitstream (no FITS framing).

    Parameters
    ----------
    data : bytes-like
        Compressed payload (as it appears in a BINTABLE
        ``COMPRESSED_DATA`` cell for ZCMPTYPE='PLIO_1'). The payload
        is a sequence of big-endian int16 opcodes — FITS BINTABLE
        columns are always stored big-endian.
    nelements : int
        Number of decompressed pixels expected (tile_h * tile_w).

    Returns
    -------
    ndarray
        Decoded tile as int32 of length ``nelements``. The caller
        casts to the FITS target dtype (typically int8/int16).
    """
    cdef:
        const unsigned char[::1] src_mv
        Py_ssize_t srcsize
        Py_ssize_t n_opcodes
        short* ll_src
        int* px_dst
        cnp.ndarray out
        Py_ssize_t i
        int written
        int npix

    if nelements <= 0:
        raise PlioError(f"plio decode: nelements must be > 0, got {nelements}")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    src_mv = data
    srcsize = src_mv.shape[0]
    if srcsize % 2 != 0:
        raise PlioError(
            f"plio decode: payload size {srcsize} is not a multiple of 2"
        )
    n_opcodes = srcsize // 2

    # Byte-swap big-endian int16 opcodes into a native short buffer
    # for pl_l2pi. ``np.frombuffer(...).byteswap()`` copies once.
    be_view = np.frombuffer(data, dtype=">i2")
    if be_view.shape[0] != n_opcodes:
        raise PlioError("plio decode: opcode count mismatch after byteswap")
    cdef cnp.ndarray native_arr = np.ascontiguousarray(be_view, dtype=np.int16)

    out = np.empty(nelements, dtype=np.int32)
    ll_src = <short*> cnp.PyArray_DATA(native_arr)
    px_dst = <int*> cnp.PyArray_DATA(out)
    npix = <int> nelements

    with nogil:
        # xs=1: pl_l2pi treats ll_src as 1-indexed (it does --ll_src
        # internally). Start at the first opcode.
        written = pl_l2pi(ll_src, <size_t> n_opcodes, 1, px_dst, npix)

    if written != npix:
        raise PlioError(
            f"pl_l2pi returned {written} pixels, expected {npix}"
        )
    return out
