# opencodecs/codecs/_qoi.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native QOI (Quite OK Image Format) codec.

QOI is an extremely simple lossless image format for RGB / RGBA uint8.
The single-header reference implementation in 3rdparty/qoi/qoi.h is
~700 lines of C. We compile it into this extension via
QOI_IMPLEMENTATION (set in setup.py).

Format derivation: bytes-in/bytes-out only — no streaming, no
multi-frame. Trivial Cython binding so we can serve as a baseline for
how the unified Codec API plugs into a native extension that has zero
external library dependencies.
"""

import numpy as np
cimport numpy as cnp

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.stdint cimport int32_t, uint8_t

from qoi cimport (
    qoi_desc,
    qoi_encode as _c_qoi_encode,
    qoi_decode as _c_qoi_decode,
    QOI_SRGB, QOI_LINEAR,
)

cnp.import_array()


class QoiError(RuntimeError):
    """Raised on QOI encode / decode failures."""


def encode(data, *, srgb: bool = True) -> bytes:
    """Encode a (H, W, 3) or (H, W, 4) uint8 array as QOI bytes."""
    cdef:
        cnp.ndarray src
        qoi_desc desc
        int out_len = 0
        void* buffer = NULL
        int channels
        bytes out
        const uint8_t[::1] dst

    src = np.ascontiguousarray(data)
    if src.dtype != np.uint8:
        raise ValueError(f'qoi: dtype must be uint8, got {src.dtype!r}')
    if src.ndim != 3:
        raise ValueError(
            f'qoi: data must be 3D (H, W, C); got ndim={src.ndim}')
    channels = <int> src.shape[2]
    if channels != 3 and channels != 4:
        raise ValueError(
            f'qoi: channels must be 3 or 4; got {channels}')
    if <unsigned long long> src.shape[0] > 0xFFFFFFFF or \
       <unsigned long long> src.shape[1] > 0xFFFFFFFF:
        raise ValueError('qoi: image too large')

    desc.width = <unsigned int> src.shape[1]
    desc.height = <unsigned int> src.shape[0]
    desc.channels = <unsigned char> channels
    desc.colorspace = QOI_SRGB if srgb else QOI_LINEAR

    with nogil:
        buffer = _c_qoi_encode(<const void*> src.data, &desc, &out_len)
    if buffer == NULL:
        raise QoiError('qoi_encode returned NULL')
    # Allocate the output bytes at the exact size + memcpy with GIL
    # released (matches imagecodecs's pattern; a hair faster than the
    # single `(<char*> buffer)[:out_len]` slice on multi-MB blobs).
    out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> out_len)
    dst = out
    try:
        with nogil:
            memcpy(<void*> &dst[0], <const void*> buffer, <size_t> out_len)
    finally:
        free(buffer)
    del dst
    return out


def decode(data, *, out=None):
    """Decode QOI bytes to a (H, W, C) uint8 ndarray (C is 3 or 4).

    Parameters
    ----------
    out : np.ndarray | None, optional
        Preallocated output array of shape (H, W, C) uint8 — must match
        the dimensions in the QOI header. Returns the same array.
        See ``_png.decode`` for the full ``out=`` contract.

        QOI's reference decoder always allocates its own output buffer
        (no API for caller-provided destinations), so when out= is
        supplied we still pay the qoi-internal alloc and then memcpy
        into the caller's buffer. The benefit is API parity + skipping
        the *second* allocation that a plain decode() does.
    """
    cdef:
        const uint8_t[::1] src
        qoi_desc desc
        void* buffer = NULL
        int srcsize
        cnp.ndarray dst
        void* dst_data
        size_t nbytes
        tuple expected_shape

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <int> src.shape[0]
    if srcsize <= 14:
        raise QoiError('qoi: input too small (header is 14 bytes)')

    with nogil:
        buffer = _c_qoi_decode(
            <const void*> &src[0], srcsize, &desc, 0)
    if buffer == NULL:
        raise QoiError('qoi_decode returned NULL (bad header / corrupt?)')

    try:
        expected_shape = (int(desc.height), int(desc.width), int(desc.channels))
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError(
                    f"qoi decode: out= must be an ndarray, "
                    f"got {type(out).__name__}")
            if out.shape != expected_shape:
                raise ValueError(
                    f"qoi decode: out= shape {out.shape} does not match "
                    f"expected {expected_shape}")
            if out.dtype != np.uint8:
                raise ValueError(
                    f"qoi decode: out= dtype must be uint8, got {out.dtype}")
            if not out.flags['C_CONTIGUOUS']:
                raise ValueError("qoi decode: out= must be C-contiguous")
            dst = out
        else:
            dst = np.empty(expected_shape, dtype=np.uint8)
        dst_data = <void*> dst.data
        nbytes = <size_t> dst.nbytes
        # Release GIL during the bulk copy — for a 12 MB RGB blob this
        # is a real win because the memcpy itself is 2-3 ms.
        with nogil:
            memcpy(dst_data, <const void*> buffer, nbytes)
    finally:
        free(buffer)
    return dst


def check_signature(data) -> bool:
    """True if `data` starts with the QOI magic 'qoif'."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:4])
    else:
        try:
            head = bytes(data)[:4]
        except Exception:
            return False
    return head == b'qoif'
