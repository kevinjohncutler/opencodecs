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
    try:
        out = (<char*> buffer)[:out_len]
    finally:
        free(buffer)
    return out


def decode(data) -> cnp.ndarray:
    """Decode QOI bytes to a (H, W, C) uint8 ndarray (C is 3 or 4)."""
    cdef:
        const uint8_t[::1] src
        qoi_desc desc
        void* buffer = NULL
        int srcsize
        cnp.ndarray dst

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
        shape = (int(desc.height), int(desc.width), int(desc.channels))
        dst = np.empty(shape, dtype=np.uint8)
        memcpy(<void*> dst.data, <const void*> buffer,
               <size_t> dst.nbytes)
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
