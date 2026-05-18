# opencodecs/codecs/_rcomp.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Rice compression (Golomb-Rice) — Cython binding to cfitsio's
``ricecomp.c``. Same wire format as ``imagecodecs.rcomp_encode``.

The pure-Python fallback used to live in ``_rcomp_codec.py``; this
extension is a ~1000x speedup on natural int16 arrays (17 ms → 0.02 ms
on a 4 KB-element block). Preserves the existing opencodecs framing
(12-byte header carrying original size + blocksize + bytes-per-pixel,
followed by the raw rice-coded payload from cfitsio).
"""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
import struct

import numpy as np
cimport numpy as cnp


cdef extern from 'ricecomp.h' nogil:
    int rcomp_int(int* a, int nx, unsigned char* c, int clen, int nblock)
    int rcomp_short(short* a, int nx, unsigned char* c, int clen, int nblock)
    int rcomp_byte(signed char* a, int nx, unsigned char* c, int clen, int nblock)
    int rdecomp_int(unsigned char* c, int clen, unsigned int* array, int nx, int nblock)
    int rdecomp_short(unsigned char* c, int clen, unsigned short* array, int nx, int nblock)
    int rdecomp_byte(unsigned char* c, int clen, unsigned char* array, int nx, int nblock)


cnp.import_array()


class RcompError(RuntimeError):
    """Raised on rcomp encode/decode failures."""


# Header carries the metadata we need at decode time. Matches the
# pure-Python fallback's layout exactly so existing blobs roundtrip.
_HEADER = struct.Struct("<IIi")   # nbytes, blocksize, bpp_signed


def encode(data, *, int blocksize=32) -> bytes:
    """Rice-encode an integer ndarray. Returns header + payload bytes.

    ``data`` may be int8/uint8/int16/uint16/int32/uint32. cfitsio's
    encoder is signed-only; unsigned inputs are interpreted as signed
    (round-trip preserves the bit pattern because we encode and
    decode through the same dtype mapping).
    """
    cdef:
        cnp.ndarray arr
        Py_ssize_t n
        int nx
        int clen
        unsigned char* out_ptr
        bytes payload
        int written
        Py_ssize_t bpp

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    if data.dtype.kind not in 'iu':
        raise RcompError(f'rcomp: requires int dtype, got {data.dtype}')

    arr = np.ascontiguousarray(data).ravel()
    n = arr.shape[0]
    if n > 0x7fffffff:
        raise RcompError(f'rcomp: too many elements ({n} > 2^31)')
    nx = <int> n
    bpp = arr.dtype.itemsize

    # cfitsio's bound: worst case ~3 bits per byte input + nblock-sized
    # header per block. 2x input + 1 KB is safe for natural data.
    clen = <int> max(arr.nbytes * 2 + 1024, 16)
    payload = PyBytes_FromStringAndSize(NULL, clen)
    out_ptr = <unsigned char*> PyBytes_AsString(payload)

    cdef cnp.ndarray buf
    cdef short* p16
    cdef int* p32
    cdef signed char* p8
    if bpp == 1:
        # Pure copy as signed bytes — covers int8/uint8.
        buf = arr.view(np.int8) if arr.dtype == np.uint8 else arr
        p8 = <signed char*> cnp.PyArray_DATA(buf)
        with nogil:
            written = rcomp_byte(p8, nx, out_ptr, clen, blocksize)
    elif bpp == 2:
        buf = arr.view(np.int16) if arr.dtype == np.uint16 else arr
        p16 = <short*> cnp.PyArray_DATA(buf)
        with nogil:
            written = rcomp_short(p16, nx, out_ptr, clen, blocksize)
    elif bpp == 4:
        buf = arr.view(np.int32) if arr.dtype == np.uint32 else arr
        p32 = <int*> cnp.PyArray_DATA(buf)
        with nogil:
            written = rcomp_int(p32, nx, out_ptr, clen, blocksize)
    else:
        raise RcompError(
            f'rcomp: unsupported dtype itemsize {bpp}; expected 1/2/4 bytes'
        )

    if written < 0:
        raise RcompError(f'rcomp_*: returned error {written}')

    # Pack header + payload-prefix. Storing nbytes in the header lets
    # the decoder allocate exactly the right output array even when
    # the payload is small enough to be ambiguous.
    header = _HEADER.pack(<unsigned int> arr.nbytes,
                           <unsigned int> blocksize, <int> bpp)
    return header + payload[:written]


def decode(data, *, out=None):
    """Decode an rcomp blob. Returns int8/int16/int32 ndarray sized
    by the header. The caller can re-view as unsigned via
    ``arr.view(arr.dtype.newbyteorder('=').kind.upper())`` or pass a
    typed ``out=`` to the codec wrapper."""
    cdef:
        Py_ssize_t total_in
        int nbytes
        unsigned int blocksize
        int bpp
        Py_ssize_t header_size
        Py_ssize_t payload_size
        int nx
        int rc
        cnp.ndarray dst

    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    total_in = len(data)
    header_size = _HEADER.size
    if total_in < header_size:
        raise RcompError(
            f'rcomp: input too short for header ({total_in} < {header_size})'
        )

    nbytes, blocksize, bpp = _HEADER.unpack(data[:header_size])
    payload_size = total_in - header_size
    if payload_size > 0x7fffffff:
        raise RcompError(f'rcomp: payload too large ({payload_size} > 2^31)')

    cdef const unsigned char[::1] payload_mv = data[header_size:]
    cdef unsigned char* payload_ptr
    if payload_size > 0:
        payload_ptr = <unsigned char*> &payload_mv[0]
    else:
        payload_ptr = NULL

    # cfitsio's rdecomp writes UNSIGNED output. Map dtype by bpp.
    nx = <int> (nbytes // bpp)
    if bpp == 1:
        dst = np.empty(nx, dtype=np.uint8)
        if nx > 0:
            with nogil:
                rc = rdecomp_byte(
                    payload_ptr, <int> payload_size,
                    <unsigned char*> cnp.PyArray_DATA(dst), nx,
                    <int> blocksize,
                )
        else:
            rc = 0
    elif bpp == 2:
        dst = np.empty(nx, dtype=np.uint16)
        if nx > 0:
            with nogil:
                rc = rdecomp_short(
                    payload_ptr, <int> payload_size,
                    <unsigned short*> cnp.PyArray_DATA(dst), nx,
                    <int> blocksize,
                )
        else:
            rc = 0
    elif bpp == 4:
        dst = np.empty(nx, dtype=np.uint32)
        if nx > 0:
            with nogil:
                rc = rdecomp_int(
                    payload_ptr, <int> payload_size,
                    <unsigned int*> cnp.PyArray_DATA(dst), nx,
                    <int> blocksize,
                )
        else:
            rc = 0
    else:
        raise RcompError(f'rcomp: unsupported bpp {bpp}')

    if rc < 0:
        raise RcompError(f'rdecomp_*: returned error {rc}')
    return dst


def check_signature(head: bytes) -> bool:
    """Rcomp blobs have no magic — return False so the registry
    doesn't auto-route on signature alone."""
    return False
