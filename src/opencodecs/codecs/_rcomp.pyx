# opencodecs/codecs/_rcomp.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Rice compression (Golomb-Rice): Cython binding to cfitsio's
``ricecomp.c``.

``encode`` writes the bare cfitsio Rice stream, the same bytes FITS
stores for a ``RICE_1`` tile (FITS Standard 4.0, section 10.4.1) and
the same bytes ``imagecodecs.rcomp_encode`` writes for an array in
native byte order (a big-endian array is coded by value here, by its
bytes there). The stream carries
no element count, pixel size or block size; FITS keeps them in the
table keywords (ZTILEn, BYTEPIX, BLOCKSIZE) and a codec caller passes
them to ``decode_raw``.

Releases up to 0.4.0 wrote a private 12-byte header in front of the
stream (``<IIi``: byte count, block size, bytes per pixel).
``decode_framed`` still reads those blobs.
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


# The private header releases up to 0.4.0 wrote in front of the Rice
# stream. Only read now, by ``decode_framed``.
_HEADER = struct.Struct("<IIi")   # nbytes, blocksize, bpp_signed


def encode(data, *, int blocksize=32) -> bytes:
    """Rice-encode an integer ndarray. Returns the bare cfitsio stream.

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

    if blocksize <= 0:
        raise RcompError('rcomp: blocksize must be positive')
    arr = np.ascontiguousarray(data, dtype=data.dtype.newbyteorder('=')).ravel()
    n = arr.shape[0]
    if n > 0x7fffffff:
        raise RcompError(f'rcomp: too many elements ({n} > 2^31)')
    nx = <int> n
    bpp = arr.dtype.itemsize
    if bpp not in (1, 2, 4):
        raise RcompError(
            f'rcomp: unsupported dtype itemsize {bpp}; expected 1/2/4 bytes'
        )
    if nx == 0:
        return b''

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

    return payload[:written]


def read_framed_header(data):
    """Return ``(nbytes, blocksize, bytes_per_pixel)`` from the private
    header releases up to 0.4.0 wrote, or None if ``data`` cannot be
    one. The header has no magic, so this checks every field for a
    value the old encoder could have written."""
    if len(data) < _HEADER.size:
        return None
    nbytes, blocksize, bpp = _HEADER.unpack(bytes(data[:_HEADER.size]))
    if bpp not in (1, 2, 4) or nbytes % bpp:
        return None
    if blocksize == 0 or blocksize > 0xffff or nbytes // bpp > 0x7fffffff:
        return None
    if nbytes and len(data) - _HEADER.size < bpp:
        return None
    return nbytes, blocksize, bpp


def decode_framed(data, *, out=None):
    """Decode a blob in the framed layout releases up to 0.4.0 wrote
    (12-byte header, then the Rice stream). Returns the unsigned
    uint8/uint16/uint32 ndarray the header sizes; the codec wrapper
    views it as the caller's dtype."""
    cdef:
        Py_ssize_t total_in
        unsigned int nbytes
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
    if bpp not in (1, 2, 4) or nbytes % bpp:
        raise RcompError('rcomp: invalid pixel size or byte count')
    if blocksize == 0 or blocksize > 0x7fffffff or nbytes // bpp > 0x7fffffff:
        raise RcompError('rcomp: invalid blocksize or element count')
    payload_size = total_in - header_size
    if payload_size > 0x7fffffff:
        raise RcompError(f'rcomp: payload too large ({payload_size} > 2^31)')

    cdef const unsigned char[::1] payload_mv = memoryview(data)[header_size:]
    cdef unsigned char* payload_ptr
    if payload_size > 0:
        payload_ptr = <unsigned char*> &payload_mv[0]
    else:
        payload_ptr = NULL

    # cfitsio's rdecomp writes UNSIGNED output. Map dtype by bpp.
    nx = <int> (nbytes // bpp)
    dtype = np.dtype(f'u{bpp}')
    if out is None:
        dst = np.empty(nx, dtype=dtype)
    else:
        if not isinstance(out, np.ndarray):
            raise TypeError('rcomp out must be an ndarray')
        if out.dtype != dtype or out.size != nx or not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError('rcomp out must be writable contiguous native unsigned storage of matching size')
        dst = out
    if bpp == 1:
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


def decode_raw(data, *, int nelements, int blocksize, int bytes_per_pixel,
               out=None):
    """Decode a bare cfitsio Rice stream.

    This is the format ``encode`` writes, FITS stores in a ``RICE_1``
    tile and ``imagecodecs.rcomp_encode`` writes. Returns a uint8 /
    uint16 / uint32 ndarray of length ``nelements``; the caller views
    it as int8 / int16 / int32 via ``arr.view(...)`` when the data was
    signed. ``out``, if given, must be writable contiguous native
    unsigned storage of that dtype and size.
    """
    cdef:
        const unsigned char[::1] payload_mv
        unsigned char* payload_ptr
        Py_ssize_t payload_size
        cnp.ndarray dst
        int rc

    if nelements < 0 or blocksize <= 0:
        raise RcompError('rcomp_raw: invalid element count or blocksize')
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    payload_mv = data
    payload_size = payload_mv.shape[0]
    if payload_size > 0x7fffffff:
        raise RcompError(f"rcomp_raw: payload too large ({payload_size} > 2^31)")
    if payload_size > 0:
        payload_ptr = <unsigned char*> &payload_mv[0]
    else:
        payload_ptr = NULL

    if bytes_per_pixel not in (1, 2, 4):
        raise RcompError(
            f"rcomp_raw: bytes_per_pixel must be 1/2/4, got {bytes_per_pixel}"
        )
    raw_dtype = np.dtype(f'u{bytes_per_pixel}')
    if out is None:
        dst = np.empty(nelements, dtype=raw_dtype)
    else:
        if not isinstance(out, np.ndarray):
            raise TypeError('rcomp out must be an ndarray')
        if (out.dtype != raw_dtype or out.size != nelements
                or not out.flags.c_contiguous or not out.flags.writeable):
            raise ValueError('rcomp out must be writable contiguous native '
                             'unsigned storage of matching size')
        dst = out

    if bytes_per_pixel == 1:
        if nelements > 0:
            with nogil:
                rc = rdecomp_byte(
                    payload_ptr, <int> payload_size,
                    <unsigned char*> cnp.PyArray_DATA(dst),
                    nelements, blocksize,
                )
        else:
            rc = 0
    elif bytes_per_pixel == 2:
        if nelements > 0:
            with nogil:
                rc = rdecomp_short(
                    payload_ptr, <int> payload_size,
                    <unsigned short*> cnp.PyArray_DATA(dst),
                    nelements, blocksize,
                )
        else:
            rc = 0
    else:
        if nelements > 0:
            with nogil:
                rc = rdecomp_int(
                    payload_ptr, <int> payload_size,
                    <unsigned int*> cnp.PyArray_DATA(dst),
                    nelements, blocksize,
                )
        else:
            rc = 0
    if rc < 0:
        raise RcompError(f"rdecomp_*: returned error {rc}")
    return dst


def check_signature(head: bytes) -> bool:
    """A Rice stream has no magic. Return False so the registry does
    not auto-route on signature alone."""
    return False
