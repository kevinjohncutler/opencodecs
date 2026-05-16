# opencodecs/codecs/_aec.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native AEC codec — CCSDS 121.0-B-2 adaptive entropy coding (libaec).

AEC is the lossless integer-array compressor used by NetCDF-4 and most
satellite/Earth-science workflows (HDF5 SZIP filter is the same codec).
For 8/16/32-bit integer data with predictable runs, ratios are usually
2–4×, often beating zstd at lower CPU cost.

Wire format
-----------
A small 16-byte opencodecs preamble is prepended to libaec's raw stream
so a self-describing blob can be decoded without out-of-band parameters::

    bytes  0..7   uint64 LE  - original payload size (bytes)
    byte   8      uint8       - bits_per_sample (1..32)
    byte   9      uint8       - block_size (8 / 16 / 32 / 64)
    bytes 10..11  uint16 LE  - rsi (1..4096)
    byte   12     uint8       - flags (AEC_DATA_*)
    bytes 13..15              - reserved (zero)
    bytes 16..              - libaec compressed stream

This makes ``opencodecs.read(blob, format='aec')`` work without
threading parameters through the API. Pass the same parameter values
during ``encode()`` to recover them.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdint cimport uint8_t

from libaec cimport (
    aec_stream,
    aec_buffer_encode, aec_buffer_decode,
    AEC_OK,
    AEC_DATA_SIGNED, AEC_DATA_PREPROCESS, AEC_DATA_MSB, AEC_DATA_3BYTE,
)

import struct as _struct


_HEADER_LEN = 16
_HEADER_FMT = '<QBBHB3x'  # uint64 size, u8 bps, u8 block, u16 rsi, u8 flags, 3 pad


class AecError(RuntimeError):
    """Raised on libaec encode/decode failures."""


_RC_NAMES = {
    -1: "AEC_CONF_ERROR (parameter out of range)",
    -2: "AEC_STREAM_ERROR (state machine corruption)",
    -3: "AEC_DATA_ERROR (input not valid)",
    -4: "AEC_MEM_ERROR (allocation failed)",
    -5: "AEC_RSI_OFFSETS_ERROR",
}


def _err(func, code):
    return AecError(f'{func} returned {_RC_NAMES.get(int(code), int(code))}')


def _build_flags(is_signed, msb, preprocess, three_byte):
    # Cython 3 reserves `signed` as a C type keyword in annotated args;
    # use a different name here.
    f = 0
    if is_signed: f |= AEC_DATA_SIGNED
    if msb:       f |= AEC_DATA_MSB
    if preprocess: f |= AEC_DATA_PREPROCESS
    if three_byte: f |= AEC_DATA_3BYTE
    return f


def _pack_header(orig_size, bits_per_sample, block_size, rsi, flags):
    return _struct.pack(_HEADER_FMT,
                        int(orig_size), int(bits_per_sample) & 0xff,
                        int(block_size) & 0xff, int(rsi) & 0xffff,
                        int(flags) & 0xff)


def _unpack_header(buf):
    if len(buf) < _HEADER_LEN:
        raise AecError("aec blob too short to contain header")
    return _struct.unpack(_HEADER_FMT, bytes(buf[:_HEADER_LEN]))


def encode(data, *,
           bits_per_sample,
           block_size=32,
           rsi=128,
           is_signed=False,
           msb=False,
           preprocess=True,
           three_byte=False):
    """AEC-compress a typed integer buffer.

    Parameters
    ----------
    data : bytes-like
        Input data, layout matching ``bits_per_sample``.
    bits_per_sample : int
        1..32. Use multiples of 8 for byte-aligned data; for n=24 set
        ``three_byte=True``.
    block_size : int
        8, 16, 32, or 64. Larger blocks compress better, encode slower.
    rsi : int
        Reference-sample interval (1..4096). 128 is a good default.
    signed, msb, preprocess, three_byte
        Sample-format flags (AEC_DATA_* in libaec.h).

    Returns
    -------
    bytes
        16-byte header + libaec-compressed stream.
    """
    cdef:
        const uint8_t[::1] src
        Py_ssize_t srcsize
        Py_ssize_t cap
        bytes payload
        unsigned char* out_ptr
        aec_stream strm
        int rc
        int flags

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = src.shape[0]
    if srcsize == 0:
        return _pack_header(0, bits_per_sample, block_size, rsi,
                            _build_flags(is_signed, msb, preprocess, three_byte))

    if not (1 <= bits_per_sample <= 32):
        raise ValueError(f"bits_per_sample must be 1..32, got {bits_per_sample}")
    if block_size not in (8, 16, 32, 64):
        raise ValueError(f"block_size must be 8/16/32/64, got {block_size}")
    if not (1 <= rsi <= 4096):
        raise ValueError(f"rsi must be 1..4096, got {rsi}")

    flags = _build_flags(is_signed, msb, preprocess, three_byte)

    # Worst case: libaec output can be slightly larger than input on
    # incompressible streams. Reserve input size + 1 KB; if encode says
    # AEC_STREAM_ERROR / output overrun, fall back to 4× input.
    cap = srcsize + 1024
    payload = PyBytes_FromStringAndSize(NULL, cap)
    out_ptr = <unsigned char*> PyBytes_AsString(payload)

    strm.next_in = <const unsigned char*> &src[0]
    strm.avail_in = <size_t> srcsize
    strm.next_out = out_ptr
    strm.avail_out = <size_t> cap
    strm.bits_per_sample = <unsigned int> bits_per_sample
    strm.block_size = <unsigned int> block_size
    strm.rsi = <unsigned int> rsi
    strm.flags = <unsigned int> flags
    strm.total_in = 0
    strm.total_out = 0
    strm.state = NULL

    with nogil:
        rc = aec_buffer_encode(&strm)
    if rc != AEC_OK:
        # One retry with bigger output buffer in case incompressible input
        # tripped a tight bound.
        cap = srcsize * 4 + 1024
        payload = PyBytes_FromStringAndSize(NULL, cap)
        out_ptr = <unsigned char*> PyBytes_AsString(payload)
        strm.next_in = <const unsigned char*> &src[0]
        strm.avail_in = <size_t> srcsize
        strm.next_out = out_ptr
        strm.avail_out = <size_t> cap
        strm.total_in = 0
        strm.total_out = 0
        strm.state = NULL
        with nogil:
            rc = aec_buffer_encode(&strm)
        if rc != AEC_OK:
            raise _err('aec_buffer_encode', rc)

    cdef Py_ssize_t out_len = <Py_ssize_t> strm.total_out
    header = _pack_header(srcsize, bits_per_sample, block_size, rsi, flags)
    return header + payload[:out_len]


def decode(data, *, out=None):
    """Decode a self-describing AEC blob (header + libaec stream).

    Parameters
    ----------
    out : int | bytearray | memoryview | None, optional
        See ``_zstd.decode`` for the full ``out=`` contract. The AEC
        header is self-describing so ``out=None`` already allocates the
        exact uncompressed size; ``out=`` is mainly useful for reusing
        a buffer across many tile decodes.
    """
    cdef:
        const uint8_t[::1] src
        uint8_t[::1] out_view             # writable view of caller buffer
        Py_ssize_t srcsize
        Py_ssize_t out_size
        bytes out_bytes
        unsigned char* out_ptr
        aec_stream strm
        int rc

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = src.shape[0]
    if srcsize < _HEADER_LEN:
        raise AecError("aec blob too short to contain header")

    orig_size, bps, block, rsi, flags = _unpack_header(bytes(src[:_HEADER_LEN]))
    if orig_size == 0:
        if out is None or isinstance(out, int):
            return b''
        return out[:0]
    # The header is the first 16 bytes of the input; for a corrupt or
    # adversarial blob those bytes can encode an absurd ``orig_size``
    # (uint64 read of random bytes -> ~10**18). Forwarding that to
    # PyBytes_FromStringAndSize attempts a multi-exabyte malloc which
    # aborts under ASAN and OOM-kills otherwise. Cap at 16 GiB — well
    # above any plausible single-call decode for the scientific data
    # libaec is used on, but small enough to bail fast on garbage.
    if orig_size > (1 << 34):
        raise AecError(
            f"aec header: orig_size {orig_size} exceeds 16 GiB sanity cap "
            "(input is probably corrupt or not an AEC blob)"
        )

    out_size = <Py_ssize_t> orig_size

    # ----- caller-supplied writable buffer (zero-alloc path) -----
    if out is not None and not isinstance(out, int):
        try:
            out_view = out
        except (TypeError, ValueError, BufferError) as e:
            raise TypeError(
                f"aec decode: out= must be int or writable buffer, "
                f"got {type(out).__name__}"
            ) from e
        if out_view.shape[0] < out_size:
            raise AecError(
                f"aec decode: out= buffer is {out_view.shape[0]} bytes "
                f"but the AEC header declares {out_size} bytes")
        out_ptr = <unsigned char*> &out_view[0]
    else:
        if isinstance(out, int):
            if out < out_size:
                raise AecError(
                    f"aec decode: out=int({out}) is less than the AEC "
                    f"header's declared {out_size} bytes")
        out_bytes = PyBytes_FromStringAndSize(NULL, out_size)
        out_ptr = <unsigned char*> PyBytes_AsString(out_bytes)

    strm.next_in = <const unsigned char*> &src[_HEADER_LEN]
    strm.avail_in = <size_t> (srcsize - _HEADER_LEN)
    strm.next_out = out_ptr
    strm.avail_out = <size_t> out_size
    strm.bits_per_sample = <unsigned int> bps
    strm.block_size = <unsigned int> block
    strm.rsi = <unsigned int> rsi
    strm.flags = <unsigned int> flags
    strm.total_in = 0
    strm.total_out = 0
    strm.state = NULL

    with nogil:
        rc = aec_buffer_decode(&strm)
    if rc != AEC_OK:
        raise _err('aec_buffer_decode', rc)

    if <Py_ssize_t> strm.total_out != out_size:
        raise AecError(
            f"aec_buffer_decode produced {strm.total_out} bytes, "
            f"expected {out_size}"
        )
    if out is not None and not isinstance(out, int):
        del out_view
        return out[:out_size]
    return out_bytes


def check_signature(data) -> bool:
    """No reliable magic bytes for libaec streams."""
    return False
