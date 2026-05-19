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
from libc.string cimport memset

from libaec cimport (
    aec_stream,
    aec_buffer_decode,
    aec_encode_init, aec_encode, aec_encode_end,
    AEC_OK, AEC_FLUSH,
    AEC_DATA_SIGNED, AEC_DATA_PREPROCESS, AEC_DATA_MSB, AEC_DATA_3BYTE,
)

import struct as _struct


DEF _HEADER_LEN = 16   # compile-time constant — usable in C pointer arithmetic
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


cdef inline void _write_header(
    unsigned char* dst,
    Py_ssize_t orig_size,
    unsigned int bps,
    unsigned int block,
    unsigned int rsi,
    unsigned int flags,
) noexcept nogil:
    """Encode the 16-byte preamble directly to ``dst`` — same wire
    format as ``_pack_header`` but skips the Python/struct.pack
    round-trip (~5 us saved per encode)."""
    cdef unsigned long long s = <unsigned long long> orig_size
    cdef int i
    # uint64 LE size at bytes 0..7
    for i in range(8):
        dst[i] = <unsigned char> ((s >> (8 * i)) & 0xff)
    dst[8]  = <unsigned char> (bps & 0xff)
    dst[9]  = <unsigned char> (block & 0xff)
    # uint16 LE rsi at bytes 10..11
    dst[10] = <unsigned char> (rsi & 0xff)
    dst[11] = <unsigned char> ((rsi >> 8) & 0xff)
    dst[12] = <unsigned char> (flags & 0xff)
    dst[13] = 0
    dst[14] = 0
    dst[15] = 0


def _unpack_header(buf):
    if len(buf) < _HEADER_LEN:
        raise AecError("aec blob too short to contain header")
    return _struct.unpack(_HEADER_FMT, bytes(buf[:_HEADER_LEN]))


def encode(data, *,
           int bits_per_sample,
           int block_size=32,
           int rsi=128,
           bint is_signed=False,
           bint msb=False,
           bint preprocess=True,
           bint three_byte=False):
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
        unsigned char* buf_ptr
        aec_stream strm
        int rc
        int flags
        Py_ssize_t out_len = 0
        bint init_ok = False
        bint out_too_small = False
        unsigned int c_bps
        unsigned int c_block
        unsigned int c_rsi
        unsigned int c_flags

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
    if block_size != 8 and block_size != 16 and block_size != 32 and block_size != 64:
        raise ValueError(f"block_size must be 8/16/32/64, got {block_size}")
    if not (1 <= rsi <= 4096):
        raise ValueError(f"rsi must be 1..4096, got {rsi}")

    # Inline _build_flags (was a Python call). Each option is a hot
    # boolean check; the extra function-frame churn dominates on the
    # 200 KB workload that bench/bench_codecs.py exercises.
    flags = 0
    if is_signed: flags |= AEC_DATA_SIGNED
    if msb:       flags |= AEC_DATA_MSB
    if preprocess: flags |= AEC_DATA_PREPROCESS
    if three_byte: flags |= AEC_DATA_3BYTE

    # libaec worst-case output bound (from upstream docs / tests):
    # ``srcsize * 67/64 + 256 + 1`` bytes — covers incompressible
    # input (where the codec stores literals with ~4.7% overhead).
    # A tighter ``srcsize + 1024`` ceiling silently truncates random
    # inputs when using the streaming aec_encode path because libaec
    # returns AEC_OK even when avail_out runs out mid-stream — only
    # checkable by re-reading avail_out, not the return code.
    #
    # Allocate `_HEADER_LEN + cap` up front and encode directly into
    # the slot after the header — saves the extra slice + concat
    # copies the naïve `header + payload[:n]` return would do
    # (~70 us per 200 KB copy at memcpy speeds, half the remaining
    # vs-ic gap).
    cap = (srcsize * 67) // 64 + 257
    payload = PyBytes_FromStringAndSize(NULL, _HEADER_LEN + cap)
    buf_ptr = <unsigned char*> PyBytes_AsString(payload)
    out_ptr = buf_ptr + _HEADER_LEN

    # Streaming init/encode/end at the C level is the same work as
    # ``aec_buffer_encode`` — the buffer wrapper just trios them. The
    # real speedup over the old code was fixing the worst-case cap
    # (above): the old ``srcsize + 1024`` ceiling tripped
    # ``aec_buffer_encode``'s internal retry path on incompressible
    # input, doubling the runtime on the random-uint16 bench
    # workload. We expose the streaming trio here because it lets us
    # detect "output too small" via ``total_in != srcsize`` and raise
    # a precise error rather than silently truncating.
    c_bps = <unsigned int> bits_per_sample
    c_block = <unsigned int> block_size
    c_rsi = <unsigned int> rsi
    c_flags = <unsigned int> flags

    try:
        with nogil:
            memset(<void*> &strm, 0, sizeof(aec_stream))
            strm.next_in = <const unsigned char*> &src[0]
            strm.avail_in = <size_t> srcsize
            strm.next_out = out_ptr
            strm.avail_out = <size_t> cap
            strm.bits_per_sample = c_bps
            strm.block_size = c_block
            strm.rsi = c_rsi
            strm.flags = c_flags
            rc = aec_encode_init(&strm)
            if rc == AEC_OK:
                init_ok = True
                rc = aec_encode(&strm, AEC_FLUSH)
                if rc == AEC_OK:
                    if strm.total_in != <size_t> srcsize:
                        out_too_small = True
                    else:
                        out_len = <Py_ssize_t> strm.total_out
        if rc != AEC_OK:
            raise _err('aec_encode', rc)
        if out_too_small:
            raise AecError(
                f"aec_encode: consumed {strm.total_in} of {srcsize} "
                f"bytes — output buffer too small"
            )
    finally:
        if init_ok:
            aec_encode_end(&strm)

    # Write header in place over the first 16 bytes of the pre-allocated
    # output (cheaper than ``struct.pack`` + Python-level concat — the
    # whole header is 16 bytes of little-endian C scalars).
    _write_header(buf_ptr, srcsize, c_bps, c_block, c_rsi, c_flags)
    return payload[: _HEADER_LEN + out_len]


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
