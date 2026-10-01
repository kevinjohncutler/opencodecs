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
``encode`` writes the bare CCSDS 121.0-B-2 coded bitstream libaec
produces, with nothing in front of it: the stream GRIB2 (template
5.42) stores in section 7 and ``imagecodecs.aec_encode`` writes. The
standard defines no container, so the coding parameters (bits per
sample, block size, reference sample interval, flags) and the sample
count travel out of band and ``decode_raw`` takes them as arguments.

Releases up to 0.4.0 wrote a private 16-byte preamble in front of the
stream instead::

    bytes  0..7   uint64 LE  - original payload size (bytes)
    byte   8      uint8       - bits_per_sample (1..32)
    byte   9      uint8       - block_size (8 / 16 / 32 / 64)
    bytes 10..11  uint16 LE  - rsi (1..4096)
    byte   12     uint8       - flags (AEC_DATA_*)
    bytes 13..15              - reserved (zero)

``decode_framed`` still reads those blobs, and ``read_framed_header``
says whether a buffer can be one. The preamble has no magic, so that
check is every field holding a value the old encoder could write.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdint cimport uint8_t
from libc.string cimport memset

from libaec cimport (
    aec_stream,
    aec_buffer_decode,
    aec_encode_init, aec_encode, aec_encode_end,
    aec_decode_init, aec_decode, aec_decode_end,
    AEC_OK, AEC_FLUSH,
    AEC_DATA_SIGNED, AEC_DATA_PREPROCESS, AEC_DATA_MSB, AEC_DATA_3BYTE,
    AEC_RESTRICTED, AEC_PAD_RSI, AEC_NOT_ENFORCE,
)

import struct as _struct


DEF _HEADER_LEN = 16   # compile-time constant — usable in C pointer arithmetic
_HEADER_FMT = '<QBBHB3x'  # uint64 size, u8 bps, u8 block, u16 rsi, u8 flags, 3 pad

# The libaec flag bits, for the Python codec layer.
FLAG_SIGNED = AEC_DATA_SIGNED
FLAG_3BYTE = AEC_DATA_3BYTE
FLAG_MSB = AEC_DATA_MSB
FLAG_PREPROCESS = AEC_DATA_PREPROCESS
FLAG_RESTRICTED = AEC_RESTRICTED
FLAG_PAD_RSI = AEC_PAD_RSI
FLAG_NOT_ENFORCE = AEC_NOT_ENFORCE
_KNOWN_FLAGS = (AEC_DATA_SIGNED | AEC_DATA_3BYTE | AEC_DATA_MSB
                | AEC_DATA_PREPROCESS | AEC_RESTRICTED | AEC_PAD_RSI
                | AEC_NOT_ENFORCE)


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


def sample_bytes(int bits_per_sample, int flags):
    """Bytes libaec reads or writes per sample for these parameters."""
    if bits_per_sample <= 8:
        return 1
    if bits_per_sample <= 16:
        return 2
    if bits_per_sample <= 24 and flags & AEC_DATA_3BYTE:
        return 3
    return 4


def _check_params(int bits_per_sample, int block_size, int rsi, int flags):
    if not (1 <= bits_per_sample <= 32):
        raise ValueError(f"bits_per_sample must be 1..32, got {bits_per_sample}")
    if flags & ~_KNOWN_FLAGS:
        raise ValueError(f"aec flags {flags:#x} has bits libaec does not define")
    if (not (flags & AEC_NOT_ENFORCE) and block_size != 8 and block_size != 16
            and block_size != 32 and block_size != 64):
        raise ValueError(f"block_size must be 8/16/32/64, got {block_size}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if not (1 <= rsi <= 4096):
        raise ValueError(f"rsi must be 1..4096, got {rsi}")


def encode(data, *,
           int bits_per_sample,
           int block_size=8,
           int rsi=2,
           int flags=AEC_DATA_PREPROCESS):
    """AEC-compress a typed integer buffer into a bare libaec stream.

    Parameters
    ----------
    data : bytes-like
        Input samples, ``sample_bytes(bits_per_sample, flags)`` bytes
        each.
    bits_per_sample : int
        1..32.
    block_size : int
        8, 16, 32, or 64 (any even size with ``AEC_NOT_ENFORCE``).
    rsi : int
        Reference sample interval in blocks (1..4096).
    flags : int
        ``AEC_DATA_*`` bits from libaec.h.

    Returns
    -------
    bytes
        The CCSDS 121.0-B-2 coded stream, with no header.
    """
    cdef:
        const uint8_t[::1] src
        Py_ssize_t srcsize
        Py_ssize_t cap
        bytes payload
        unsigned char* out_ptr
        aec_stream strm
        int rc
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
    _check_params(bits_per_sample, block_size, rsi, flags)
    if srcsize == 0:
        return b''
    if srcsize % sample_bytes(bits_per_sample, flags):
        raise ValueError(
            f"aec encode: {srcsize} bytes is not a whole number of "
            f"{sample_bytes(bits_per_sample, flags)}-byte samples")

    # libaec worst-case output bound (from upstream docs / tests):
    # ``srcsize * 67/64 + 256 + 1`` bytes — covers incompressible
    # input (where the codec stores literals with ~4.7% overhead).
    # A tighter ``srcsize + 1024`` ceiling silently truncates random
    # inputs when using the streaming aec_encode path because libaec
    # returns AEC_OK even when avail_out runs out mid-stream — only
    # checkable by re-reading avail_out, not the return code.
    cap = (srcsize * 67) // 64 + 257
    payload = PyBytes_FromStringAndSize(NULL, cap)
    out_ptr = <unsigned char*> PyBytes_AsString(payload)

    # Streaming init/encode/end at the C level is the same work as
    # ``aec_buffer_encode``: the buffer wrapper just trios them. We
    # expose the streaming trio here because it lets us detect "output
    # too small" via ``total_in != srcsize`` and raise a precise error
    # rather than silently truncating.
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

    return payload[:out_len]


cdef inline int _decode_step(aec_stream* strm, unsigned char* dst,
                             size_t cap) noexcept nogil:
    strm.next_out = dst
    strm.avail_out = cap
    return aec_decode(strm, AEC_FLUSH)


def decode_raw(data, *,
               int bits_per_sample,
               int block_size=8,
               int rsi=2,
               int flags=AEC_DATA_PREPROCESS,
               out=None,
               Py_ssize_t size=-1):
    """Decode a bare libaec stream with out-of-band parameters.

    The stream does not record how many samples it holds, so the
    output size comes from the caller when known:

    * ``out`` (a writable contiguous byte buffer): decode into it and
      return ``out[:n]``, ``n`` the bytes the stream decoded to.
    * ``size >= 0``: return exactly ``size`` bytes, raising if the
      stream holds fewer.
    * neither: decode the whole stream, which can run past the data
      (see below), as ``imagecodecs.aec_decode`` does.

    A stream can decode to more samples than were encoded: libaec pads
    the last block, and codes trailing all-zero blocks as "zero blocks
    to the end of the segment", a segment being 64 blocks or the
    reference sample interval if shorter (the whole interval with
    ``AEC_PAD_RSI``). With ``out`` or ``size``, a stream that holds more
    samples than fit, beyond that much padding, raises rather than
    being cut short.
    """
    cdef:
        const uint8_t[::1] src
        uint8_t[::1] dst_view
        Py_ssize_t srcsize
        Py_ssize_t sb
        Py_ssize_t cap
        Py_ssize_t total
        Py_ssize_t allowed
        unsigned char* dst_ptr
        unsigned char* scratch_ptr
        bytes out_bytes = None
        bytes scratch
        bytearray grow
        aec_stream strm
        int rc
        bint init_ok = False

    _check_params(bits_per_sample, block_size, rsi, flags)
    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = src.shape[0]
    sb = sample_bytes(bits_per_sample, flags)

    memset(<void*> &strm, 0, sizeof(aec_stream))
    if srcsize > 0:
        strm.next_in = <const unsigned char*> &src[0]
    strm.avail_in = <size_t> srcsize
    strm.bits_per_sample = <unsigned int> bits_per_sample
    strm.block_size = <unsigned int> block_size
    strm.rsi = <unsigned int> rsi
    strm.flags = <unsigned int> flags

    rc = aec_decode_init(&strm)
    if rc != AEC_OK:
        raise _err('aec_decode_init', rc)
    init_ok = True
    try:
        if out is not None or size >= 0:
            if out is not None:
                try:
                    dst_view = out
                except (TypeError, ValueError, BufferError) as e:
                    raise TypeError(
                        f"aec decode: out= must be a writable buffer, "
                        f"got {type(out).__name__}") from e
                cap = dst_view.shape[0]
                dst_ptr = &dst_view[0] if cap > 0 else NULL
            else:
                cap = size
                out_bytes = PyBytes_FromStringAndSize(NULL, cap)
                dst_ptr = <unsigned char*> PyBytes_AsString(out_bytes)
            # Whole samples only, so a short final sample is never cut.
            cap -= cap % sb
            if cap > 0:
                with nogil:
                    rc = _decode_step(&strm, dst_ptr, <size_t> cap)
                if rc != AEC_OK:
                    raise _err('aec_decode', rc)
            total = <Py_ssize_t> strm.total_out
            if total == cap and srcsize > 0:
                # Full: decode what is left into a scratch buffer. Up to
                # a segment of padding (see the docstring) can follow
                # the data; anything past that did not fit.
                if flags & AEC_PAD_RSI:
                    allowed = (<Py_ssize_t> block_size * rsi - 1) * sb
                else:
                    allowed = (<Py_ssize_t> block_size * min(rsi, 64) - 1) * sb
                scratch = PyBytes_FromStringAndSize(NULL, allowed + sb)
                scratch_ptr = <unsigned char*> PyBytes_AsString(scratch)
                with nogil:
                    rc = _decode_step(&strm, scratch_ptr,
                                      <size_t> (allowed + sb))
                if rc != AEC_OK:
                    raise _err('aec_decode', rc)
                if <Py_ssize_t> strm.total_out - total > allowed:
                    raise AecError(
                        f"aec decode: the output holds {cap} bytes but the "
                        f"stream decodes to more than {total + allowed} bytes")
            if out is not None:
                dst_view = None
                return out[:total]
            if total != size:
                raise AecError(
                    f"aec decode: the stream decoded to {total} bytes, "
                    f"expected {size}")
            return out_bytes

        # Unknown size: grow until libaec stops with room to spare,
        # which means the input is used up.
        cap = max(srcsize * 4, 4096)
        cap += sb - cap % sb if cap % sb else 0
        grow = bytearray(cap)
        total = 0
        while True:
            dst_view = grow
            with nogil:
                rc = _decode_step(&strm, &dst_view[total],
                                  <size_t> (cap - total))
            dst_view = None
            if rc != AEC_OK:
                raise _err('aec_decode', rc)
            total = <Py_ssize_t> strm.total_out
            if total < cap:
                break
            grow.extend(bytes(cap))
            cap *= 2
        del grow[total:]
        return bytes(grow)
    finally:
        if init_ok:
            aec_decode_end(&strm)


def read_framed_header(data):
    """Return ``(size, bits_per_sample, block_size, rsi, flags)`` from
    the preamble releases up to 0.4.0 wrote, or None if ``data`` cannot
    start with one."""
    if len(data) < _HEADER_LEN:
        return None
    raw = bytes(data[:_HEADER_LEN])
    if raw[13:16] != b'\x00\x00\x00':
        return None
    size, bps, block, rsi, flags = _struct.unpack(_HEADER_FMT, raw)
    if size > (1 << 34):
        return None
    if not (1 <= bps <= 32) or block not in (8, 16, 32, 64):
        return None
    if not (1 <= rsi <= 4096):
        return None
    # The old encoder only ever set SIGNED, 3BYTE, MSB and PREPROCESS.
    if flags & ~(AEC_DATA_SIGNED | AEC_DATA_3BYTE | AEC_DATA_MSB
                 | AEC_DATA_PREPROCESS):
        return None
    if size % sample_bytes(bps, flags):
        return None
    if size and len(data) == _HEADER_LEN:
        return None
    return size, bps, block, rsi, flags


def decode_framed(data, *, out=None):
    """Decode a blob in the framed layout releases up to 0.4.0 wrote
    (16-byte preamble + libaec stream).

    ``out`` is an int or a writable byte buffer, as for the byte codecs.
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

    orig_size, bps, block, rsi, flags = _struct.unpack(
        _HEADER_FMT, bytes(src[:_HEADER_LEN]))
    if orig_size == 0:
        if out is None or isinstance(out, int):
            return b''
        return out[:0]
    # A corrupt header can encode an absurd ``orig_size``; forwarding
    # that to PyBytes_FromStringAndSize attempts a multi-exabyte malloc.
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
    """A CCSDS stream has no magic bytes."""
    return False
