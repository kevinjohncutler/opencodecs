# opencodecs/codecs/_snappy.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native Snappy codec — Google's fast block compression.

Bytes-in / bytes-out wrapper around libsnappy's C API (snappy-c.h).
Snappy targets the ~500 MB/s throughput regime with ~2x compression
ratios — useful in Parquet/Hadoop/Bigtable pipelines where speed
matters more than tight ratios.

``encode`` / ``decode`` are the raw block format (no framing, no
checksums), the same as imagecodecs.snappy_encode / snappy_decode.
``framed_encode`` / ``framed_decode`` are Snappy's framing format, the
``.sz`` stream format, with CRC-32C chunk checksums.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t

from snappy cimport (
    snappy_status, SNAPPY_OK, SNAPPY_INVALID_INPUT, SNAPPY_BUFFER_TOO_SMALL,
    snappy_compress, snappy_uncompress,
    snappy_max_compressed_length, snappy_uncompressed_length,
    snappy_validate_compressed_buffer,
)


class SnappyError(RuntimeError):
    """Raised on Snappy encode/decode failures."""


def encode(data) -> bytes:
    """Compress bytes-like input as a raw Snappy block.

    Accepts any buffer-protocol object (bytes, bytearray, memoryview,
    numpy uint8 arrays). Returns the compressed bytes; raises
    :class:`SnappyError` on internal failure (rare — Snappy doesn't
    have many failure modes on the encode side).
    """
    cdef:
        const uint8_t[::1] src
        const uint8_t[::1] dst
        size_t srcsize, dstcap, dstlen
        bytes out
        snappy_status rc

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <size_t> src.shape[0]

    dstcap = snappy_max_compressed_length(srcsize)
    out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dstcap)
    # Memoryview output pattern (matches _zstd / _brotli / _lerc) —
    # measurably faster than PyBytes_AsString + _PyBytes_Resize on
    # multi-MB outputs. See _zstd.encode for the empirical rationale.
    dst = out
    dstlen = dstcap

    cdef const char* src_p = NULL
    if srcsize > 0:
        src_p = <const char*> &src[0]

    with nogil:
        rc = snappy_compress(src_p, srcsize, <char*> &dst[0], &dstlen)
    if rc != SNAPPY_OK:
        raise SnappyError(f"snappy_compress failed: status={rc}")
    del dst
    return out[:dstlen]


def decode(data, *, out=None):
    """Decompress a Snappy block.

    The uncompressed size is stored in the block header (Snappy's
    first varint), so we can pre-allocate the exact output buffer
    in one shot — no growing-buffer retry loop.

    Parameters
    ----------
    out : int | bytearray | memoryview | None, optional
        See ``_zstd.decode`` for the full ``out=`` contract.
    """
    cdef:
        const uint8_t[::1] src
        const uint8_t[::1] dst                # output bytes view
        uint8_t[::1] out_view                 # writable view of caller buffer
        size_t srcsize
        size_t dstlen = 0
        size_t out_len
        bytes out_bytes
        snappy_status rc
        char* dst_ptr

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if srcsize == 0:
        if out is None or isinstance(out, int):
            return b''
        return out[:0]

    cdef const char* src_p = <const char*> &src[0]
    rc = snappy_uncompressed_length(src_p, srcsize, &dstlen)
    if rc != SNAPPY_OK:
        raise SnappyError(
            f"snappy_uncompressed_length failed: status={rc} "
            f"(not a valid Snappy block?)"
        )

    # ----- caller-supplied writable buffer (zero-alloc path) -----
    if out is not None and not isinstance(out, int):
        try:
            out_view = out
        except (TypeError, ValueError, BufferError) as e:
            raise TypeError(
                f"snappy decode: out= must be int or writable buffer, "
                f"got {type(out).__name__}"
            ) from e
        if out_view.shape[0] < dstlen:
            raise SnappyError(
                f"snappy decode: out= buffer is {out_view.shape[0]} bytes "
                f"but the Snappy header declares {dstlen} bytes")
        out_len = <size_t> out_view.shape[0]
        dst_ptr = <char*> &out_view[0]
        with nogil:
            rc = snappy_uncompress(src_p, srcsize, dst_ptr, &out_len)
        if rc != SNAPPY_OK:
            raise SnappyError(f"snappy_uncompress failed: status={rc}")
        del out_view
        return out[:out_len]

    # ----- fresh bytes allocation -----
    if isinstance(out, int):
        if out < dstlen:
            raise SnappyError(
                f"snappy decode: out=int({out}) is less than the "
                f"Snappy header's declared {dstlen} bytes")
    out_bytes = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dstlen)
    dst = out_bytes
    out_len = dstlen   # snappy_uncompress overwrites
    with nogil:
        rc = snappy_uncompress(src_p, srcsize, <char*> &dst[0], &out_len)
    if rc != SNAPPY_OK:
        raise SnappyError(f"snappy_uncompress failed: status={rc}")
    del dst
    if out_len != dstlen:
        # Shouldn't happen for valid input — header says X bytes, we
        # got Y. Slice down to be safe.
        return out_bytes[:out_len]
    return out_bytes


def check_signature(data) -> bool:
    """Best-effort Snappy block detection.

    Snappy raw blocks have no fixed magic — they start with a varint
    encoding the uncompressed length. We validate the full buffer
    via ``snappy_validate_compressed_buffer`` (single linear scan,
    no allocation). It's not free, but for the byte sizes typical of
    a sniff (a few KB), the cost is negligible — and a header-only
    check is wrong far too often to be useful.
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        snappy_status rc

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        try:
            src = bytes(data)
        except Exception:
            return False
    srcsize = <size_t> src.shape[0]
    if srcsize < 1 or srcsize > 0x7fffffff:
        return False
    rc = snappy_validate_compressed_buffer(
        <const char*> &src[0], srcsize)
    return rc == SNAPPY_OK


# ---------------------------------------------------------------------------
# Snappy framing format (the ".sz" stream format)
# ---------------------------------------------------------------------------
#
# Google's framing_format.txt defines it: a stream identifier chunk, then
# chunks of [type:1][length:3 LE][data]. Type 0x00 holds a compressed
# block and 0x01 a stored one, each preceded by the masked CRC-32C of its
# uncompressed bytes and each at most 65536 bytes uncompressed; types
# 0x80-0xfe are skipped and 0x02-0x7f are reserved and fatal. Its file
# extension is .sz. The raw block functions above are a different thing,
# the block format with no framing, which is what imagecodecs.snappy_*
# reads and writes.

from libc.stdint cimport uint32_t
from libc.string cimport memcpy
from cpython.bytes cimport PyBytes_AsString

cdef extern from *:
    """
    /* CRC-32C (Castagnoli, reflected polynomial 0x82F63B78), slice-by-8. */
    static uint32_t oc_crc32c_table[8][256];

    static void oc_crc32c_init(void) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++)
                c = (c & 1) ? (c >> 1) ^ 0x82F63B78u : c >> 1;
            oc_crc32c_table[0][i] = c;
        }
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = oc_crc32c_table[0][i];
            for (int t = 1; t < 8; t++) {
                c = oc_crc32c_table[0][c & 0xFF] ^ (c >> 8);
                oc_crc32c_table[t][i] = c;
            }
        }
    }

    static uint32_t oc_crc32c(const uint8_t *p, size_t n) {
        uint32_t c = 0xFFFFFFFFu;
        while (n >= 8) {
            uint32_t lo = c ^ ((uint32_t)p[0] | ((uint32_t)p[1] << 8)
                               | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
            c = oc_crc32c_table[7][lo & 0xFF] ^ oc_crc32c_table[6][(lo >> 8) & 0xFF]
              ^ oc_crc32c_table[5][(lo >> 16) & 0xFF] ^ oc_crc32c_table[4][lo >> 24]
              ^ oc_crc32c_table[3][p[4]] ^ oc_crc32c_table[2][p[5]]
              ^ oc_crc32c_table[1][p[6]] ^ oc_crc32c_table[0][p[7]];
            p += 8;
            n -= 8;
        }
        while (n--) c = oc_crc32c_table[0][(c ^ *p++) & 0xFF] ^ (c >> 8);
        return c ^ 0xFFFFFFFFu;
    }

    /* framing_format.txt section 3: the stored checksum is masked. */
    static uint32_t oc_snappy_masked_crc(const uint8_t *p, size_t n) {
        uint32_t c = oc_crc32c(p, n);
        return ((c >> 15) | (c << 17)) + 0xa282ead8u;
    }
    """
    void oc_crc32c_init() nogil
    uint32_t oc_crc32c(const uint8_t* p, size_t n) nogil
    uint32_t oc_snappy_masked_crc(const uint8_t* p, size_t n) nogil

oc_crc32c_init()

FRAMED_MAGIC = b'\xff\x06\x00\x00sNaPpY'
cdef size_t _FRAMED_MAX_BLOCK = 65536


def crc32c(data) -> int:
    """CRC-32C of ``data`` (exposed for tests)."""
    cdef const uint8_t[::1] src
    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    if src.shape[0] == 0:
        return oc_crc32c(NULL, 0)
    return oc_crc32c(&src[0], <size_t> src.shape[0])


cdef inline void _put_le(uint8_t* p, uint32_t v, int nbytes) noexcept nogil:
    cdef int k
    for k in range(nbytes):
        p[k] = <uint8_t> (v >> (8 * k))


def framed_encode(data) -> bytes:
    """Encode as a Snappy framing-format stream (the ``.sz`` format).

    The stream identifier, then one chunk per 65536 input bytes. A chunk
    is stored compressed unless that saves less than 1/8 of it, in which
    case it is stored uncompressed (the rule the Go and Python reference
    writers use); either way it carries the masked CRC-32C of its data.
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize, nchunks, cap, pos = 0, wpos, take, clen, i
        bytes out
        uint8_t* dst
        uint32_t crc
        snappy_status rc
    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    nchunks = (srcsize + _FRAMED_MAX_BLOCK - 1) // _FRAMED_MAX_BLOCK
    cap = 10 + nchunks * (8 + snappy_max_compressed_length(_FRAMED_MAX_BLOCK))
    out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> cap)
    dst = <uint8_t*> PyBytes_AsString(out)
    memcpy(dst, <const char*> b'\xff\x06\x00\x00sNaPpY', 10)
    wpos = 10
    rc = SNAPPY_OK
    with nogil:
        for i in range(nchunks):
            take = srcsize - pos
            if take > _FRAMED_MAX_BLOCK:
                take = _FRAMED_MAX_BLOCK
            crc = oc_snappy_masked_crc(&src[pos], take)
            clen = cap - wpos - 8
            rc = snappy_compress(<const char*> &src[pos], take,
                                 <char*> (dst + wpos + 8), &clen)
            if rc != SNAPPY_OK:
                break
            if clen >= take - take // 8:
                dst[wpos] = 0x01
                memcpy(dst + wpos + 8, &src[pos], take)
                clen = take
            else:
                dst[wpos] = 0x00
            _put_le(dst + wpos + 1, <uint32_t> (clen + 4), 3)
            _put_le(dst + wpos + 4, crc, 4)
            wpos += 8 + clen
            pos += take
    if rc != SNAPPY_OK:
        raise SnappyError(f"snappy_compress failed: status={rc}")
    return out[:wpos]


def framed_decode(data, *, bint verify=True):
    """Decode a Snappy framing-format stream (the ``.sz`` format).

    Every chunk's masked CRC-32C is checked unless ``verify=False``.
    Concatenated streams (a repeated stream identifier) decode to their
    concatenation; padding and reserved skippable chunks are skipped;
    reserved unskippable chunks, a missing stream identifier and a
    truncated chunk raise :class:`SnappyError`.
    """
    cdef:
        const uint8_t[::1] src
        const uint8_t* p
        size_t n, pos = 0, length, total = 0, ulen, wpos = 0, outlen
        int ctype
        uint32_t want
        bytes out
        uint8_t* dst
        list chunks = []
        snappy_status rc
    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    n = <size_t> src.shape[0]
    if n < 10 or bytes(src[:10]) != FRAMED_MAGIC:
        raise SnappyError("snappy framed: missing stream identifier")
    p = &src[0]
    # Pass 1: walk the chunks and size the output exactly.
    while pos < n:
        if n - pos < 4:
            raise SnappyError(f"snappy framed: truncated chunk header at offset {pos}")
        ctype = p[pos]
        length = <size_t> p[pos + 1] | (<size_t> p[pos + 2] << 8) | (<size_t> p[pos + 3] << 16)
        if length > n - pos - 4:
            raise SnappyError(f"snappy framed: chunk at offset {pos} is truncated")
        if ctype == 0xFF:
            if length != 6 or bytes(src[pos + 4:pos + 10]) != b'sNaPpY':
                raise SnappyError(f"snappy framed: bad stream identifier at offset {pos}")
        elif ctype == 0x00 or ctype == 0x01:
            if length < 4:
                raise SnappyError(f"snappy framed: chunk at offset {pos} has no checksum")
            if ctype == 0x01:
                ulen = length - 4
            else:
                rc = snappy_uncompressed_length(
                    <const char*> (p + pos + 8), length - 4, &ulen)
                if rc != SNAPPY_OK:
                    raise SnappyError(
                        f"snappy framed: chunk at offset {pos} is not a Snappy block")
            if ulen > _FRAMED_MAX_BLOCK:
                raise SnappyError(
                    f"snappy framed: chunk at offset {pos} holds {ulen} bytes, "
                    f"more than the format's 65536")
            chunks.append((pos, ctype, length, ulen))
            total += ulen
        elif ctype <= 0x7F:
            raise SnappyError(
                f"snappy framed: reserved unskippable chunk 0x{ctype:02x} "
                f"at offset {pos}")
        # 0x80-0xfe: padding and reserved skippable chunks.
        pos += 4 + length
    # Pass 2: decode and verify.
    out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> total)
    dst = <uint8_t*> PyBytes_AsString(out)
    for chunk_pos, ctype, chunk_len, chunk_ulen in chunks:
        pos = <size_t> chunk_pos
        length = <size_t> chunk_len - 4        # payload after the checksum
        ulen = <size_t> chunk_ulen
        want = (<uint32_t> p[pos + 4] | (<uint32_t> p[pos + 5] << 8)
                | (<uint32_t> p[pos + 6] << 16) | (<uint32_t> p[pos + 7] << 24))
        if ctype == 0x01:
            memcpy(dst + wpos, p + pos + 8, ulen)
        else:
            outlen = ulen
            with nogil:
                rc = snappy_uncompress(<const char*> (p + pos + 8), length,
                                       <char*> (dst + wpos), &outlen)
            if rc != SNAPPY_OK or outlen != ulen:
                raise SnappyError(
                    f"snappy framed: chunk at offset {pos} failed to decode")
        if verify and oc_snappy_masked_crc(dst + wpos, ulen) != want:
            raise SnappyError(f"snappy framed: checksum mismatch in chunk at offset {pos}")
        wpos += ulen
    return out


def framed_check_signature(data) -> bool:
    """True if ``data`` starts with the framing format's stream identifier."""
    try:
        return bytes(data[:10]) == FRAMED_MAGIC
    except Exception:
        return False


__all__ = ["encode", "decode", "check_signature", "SnappyError",
           "framed_encode", "framed_decode", "framed_check_signature",
           "crc32c"]
