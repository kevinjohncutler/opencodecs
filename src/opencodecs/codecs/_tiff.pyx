# opencodecs/codecs/_tiff.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native TIFF IFD walker + tile-decode dispatcher.

Replaces tifffile's pure-Python IFD parsing with a Cython implementation
that:

  * Parses both classic TIFF 6.0 and BigTIFF (TIFF 2.0) headers
  * Walks the IFD chain extracting tags into a dict-per-IFD
  * Computes tile/strip layout (offset table, byte counts, tile sizes)
  * Dispatches tile decode to opencodecs's existing native codecs
    (deflate, jpeg, jpeg2k, zstd, jxl, lerc, webp) — no libtiff dep

What this module does NOT do (intentionally, deferred to caller):

  * I/O — accepts a callable ``read_at(offset, n) -> bytes`` so the same
    parser drives local files, mmap, HTTP-range requests, S3, etc.
  * Color-space conversion — returns the raw decoded sample buffer
    (caller handles YCbCr→RGB, CFA→RGB, etc.)
  * Concurrency — single-threaded; the existing tiff_reader.py
    wraps this in a thread pool for parallel-tile decode.

All 4-byte-offset values in classic TIFF and 8-byte-offset values in
BigTIFF are read into ``uint64`` regardless of the underlying width;
upper bits are 0 for classic TIFF.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t
from libc.string cimport memcpy

import struct as _struct


# TIFF LZW, both directions, in ``3rdparty/oc_tifflzw/``. Both are
# opencodecs' own code (MIT); the bit-stream and dictionary handling
# live in C because that is where the hot loop belongs.
cdef extern from "oc_tifflzw.h" nogil:
    Py_ssize_t oc_tifflzw_decode(
        const uint8_t* input, size_t input_len,
        uint8_t* output, size_t output_len,
    )
    size_t oc_tifflzw_encode_bound(size_t input_len)
    Py_ssize_t oc_tifflzw_encode(
        const uint8_t* input, size_t input_len,
        uint8_t* output, size_t output_len,
    )


# ---------------------------------------------------------------------------
# TIFF data-type sizes (per TIFF 6.0 §2 + TIFF 2.0 BigTIFF additions).
# ---------------------------------------------------------------------------

# Index into this table by the 2-byte type field of an IFD entry.
# 0 sentinel for "unknown / out-of-range type" so callers can detect it.
cdef int _TYPE_SIZE[20]
_TYPE_SIZE[:] = [
    0,    # 0
    1,    # 1  BYTE
    1,    # 2  ASCII (1 byte/char)
    2,    # 3  SHORT (uint16)
    4,    # 4  LONG  (uint32)
    8,    # 5  RATIONAL (2 × uint32)
    1,    # 6  SBYTE
    1,    # 7  UNDEFINED
    2,    # 8  SSHORT
    4,    # 9  SLONG
    8,    # 10 SRATIONAL
    4,    # 11 FLOAT
    8,    # 12 DOUBLE
    4,    # 13 IFD (uint32 offset)
    0, 0, # 14, 15 reserved
    8,    # 16 LONG8 (uint64; BigTIFF)
    8,    # 17 SLONG8 (int64; BigTIFF)
    8,    # 18 IFD8 (uint64 offset; BigTIFF)
    0,    # 19
]


# Constants for the tags we always need.
TAG_IMAGE_WIDTH        = 256
TAG_IMAGE_LENGTH       = 257
TAG_BITS_PER_SAMPLE    = 258
TAG_COMPRESSION        = 259
TAG_PHOTOMETRIC        = 262
TAG_STRIP_OFFSETS      = 273
TAG_SAMPLES_PER_PIXEL  = 277
TAG_ROWS_PER_STRIP     = 278
TAG_STRIP_BYTE_COUNTS  = 279
TAG_PLANAR_CONFIG      = 284
TAG_PREDICTOR          = 317
TAG_TILE_WIDTH         = 322
TAG_TILE_LENGTH        = 323
TAG_TILE_OFFSETS       = 324
TAG_TILE_BYTE_COUNTS   = 325
TAG_SAMPLE_FORMAT      = 339
TAG_JPEG_TABLES        = 347
TAG_SUB_IFDS           = 330   # bioformats / OME-TIFF pyramid sub-resolutions


# Compression codes opencodecs recognizes. Codes outside this set are
# preserved in the page metadata and the caller can dispatch elsewhere.
CMP_NONE          = 1
CMP_CCITT         = 2     # CCITT 1D — not native, raise (legacy fax)
CMP_CCITT_T4      = 3     # legacy fax
CMP_CCITT_T6      = 4
CMP_LZW           = 5     # vendored decoder
CMP_OLD_JPEG      = 6     # deprecated — TIFF 6 § "old JPEG" path; rare
CMP_JPEG          = 7     # libjpeg-turbo via opencodecs._jpeg
CMP_DEFLATE       = 8     # zlib via opencodecs._deflate (RFC 1950)
CMP_PACKBITS      = 32773 # vendored decoder
CMP_LZMA          = 34925
CMP_ZSTD          = 50000 # opencodecs._zstd
CMP_WEBP          = 50001 # opencodecs._webp
CMP_JXL           = 50002 # opencodecs._jxl  (TIFF 6 community-assigned)
CMP_JPEG2000      = 34712 # opencodecs._jpeg2k
CMP_LERC          = 34887 # opencodecs._lerc — official TIFF LERC code (registered 2016).
                          # 33003 is an earlier/legacy alias seen in some
                          # very old GDAL output; we accept both at the
                          # dispatcher level.
CMP_LERC_LEGACY   = 33003
CMP_ADOBE_DEFLATE = 32946 # the code tifffile writes for `compression="deflate"`
                          # (TIFF 6 sec'd deflate as 8; Adobe added 32946 with
                          # identical semantics. Both decode through zlib.)

# Thermo Fisher EER (Electron Event Representation) — cryo-EM detector
# raw output. Three compression codes carry slightly different EER
# variants; per-frame bit-field widths come from private tags
# 65007/65008/65009.
CMP_EER_V0        = 65000
CMP_EER_V1        = 65001
CMP_EER_V2        = 65002
# EER private tags (TFS-assigned; documented in EER spec v3, M. Leichsenring 2023).
TAG_EER_SKIPBITS  = 65007
TAG_EER_HORZBITS  = 65008
TAG_EER_VERTBITS  = 65009


class TiffError(RuntimeError):
    """Raised on malformed TIFF input."""


# ---------------------------------------------------------------------------
# Header parse — returns (byte_order, is_bigtiff, first_ifd_offset)
# ---------------------------------------------------------------------------


def parse_header(read_at):
    """Parse a TIFF header via a ``read_at(offset, n) -> bytes`` callable.

    Returns ``(byte_order, is_bigtiff, first_ifd_offset)`` where
    byte_order is ``'<'`` (little-endian) or ``'>'`` (big-endian) per
    Python's struct conventions.
    """
    head = read_at(0, 16)
    if len(head) < 8:
        raise TiffError("TIFF: file too short for header")
    bo = head[:2]
    if bo == b"II":
        byte_order = "<"
    elif bo == b"MM":
        byte_order = ">"
    else:
        raise TiffError(f"TIFF: bad byte-order mark {bo!r}")
    magic = _struct.unpack(byte_order + "H", head[2:4])[0]
    if magic == 0x002A:
        # Classic TIFF: 4-byte first IFD offset.
        first = _struct.unpack(byte_order + "I", head[4:8])[0]
        return byte_order, False, int(first)
    if magic == 0x002B:
        # BigTIFF: 2 bytes "8" (offset size), 2 bytes constant 0,
        # 8-byte first IFD offset.
        if len(head) < 16:
            raise TiffError("TIFF: BigTIFF needs 16-byte header")
        offset_size, const = _struct.unpack(byte_order + "HH", head[4:8])
        if offset_size != 8 or const != 0:
            raise TiffError(
                f"TIFF: invalid BigTIFF marker (offset_size={offset_size}, "
                f"const={const})"
            )
        first = _struct.unpack(byte_order + "Q", head[8:16])[0]
        return byte_order, True, int(first)
    raise TiffError(f"TIFF: unknown magic 0x{magic:04x}")


# ---------------------------------------------------------------------------
# IFD walker
# ---------------------------------------------------------------------------


def _read_value(byte_order, ifd_data, entry_offset, is_bigtiff):
    """Read one IFD entry. Returns (tag, type, count, value_or_offset).

    ``value_or_offset`` is the raw 4 (classic) or 8 (BigTIFF) bytes from
    the entry's value slot — interpretation depends on type & count.
    """
    bo = byte_order
    if is_bigtiff:
        # 20-byte entry: 2 tag, 2 type, 8 count, 8 value-or-offset
        tag, dtype, count = _struct.unpack_from(bo + "HHQ", ifd_data, entry_offset)
        value_bytes = ifd_data[entry_offset + 12:entry_offset + 20]
    else:
        # 12-byte entry: 2 tag, 2 type, 4 count, 4 value-or-offset
        tag, dtype, count = _struct.unpack_from(bo + "HHI", ifd_data, entry_offset)
        value_bytes = ifd_data[entry_offset + 8:entry_offset + 12]
    return int(tag), int(dtype), int(count), value_bytes


def _resolve_value(read_at, byte_order, is_bigtiff, dtype, count, value_bytes):
    """Resolve an IFD entry's payload.

    If the data fits in the inline 4 (classic) or 8 (BigTIFF) bytes,
    decode in place. Otherwise treat ``value_bytes`` as an offset and
    read ``count * size_of(dtype)`` bytes from there.
    """
    if dtype < 1 or dtype >= 20:
        return None  # unknown type; skip
    item_size = _TYPE_SIZE[dtype]
    if item_size == 0:
        return None
    total = count * item_size
    inline_cap = 8 if is_bigtiff else 4
    if total <= inline_cap:
        raw = value_bytes[:total]
    else:
        bo = byte_order
        if is_bigtiff:
            offset = _struct.unpack(bo + "Q", value_bytes)[0]
        else:
            offset = _struct.unpack(bo + "I", value_bytes)[0]
        raw = read_at(int(offset), total)
        if len(raw) != total:
            raise TiffError(
                f"TIFF: out-of-band IFD value short read at {offset}: "
                f"got {len(raw)}, want {total}"
            )

    bo = byte_order
    if dtype == 1 or dtype == 7:                    # BYTE / UNDEFINED
        return bytes(raw)
    if dtype == 2:                                  # ASCII
        # Strip trailing NULs and any extra noise.
        return bytes(raw).rstrip(b'\x00').decode('ascii', 'replace')
    if dtype == 3:                                  # SHORT
        return _struct.unpack(bo + f"{count}H", raw) if count > 1 \
            else _struct.unpack(bo + "H", raw[:2])[0]
    if dtype == 4:                                  # LONG
        return _struct.unpack(bo + f"{count}I", raw) if count > 1 \
            else _struct.unpack(bo + "I", raw[:4])[0]
    if dtype == 5:                                  # RATIONAL (num/den)
        out = []
        for i in range(count):
            num, den = _struct.unpack_from(bo + "II", raw, i * 8)
            out.append((int(num), int(den)))
        return tuple(out) if count > 1 else out[0]
    if dtype == 6:                                  # SBYTE
        return _struct.unpack(bo + f"{count}b", raw)
    if dtype == 8:                                  # SSHORT
        return _struct.unpack(bo + f"{count}h", raw) if count > 1 \
            else _struct.unpack(bo + "h", raw[:2])[0]
    if dtype == 9:                                  # SLONG
        return _struct.unpack(bo + f"{count}i", raw) if count > 1 \
            else _struct.unpack(bo + "i", raw[:4])[0]
    if dtype == 10:                                 # SRATIONAL
        out = []
        for i in range(count):
            num, den = _struct.unpack_from(bo + "ii", raw, i * 8)
            out.append((int(num), int(den)))
        return tuple(out) if count > 1 else out[0]
    if dtype == 11:                                 # FLOAT
        return _struct.unpack(bo + f"{count}f", raw) if count > 1 \
            else _struct.unpack(bo + "f", raw[:4])[0]
    if dtype == 12:                                 # DOUBLE
        return _struct.unpack(bo + f"{count}d", raw) if count > 1 \
            else _struct.unpack(bo + "d", raw[:8])[0]
    if dtype == 13:                                 # IFD (uint32 offset)
        return _struct.unpack(bo + f"{count}I", raw) if count > 1 \
            else _struct.unpack(bo + "I", raw[:4])[0]
    if dtype == 16:                                 # LONG8
        return _struct.unpack(bo + f"{count}Q", raw) if count > 1 \
            else _struct.unpack(bo + "Q", raw[:8])[0]
    if dtype == 17:                                 # SLONG8
        return _struct.unpack(bo + f"{count}q", raw) if count > 1 \
            else _struct.unpack(bo + "q", raw[:8])[0]
    if dtype == 18:                                 # IFD8 (uint64 offset)
        return _struct.unpack(bo + f"{count}Q", raw) if count > 1 \
            else _struct.unpack(bo + "Q", raw[:8])[0]
    return None


def parse_ifd(read_at, byte_order, is_bigtiff, ifd_offset):
    """Read one IFD at ``ifd_offset``. Returns (tags_dict, next_ifd_offset).

    Tags are keyed by their integer ID. Values are scalars for count==1,
    tuples otherwise (per TIFF convention). Unknown types are skipped.
    """
    if ifd_offset == 0:
        return {}, 0
    # Read the entry-count word, then the body all at once for fewer I/O hops.
    count_size = 8 if is_bigtiff else 2
    count_bytes = read_at(ifd_offset, count_size)
    if len(count_bytes) < count_size:
        raise TiffError("TIFF: short read on IFD entry count")
    if is_bigtiff:
        n_entries = _struct.unpack(byte_order + "Q", count_bytes)[0]
    else:
        n_entries = _struct.unpack(byte_order + "H", count_bytes)[0]

    entry_size = 20 if is_bigtiff else 12
    next_offset_size = 8 if is_bigtiff else 4
    body_size = n_entries * entry_size + next_offset_size
    body = read_at(ifd_offset + count_size, body_size)
    if len(body) < body_size:
        raise TiffError(
            f"TIFF: short read on IFD body (got {len(body)}, want {body_size})"
        )

    tags = {}
    for i in range(n_entries):
        off = i * entry_size
        tag, dtype, ent_count, value_bytes = _read_value(
            byte_order, body, off, is_bigtiff,
        )
        try:
            value = _resolve_value(
                read_at, byte_order, is_bigtiff,
                dtype, ent_count, value_bytes,
            )
        except TiffError:
            value = None  # broken entry; skip rather than abort whole IFD
        tags[tag] = (dtype, ent_count, value)

    if is_bigtiff:
        next_ifd = _struct.unpack(byte_order + "Q",
                                  body[n_entries * entry_size:
                                       n_entries * entry_size + 8])[0]
    else:
        next_ifd = _struct.unpack(byte_order + "I",
                                  body[n_entries * entry_size:
                                       n_entries * entry_size + 4])[0]
    return tags, int(next_ifd)


def parse_all_ifds(read_at):
    """Walk the IFD chain eagerly, parsing every tag.

    Slower than ``parse_ifd_chain``; kept for callers that genuinely
    want every tag resolved up front (rare). Most callers should use
    ``parse_ifd_chain`` + on-demand ``parse_ifd``.
    """
    byte_order, is_bigtiff, off = parse_header(read_at)
    out = []
    visited = set()
    while off != 0:
        if off in visited:
            raise TiffError(f"TIFF: cyclic IFD chain at offset {off}")
        visited.add(off)
        tags, next_off = parse_ifd(read_at, byte_order, is_bigtiff, off)
        out.append(tags)
        off = next_off
    return byte_order, is_bigtiff, out


cdef inline uint16_t _read_u16(const uint8_t* p, bint big_endian) nogil:
    if big_endian:
        return (<uint16_t>p[0] << 8) | <uint16_t>p[1]
    return <uint16_t>p[0] | (<uint16_t>p[1] << 8)


cdef inline uint32_t _read_u32(const uint8_t* p, bint big_endian) nogil:
    cdef uint32_t v
    if big_endian:
        v = (<uint32_t>p[0] << 24) | (<uint32_t>p[1] << 16) \
            | (<uint32_t>p[2] << 8) | <uint32_t>p[3]
    else:
        v = <uint32_t>p[0] | (<uint32_t>p[1] << 8) \
            | (<uint32_t>p[2] << 16) | (<uint32_t>p[3] << 24)
    return v


cdef inline uint64_t _read_u64(const uint8_t* p, bint big_endian) nogil:
    cdef uint64_t v = 0
    cdef int i
    if big_endian:
        for i in range(8):
            v = (v << 8) | <uint64_t>p[i]
    else:
        for i in range(7, -1, -1):
            v = (v << 8) | <uint64_t>p[i]
    return v


def parse_ifd_chain(read_at):
    """Walk the IFD chain by offsets only, without resolving tag values.

    Returns ``(byte_order, is_bigtiff, [ifd_offset, ...])``.

    Fast path: when the underlying source is bytes / bytearray /
    memoryview backed by a single contiguous buffer (every TIFF
    that fits in memory), the whole walk runs in nogil Cython with
    raw pointer arithmetic — no Python calls, no struct.unpack,
    no per-IFD read_at trampoline. This is the difference between
    "few µs per IFD" and "few ns per IFD" — i.e. opening a 10000-page
    OME-TIFF in 0.3 ms vs 30 ms.
    """
    cdef const uint8_t[::1] view
    cdef const uint8_t* buf
    cdef Py_ssize_t bufsize
    cdef bint big_endian
    cdef Py_ssize_t entry_size
    cdef Py_ssize_t count_size
    cdef Py_ssize_t next_size
    cdef uint64_t cur
    cdef uint64_t n_entries_64
    cdef uint32_t n_entries_32
    cdef Py_ssize_t skip
    cdef uint64_t next_off
    cdef Py_ssize_t MAX_IFDS = 1 << 24   # 16 M IFDs is plenty
    cdef Py_ssize_t cap
    cdef Py_ssize_t n
    cdef uint64_t* offsets_buf
    cdef uint64_t* tmp

    byte_order, is_bigtiff, off = parse_header(read_at)
    if off == 0:
        return byte_order, is_bigtiff, []

    big_endian = (byte_order == ">")
    entry_size = 20 if is_bigtiff else 12
    count_size = 8 if is_bigtiff else 2
    next_size = 8 if is_bigtiff else 4
    cur = <uint64_t> off

    # Try the fast path: did the caller wrap a contiguous in-memory
    # buffer? The bytes/memoryview-input TiffStream sets `read_at._buf`
    # to the underlying memoryview.
    direct = getattr(read_at, "_buf", None)
    if direct is not None:
        try:
            view = direct
        except Exception:
            view = None
        if view is not None:
            bufsize = view.shape[0]
            buf = &view[0]
            # Collect into a typed C array first, then convert to a
            # Python list once at the end. Avoids list.append overhead
            # (one PyObject creation + GC bookkeeping per IFD).
            # Cap protects against malicious / corrupted IFD chains.
            cap = 64
            n = 0
            offsets_buf = <uint64_t*> PyMem_Malloc(cap * sizeof(uint64_t))
            if offsets_buf == NULL:
                raise MemoryError()
            try:
                while cur != 0:
                    if cur + count_size > <uint64_t>bufsize:
                        raise TiffError("TIFF: short read on IFD entry count")
                    if n >= MAX_IFDS:
                        raise TiffError(
                            f"TIFF: IFD chain too long (>{MAX_IFDS}) — "
                            "likely cyclic or corrupted"
                        )
                    if n == cap:
                        cap *= 2
                        tmp = <uint64_t*> PyMem_Realloc(
                            offsets_buf, cap * sizeof(uint64_t))
                        if tmp == NULL:
                            raise MemoryError()
                        offsets_buf = tmp
                    offsets_buf[n] = cur
                    n += 1

                    if is_bigtiff:
                        n_entries_64 = _read_u64(buf + cur, big_endian)
                        skip = count_size + <Py_ssize_t>(n_entries_64 * entry_size)
                    else:
                        n_entries_32 = <uint32_t>_read_u16(buf + cur, big_endian)
                        skip = count_size + <Py_ssize_t>(n_entries_32 * entry_size)
                    if cur + skip + next_size > <uint64_t>bufsize:
                        raise TiffError("TIFF: short read on next-IFD offset")
                    if is_bigtiff:
                        next_off = _read_u64(buf + cur + skip, big_endian)
                    else:
                        next_off = <uint64_t>_read_u32(buf + cur + skip, big_endian)
                    if next_off != 0 and next_off <= cur:
                        # Cycle detection without a Python set: any
                        # well-formed TIFF puts later IFDs at higher
                        # offsets (TIFF spec requires it). A non-zero
                        # backward jump is a cycle or attack.
                        raise TiffError(
                            f"TIFF: backward IFD jump from {cur} to {next_off}"
                        )
                    cur = next_off

                # Convert C array to Python list once at the end.
                out = [<object>offsets_buf[i] for i in range(n)]
            finally:
                PyMem_Free(offsets_buf)
            return byte_order, is_bigtiff, out

    # Slow path: dispatch through read_at (used for file handles,
    # HTTP-range data sources, etc.). Same algorithm, Python overhead.
    out = []
    visited = set()
    while off != 0:
        if off in visited:
            raise TiffError(f"TIFF: cyclic IFD chain at offset {off}")
        visited.add(off)
        out.append(int(off))
        ec = read_at(off, count_size)
        if len(ec) < count_size:
            raise TiffError("TIFF: short read on IFD entry count")
        if is_bigtiff:
            n_e = _struct.unpack(byte_order + "Q", bytes(ec))[0]
        else:
            n_e = _struct.unpack(byte_order + "H", bytes(ec))[0]
        skip_off = count_size + n_e * entry_size
        next_off_bytes = read_at(off + skip_off, next_size)
        if len(next_off_bytes) < next_size:
            raise TiffError("TIFF: short read on next-IFD offset")
        if is_bigtiff:
            off = int(_struct.unpack(byte_order + "Q",
                                     bytes(next_off_bytes))[0])
        else:
            off = int(_struct.unpack(byte_order + "I",
                                     bytes(next_off_bytes))[0])
    return byte_order, is_bigtiff, out


def copy_strips_from_buffer(
    const uint8_t[::1] src not None,
    uint8_t[::1] dst not None,
    object offsets,
    object byte_counts,
):
    """Copy N uncompressed strips from src into dst.

    Both `src` and `dst` are contiguous uint8 views — `src` is the
    entire TIFF buffer, `dst` is a flat uint8 view of the destination
    ndarray. Strips are copied row-major in `offsets` order, contiguously
    appended in `dst`.

    Optimization: when all strips are end-to-end in the file (the TIFF
    spec doesn't require this but virtually every writer produces it),
    collapse the N separate copies into a single big memcpy. For 64
    strips of 512 KB this turns 64×memcpy + per-strip Python entry into
    one 32 MB memcpy — matches what tifffile does internally.
    """
    cdef Py_ssize_t n_strips = len(offsets)
    cdef Py_ssize_t i
    cdef uint64_t off
    cdef uint64_t nbytes
    cdef Py_ssize_t srcsize = src.shape[0]
    cdef Py_ssize_t dstsize = dst.shape[0]
    cdef Py_ssize_t write_off = 0
    cdef const uint8_t* sp = &src[0]
    cdef uint8_t* dp = &dst[0]
    cdef uint64_t prev_end
    cdef bint contiguous = True
    cdef uint64_t total_bytes = 0
    cdef uint64_t first_off = 0

    if n_strips == 0:
        return

    # Single pass: validate + detect end-to-end contiguity.
    first_off = <uint64_t> offsets[0]
    prev_end = first_off
    for i in range(n_strips):
        off = <uint64_t> offsets[i]
        nbytes = <uint64_t> byte_counts[i]
        if off + nbytes > <uint64_t>srcsize:
            raise TiffError(
                f"TIFF: strip {i} extends past end of buffer"
            )
        if i > 0 and off != prev_end:
            contiguous = False
        prev_end = off + nbytes
        total_bytes += nbytes
    if total_bytes > <uint64_t>dstsize:
        raise TiffError("TIFF: combined strip bytes overflow output buffer")

    if contiguous:
        # One big memcpy — equivalent to a sequential read of the whole
        # strip range; tifffile takes this path on contiguous TIFFs.
        with nogil:
            memcpy(dp, sp + first_off, <Py_ssize_t> total_bytes)
        return

    # Fragmented strips (rare): fall back to per-strip memcpy.
    for i in range(n_strips):
        off = <uint64_t> offsets[i]
        nbytes = <uint64_t> byte_counts[i]
        with nogil:
            memcpy(dp + write_off, sp + off, <Py_ssize_t> nbytes)
        write_off += <Py_ssize_t> nbytes


# ---------------------------------------------------------------------------
# PackBits (TIFF compression code 32773)
# ---------------------------------------------------------------------------
#
# Apple's run-length encoding scheme. For each input byte n (interpreted
# as int8):
#   n in 0..127      → copy next n+1 input bytes literally
#   n == -128 (0x80) → no-op
#   n in -127..-1    → replicate next byte (1 - n) times
# All TIFF writers fit into < 256 KB output per strip in practice, but
# we still bound-check on every emit.

def packbits_decode(data, expected_size: int = -1) -> bytes:
    """Decode a PackBits-compressed strip / tile to bytes.

    `expected_size` (when known from tile_h * tile_w * itemsize) is used
    as the output capacity. Pass -1 to size at 2x source (works for
    typical TIFF strip RLE; raises if the encoded data overruns).
    """
    cdef:
        const uint8_t[::1] src
        Py_ssize_t srcsize
        Py_ssize_t i = 0
        Py_ssize_t out_off = 0
        Py_ssize_t out_cap
        int8_t n
        uint8_t b
        Py_ssize_t k
        Py_ssize_t kk
        bytes out
        uint8_t* dst

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = src.shape[0]
    out_cap = expected_size if expected_size > 0 else max(srcsize * 2, 64)
    out = PyBytes_FromStringAndSize(NULL, out_cap)
    dst = <uint8_t*> PyBytes_AsString(out)

    while i < srcsize:
        n = <int8_t> src[i]
        i += 1
        if n >= 0:
            # Literal run of n+1 bytes.
            k = n + 1
            if i + k > srcsize:
                raise TiffError("packbits: literal run extends past EOF")
            if out_off + k > out_cap:
                raise TiffError(
                    f"packbits: output overflow (need >{out_cap} bytes; "
                    "pass a larger expected_size)"
                )
            memcpy(dst + out_off, &src[i], k)
            i += k
            out_off += k
        elif n == -128:
            # No-op marker.
            continue
        else:
            # Replicate next byte (1 - n) times.
            k = 1 - n
            if i >= srcsize:
                raise TiffError("packbits: replicate marker has no payload")
            if out_off + k > out_cap:
                raise TiffError(
                    f"packbits: output overflow (need >{out_cap} bytes)"
                )
            b = src[i]
            i += 1
            for kk in range(k):
                dst[out_off] = b
                out_off += 1
    return out[:out_off]


# ---------------------------------------------------------------------------
# LZW (TIFF compression code 5) — variable-width 9..12 bit, MSB-first.
# ---------------------------------------------------------------------------
#
# TIFF's LZW is the "old style" variant defined in TIFF 6.0 Section 13:
#   * Bits packed MSB-first within each byte (vs GIF's LSB-first)
#   * 9-bit codes initially; width grows to 10/11/12 as the dictionary fills
#   * Code 256 = clear-code → reset dictionary, drop back to 9-bit
#   * Code 257 = end-of-information
#   * Width grows when next-code-to-add equals 2^width - 1 (off-by-one
#     vs canonical LZW — TIFF historical quirk).
#
# Decoder is in 3rdparty/oc_tifflzw/oc_tifflzw.c (flat-tables, stack
# emit; see oc_giflzw for the design rationale).


def lzw_encode(data) -> bytes:
    """Encode bytes as a TIFF-flavor LZW strip / tile.

    Variable-width (9..12 bit) MSB-first codes with CLEAR / EOI
    markers, compatible with libtiff, tifffile, and any TIFF reader
    that handles ``Compression = 5``. Implemented in
    ``3rdparty/oc_tifflzw/`` alongside the matching decoder; both are
    opencodecs' own code.
    """
    cdef:
        const uint8_t[::1] src
        ssize_t srcsize
        ssize_t dstsize
        ssize_t written
        bytes out
        uint8_t* dst

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = src.shape[0]
    dstsize = <Py_ssize_t> oc_tifflzw_encode_bound(<size_t> srcsize)
    # encode_size can return very small values for zero-byte input;
    # bottom-out at 3 to keep the LZW_WRITE_DST sanity check happy
    # (needs at least CLEAR + EOI).
    if dstsize < 16:
        dstsize = 16
    out = PyBytes_FromStringAndSize(NULL, dstsize)
    dst = <uint8_t*> PyBytes_AsString(out)

    if srcsize == 0:
        with nogil:
            written = oc_tifflzw_encode(NULL, 0, dst, <size_t> dstsize)
    else:
        with nogil:
            written = oc_tifflzw_encode(&src[0], <size_t> srcsize,
                                        dst, <size_t> dstsize)
    if written < 0:
        raise RuntimeError(
            f"lzw_encode: encoder returned error {written}"
        )
    return out[:written]


def lzw_decode(data, expected_size: int = -1) -> bytes:
    """Decode a TIFF-flavor LZW strip / tile.

    Backed by the vendored ``oc_tifflzw`` C decoder
    (``3rdparty/oc_tifflzw/``) — flat prefix/suffix/first_byte tables,
    stack-based string emit. Measured ~2x faster than the previous
    pure-Cython per-string-malloc implementation and faster than
    imagecodecs.lzw_decode.

    ``expected_size`` is the exact uncompressed byte count (from the
    TIFF strip / tile size). Must be > 0 — pass it from the calling
    side; we don't have a sensible default because LZW doesn't carry
    the uncompressed size in-band.
    """
    cdef:
        const uint8_t[::1] src
        Py_ssize_t srcsize
        bytes out
        uint8_t* dst
        Py_ssize_t out_cap
        Py_ssize_t n_written

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = src.shape[0]

    if expected_size <= 0:
        # Best-effort: most TIFF strips compress 2-6x; 8x covers most
        # cases. Callers should pass expected_size for correctness.
        out_cap = max(srcsize * 8, 256)
    else:
        out_cap = expected_size

    out = PyBytes_FromStringAndSize(NULL, out_cap)
    dst = <uint8_t*> PyBytes_AsString(out)

    with nogil:
        n_written = oc_tifflzw_decode(
            &src[0], <size_t> srcsize,
            dst, <size_t> out_cap,
        )
    if n_written < 0:
        raise TiffError(
            f"oc_tifflzw_decode failed: rc={n_written} "
            f"(srcsize={srcsize}, expected_size={out_cap})"
        )
    return out[:n_written]


# ---------------------------------------------------------------------------
# Predictors — TIFF tag 317. Reverse the encoder-side delta.
# ---------------------------------------------------------------------------
#
# Predictor 1 = no predictor (identity).
# Predictor 2 = horizontal differencing: each sample (after the first
#   in a row) was stored as (sample - sample_to_left). Inverse is a
#   prefix-sum along the last axis (within each row, per channel).
# Predictor 3 = floating-point predictor (TIFF Tech Note 3): the float
#   bytes are byte-rearranged + horizontal-differenced. Reverse that.

ctypedef fused _uint_t:
    uint8_t
    uint16_t
    uint32_t


cdef void _undo_rows(_uint_t* p0, Py_ssize_t rows, Py_ssize_t cols,
                     Py_ssize_t samples, Py_ssize_t row_elems) noexcept nogil:
    """Undo horizontal differencing: a running sum along each row, per sample.

    ``p0`` points at ``rows`` rows ``row_elems`` elements apart, each of
    ``cols`` pixels of ``samples`` interleaved samples; every sample after
    the first pixel gets the one to its left added. TIFF predictor 2 is
    exactly this, and predictor 3's first step is this on the row's bytes.

    Written so that no compiler has to guess. With one sample the running
    sum lives in a 32-bit register whatever the sample width; only its
    low bits are stored, and a sum modulo 2**32 agrees with one modulo
    2**8 or 2**16 there, so the output is the same. A sum kept as
    uint16_t is truncated on every step of a loop that cannot run in
    parallel, which MSVC compiled 4x slower. That loop is unrolled by
    hand, because Apple Clang unrolls the 16-bit form but not the 32-bit
    one. Three and four samples keep sums of the sample's own width:
    GCC then adds a whole pixel at once (one paddb for four uint8
    samples) and stores it once, where 32-bit sums cost a store per
    sample and made a uint8 RGB read 2% slower. And the obvious
    ``p[c] += p[c - samples]`` rereads the value it just stored, 3 to 6x
    slower than a register on every compiler measured, so other sample
    counts walk one sample's chain at a time.

    One-sample uint16 predictor 2 over a 4096 x 4096 image as 256 x 256
    tiles, ms (lower is better):

      =========================  =====  ========  ====  ===========
      form                       MSVC   clang-cl  GCC   Apple Clang
      =========================  =====  ========  ====  ===========
      16-bit sum (0.4.0)         34.0   7.7       11.1  5.41
      32-bit sum                  8.1   7.8        9.3  6.56
      32-bit sum, unrolled x4     6.96  6.79      10.1  5.41
      =========================  =====  ========  ====  ===========
    """
    cdef Py_ssize_t r, c, k
    cdef uint32_t s0
    cdef _uint_t t0, t1, t2, t3
    cdef _uint_t* p
    for r in range(rows):
        p = p0 + r * row_elems
        if samples == 1:
            s0 = p[0]
            c = 1
            while c + 4 <= cols:
                s0 = s0 + p[c]; p[c] = <_uint_t> s0
                s0 = s0 + p[c + 1]; p[c + 1] = <_uint_t> s0
                s0 = s0 + p[c + 2]; p[c + 2] = <_uint_t> s0
                s0 = s0 + p[c + 3]; p[c + 3] = <_uint_t> s0
                c += 4
            while c < cols:
                s0 = s0 + p[c]; p[c] = <_uint_t> s0
                c += 1
        elif samples == 3:
            t0 = p[0]; t1 = p[1]; t2 = p[2]
            for c in range(1, cols):
                t0 = <_uint_t> (t0 + p[c * 3]); p[c * 3] = t0
                t1 = <_uint_t> (t1 + p[c * 3 + 1]); p[c * 3 + 1] = t1
                t2 = <_uint_t> (t2 + p[c * 3 + 2]); p[c * 3 + 2] = t2
        elif samples == 4:
            t0 = p[0]; t1 = p[1]; t2 = p[2]; t3 = p[3]
            for c in range(1, cols):
                t0 = <_uint_t> (t0 + p[c * 4]); p[c * 4] = t0
                t1 = <_uint_t> (t1 + p[c * 4 + 1]); p[c * 4 + 1] = t1
                t2 = <_uint_t> (t2 + p[c * 4 + 2]); p[c * 4 + 2] = t2
                t3 = <_uint_t> (t3 + p[c * 4 + 3]); p[c * 4 + 3] = t3
        else:
            # One chain per sample, each walked with its sum in a register.
            for k in range(samples):
                s0 = p[k]
                for c in range(1, cols):
                    s0 = s0 + p[c * samples + k]
                    p[c * samples + k] = <_uint_t> s0


def undo_horizontal_u8(uint8_t[:, :, ::1] arr not None):
    """In-place undo of predictor 2 on a (rows, cols, samples) uint8 array."""
    if arr.shape[0] and arr.shape[1] > 1:
        with nogil:
            _undo_rows(&arr[0, 0, 0], arr.shape[0], arr.shape[1], arr.shape[2],
                       arr.shape[1] * arr.shape[2])


def undo_horizontal_u16(uint16_t[:, :, ::1] arr not None):
    """In-place undo of predictor 2 on a (rows, cols, samples) uint16 array."""
    if arr.shape[0] and arr.shape[1] > 1:
        with nogil:
            _undo_rows(&arr[0, 0, 0], arr.shape[0], arr.shape[1], arr.shape[2],
                       arr.shape[1] * arr.shape[2])


def undo_horizontal_u32(uint32_t[:, :, ::1] arr not None):
    """In-place undo of predictor 2 on a (rows, cols, samples) uint32 array."""
    if arr.shape[0] and arr.shape[1] > 1:
        with nogil:
            _undo_rows(&arr[0, 0, 0], arr.shape[0], arr.shape[1], arr.shape[2],
                       arr.shape[1] * arr.shape[2])


def undo_floating_point(uint8_t[:, :, ::1] arr not None, int bytes_per_sample):
    """Undo TIFF predictor 3 in a writable contiguous byte view.

    Shape is (rows, columns, samples_per_pixel * bytes_per_sample).
    Shuffled byte planes are most-significant first regardless of file
    byte order. Reconstruct native-endian sample bytes in place.
    """
    cdef Py_ssize_t r
    cdef Py_ssize_t rows = arr.shape[0]
    cdef Py_ssize_t cols = arr.shape[1]
    cdef Py_ssize_t pixel_bytes = arr.shape[2]
    cdef Py_ssize_t bps = bytes_per_sample
    cdef Py_ssize_t spp, total_bytes, n_samples
    cdef Py_ssize_t lane, samp_i, src_idx, dst_idx
    cdef uint8_t* row_p
    cdef uint8_t* tmp
    cdef uint16_t endian_probe = 1
    cdef bint little_endian = (<uint8_t*>&endian_probe)[0] == 1

    if bps != 2 and bps != 4 and bps != 8:
        raise TiffError(
            f"predictor 3: bytes_per_sample must be 2/4/8, got {bps}"
        )
    if pixel_bytes == 0 or pixel_bytes % bps != 0:
        raise TiffError("predictor 3: last dimension must contain whole samples")
    if rows == 0 or cols == 0:
        return
    total_bytes = cols * pixel_bytes
    if arr.strides[1] != pixel_bytes or arr.strides[0] != total_bytes:
        raise TiffError("predictor 3: expected contiguous rows and columns")
    spp = pixel_bytes // bps
    n_samples = cols * spp
    tmp = <uint8_t*> PyMem_Malloc(total_bytes)
    if tmp == NULL:
        raise MemoryError()
    try:
        with nogil:
            for r in range(rows):
                row_p = &arr[r, 0, 0]
                # Each channel has its own recurrence across the byte planes.
                _undo_rows(row_p, 1, total_bytes // spp, spp, total_bytes)
                for samp_i in range(n_samples):
                    for lane in range(bps):
                        src_idx = lane * n_samples + samp_i
                        dst_idx = samp_i * bps + (bps - 1 - lane if little_endian else lane)
                        tmp[dst_idx] = row_p[src_idx]
                memcpy(row_p, tmp, total_bytes)
    finally:
        PyMem_Free(tmp)


def check_signature(data) -> bool:
    """Recognize TIFF (II/MM + 0x002A or 0x002B magic) from the first 4 bytes."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        try:
            data = bytes(data)
        except Exception:
            return False
    if len(data) < 4:
        return False
    head = bytes(data[:4])
    if head[:2] == b"II":
        return head[2:4] in (b"\x2a\x00", b"\x2b\x00")
    if head[:2] == b"MM":
        return head[2:4] in (b"\x00\x2a", b"\x00\x2b")
    return False



# ---------------------------------------------------------------------------
# Whole segments into the output: decode, un-predict and place, one call
# ---------------------------------------------------------------------------
#
# The per-segment path decoded a tile in one native call, undid the
# predictor in a second and copied it into the output with numpy, handing
# the GIL back and forth between each. With several reader threads those
# handoffs queue, and eight threads reading a tiled 4096 x 4096 TIFF got
# 0.69x the throughput of the same eight reading serially. Here a batch of
# segments goes through all three steps under one GIL release.

from cpython.pycapsule cimport PyCapsule_GetPointer
from libc.stddef cimport ptrdiff_t
from libc.stdlib cimport malloc, free
from libc.string cimport memset

cdef extern from "oc_decoder_vtable.h":
    ctypedef struct oc_decoder_vtable:
        void* (*create)() noexcept nogil
        void (*destroy)(void*) noexcept nogil
        ptrdiff_t (*decode)(void*, const uint8_t*, size_t, uint8_t*, size_t) noexcept nogil
    const char* OC_DECODER_VTABLE_CAPSULE

cdef enum:
    _SEG_NONE = 0
    _SEG_LZW = 1
    _SEG_PACKBITS = 2
    _SEG_EXTERNAL = 3

#: ``codec`` values for :func:`decode_segments_into`.
SEGMENT_NONE = _SEG_NONE
SEGMENT_LZW = _SEG_LZW
SEGMENT_PACKBITS = _SEG_PACKBITS
SEGMENT_EXTERNAL = _SEG_EXTERNAL   # ``decoder`` is a capsule from _deflate or _zstd


cdef Py_ssize_t _packbits_into(const uint8_t* src, Py_ssize_t n,
                               uint8_t* dst, Py_ssize_t cap) noexcept nogil:
    """PackBits, as packbits_decode does it; -1 on a malformed stream."""
    cdef Py_ssize_t i = 0, o = 0, k
    cdef int8_t c
    while i < n:
        c = <int8_t> src[i]
        i += 1
        if c >= 0:
            k = <Py_ssize_t> c + 1
            if i + k > n or o + k > cap:
                return -1
            memcpy(dst + o, src + i, <size_t> k)
            i += k
            o += k
        elif c != -128:
            k = 1 - <Py_ssize_t> c
            if i >= n or o + k > cap:
                return -1
            memset(dst + o, src[i], <size_t> k)
            i += 1
            o += k
    return o


cdef void _undo_horizontal(uint8_t* buf, Py_ssize_t rows, Py_ssize_t cols,
                           Py_ssize_t samples, Py_ssize_t itemsize,
                           Py_ssize_t row_bytes) noexcept nogil:
    """Predictor 2 on the first ``cols`` pixels of ``rows`` rows."""
    if itemsize == 1:
        _undo_rows(<uint8_t*> buf, rows, cols, samples, row_bytes)
    elif itemsize == 2:
        _undo_rows(<uint16_t*> buf, rows, cols, samples, row_bytes // 2)
    else:
        _undo_rows(<uint32_t*> buf, rows, cols, samples, row_bytes // 4)


cdef void _undo_float(uint8_t* buf, Py_ssize_t rows, Py_ssize_t cols,
                      Py_ssize_t samples, Py_ssize_t itemsize,
                      Py_ssize_t row_bytes, uint8_t* tmp) noexcept nogil:
    """Predictor 3 on ``rows`` whole rows; see undo_floating_point."""
    cdef Py_ssize_t r, lane, samp_i, n_samples = cols * samples
    cdef Py_ssize_t total = cols * samples * itemsize
    cdef uint8_t* row_p
    cdef uint16_t endian_probe = 1
    cdef bint little_endian = (<uint8_t*> &endian_probe)[0] == 1
    _undo_rows(buf, rows, cols * itemsize, samples, row_bytes)
    for r in range(rows):
        row_p = buf + r * row_bytes
        for samp_i in range(n_samples):
            for lane in range(itemsize):
                tmp[samp_i * itemsize
                    + (itemsize - 1 - lane if little_endian else lane)] = \
                    row_p[lane * n_samples + samp_i]
        memcpy(row_p, tmp, <size_t> total)


def decode_segments_into(segments, out, const Py_ssize_t[:, ::1] geometry, *,
                         int codec, decoder=None, Py_ssize_t segment_cols,
                         Py_ssize_t samples, Py_ssize_t itemsize,
                         int predictor):
    """Decode TIFF segments, undo their predictor and place them, one call.

    ``segments`` holds each segment's stored bytes (bytes or any buffer).
    ``out`` is the image as a C-contiguous 2D byte view, one row per image
    row. Row ``i`` of ``geometry`` describes segment ``i`` as
    ``(decoded_rows, crop_rows, crop_cols, dst_row, dst_col)``: the segment
    decodes to ``decoded_rows`` rows of ``segment_cols`` pixels of
    ``samples`` samples of ``itemsize`` bytes, and its top-left
    ``crop_rows`` x ``crop_cols`` pixels land at ``out`` pixel
    ``(dst_row, dst_col)``. ``predictor`` is TIFF's 1, 2 or 3; samples must
    already be in native byte order unless it is 3, whose byte planes are
    order-independent.

    Every segment runs under one GIL release. A segment that does not
    decode to exactly its expected size is left unwritten and its index
    returned, so the caller can redo it on the general path, which raises
    the precise error.
    """
    cdef:
        uint8_t[:, ::1] dst = out
        Py_ssize_t nseg = len(segments)
        Py_ssize_t s, r, expected, got, max_rows = 0
        Py_ssize_t pixel_bytes = samples * itemsize
        Py_ssize_t row_bytes = segment_cols * pixel_bytes
        Py_ssize_t dec_rows, crop_rows, crop_cols, dst_row, dst_col
        const uint8_t[::1] view
        const uint8_t** srcs = NULL
        Py_ssize_t* lens = NULL
        char* failed = NULL
        uint8_t* scratch = NULL
        uint8_t* tmp = NULL
        const uint8_t* data
        uint8_t* target
        bint direct
        oc_decoder_vtable* vt = NULL
        void* ctx = NULL
        list keep = []

    if geometry.shape[0] != nseg or geometry.shape[1] != 5:
        raise ValueError(f"geometry must be ({nseg}, 5), got "
                         f"({geometry.shape[0]}, {geometry.shape[1]})")
    if samples < 1 or segment_cols < 1:
        raise ValueError("samples and segment_cols must be positive")
    if predictor == 2 and itemsize not in (1, 2, 4):
        raise ValueError(f"predictor 2 takes 1, 2 or 4 byte samples, not {itemsize}")
    if predictor == 3 and itemsize not in (2, 4, 8):
        raise ValueError(f"predictor 3 takes 2, 4 or 8 byte samples, not {itemsize}")
    if predictor not in (1, 2, 3):
        raise ValueError(f"TIFF predictor {predictor} not supported")
    for s in range(nseg):
        dec_rows = geometry[s, 0]; crop_rows = geometry[s, 1]
        crop_cols = geometry[s, 2]; dst_row = geometry[s, 3]; dst_col = geometry[s, 4]
        if (crop_rows < 0 or crop_cols < 0 or crop_rows > dec_rows
                or crop_cols > segment_cols or dst_row < 0 or dst_col < 0
                or dst_row + crop_rows > dst.shape[0]
                or (dst_col + crop_cols) * pixel_bytes > dst.shape[1]):
            raise ValueError(f"segment {s} does not fit the output")
        if dec_rows > max_rows:
            max_rows = dec_rows
    if codec == _SEG_EXTERNAL:
        vt = <oc_decoder_vtable*> PyCapsule_GetPointer(decoder, OC_DECODER_VTABLE_CAPSULE)
    elif codec not in (_SEG_NONE, _SEG_LZW, _SEG_PACKBITS):
        raise ValueError(f"unknown segment codec {codec}")
    if nseg == 0:
        return []

    srcs = <const uint8_t**> malloc(nseg * sizeof(uint8_t*))
    lens = <Py_ssize_t*> malloc(nseg * sizeof(Py_ssize_t))
    failed = <char*> malloc(nseg)
    scratch = <uint8_t*> malloc(max(1, max_rows * row_bytes))
    tmp = <uint8_t*> malloc(max(1, row_bytes))
    try:
        if (srcs == NULL or lens == NULL or failed == NULL or scratch == NULL
                or tmp == NULL):
            raise MemoryError()
        for s in range(nseg):
            # The memoryview in ``keep`` holds each buffer's export for the
            # whole call, so its pointer stays valid without the GIL.
            keep.append(memoryview(segments[s]).cast("B"))
            view = keep[s]
            lens[s] = view.shape[0]
            srcs[s] = &view[0] if view.shape[0] else NULL
            failed[s] = 0
        with nogil:
            if vt != NULL:
                ctx = vt.create()
            for s in range(nseg):
                dec_rows = geometry[s, 0]; crop_rows = geometry[s, 1]
                crop_cols = geometry[s, 2]; dst_row = geometry[s, 3]
                dst_col = geometry[s, 4]
                expected = dec_rows * row_bytes
                if srcs[s] == NULL and expected:
                    failed[s] = 1
                    continue
                # A whole strip is whole output rows: decode straight into
                # them. Tile rows are strided in the output, so a tile goes
                # through scratch and is copied in.
                direct = (crop_rows == dec_rows and crop_cols == segment_cols
                          and dst_col == 0 and dst.shape[1] == row_bytes
                          and expected > 0)
                target = &dst[dst_row, 0] if direct else scratch
                if codec == _SEG_NONE:
                    if lens[s] != expected:
                        failed[s] = 1
                        continue
                    if predictor == 1 and not direct:
                        data = srcs[s]
                    else:
                        memcpy(target, srcs[s], <size_t> expected)
                        data = target
                else:
                    if codec == _SEG_LZW:
                        got = oc_tifflzw_decode(srcs[s], <size_t> lens[s],
                                                target, <size_t> expected)
                    elif codec == _SEG_PACKBITS:
                        got = _packbits_into(srcs[s], lens[s], target, expected)
                    elif ctx != NULL:
                        got = vt.decode(ctx, srcs[s], <size_t> lens[s],
                                        target, <size_t> expected)
                    else:
                        got = -1
                    if got != expected:
                        # A direct decode may have written part of these
                        # rows; the caller redoes the whole segment.
                        failed[s] = 1
                        continue
                    data = target
                if predictor == 2:
                    _undo_horizontal(target, crop_rows, crop_cols, samples,
                                     itemsize, row_bytes)
                elif predictor == 3:
                    _undo_float(target, crop_rows, segment_cols, samples,
                                itemsize, row_bytes, tmp)
                if direct:
                    continue
                for r in range(crop_rows):
                    memcpy(&dst[dst_row + r, dst_col * pixel_bytes],
                           data + r * row_bytes,
                           <size_t> (crop_cols * pixel_bytes))
            if vt != NULL and ctx != NULL:
                vt.destroy(ctx)
        return [s for s in range(nseg) if failed[s]]
    finally:
        free(srcs)
        free(lens)
        free(failed)
        free(scratch)
        free(tmp)
