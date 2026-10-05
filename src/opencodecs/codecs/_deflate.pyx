# opencodecs/codecs/_deflate.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native zlib / deflate codec — bytes-in / bytes-out compression.

Produces and consumes zlib-format streams (RFC 1950: deflate + 2-byte
header + 4-byte adler32) by default, bare DEFLATE streams (RFC 1951)
with ``raw=True``, and gzip members (RFC 1952) through
:func:`gzip_encode`. Matches imagecodecs's ``deflate_encode`` /
``deflate_decode`` (including ``raw=``) and ``gzip_encode``
bit-for-bit when linked to libdeflate, as imagecodecs is.

Three backends, picked at compile time by setup.py probes:

  1. ``libdeflate`` (https://github.com/ebiggers/libdeflate, MIT) —
     by far the fastest one-shot zlib encode/decode; what imagecodecs
     uses. Selected by ``-DOPENCODECS_HAVE_LIBDEFLATE``.
  2. ``zlib-ng-compat`` — drop-in zlib replacement with a 1.3-1.5x
     speedup over stdlib zlib. Selected automatically when its
     ``libz`` is on the linker path (no ifdef needed; it exports the
     same ``compress2`` / ``uncompress`` symbols).
  3. System ``zlib`` — the fallback.

Backends 2 and 3 share the .pyx path because they share the zlib
API; libdeflate has a different API and lives behind the macro.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING
from libc.stdint cimport uint8_t
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy
from libc.stddef cimport size_t

from zlib_h cimport (
    Z_OK, Z_DEFAULT_COMPRESSION,
    compress2, uncompress, compressBound,
    uLongf, uLong,
)


# ---------------------------------------------------------------------------
# libdeflate — conditional binding
# ---------------------------------------------------------------------------
#
# setup.py sets OPENCODECS_HAVE_LIBDEFLATE = 1 when it found
# <libdeflate.h> + libdeflate on the host. In that build the macro
# guards include <libdeflate.h>; the Cython-emitted calls go straight
# to the real library. Otherwise the macro substitutes inline static
# stubs so the .so still links; runtime branches on
# ``OPENCODECS_HAVE_LIBDEFLATE`` keep those stubs from ever being
# called.

cdef extern from *:
    """
    #ifndef OPENCODECS_HAVE_LIBDEFLATE
    #define OPENCODECS_HAVE_LIBDEFLATE 0
    #endif

    #if OPENCODECS_HAVE_LIBDEFLATE
      #include <libdeflate.h>
      /* libdeflate.h uses bare `struct` / `enum` tags. Add typedefs
         so Cython-generated code (which writes `libdeflate_compressor *`
         without the `struct` prefix) compiles. */
      typedef struct libdeflate_compressor libdeflate_compressor;
      typedef struct libdeflate_decompressor libdeflate_decompressor;
      typedef enum libdeflate_result libdeflate_result;
    #else
      /* Link-time stubs. Symbols never called at runtime because all
         libdeflate code-paths are gated on the macro above. */
      typedef struct libdeflate_compressor libdeflate_compressor;
      typedef struct libdeflate_decompressor libdeflate_decompressor;
      typedef int libdeflate_result;
      #define LIBDEFLATE_SUCCESS 0
      #define LIBDEFLATE_INSUFFICIENT_SPACE 3
      static inline libdeflate_compressor*
      libdeflate_alloc_compressor(int lvl) { (void)lvl; return 0; }
      static inline void
      libdeflate_free_compressor(libdeflate_compressor* c) { (void)c; }
      static inline libdeflate_decompressor*
      libdeflate_alloc_decompressor(void) { return 0; }
      static inline void
      libdeflate_free_decompressor(libdeflate_decompressor* d) { (void)d; }
      static inline size_t
      libdeflate_zlib_compress(libdeflate_compressor* c,
                               const void* a, size_t b,
                               void* d, size_t e) {
          (void)c; (void)a; (void)b; (void)d; (void)e; return 0;
      }
      static inline size_t
      libdeflate_zlib_compress_bound(libdeflate_compressor* c, size_t b) {
          (void)c; (void)b; return 0;
      }
      static inline libdeflate_result
      libdeflate_zlib_decompress(libdeflate_decompressor* d,
                                 const void* a, size_t b,
                                 void* o, size_t oa,
                                 size_t* actual) {
          (void)d; (void)a; (void)b; (void)o; (void)oa; (void)actual;
          return 1;
      }
      static inline size_t
      libdeflate_deflate_compress(libdeflate_compressor* c,
                                  const void* a, size_t b,
                                  void* d, size_t e) {
          (void)c; (void)a; (void)b; (void)d; (void)e; return 0;
      }
      static inline size_t
      libdeflate_deflate_compress_bound(libdeflate_compressor* c, size_t b) {
          (void)c; (void)b; return 0;
      }
      static inline size_t
      libdeflate_gzip_compress(libdeflate_compressor* c,
                               const void* a, size_t b,
                               void* d, size_t e) {
          (void)c; (void)a; (void)b; (void)d; (void)e; return 0;
      }
      static inline size_t
      libdeflate_gzip_compress_bound(libdeflate_compressor* c, size_t b) {
          (void)c; (void)b; return 0;
      }
      static inline libdeflate_result
      libdeflate_deflate_decompress(libdeflate_decompressor* d,
                                    const void* a, size_t b,
                                    void* o, size_t oa,
                                    size_t* actual) {
          (void)d; (void)a; (void)b; (void)o; (void)oa; (void)actual;
          return 1;
      }
      static inline libdeflate_result
      libdeflate_gzip_decompress_ex(libdeflate_decompressor* d,
                                    const void* a, size_t b,
                                    void* o, size_t oa,
                                    size_t* actual_in, size_t* actual_out) {
          (void)d; (void)a; (void)b; (void)o; (void)oa;
          (void)actual_in; (void)actual_out;
          return 1;
      }
    #endif

    /* One entry point per wrapper, so the encode and decode loops below
       stay single: 0 = zlib (RFC 1950), 1 = raw DEFLATE (RFC 1951),
       2 = gzip (RFC 1952, encode only). */
    static inline size_t
    oc_ld_compress(int fmt, libdeflate_compressor* c, const void* a,
                   size_t b, void* d, size_t e) {
        if (fmt == 1) return libdeflate_deflate_compress(c, a, b, d, e);
        if (fmt == 2) return libdeflate_gzip_compress(c, a, b, d, e);
        return libdeflate_zlib_compress(c, a, b, d, e);
    }
    static inline size_t
    oc_ld_compress_bound(int fmt, libdeflate_compressor* c, size_t b) {
        if (fmt == 1) return libdeflate_deflate_compress_bound(c, b);
        if (fmt == 2) return libdeflate_gzip_compress_bound(c, b);
        return libdeflate_zlib_compress_bound(c, b);
    }
    static inline libdeflate_result
    oc_ld_decompress(int fmt, libdeflate_decompressor* d, const void* a,
                     size_t b, void* o, size_t oa, size_t* actual) {
        if (fmt == 1)
            return libdeflate_deflate_decompress(d, a, b, o, oa, actual);
        return libdeflate_zlib_decompress(d, a, b, o, oa, actual);
    }
    """
    int OPENCODECS_HAVE_LIBDEFLATE
    int LIBDEFLATE_SUCCESS
    int LIBDEFLATE_INSUFFICIENT_SPACE
    ctypedef struct libdeflate_compressor:
        pass
    ctypedef struct libdeflate_decompressor:
        pass
    ctypedef int libdeflate_result
    libdeflate_compressor* libdeflate_alloc_compressor(int level) nogil
    void libdeflate_free_compressor(libdeflate_compressor* c) nogil
    libdeflate_decompressor* libdeflate_alloc_decompressor() nogil
    void libdeflate_free_decompressor(libdeflate_decompressor* d) nogil
    size_t libdeflate_zlib_compress(
        libdeflate_compressor* c,
        const void* indata, size_t in_nbytes,
        void* outdata, size_t out_nbytes_avail,
    ) nogil
    size_t libdeflate_zlib_compress_bound(
        libdeflate_compressor* c, size_t in_nbytes,
    ) nogil
    libdeflate_result libdeflate_zlib_decompress(
        libdeflate_decompressor* d,
        const void* indata, size_t in_nbytes,
        void* outdata, size_t out_nbytes_avail,
        size_t* actual_out_nbytes_ret,
    ) nogil
    size_t oc_ld_compress(
        int fmt, libdeflate_compressor* c,
        const void* indata, size_t in_nbytes,
        void* outdata, size_t out_nbytes_avail,
    ) nogil
    size_t oc_ld_compress_bound(
        int fmt, libdeflate_compressor* c, size_t in_nbytes,
    ) nogil
    libdeflate_result oc_ld_decompress(
        int fmt, libdeflate_decompressor* d,
        const void* indata, size_t in_nbytes,
        void* outdata, size_t out_nbytes_avail,
        size_t* actual_out_nbytes_ret,
    ) nogil
    libdeflate_result libdeflate_gzip_decompress_ex(
        libdeflate_decompressor* d,
        const void* indata, size_t in_nbytes,
        void* outdata, size_t out_nbytes_avail,
        size_t* actual_in_nbytes_ret,
        size_t* actual_out_nbytes_ret,
    ) nogil


import zlib as _stdlib_zlib

cdef enum:
    _FMT_ZLIB = 0
    _FMT_RAW = 1
    _FMT_GZIP = 2


def backend() -> str:
    """Return the deflate backend this build was linked against.

    One of ``"libdeflate"`` or ``"zlib"`` (which may itself be linked
    to zlib-ng-compat — check via ``otool`` / ``ldd``). Useful for
    benchmarks and CI sanity-checks."""
    return "libdeflate" if OPENCODECS_HAVE_LIBDEFLATE else "zlib"


class ZlibError(RuntimeError):
    """Raised on zlib encode/decode failures."""


def encode(data, *, level: int | None = None, bint raw=False) -> bytes:
    """Encode bytes-like input as a zlib stream (RFC 1950).

    ``raw=True`` writes a bare DEFLATE stream (RFC 1951) with no zlib
    header or Adler-32 trailer, as ZIP members and ``wbits=-15`` expect;
    this is imagecodecs's ``deflate_encode(raw=True)``.
    """
    return _encode(data, level, _FMT_RAW if raw else _FMT_ZLIB)


def gzip_encode(data, *, level: int | None = None) -> bytes:
    """Encode bytes-like input as one gzip member (RFC 1952).

    The header is fixed, so the output depends only on the input and the
    level, on every platform: MTIME 0, no name, OS 255 ("unknown", the
    value libdeflate, CPython 3.13+ and imagecodecs write) and XFL 4 for
    the fastest levels (below 2) or 2 for the slowest (8 and up), as RFC
    1952 section 2.3.1 defines it. With libdeflate linked the member is
    byte-identical to imagecodecs's ``gzip_encode``.
    """
    return _encode(data, level, _FMT_GZIP)


def _fallback_encode(data, int lvl, int fmt) -> bytes:
    """The zlib-library path for a raw DEFLATE stream or a gzip member.

    Used only when libdeflate is not linked. Python's zlib is the same
    zlib this module links in that case. The gzip header is assembled
    here rather than by zlib, because zlib writes its own OS_CODE (3 on
    Unix, 10 on Windows, 19 on macOS), which would make the bytes depend
    on the platform.
    """
    cdef bytes body
    view = memoryview(data).cast('B')
    if lvl > 9:
        lvl = 9
    compressor = _stdlib_zlib.compressobj(lvl, _stdlib_zlib.DEFLATED, -15)
    body = compressor.compress(view) + compressor.flush()
    if fmt == _FMT_RAW:
        return body
    xfl = 4 if lvl < 2 else (2 if lvl >= 8 else 0)
    header = bytes((0x1F, 0x8B, 8, 0, 0, 0, 0, 0, xfl, 255))
    trailer = ((_stdlib_zlib.crc32(view) & 0xFFFFFFFF).to_bytes(4, 'little')
               + (len(view) & 0xFFFFFFFF).to_bytes(4, 'little'))
    return header + body + trailer


# DEFLATE cannot expand its input more than about 1032 to 1 (a 258-byte
# match coded in two bits). A size above this many bytes per compressed
# byte cannot be the output of the input in hand, so it is never
# allocated: a corrupt trailer claiming 4 GB for a 30-byte file is
# refused rather than honored.
cdef size_t _MAX_INFLATE_RATIO = 1040


cdef inline size_t _inflate_bound(size_t in_nbytes) noexcept nogil:
    if in_nbytes > (<size_t> -1 - 65536) // _MAX_INFLATE_RATIO:
        return <size_t> -1
    return in_nbytes * _MAX_INFLATE_RATIO + 65536


def gzip_decode(data) -> bytes:
    """Inflate every member of a gzip stream (RFC 1952) with libdeflate.

    Members are decoded in order and their outputs joined. Zero bytes
    after a member are skipped, as the stdlib ``gzip.decompress``
    skips them; anything else after a member must be another member.
    Each member's CRC-32 and ISIZE are checked by libdeflate.

    The output size comes from the trailer. ISIZE is the LAST member's
    length mod 2**32, so for the common single-member file under 4 GiB
    the result is allocated once at its exact size and written in place.
    Anything else (concatenated members, a member of 4 GiB or more whose
    ISIZE has wrapped) takes a second path that grows a buffer and
    retries the member that did not fit.

    Raises ZlibError for anything libdeflate rejects. The caller is
    expected to hand such input to the stdlib, which then raises its
    own, more specific error, or decodes it.
    """
    cdef:
        const uint8_t[::1] src
        size_t n, isize, pos, done, cap, avail, bound, used_in, used_out
        size_t members
        libdeflate_decompressor* decompressor = NULL
        libdeflate_result rc
        bytes first
        uint8_t* buf = NULL
        uint8_t* grown
        uint8_t dummy = 0
        uint8_t* dst

    if not OPENCODECS_HAVE_LIBDEFLATE:
        raise ZlibError("gzip decode: libdeflate is not linked")
    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    n = <size_t> src.shape[0]
    if n == 0:
        return b""
    if n < 18:
        # Shorter than an empty member's header and trailer.
        raise ZlibError("gzip decode: truncated stream")

    isize = (<size_t> src[n - 4] | (<size_t> src[n - 3] << 8)
             | (<size_t> src[n - 2] << 16) | (<size_t> src[n - 1] << 24))
    bound = _inflate_bound(n)

    with nogil:
        decompressor = libdeflate_alloc_decompressor()
    if decompressor == NULL:
        raise MemoryError("libdeflate_alloc_decompressor returned NULL")
    try:
        # One member, sized by its own trailer: decode in place.
        if isize <= bound:
            first = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> isize)
            dst = <uint8_t*> PyBytes_AS_STRING(first) if isize else &dummy
            with nogil:
                rc = libdeflate_gzip_decompress_ex(
                    decompressor, <const void*> &src[0], n,
                    <void*> dst, isize, &used_in, &used_out)
            if rc == LIBDEFLATE_SUCCESS and used_out == isize:
                pos = used_in
                while pos < n and src[pos] == 0:
                    pos += 1
                if pos == n:
                    return first
            first = None

        # Several members, or a member whose ISIZE wrapped.
        cap = isize if isize > 65536 else 65536
        if cap > bound:
            cap = bound
        buf = <uint8_t*> malloc(cap)
        if buf == NULL:
            raise MemoryError()
        pos = 0
        done = 0
        members = 0
        while True:
            if members:
                while pos < n and src[pos] == 0:
                    pos += 1
                if pos == n:
                    break
            avail = cap - done
            with nogil:
                rc = libdeflate_gzip_decompress_ex(
                    decompressor, <const void*> &src[pos], n - pos,
                    <void*> (buf + done), avail, &used_in, &used_out)
            if rc == LIBDEFLATE_INSUFFICIENT_SPACE:
                # avail can be 0: the previous member filled the buffer.
                bound = _inflate_bound(n - pos)
                if avail >= bound:
                    raise ZlibError("gzip decode: member does not end")
                avail = avail * 2 if avail > 32768 else 65536
                if avail > bound:
                    avail = bound
                grown = <uint8_t*> realloc(buf, done + avail)
                if grown == NULL:
                    raise MemoryError()
                buf = grown
                cap = done + avail
                continue
            if rc != LIBDEFLATE_SUCCESS:
                raise ZlibError(f"gzip decode: libdeflate rc={rc}")
            pos += used_in
            done += used_out
            members += 1
            if pos == n:
                break
        return PyBytes_FromStringAndSize(<char*> buf, <Py_ssize_t> done)
    finally:
        free(buf)
        with nogil:
            libdeflate_free_decompressor(decompressor)


def _fallback_raw_decode(data) -> bytes:
    """Inflate one bare DEFLATE stream with the zlib library."""
    inflater = _stdlib_zlib.decompressobj(-15)
    try:
        result = inflater.decompress(data)
    except _stdlib_zlib.error as exc:
        raise ZlibError(f"raw deflate decode failed: {exc}") from None
    if not inflater.eof:
        raise ZlibError("raw deflate decode failed: truncated stream")
    return result


cdef object _encode(object data, object level, int fmt):
    cdef:
        const uint8_t[::1] src
        const uint8_t[::1] dst_mv    # memoryview cast for output bytes
        size_t srcsize_s, dstcap_s, written
        uLong srcsize_z
        uLongf dstsize_z
        int rc, lvl
        bytes out
        const uint8_t* src_ptr = NULL
        libdeflate_compressor* compressor = NULL

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize_s = <size_t> src.shape[0]
    if srcsize_s > 0:
        src_ptr = <const uint8_t*> &src[0]

    if level is None:
        lvl = 6   # libdeflate's default; matches zlib's Z_DEFAULT_COMPRESSION
    else:
        lvl = int(level)
        if lvl < 0:
            lvl = 0
        if lvl > 12 and OPENCODECS_HAVE_LIBDEFLATE:
            lvl = 12   # libdeflate accepts 0..12
        elif lvl > 9 and not OPENCODECS_HAVE_LIBDEFLATE:
            lvl = 9

    # libdeflate path — preferred when linked.
    if OPENCODECS_HAVE_LIBDEFLATE:
        with nogil:
            compressor = libdeflate_alloc_compressor(lvl)
        if compressor == NULL:
            raise ZlibError(
                f"libdeflate_alloc_compressor returned NULL "
                f"(invalid level {lvl}?)"
            )
        try:
            dstcap_s = oc_ld_compress_bound(fmt, compressor, srcsize_s)
            out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dstcap_s)
            # memoryview-cast + slice pattern (see _zstd.encode for the
            # empirical rationale — faster than PyBytes_AsString +
            # _PyBytes_Resize for multi-MB blobs).
            dst_mv = out
            with nogil:
                written = oc_ld_compress(
                    fmt, compressor, src_ptr, srcsize_s,
                    <void*> &dst_mv[0], dstcap_s,
                )
            if written == 0:
                raise ZlibError("libdeflate compress returned 0")
            del dst_mv
            return out[:written]
        finally:
            with nogil:
                libdeflate_free_compressor(compressor)

    # zlib (system / zlib-ng-compat) fallback.
    if fmt != _FMT_ZLIB:
        return _fallback_encode(src, lvl, fmt)
    srcsize_z = <uLong> srcsize_s
    if level is None:
        lvl = Z_DEFAULT_COMPRESSION
    dstsize_z = compressBound(srcsize_z)
    out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dstsize_z)
    dst_mv = out
    with nogil:
        rc = compress2(
            <uint8_t*> &dst_mv[0], &dstsize_z, src_ptr, srcsize_z, lvl,
        )
    if rc != Z_OK:
        raise ZlibError(f"compress2 failed: {rc}")
    del dst_mv
    return out[:dstsize_z]


def decode(data, *, out=None, bint raw=False):
    """Decode a zlib stream (RFC 1950), or with ``raw=True`` a bare
    DEFLATE stream (RFC 1951), as imagecodecs's ``deflate_decode`` does.

    Parameters
    ----------
    out : int | bytearray | memoryview | None, optional
        See ``_zstd.decode`` for the full ``out=`` contract.
    raw : bool
        The input is a bare DEFLATE stream with no zlib wrapper.
    """
    cdef int fmt = _FMT_RAW if raw else _FMT_ZLIB
    cdef:
        const uint8_t[::1] src
        uint8_t[::1] out_view             # writable view of caller buffer
        size_t srcsize_s, dstcap_s, written
        uLong srcsize_z
        uLongf dstcap_z, dstsize_z
        int rc
        libdeflate_decompressor* decompressor = NULL
        libdeflate_result ld_rc
        uint8_t* buf = NULL
        bint can_grow
        bytes out_bytes

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize_s = <size_t> src.shape[0]
    if srcsize_s == 0:
        if out is None or isinstance(out, int):
            return b""
        return out[:0]

    # ----- caller-supplied writable buffer (zero-alloc path) -----
    if out is not None and not isinstance(out, int):
        try:
            out_view = out
        except (TypeError, ValueError, BufferError) as e:
            raise TypeError(
                f"deflate decode: out= must be int or writable buffer, "
                f"got {type(out).__name__}"
            ) from e
        dstcap_s = <size_t> out_view.shape[0]
        if OPENCODECS_HAVE_LIBDEFLATE:
            with nogil:
                decompressor = libdeflate_alloc_decompressor()
            if decompressor == NULL:
                raise MemoryError("libdeflate_alloc_decompressor returned NULL")
            try:
                with nogil:
                    ld_rc = oc_ld_decompress(
                        fmt, decompressor,
                        <const void*> &src[0], srcsize_s,
                        <void*> &out_view[0], dstcap_s, &written,
                    )
                if ld_rc == LIBDEFLATE_INSUFFICIENT_SPACE:
                    raise ZlibError(
                        "deflate decode: out= buffer too small")
                if ld_rc != LIBDEFLATE_SUCCESS:
                    raise ZlibError(
                        f"libdeflate decompress failed: rc={ld_rc}")
                del out_view
                return out[:written]
            finally:
                with nogil:
                    libdeflate_free_decompressor(decompressor)
        # zlib fallback path with out= buffer
        if fmt == _FMT_RAW:
            result = <bytes> _fallback_raw_decode(src)
            if len(result) > <Py_ssize_t> dstcap_s:
                raise ZlibError("deflate decode: out= buffer too small")
            if len(result):
                memcpy(&out_view[0], <const char*> result, len(result))
            del out_view
            return out[:len(result)]
        srcsize_z = <uLong> srcsize_s
        dstsize_z = <uLongf> dstcap_s
        with nogil:
            rc = uncompress(&out_view[0], &dstsize_z, &src[0], srcsize_z)
        if rc == -5:
            raise ZlibError("deflate decode: out= buffer too small")
        if rc != Z_OK:
            raise ZlibError(f"uncompress failed: {rc}")
        del out_view
        return out[:dstsize_z]

    # ----- pre-sized bytes (out=int) or grow-loop (out=None) -----
    if isinstance(out, int):
        if out < 0:
            raise ValueError("deflate decode: out=int(N) requires N >= 0")
        dstcap_s = <size_t> out
        can_grow = False
    else:
        dstcap_s = max(<size_t> 4 * srcsize_s, <size_t> 65536)
        can_grow = True

    if OPENCODECS_HAVE_LIBDEFLATE:
        with nogil:
            decompressor = libdeflate_alloc_decompressor()
        if decompressor == NULL:
            raise MemoryError("libdeflate_alloc_decompressor returned NULL")
        try:
            while True:
                buf = <uint8_t*> realloc(buf, dstcap_s)
                if buf == NULL:
                    raise MemoryError()
                with nogil:
                    ld_rc = oc_ld_decompress(
                        fmt, decompressor,
                        <const void*> &src[0], srcsize_s,
                        <void*> buf, dstcap_s, &written,
                    )
                if ld_rc == LIBDEFLATE_SUCCESS:
                    try:
                        return PyBytes_FromStringAndSize(
                            <char*> buf, <Py_ssize_t> written,
                        )
                    finally:
                        free(buf)
                if ld_rc == LIBDEFLATE_INSUFFICIENT_SPACE and can_grow \
                        and dstcap_s < <size_t>(1 << 30):
                    dstcap_s *= 2
                    continue
                free(buf)
                if not can_grow and ld_rc == LIBDEFLATE_INSUFFICIENT_SPACE:
                    raise ZlibError(
                        "deflate decode: out= int hint too small")
                raise ZlibError(
                    f"libdeflate decompress failed: rc={ld_rc}"
                )
        finally:
            with nogil:
                libdeflate_free_decompressor(decompressor)

    # zlib fallback.
    if fmt == _FMT_RAW:
        result = _fallback_raw_decode(src)
        if not can_grow and len(result) > <Py_ssize_t> dstcap_s:
            raise ZlibError("deflate decode: out= int hint too small")
        return result
    srcsize_z = <uLong> srcsize_s
    dstcap_z = <uLong> dstcap_s
    while True:
        buf = <uint8_t*> realloc(buf, dstcap_z)
        if buf == NULL:
            raise MemoryError()
        dstsize_z = dstcap_z
        with nogil:
            rc = uncompress(buf, &dstsize_z, &src[0], srcsize_z)
        if rc == Z_OK:
            try:
                out_bytes = PyBytes_FromStringAndSize(
                    <char*> buf, <Py_ssize_t> dstsize_z,
                )
                return out_bytes
            finally:
                free(buf)
        # Z_BUF_ERROR (-5) → grow + retry. Cap at 1 GiB.
        if rc == -5 and can_grow and dstcap_z < <uLong>(1 << 30):
            dstcap_z *= 2
            continue
        free(buf)
        if not can_grow and rc == -5:
            raise ZlibError("deflate decode: out= int hint too small")
        raise ZlibError(f"uncompress failed: {rc}")


def check_signature(data) -> bool:
    """True if `data` looks like a zlib stream (CMF byte 0x78 typical)."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:2])
    else:
        try:
            head = bytes(data)[:2]
        except Exception:
            return False
    if len(head) < 2:
        return False
    # zlib: CMF (0x78 typical) + FLG; (CMF*256 + FLG) % 31 == 0.
    return (head[0] & 0x0F) == 0x08 and ((head[0] * 256 + head[1]) % 31 == 0)


# ---------------------------------------------------------------------------
# C-level decoder for other extensions (see oc_decoder_vtable.h)
# ---------------------------------------------------------------------------

from cpython.pycapsule cimport PyCapsule_New
from libc.stddef cimport ptrdiff_t

cdef extern from "oc_decoder_vtable.h":
    ctypedef struct oc_decoder_vtable:
        void* (*create)() noexcept nogil
        void (*destroy)(void*) noexcept nogil
        ptrdiff_t (*decode)(void*, const uint8_t*, size_t, uint8_t*, size_t) noexcept nogil
    const char* OC_DECODER_VTABLE_CAPSULE


cdef void* _vt_create() noexcept nogil:
    # zlib needs no state; any non-NULL value means "ready".
    if OPENCODECS_HAVE_LIBDEFLATE:
        return <void*> libdeflate_alloc_decompressor()
    return <void*> 1


cdef void _vt_destroy(void* ctx) noexcept nogil:
    if OPENCODECS_HAVE_LIBDEFLATE and ctx != NULL:
        libdeflate_free_decompressor(<libdeflate_decompressor*> ctx)


cdef ptrdiff_t _vt_decode(void* ctx, const uint8_t* src, size_t n,
                          uint8_t* dst, size_t cap) noexcept nogil:
    """A zlib stream, as TIFF deflate (compression 8 and 32946) stores it."""
    cdef size_t written = 0
    cdef uLongf dstsize
    cdef int rc
    if ctx == NULL:
        return -1
    if OPENCODECS_HAVE_LIBDEFLATE:
        if libdeflate_zlib_decompress(<libdeflate_decompressor*> ctx,
                                      <const void*> src, n, <void*> dst, cap,
                                      &written) != LIBDEFLATE_SUCCESS:
            return -1
        return <ptrdiff_t> written
    dstsize = <uLongf> cap
    rc = uncompress(dst, &dstsize, src, <uLong> n)
    if rc != Z_OK:
        return -1
    return <ptrdiff_t> dstsize


cdef oc_decoder_vtable _VTABLE
_VTABLE.create = _vt_create
_VTABLE.destroy = _vt_destroy
_VTABLE.decode = _vt_decode


def decoder_capsule():
    """This module's decompressor as a C table, for a nogil caller in C."""
    return PyCapsule_New(<void*> &_VTABLE, OC_DECODER_VTABLE_CAPSULE, NULL)
