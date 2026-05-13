# opencodecs/codecs/_deflate.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native zlib / deflate codec — bytes-in / bytes-out compression.

Produces and consumes zlib-format streams (deflate + 2-byte header +
4-byte adler32). Matches imagecodecs's ``zlib_encode`` /
``zlib_decode`` bit-for-bit on encode-with-fixed-level workloads.

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

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdint cimport uint8_t
from libc.stdlib cimport realloc, free
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
    #endif
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


def backend() -> str:
    """Return the deflate backend this build was linked against.

    One of ``"libdeflate"`` or ``"zlib"`` (which may itself be linked
    to zlib-ng-compat — check via ``otool`` / ``ldd``). Useful for
    benchmarks and CI sanity-checks."""
    return "libdeflate" if OPENCODECS_HAVE_LIBDEFLATE else "zlib"


class ZlibError(RuntimeError):
    """Raised on zlib encode/decode failures."""


def encode(data, *, level: int | None = None) -> bytes:
    """Encode bytes-like input as a zlib stream."""
    cdef:
        const uint8_t[::1] src
        size_t srcsize_s, dstcap_s, written
        uLong srcsize_z
        uLongf dstsize_z
        int rc, lvl
        bytes out
        const uint8_t* src_ptr = NULL
        uint8_t* dst_ptr
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
            dstcap_s = libdeflate_zlib_compress_bound(compressor, srcsize_s)
            out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dstcap_s)
            dst_ptr = <uint8_t*> PyBytes_AsString(out)
            with nogil:
                written = libdeflate_zlib_compress(
                    compressor, src_ptr, srcsize_s, dst_ptr, dstcap_s,
                )
            if written == 0:
                raise ZlibError("libdeflate_zlib_compress returned 0")
            return out[:written]
        finally:
            with nogil:
                libdeflate_free_compressor(compressor)

    # zlib (system / zlib-ng-compat) fallback.
    srcsize_z = <uLong> srcsize_s
    if level is None:
        lvl = Z_DEFAULT_COMPRESSION
    dstsize_z = compressBound(srcsize_z)
    out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> dstsize_z)
    dst_ptr = <uint8_t*> PyBytes_AsString(out)
    with nogil:
        rc = compress2(dst_ptr, &dstsize_z, src_ptr, srcsize_z, lvl)
    if rc != Z_OK:
        raise ZlibError(f"compress2 failed: {rc}")
    return out[:dstsize_z]


def decode(data) -> bytes:
    """Decode a zlib stream to bytes."""
    cdef:
        const uint8_t[::1] src
        size_t srcsize_s, dstcap_s, written
        uLong srcsize_z
        uLongf dstcap_z, dstsize_z
        int rc
        libdeflate_decompressor* decompressor = NULL
        libdeflate_result ld_rc
        uint8_t* buf = NULL
        bytes out

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize_s = <size_t> src.shape[0]
    if srcsize_s == 0:
        return b""

    if OPENCODECS_HAVE_LIBDEFLATE:
        with nogil:
            decompressor = libdeflate_alloc_decompressor()
        if decompressor == NULL:
            raise MemoryError("libdeflate_alloc_decompressor returned NULL")
        try:
            # Start at 4x source like the zlib path; grow on
            # INSUFFICIENT_SPACE. Cap at 1 GiB so a malformed stream
            # can't loop forever.
            dstcap_s = max(<size_t> 4 * srcsize_s, <size_t> 65536)
            while True:
                buf = <uint8_t*> realloc(buf, dstcap_s)
                if buf == NULL:
                    raise MemoryError()
                with nogil:
                    ld_rc = libdeflate_zlib_decompress(
                        decompressor,
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
                if ld_rc == LIBDEFLATE_INSUFFICIENT_SPACE and \
                        dstcap_s < <size_t>(1 << 30):
                    dstcap_s *= 2
                    continue
                free(buf)
                raise ZlibError(
                    f"libdeflate_zlib_decompress failed: rc={ld_rc}"
                )
        finally:
            with nogil:
                libdeflate_free_decompressor(decompressor)

    # zlib fallback.
    srcsize_z = <uLong> srcsize_s
    dstcap_z = max(<uLong> 4 * srcsize_z, <uLong> 65536)
    while True:
        buf = <uint8_t*> realloc(buf, dstcap_z)
        if buf == NULL:
            raise MemoryError()
        dstsize_z = dstcap_z
        with nogil:
            rc = uncompress(buf, &dstsize_z, &src[0], srcsize_z)
        if rc == Z_OK:
            try:
                out = PyBytes_FromStringAndSize(
                    <char*> buf, <Py_ssize_t> dstsize_z,
                )
                return out
            finally:
                free(buf)
        # Z_BUF_ERROR (-5) → grow + retry. Cap at 1 GiB.
        if rc == -5 and dstcap_z < <uLong>(1 << 30):
            dstcap_z *= 2
            continue
        free(buf)
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
