# opencodecs/codecs/_isal.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""ISA-L (Intel Storage Acceleration Library) zlib codec.

ISA-L's ``igzip`` deflate engine is the fastest deflate / inflate
implementation on x86_64 — typically 1.5-3x faster than libdeflate
(which is itself ~2x faster than zlib). Available via
``apt install libisal-dev`` on Linux, no current Mac build (ISA-L
is x86_64-only). When this extension fails to build (no libisal on
the host) the DeflateCodec falls back to libdeflate / zlib silently.

Wire format is the same zlib stream RFC 1950 produces — fully
interoperable with anything that reads zlib / gzip / deflate.
Compression levels are 0-3 (ISA-L's native range); the wrapper
clamps caller-passed levels to that range.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdint cimport uint8_t


cdef extern from *:
    """
    #include <stddef.h>
    #include <stdint.h>
    extern ssize_t opencodecs_isal_zlib_encode(
        const uint8_t*, size_t, uint8_t*, size_t, int);
    extern ssize_t opencodecs_isal_zlib_decode(
        const uint8_t*, size_t, uint8_t*, size_t);
    """
    Py_ssize_t opencodecs_isal_zlib_encode(
        const uint8_t* src, size_t srcsize,
        uint8_t* dst, size_t dstcap, int level) nogil
    Py_ssize_t opencodecs_isal_zlib_decode(
        const uint8_t* src, size_t srcsize,
        uint8_t* dst, size_t dstcap) nogil


class IsalError(RuntimeError):
    """Raised on ISA-L encode/decode failures."""


def version() -> str:
    """ISA-L's zlib codec is available."""
    return "isa-l"


def encode(data, *, level: int | None = None) -> bytes:
    """Encode ``data`` as a zlib-format byte stream.

    ``level`` is clamped to 0..3 (ISA-L's native level range). Level
    1 is the default — same speed/ratio sweet spot ISA-L's own
    ``igzip`` CLI uses.
    """
    cdef:
        const uint8_t[::1] src
        Py_ssize_t srcsize
        Py_ssize_t cap
        bytes out
        unsigned char* out_ptr
        Py_ssize_t written
        int lvl

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = src.shape[0]
    # zlib worst-case bound: srcsize + 5 bytes per 16 KiB block + 6
    # (zlib header + adler32). ISA-L can exceed zlib's "compressed_bound"
    # because it writes literals more conservatively at low levels,
    # so add ~10% slack on top.
    cap = srcsize + (srcsize // 8) + 64
    out = PyBytes_FromStringAndSize(NULL, cap)
    out_ptr = <unsigned char*> PyBytes_AsString(out)

    lvl = 1 if level is None else int(level)
    if lvl < 0: lvl = 0
    if lvl > 3: lvl = 3

    cdef const unsigned char* src_ptr = NULL
    if srcsize > 0:
        src_ptr = &src[0]
    with nogil:
        written = opencodecs_isal_zlib_encode(
            src_ptr, <size_t> srcsize, out_ptr, <size_t> cap, lvl,
        )
    if written < 0:
        raise IsalError(f"isal_deflate_stateless returned {written}")
    return out[:written]


def decode(data, *, out=None):
    """Decode a zlib-format byte stream.

    ``out`` is an optional preallocated writable buffer (bytes /
    bytearray / memoryview). If omitted, a heuristic upper-bound
    buffer is allocated and grown on overflow — same retry pattern
    as ``opencodecs.codecs._deflate.decode``.
    """
    cdef:
        const uint8_t[::1] src
        uint8_t[::1] out_view
        Py_ssize_t srcsize
        Py_ssize_t cap
        bytes out_bytes
        unsigned char* out_ptr
        Py_ssize_t written

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = src.shape[0]
    if srcsize == 0:
        return b"" if out is None else out[:0]

    # ----- caller-supplied buffer (zero-alloc path) -----
    if out is not None and not isinstance(out, int):
        try:
            out_view = out
        except (TypeError, ValueError, BufferError) as e:
            raise TypeError(
                f"isal decode: out= must be int or writable buffer, "
                f"got {type(out).__name__}"
            ) from e
        out_ptr = &out_view[0]
        with nogil:
            written = opencodecs_isal_zlib_decode(
                &src[0], <size_t> srcsize, out_ptr,
                <size_t> out_view.shape[0],
            )
        if written < 0:
            raise IsalError(f"isal_inflate_stateless returned {written}")
        del out_view
        return out[:written]

    # ----- fresh allocation path with retry on overflow -----
    cap = srcsize * 4 if not isinstance(out, int) else int(out)
    if cap < 1024:
        cap = 1024
    for _ in range(8):
        out_bytes = PyBytes_FromStringAndSize(NULL, cap)
        out_ptr = <unsigned char*> PyBytes_AsString(out_bytes)
        with nogil:
            written = opencodecs_isal_zlib_decode(
                &src[0], <size_t> srcsize, out_ptr, <size_t> cap,
            )
        if written >= 0:
            return out_bytes[:written]
        # ISAL_OUT_OVERFLOW == -2; retry with bigger buffer. Other
        # negative codes are hard errors.
        if written != -2:
            raise IsalError(f"isal_inflate_stateless returned {written}")
        cap *= 2
    raise IsalError(
        f"isal decode: output exceeded {cap} bytes after 8 doublings"
    )


def check_signature(data) -> bool:
    """zlib stream magic: first byte's low nibble = 8 (deflate), high
    nibble = window-size shift. Same check the _deflate module uses."""
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
    return (head[0] & 0x0F) == 0x08 and ((head[0] << 8) | head[1]) % 31 == 0
