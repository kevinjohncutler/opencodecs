# opencodecs/codecs/_brunsli.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native Brunsli — lossless JPEG transcoder (~15-25% smaller).

Brunsli (Google, 2019) repacks an existing JPEG bitstream into a
smaller container while preserving every DCT coefficient bit-exactly.
Decoding reproduces the original JPEG byte-for-byte (the same JPEG
bytestream that went in). This makes Brunsli ideal for "JPEG-on-disk"
storage savings without any visual quality change vs. the source JPEG.

This module exposes a pure bytes-to-bytes API::

    brunsli_bytes = encode_jpeg(jpeg_bytes)   # ~80% of original size
    jpeg_bytes    = decode_jpeg(brunsli_bytes)  # byte-identical recovery

The higher-level :class:`BrunsliCodec` adapter chains through the
opencodecs JPEG codec so callers can also pass / receive ndarrays.

Brunsli files start with a 4-byte signature ``\\x0A\\x04\\x42\\xD2`` (the
"Brunsli marker"). ``check_signature`` matches this prefix.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy

from brunsli cimport EncodeBrunsli, DecodeBrunsli


# Growable sink — a callback-driven byte buffer that EncodeBrunsli /
# DecodeBrunsli write through the C function pointer interface.
cdef struct _Sink:
    uint8_t* data
    size_t size      # bytes written
    size_t cap       # capacity allocated
    void* writer_state


cdef size_t _sink_write(void* sink, const uint8_t* buf, size_t n) noexcept nogil:
    cdef _Sink* s = <_Sink*> sink
    if s.writer_state != NULL:
        with gil:
            try:
                (<object> s.writer_state).write(
                    PyBytes_FromStringAndSize(<const char*> buf, <Py_ssize_t> n))
            except BaseException as exc:
                (<object> s.writer_state).error = exc
                return 0
        return n
    cdef size_t need = s.size + n
    cdef size_t new_cap
    cdef uint8_t* new_data
    if need > s.cap:
        new_cap = s.cap * 2 if s.cap else 4096
        while new_cap < need:
            new_cap *= 2
        new_data = <uint8_t*> realloc(s.data, new_cap)
        if new_data == NULL:
            return 0
        s.data = new_data
        s.cap = new_cap
    memcpy(s.data + s.size, buf, n)
    s.size += n
    return n


_BRUNSLI_MAGIC = b'\x0A\x04\x42\xd2'   # Brunsli marker (first 4 bytes)


class BrunsliError(RuntimeError):
    """Raised on Brunsli encode/decode failures."""


def encode_jpeg(data, *, dest=None):
    """Transcode a complete JPEG to bytes or directly to a destination."""
    return _transcode(data, True, dest)


def decode_jpeg(data, *, dest=None):
    """Recover the original JPEG bytes or write them to a destination."""
    return _transcode(data, False, dest)


cdef _transcode(data, bint encode, dest):
    cdef const uint8_t[::1] src
    cdef _Sink sink
    cdef int rc
    cdef object state = None
    cdef object context
    cdef size_t srcsize
    from contextlib import nullcontext
    from opencodecs.core._write_helpers import binary_destination
    from opencodecs.core.buffers import CallbackDestination

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    if src.shape[0] < 4:
        raise BrunsliError("transcode input is too short")
    if encode and (src[0] != 0xff or src[1] != 0xd8):
        raise BrunsliError("encode_jpeg: input does not look like a JPEG (missing SOI 0xFFD8 marker)")
    sink.data = NULL
    sink.size = 0
    sink.cap = 0
    sink.writer_state = NULL
    srcsize = <size_t> src.shape[0]
    context = nullcontext(None) if dest is None else binary_destination(dest)
    try:
        with context as target:
            if dest is not None:
                state = CallbackDestination(target)
                sink.writer_state = <void*> state
            with nogil:
                if encode:
                    rc = EncodeBrunsli(srcsize, &src[0], <void*> &sink, _sink_write)
                else:
                    rc = DecodeBrunsli(srcsize, &src[0], <void*> &sink, _sink_write)
            if state is not None and state.error is not None:
                raise state.error
            if rc != 1:
                raise BrunsliError(f"Brunsli transcode returned {rc}")
            if dest is None:
                return PyBytes_FromStringAndSize(<const char*> sink.data, <Py_ssize_t> sink.size)
            return None
    finally:
        if sink.data != NULL:
            free(sink.data)


def check_signature(data) -> bool:
    """Return True if ``data`` begins with the Brunsli 4-byte marker."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:4])
    else:
        try:
            head = bytes(data)[:4]
        except Exception:
            return False
    return head == _BRUNSLI_MAGIC
