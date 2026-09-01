# opencodecs/codecs/_rgbe.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Radiance HDR (RGBE) codec — Cython binding to the vendored Bruce
Walter / Greg Ward C library (``3rdparty/rgbe/``).

The encoder writes a full Radiance ``.hdr`` file: a text header
(``#?RADIANCE``, ``FORMAT=32-bit_rle_rgbe``, ``Y H +X W`` resolution
line) followed by run-length-encoded RGBE pixels — the same on-disk
format consumed by Blender, Mitsuba, and every other Radiance-aware
imaging tool. Decoder accepts that format and returns an
``(H, W, 3) float32`` array.

For Pareto-default behaviour we always emit RLE-compressed pixels
when a header is written: RLE is ~1.4-2× smaller than raw on real
HDR photographic data, decoding is essentially free, and the
Radiance reference tools assume RLE inside a header'd file.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport int32_t

import numpy as np
cimport numpy as cnp

from rgbe cimport (
    rgbe_stream_t, rgbe_stream_new, rgbe_stream_del,
    rgbe_header_info,
    RGBE_WriteHeader, RGBE_ReadHeader, RGBE_ReadHeaderOriented,
    RGBE_WritePixels, RGBE_ReadPixels,
    RGBE_WritePixels_RLE, RGBE_ReadPixels_RLE,
    RGBE_RETURN_SUCCESS,
    RGBE_ORIENT_NONE, RGBE_ORIENT_FLIP_X,
    RGBE_ORIENT_FLIP_Y, RGBE_ORIENT_TRANSPOSE,
)


cnp.import_array()


class RgbeError(RuntimeError):
    """Raised on RGBE encode/decode failures."""


# Reasonable upper bound for output: 4 bytes per pixel for the raw
# encoding + 256 bytes for the header. RLE never inflates because
# the codec falls back to raw whenever a run wouldn't shorten the
# scanline (see RGBE_WritePixels_RLE in rgbe.c).
cdef inline Py_ssize_t _max_encoded_size(Py_ssize_t pixels) nogil:
    return pixels * 4 + 1024


def encode(arr) -> bytes:
    """Encode an ``(H, W, 3)`` float32 RGB image as a Radiance HDR file.

    Returns the full file bytes (header + RLE-compressed RGBE pixels).
    Input must be C-contiguous and finite; NaN / Inf are not part of
    the Radiance format and silently produce invalid output.
    """
    cdef:
        cnp.ndarray contig
        rgbe_stream_t* stream = NULL
        Py_ssize_t cap, written
        int rc
        int width, height
        bytes out
        const unsigned char[::1] dst_mv

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=np.float32)
    contig = np.ascontiguousarray(arr, dtype=np.float32)
    if contig.ndim != 3 or contig.shape[2] != 3:
        # ``contig.shape`` is a C array in Cython; round-trip through
        # numpy.shape() to get the Python tuple for the error message.
        raise ValueError(
            f"rgbe encode: expected (H, W, 3) float32, "
            f"got shape {np.shape(contig)} dtype {contig.dtype}"
        )

    height = <int> contig.shape[0]
    width = <int> contig.shape[1]
    if height <= 0 or width <= 0:
        raise ValueError(f"rgbe encode: empty image {np.shape(contig)}")

    cap = _max_encoded_size(<Py_ssize_t> width * <Py_ssize_t> height)
    out = PyBytes_FromStringAndSize(NULL, cap)
    # const memoryview lets us write through the bytes buffer's char*
    # without forcing Cython to take a writable export (same trick
    # used in _zfp.pyx / _zstd.pyx).
    dst_mv = out
    stream = rgbe_stream_new(<size_t> cap, <char*> &dst_mv[0])
    if stream == NULL:
        raise MemoryError("rgbe_stream_new returned NULL")

    try:
        with nogil:
            rc = RGBE_WriteHeader(stream, width, height, NULL)
        if rc != RGBE_RETURN_SUCCESS:
            raise RgbeError(f"RGBE_WriteHeader returned {rc}")
        with nogil:
            rc = RGBE_WritePixels_RLE(
                stream,
                <const float*> contig.data,
                width, height,
            )
        if rc != RGBE_RETURN_SUCCESS:
            raise RgbeError(f"RGBE_WritePixels_RLE returned {rc}")
        written = <Py_ssize_t> stream.pos
    finally:
        rgbe_stream_del(stream)
    del dst_mv
    return out[:written]


def decode(data, *, out=None):
    """Decode a Radiance HDR (RGBE) byte stream into an ``(H, W, 3)``
    float32 array.

    ``out`` is an optional preallocated ``np.ndarray`` of the right
    shape and dtype; the codec writes into it directly.
    """
    cdef:
        const unsigned char[::1] src
        rgbe_stream_t* stream = NULL
        Py_ssize_t srcsize
        int rc
        int width = 0
        int height = 0
        int orientation = 0
        int transposed = 0
        int scan_w = 0
        int scan_n = 0
        cnp.ndarray out_arr
        cnp.ndarray raw_arr

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = src.shape[0]
    if srcsize < 8:
        raise RgbeError("rgbe decode: stream too short")

    stream = rgbe_stream_new(<size_t> srcsize, <char*> &src[0])
    if stream == NULL:
        raise MemoryError("rgbe_stream_new returned NULL")

    try:
        with nogil:
            rc = RGBE_ReadHeaderOriented(
                stream, &width, &height, NULL, &orientation
            )
        if rc != RGBE_RETURN_SUCCESS:
            raise RgbeError(f"RGBE_ReadHeader returned {rc}")
        if width <= 0 or height <= 0:
            raise RgbeError(
                f"rgbe decode: bad dimensions {height}x{width}"
            )

        # Radiance stores the image in the order the resolution line
        # describes. Row-major is the usual case; an X-major file stores
        # height-long scanlines, width of them, and is transposed after
        # decoding. Flips are applied afterwards too, because reversing a
        # decoded array is cheaper and clearer than decoding backwards.
        transposed = (orientation & RGBE_ORIENT_TRANSPOSE) != 0
        scan_w = height if transposed else width
        scan_n = width if transposed else height
        shape = (height, width, 3)
        raw_shape = (scan_n, scan_w, 3)
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError(
                    f"rgbe decode: out= must be ndarray, "
                    f"got {type(out).__name__}"
                )
            if out.shape != shape:
                raise ValueError(
                    f"rgbe decode: out= shape {out.shape} != expected {shape}"
                )
            if out.dtype != np.float32:
                raise ValueError(
                    f"rgbe decode: out= dtype {out.dtype} != float32"
                )
            if not out.flags['C_CONTIGUOUS']:
                raise ValueError("rgbe decode: out= must be C-contiguous")
            out_arr = out
        else:
            out_arr = np.empty(shape, dtype=np.float32)

        # Decode into a buffer shaped the way the file stores it. For the
        # common orientation that is out_arr itself, so nothing is copied.
        if orientation == RGBE_ORIENT_NONE:
            raw_arr = out_arr
        else:
            raw_arr = np.empty(raw_shape, dtype=np.float32)

        with nogil:
            rc = RGBE_ReadPixels_RLE(
                stream,
                <float*> raw_arr.data,
                scan_w, scan_n,
            )
        if rc != RGBE_RETURN_SUCCESS:
            raise RgbeError(f"RGBE_ReadPixels_RLE returned {rc}")

        if orientation != RGBE_ORIENT_NONE:
            view = raw_arr
            if transposed:
                view = view.transpose(1, 0, 2)
            if orientation & RGBE_ORIENT_FLIP_Y:
                view = view[::-1]
            if orientation & RGBE_ORIENT_FLIP_X:
                view = view[:, ::-1]
            if view.shape != shape:
                raise RgbeError(
                    f"rgbe decode: oriented shape {view.shape} != {shape}"
                )
            out_arr[...] = view
    finally:
        rgbe_stream_del(stream)
    return out_arr


def check_signature(data) -> bool:
    """Recognise a Radiance HDR header magic word."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:11])
    else:
        try:
            head = bytes(data)[:11]
        except Exception:
            return False
    # Radiance files start with "#?RADIANCE" or "#?RGBE" (older Greg
    # Ward variants); the C reader accepts either.
    return head.startswith(b"#?")
