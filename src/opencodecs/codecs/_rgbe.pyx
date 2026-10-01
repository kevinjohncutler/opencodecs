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

By default the encoder emits RLE-compressed pixels after the header:
RLE is ~1.4-2× smaller than raw on real HDR photographic data,
decoding is essentially free, and the Radiance reference tools assume
RLE inside a header'd file. ``header=False`` writes the bare pixel
stream (flat RGBE quadruples unless ``rle=True``), and ``rle=False``
writes flat pixels after a header; both follow the parameter names of
imagecodecs ``rgbe_encode`` / ``rgbe_decode``. imagecodecs ignores
``rle=False`` when it writes a header and writes RLE anyway, so that one
combination gives different bytes here (flat scanlines, which
imagecodecs and Radiance read). The decoder honors every orientation the
resolution line can state (Radiance file format, "Resolution String"),
and, like Radiance's own reader, accepts a ``#?`` header with no
``FORMAT`` line.
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


# Upper bound on the encoded size. Flat pixels take 4 bytes each. An RLE
# scanline (RGBE_WritePixels_RLE in rgbe.c) is a 4-byte marker followed
# by the four channels, each written by RGBE_WriteBytes_RLE as literal
# chunks of at most 128 bytes (one count byte per chunk) and runs of 4 to
# 127 bytes (two bytes per run, which never grows the data). Each run
# also ends a literal stretch, so a channel of W bytes takes at most
# W + ceil(W / 128) + 1 bytes, the + 1 covering the chunk a run splits.
# Noisy scanlines reach that bound, so 4 bytes per pixel is not enough.
# The header the writer emits is under 100 bytes; 1024 leaves room.
cdef inline Py_ssize_t _max_encoded_size(
        Py_ssize_t width, Py_ssize_t height) nogil:
    cdef Py_ssize_t per_channel = width + (width + 127) // 128 + 2
    return height * (4 + 4 * per_channel) + 1024


def encode(arr, *, header=None, rle=None) -> bytes:
    """Encode an ``(H, W, 3)`` float32 RGB image as a Radiance HDR file.

    Returns the full file bytes (header + RLE-compressed RGBE pixels).
    Input must be C-contiguous and finite; NaN / Inf are not part of
    the Radiance format and silently produce invalid output.

    ``header`` (default True) writes the ``#?RADIANCE`` header and
    resolution line; False writes only the pixels, and then any
    ``(..., 3)`` array is accepted. ``rle`` (default: True with a
    header, False without) selects run-length-encoded scanlines over
    flat RGBE quadruples. RLE needs scanlines 8 to 32767 pixels wide;
    other widths are written flat, as the Radiance writer does.
    ``rle=False`` with a header is honored, where imagecodecs writes RLE
    regardless, so only that combination differs from its bytes.
    """
    cdef:
        cnp.ndarray contig
        rgbe_stream_t* stream = NULL
        Py_ssize_t cap, written
        int rc
        int width, height
        bytes out
        const unsigned char[::1] dst_mv

    if header is None:
        header = True
    if rle is None:
        rle = bool(header)
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=np.float32)
    contig = np.ascontiguousarray(arr, dtype=np.float32)
    if not header and contig.ndim >= 1 and contig.shape[contig.ndim - 1] == 3:
        if contig.ndim == 1:
            contig = contig.reshape(1, 1, 3)
        else:
            contig = contig.reshape(-1, contig.shape[contig.ndim - 2], 3)
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

    cap = _max_encoded_size(<Py_ssize_t> width, <Py_ssize_t> height)
    out = PyBytes_FromStringAndSize(NULL, cap)
    # const memoryview lets us write through the bytes buffer's char*
    # without forcing Cython to take a writable export (same trick
    # used in _zfp.pyx / _zstd.pyx).
    dst_mv = out
    stream = rgbe_stream_new(<size_t> cap, <char*> &dst_mv[0])
    if stream == NULL:
        raise MemoryError("rgbe_stream_new returned NULL")

    try:
        if header:
            with nogil:
                rc = RGBE_WriteHeader(stream, width, height, NULL)
            if rc != RGBE_RETURN_SUCCESS:
                raise RgbeError(f"RGBE_WriteHeader returned {rc}")
        if rle:
            with nogil:
                rc = RGBE_WritePixels_RLE(
                    stream,
                    <const float*> contig.data,
                    width, height,
                )
        else:
            with nogil:
                rc = RGBE_WritePixels(
                    stream, <const float*> contig.data, width * height,
                )
        if rc != RGBE_RETURN_SUCCESS:
            raise RgbeError(f"RGBE_WritePixels returned {rc}")
        written = <Py_ssize_t> stream.pos
    finally:
        rgbe_stream_del(stream)
    del dst_mv
    return out[:written]


def decode(data, *, header=None, rle=None, out=None):
    """Decode a Radiance HDR (RGBE) byte stream into an ``(H, W, 3)``
    float32 array.

    ``out`` is an optional preallocated ``np.ndarray`` of the right
    shape and dtype; the codec writes into it directly.

    A header needs a ``FORMAT=32-bit_rle_rgbe`` (or ``_xyze``) line,
    unless it starts with the ``#?`` magic and has no ``FORMAT`` line at
    all, which Radiance reads as its default picture format.

    ``header=False`` reads a bare pixel stream, whose shape only
    ``out`` can give. The default reads a header when the data starts
    with the ``#?`` magic or a header without it parses (it then needs a
    ``FORMAT`` line), and treats the data as a bare pixel stream only
    when ``out`` is given and neither holds. ``out`` for a bare stream
    is an ``(..., W, 3)`` float32 array that the stream must fill
    exactly: input left over raises ``ValueError``, as in imagecodecs.
    RLE and flat scanlines are told apart from the data itself, so
    ``rle`` is accepted for imagecodecs compatibility and needs no
    value.
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
    if header is None:
        # A header when nothing else could give the shape (no out=), when
        # the data starts with the "#?" magic, or when a header parses
        # without the magic (Radiance does not require it, but then it
        # needs a FORMAT line). Otherwise, with out= given, a bare pixel
        # stream, which must fill out exactly: text that is not a header
        # is caught when it leaves input over (ValueError, as in
        # imagecodecs) or runs out of pixels.
        header = out is None or (
            srcsize >= 2 and src[0] == 0x23 and src[1] == 0x3F
        ) or _header_parses(src)
    if not header:
        return _decode_headerless(src, out)
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


cdef bint _header_parses(const unsigned char[::1] src):
    """True when ``src`` starts with a Radiance header the reader accepts."""
    cdef:
        rgbe_stream_t* stream = NULL
        int rc
        int width = 0
        int height = 0
        int orientation = 0
    if src.shape[0] < 8:
        return False
    stream = rgbe_stream_new(<size_t> src.shape[0], <char*> &src[0])
    if stream == NULL:
        raise MemoryError("rgbe_stream_new returned NULL")
    try:
        with nogil:
            rc = RGBE_ReadHeaderOriented(
                stream, &width, &height, NULL, &orientation)
    finally:
        rgbe_stream_del(stream)
    return rc == RGBE_RETURN_SUCCESS


def _decode_headerless(const unsigned char[::1] src, out):
    """Decode a pixel stream without a header into ``out``."""
    cdef:
        rgbe_stream_t* stream = NULL
        Py_ssize_t srcsize = src.shape[0]
        int rc
        int scan_w, scan_n
        cnp.ndarray out_arr
    if out is None:
        raise ValueError(
            "rgbe decode: header=False needs out= to give the shape")
    if not isinstance(out, np.ndarray):
        raise TypeError(
            f"rgbe decode: out= must be ndarray, got {type(out).__name__}")
    if out.dtype != np.float32 or not out.flags['C_CONTIGUOUS']:
        raise ValueError("rgbe decode: out= must be C-contiguous float32")
    if out.ndim < 2 or out.shape[out.ndim - 1] != 3 or out.size == 0:
        raise ValueError(
            f"rgbe decode: out= shape {out.shape} is not (..., W, 3)")
    out_arr = out
    scan_w = <int> out.shape[out.ndim - 2]
    scan_n = <int> (out.size // (3 * scan_w))
    if srcsize < 1:
        raise RgbeError("rgbe decode: stream too short")
    stream = rgbe_stream_new(<size_t> srcsize, <char*> &src[0])
    if stream == NULL:
        raise MemoryError("rgbe_stream_new returned NULL")
    try:
        with nogil:
            rc = RGBE_ReadPixels_RLE(
                stream, <float*> out_arr.data, scan_w, scan_n)
        if rc != RGBE_RETURN_SUCCESS:
            raise RgbeError(f"RGBE_ReadPixels_RLE returned {rc}")
        if <Py_ssize_t> stream.pos != srcsize:
            # imagecodecs raises the same for a bare stream longer than
            # out; it is also how header text without a FORMAT line,
            # read as pixels, shows itself.
            raise ValueError(
                f"rgbe decode: not all input decoded, "
                f"{srcsize - <Py_ssize_t> stream.pos} bytes left over")
    finally:
        rgbe_stream_del(stream)
    return out_arr


def check_signature(data) -> bool:
    """Recognize a Radiance HDR header magic word."""
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
