# opencodecs/codecs/_jpeg.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native JPEG codec via libjpeg-turbo (TurboJPEG API v3).

Encode: 2D uint8 (grayscale) or (H, W, 3) uint8 RGB.
Decode: returns (H, W) for grayscale JPEGs, (H, W, 3) for color.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t

import numpy as np
cimport numpy as cnp

from turbojpeg cimport (
    tjhandle, tj3Init, tj3Destroy, tj3GetErrorStr,
    tj3Set, tj3Get, tj3Free,
    tj3Compress8, tj3DecompressHeader, tj3Decompress8,
    TJINIT_COMPRESS, TJINIT_DECOMPRESS,
    TJPF_GRAY, TJPF_RGB,
    TJSAMP_GRAY, TJSAMP_444, TJSAMP_422, TJSAMP_420, TJSAMP_440, TJSAMP_411,
    TJPARAM_QUALITY, TJPARAM_SUBSAMP,
    TJPARAM_JPEGWIDTH, TJPARAM_JPEGHEIGHT,
)

cnp.import_array()


class JpegError(RuntimeError):
    """Raised on JPEG encode/decode failures."""


_SUBSAMP_MAP = {
    "444": TJSAMP_444,
    "422": TJSAMP_422,
    "420": TJSAMP_420,
    "440": TJSAMP_440,
    "411": TJSAMP_411,
    "gray": TJSAMP_GRAY,
    "grayscale": TJSAMP_GRAY,
}


def encode(data, *, level: int | None = None,
           subsampling: object = None) -> bytes:
    """Encode a 2D or 3D uint8 array as JPEG.

    ``level`` is the JPEG quality 0-100 (default 75).
    ``subsampling`` chooses the chroma subsampling for color JPEGs:
    "420" (default — same as imagecodecs / cjpeg / every JPEG tool),
    "422", "444", "440", "411". Higher ratios produce smaller files
    and encode/decode faster at a small chroma-resolution cost; "444"
    keeps full chroma. Pass ``"444"`` to match opencodecs's previous
    behavior. Ignored for grayscale input.
    """
    cdef:
        cnp.ndarray arr
        tjhandle handle = NULL
        unsigned char* out_ptr = NULL
        size_t out_size = 0
        int rc
        int pf
        int subsamp
        int quality
        int height, width
        int pitch
        bytes out

    if not isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data, dtype=np.uint8)
    else:
        if data.dtype != np.uint8:
            raise JpegError(f'JPEG encode: unsupported dtype {data.dtype}')
        arr = np.ascontiguousarray(data)

    if arr.ndim == 2:
        pf = TJPF_GRAY
        subsamp = TJSAMP_GRAY
        height = <int> arr.shape[0]
        width = <int> arr.shape[1]
        pitch = width
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pf = TJPF_RGB
        # 4:2:0 is the JPEG-encoder universal default — matches
        # imagecodecs and cjpeg. Halves chroma data → ~2x faster
        # encode + decode and ~2x smaller files.
        if subsampling is None:
            subsamp = TJSAMP_420
        else:
            key = str(subsampling).lower().strip()
            if key not in _SUBSAMP_MAP:
                raise JpegError(
                    f'JPEG encode: unknown subsampling {subsampling!r}; '
                    f'expected one of {sorted(_SUBSAMP_MAP)}')
            subsamp = _SUBSAMP_MAP[key]
        height = <int> arr.shape[0]
        width = <int> arr.shape[1]
        pitch = 3 * width
    else:
        raise JpegError(
            f'JPEG encode: unsupported ndim={arr.ndim}; '
            'expected 2D grayscale or (H, W, 3) RGB')

    quality = 75 if level is None else int(level)
    if quality < 1: quality = 1
    if quality > 100: quality = 100

    handle = tj3Init(TJINIT_COMPRESS)
    if handle == NULL:
        raise JpegError('tj3Init(COMPRESS) failed')
    try:
        if tj3Set(handle, TJPARAM_QUALITY, quality) < 0:
            raise JpegError(
                f'tj3Set(QUALITY): {tj3GetErrorStr(handle).decode()}')
        if tj3Set(handle, TJPARAM_SUBSAMP, subsamp) < 0:
            raise JpegError(
                f'tj3Set(SUBSAMP): {tj3GetErrorStr(handle).decode()}')
        with nogil:
            rc = tj3Compress8(
                handle, <const unsigned char*> cnp.PyArray_DATA(arr),
                width, pitch, height, pf, &out_ptr, &out_size,
            )
        if rc < 0:
            raise JpegError(f'tj3Compress8: {tj3GetErrorStr(handle).decode()}')
        try:
            out = PyBytes_FromStringAndSize(
                <char*> out_ptr, <Py_ssize_t> out_size)
            return out
        finally:
            tj3Free(out_ptr)
    finally:
        tj3Destroy(handle)


def decode(data) -> np.ndarray:
    """Decode JPEG bytes into a uint8 array."""
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        tjhandle handle = NULL
        int rc
        int width, height
        int pf
        int channels
        cnp.ndarray out
        cnp.npy_intp shape[3]
        int ndim

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if srcsize < 3:
        raise JpegError('input too short to be JPEG')

    handle = tj3Init(TJINIT_DECOMPRESS)
    if handle == NULL:
        raise JpegError('tj3Init(DECOMPRESS) failed')
    try:
        rc = tj3DecompressHeader(handle, &src[0], srcsize)
        if rc < 0:
            raise JpegError(
                f'tj3DecompressHeader: {tj3GetErrorStr(handle).decode()}')
        width = tj3Get(handle, TJPARAM_JPEGWIDTH)
        height = tj3Get(handle, TJPARAM_JPEGHEIGHT)
        # TJSAMP_GRAY indicates a single-component (grayscale) JPEG;
        # everything else we coerce to RGB.
        if tj3Get(handle, TJPARAM_SUBSAMP) == TJSAMP_GRAY:
            pf = TJPF_GRAY
            channels = 1
            ndim = 2
        else:
            pf = TJPF_RGB
            channels = 3
            ndim = 3
            shape[2] = 3

        shape[0] = height
        shape[1] = width
        out = cnp.PyArray_EMPTY(ndim, shape, cnp.NPY_UINT8, 0)
        with nogil:
            rc = tj3Decompress8(
                handle, &src[0], srcsize,
                <unsigned char*> cnp.PyArray_DATA(out),
                width * channels, pf,
            )
        if rc < 0:
            raise JpegError(
                f'tj3Decompress8: {tj3GetErrorStr(handle).decode()}')
        return out
    finally:
        tj3Destroy(handle)


def check_signature(data) -> bool:
    """True if `data` starts with a JPEG SOI marker (0xFFD8)."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:2])
    else:
        try:
            head = bytes(data)[:2]
        except Exception:
            return False
    return len(head) >= 2 and head[0] == 0xFF and head[1] == 0xD8
