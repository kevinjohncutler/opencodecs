# opencodecs/codecs/_mozjpeg.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native JPEG encoder via Mozilla's MozJPEG (libjpeg-turbo fork).

MozJPEG's value proposition: it produces JPEG files ~10-15% smaller
than libjpeg-turbo at the same quality setting, by using progressive
encoding, trellis quantization, and better quantization tables.
Standard JPEG bitstreams — any decoder reads them.

Why a separate module
=====================

MozJPEG only ships the older TurboJPEG v2 C API (``tj*`` symbols).
opencodecs's regular ``_jpeg.pyx`` uses v3 (``tj3*``) which gives us
finer-grained parameter control on libjpeg-turbo 3.0+. Rather than
losing that, we keep ``_jpeg.pyx`` against libjpeg-turbo v3 and add
this separate ``_mozjpeg.pyx`` against MozJPEG's v2 API.

Encode is the differentiator; for decode use either codec (output
bitstreams are interoperable JPEGs).
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t

import numpy as np
cimport numpy as cnp

from mozjpeg cimport (
    tjhandle, tjInitCompress, tjInitDecompress, tjDestroy,
    tjGetErrorStr2, tjCompress2, tjDecompressHeader3, tjDecompress2,
    tjFree,
    TJPF_GRAY, TJPF_RGB,
    TJSAMP_GRAY, TJSAMP_444, TJSAMP_422, TJSAMP_420, TJSAMP_440, TJSAMP_411,
)

cnp.import_array()


class MozJpegError(RuntimeError):
    """Raised on MozJPEG encode/decode failures."""


_SUBSAMP_MAP = {
    "444": TJSAMP_444,
    "422": TJSAMP_422,
    "420": TJSAMP_420,
    "440": TJSAMP_440,
    "411": TJSAMP_411,
}


def encode(data, *, level: int | None = None,
           subsampling: object = None,
           progressive: bool = True) -> bytes:
    """Encode a 2D or 3D uint8 array as JPEG via MozJPEG.

    Parameters
    ----------
    data
        2D uint8 (grayscale) or (H, W, 3) uint8 RGB.
    level
        Quality 0-100 (default 75). MozJPEG's quality scale matches
        libjpeg-turbo's exactly so a/b switching is transparent.
    subsampling
        ``"420"`` (default — universal JPEG default), ``"422"``,
        ``"444"``, ``"440"``, ``"411"``. Ignored for grayscale.
    progressive
        Default ``True`` — MozJPEG's progressive encode is the source
        of most of its size advantage over libjpeg-turbo. Set
        ``False`` for baseline (sequential) encode.
    """
    cdef:
        cnp.ndarray arr
        tjhandle handle = NULL
        unsigned char* out_ptr = NULL
        unsigned long out_size = 0
        int rc
        int pf
        int subsamp
        int quality
        int height, width
        int pitch
        int flags
        bytes out

    if not isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data, dtype=np.uint8)
    else:
        if data.dtype != np.uint8:
            raise MozJpegError(
                f'MozJPEG encode: unsupported dtype {data.dtype}')
        arr = np.ascontiguousarray(data)

    if arr.ndim == 2:
        pf = TJPF_GRAY
        subsamp = TJSAMP_GRAY
        height = <int> arr.shape[0]
        width = <int> arr.shape[1]
        pitch = width
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pf = TJPF_RGB
        if subsampling is None:
            subsamp = TJSAMP_420
        else:
            key = str(subsampling).lower().strip()
            if key not in _SUBSAMP_MAP:
                raise MozJpegError(
                    f'MozJPEG encode: unknown subsampling {subsampling!r}; '
                    f'expected one of {sorted(_SUBSAMP_MAP)}')
            subsamp = _SUBSAMP_MAP[key]
        height = <int> arr.shape[0]
        width = <int> arr.shape[1]
        pitch = 3 * width
    else:
        raise MozJpegError(
            f'MozJPEG encode: unsupported ndim={arr.ndim}; '
            'expected 2D grayscale or (H, W, 3) RGB')

    # Default quality 95 — matches imagecodecs.mozjpeg_encode and our
    # own _jpeg.pyx default, per the Pareto-better-or-equal policy in
    # docs/codec_api_conventions.md "Default settings".
    quality = 95 if level is None else int(level)
    if quality < 1: quality = 1
    if quality > 100: quality = 100

    # TJFLAG_PROGRESSIVE | TJFLAG_ACCURATEDCT
    # Progressive is where MozJPEG's main quality/size advantage comes
    # from; accurate DCT keeps decoded pixels closer to source.
    flags = 4096   # TJFLAG_ACCURATEDCT
    if progressive:
        flags |= 16384   # TJFLAG_PROGRESSIVE

    handle = tjInitCompress()
    if handle == NULL:
        raise MozJpegError('tjInitCompress failed')
    try:
        rc = tjCompress2(
            handle, <const unsigned char*> cnp.PyArray_DATA(arr),
            width, pitch, height, pf,
            &out_ptr, &out_size, subsamp, quality, flags,
        )
        if rc < 0:
            err = tjGetErrorStr2(handle).decode('ascii', errors='replace')
            raise MozJpegError(f'tjCompress2: {err}')
        try:
            out = PyBytes_FromStringAndSize(
                <char*> out_ptr, <Py_ssize_t> out_size)
            return out
        finally:
            tjFree(out_ptr)
    finally:
        tjDestroy(handle)


def decode(data, *, out=None) -> np.ndarray:
    """Decode JPEG bytes into a uint8 array.

    Decode is standard JPEG — the output is identical regardless of
    which library encoded the input. We expose this for symmetry with
    the rest of the codec module surface.
    """
    cdef:
        const uint8_t[::1] src
        unsigned long srcsize
        tjhandle handle = NULL
        int rc
        int width, height, subsamp, colorspace
        int pf
        int channels
        cnp.ndarray out_arr
        cnp.npy_intp shape[3]
        int ndim
        tuple expected_shape

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <unsigned long> src.shape[0]
    if srcsize < 3:
        raise MozJpegError('input too short to be JPEG')

    handle = tjInitDecompress()
    if handle == NULL:
        raise MozJpegError('tjInitDecompress failed')
    try:
        rc = tjDecompressHeader3(
            handle, &src[0], srcsize, &width, &height, &subsamp, &colorspace,
        )
        if rc < 0:
            err = tjGetErrorStr2(handle).decode('ascii', errors='replace')
            raise MozJpegError(f'tjDecompressHeader3: {err}')

        if subsamp == TJSAMP_GRAY:
            pf = TJPF_GRAY
            channels = 1
            ndim = 2
            expected_shape = (height, width)
        else:
            pf = TJPF_RGB
            channels = 3
            ndim = 3
            shape[2] = 3
            expected_shape = (height, width, 3)

        shape[0] = height
        shape[1] = width
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError(
                    f"mozjpeg decode: out= must be an ndarray, "
                    f"got {type(out).__name__}")
            if out.shape != expected_shape:
                raise ValueError(
                    f"mozjpeg decode: out= shape {out.shape} does not match "
                    f"expected {expected_shape}")
            if out.dtype != np.uint8:
                raise ValueError(
                    f"mozjpeg decode: out= dtype must be uint8, "
                    f"got {out.dtype}")
            if not out.flags['C_CONTIGUOUS']:
                raise ValueError("mozjpeg decode: out= must be C-contiguous")
            out_arr = out
        else:
            out_arr = cnp.PyArray_EMPTY(ndim, shape, cnp.NPY_UINT8, 0)
        rc = tjDecompress2(
            handle, &src[0], srcsize,
            <unsigned char*> cnp.PyArray_DATA(out_arr),
            width, width * channels, height, pf, 0,
        )
        if rc < 0:
            err = tjGetErrorStr2(handle).decode('ascii', errors='replace')
            raise MozJpegError(f'tjDecompress2: {err}')
        return out_arr
    finally:
        tjDestroy(handle)


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
