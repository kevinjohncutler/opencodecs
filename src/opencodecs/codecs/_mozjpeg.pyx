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
    tjFree, tjscalingfactor, tjGetScalingFactors,
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


def supported_scaling_factors() -> list[tuple[int, int]]:
    """MozJPEG's allowed decode-time scaling factors, ``(num, denom)``.

    Pass any of these to ``decode(scale=...)``. The ratios come from
    the library rather than a hardcoded list, because MozJPEG tracks
    libjpeg-turbo's set and it has changed across versions.
    """
    cdef int n = 0
    cdef tjscalingfactor* arr = tjGetScalingFactors(&n)
    if arr == NULL or n <= 0:
        return []
    return [(int(arr[i].num), int(arr[i].denom)) for i in range(n)]


def _resolve_scale(scale, scale_num, scale_denom):
    """Coerce a caller's scale into a supported ``(num, denom)``.

    Deliberately the same surface as ``_jpeg._resolve_scale``: an int N
    means 1/N, a float snaps to the nearest supported ratio, a tuple is
    explicit. Two JPEG codecs that take the same argument differently
    would be a trap, and mozjpeg is a drop-in for jpeg at decode.
    """
    if scale is None:
        if scale_num is None and scale_denom is None:
            return (1, 1)
        if scale_num is None:
            scale_num = 1
        if scale_denom is None:
            scale_denom = 1
        return (int(scale_num), int(scale_denom))

    if isinstance(scale, (tuple, list)):
        if len(scale) != 2:
            raise ValueError(
                f"mozjpeg decode: scale tuple must be (num, denom); "
                f"got {scale!r}")
        return (int(scale[0]), int(scale[1]))

    if isinstance(scale, int) and not isinstance(scale, bool):
        if scale < 1:
            raise ValueError(
                f"mozjpeg decode: scale int must be >= 1 (means '1/N'); "
                f"got {scale!r}")
        return (1, int(scale))

    f = float(scale)
    if f <= 0:
        raise ValueError(f"mozjpeg decode: scale must be > 0; got {scale!r}")
    factors = supported_scaling_factors()
    if not factors:
        return (1, 1)
    return min(factors, key=lambda nd: abs(nd[0] / nd[1] - f))


def decode(data, *, out=None, scale=None,
           scale_num=None, scale_denom=None) -> np.ndarray:
    """Decode JPEG bytes into a uint8 array.

    Decode is standard JPEG — the output is identical regardless of
    which library encoded the input. We expose this for symmetry with
    the rest of the codec module surface.

    Parameters
    ----------
    scale : int | float | tuple, optional
        Decode at a fraction of the stored size using the DCT-domain
        shortcut: the decoder runs a smaller inverse DCT and never
        reconstructs the high-frequency detail, so a 1/8 decode costs a
        fraction of a full one rather than being a resize of it.
        ``8`` means 1/8; a float snaps to the nearest supported ratio;
        a ``(num, denom)`` tuple is explicit.
        :func:`supported_scaling_factors` lists what is allowed.
    scale_num, scale_denom : int, optional
        The ratio spelled out, as an alternative to ``scale``.
    """
    cdef:
        const uint8_t[::1] src
        unsigned long srcsize
        tjhandle handle = NULL
        int rc
        int width, height, subsamp, colorspace
        int full_width, full_height
        int s_num, s_den
        int pf
        int channels
        cnp.ndarray out_arr
        cnp.npy_intp shape[3]
        int ndim
        tuple expected_shape
        unsigned char* dst_ptr
        int pitch

    s_num, s_den = _resolve_scale(scale, scale_num, scale_denom)
    if (s_num, s_den) != (1, 1):
        if (s_num, s_den) not in supported_scaling_factors():
            raise ValueError(
                f"mozjpeg decode: scaling factor {s_num}/{s_den} is not "
                f"supported; available: {supported_scaling_factors()}")

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
        with nogil:
            rc = tjDecompressHeader3(
                handle, &src[0], srcsize,
                &full_width, &full_height, &subsamp, &colorspace)
        if rc < 0:
            err = tjGetErrorStr2(handle).decode('ascii', errors='replace')
            raise MozJpegError(f'tjDecompressHeader3: {err}')

        # TurboJPEG v2 has no SetScalingFactor: tjDecompress2 picks the
        # scaling factor from the destination size it is given, so the
        # scaled extent has to be computed here. TJSCALED(d, f) is
        # (d * num + denom - 1) // denom.
        width = (full_width * s_num + s_den - 1) // s_den
        height = (full_height * s_num + s_den - 1) // s_den

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
        # The Huffman and inverse-DCT passes are the expensive half,
        # and TurboJPEG touches nothing Python in them: source and
        # destination are both raw pointers, one into the input buffer
        # and one into an array this function owns. Holding the GIL
        # through it made every caller decoding JPEGs on threads
        # measure 0.99x -- a thread pool paying overhead to take
        # turns, which is worse than not having one.
        dst_ptr = <unsigned char*> cnp.PyArray_DATA(out_arr)
        pitch = width * channels
        with nogil:
            rc = tjDecompress2(
                handle, &src[0], srcsize, dst_ptr,
                width, pitch, height, pf, 0)
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
