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
    tj3SetICCProfile, tj3GetICCProfile,
    tj3SetScalingFactor, tj3GetScalingFactors, tjscalingfactor,
    TJINIT_COMPRESS, TJINIT_DECOMPRESS,
    TJPF_GRAY, TJPF_RGB,
    TJSAMP_GRAY, TJSAMP_444, TJSAMP_422, TJSAMP_420, TJSAMP_440, TJSAMP_411,
    TJPARAM_QUALITY, TJPARAM_SUBSAMP,
    TJPARAM_JPEGWIDTH, TJPARAM_JPEGHEIGHT,
    TJPARAM_LOSSLESS, TJPARAM_LOSSLESSPSV,
)

from cpython.bytes cimport PyBytes_FromStringAndSize

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
           subsampling: object = None,
           iccprofile: bytes | None = None,
           lossless: bool = False) -> bytes:
    """Encode a 2D or 3D uint8 array as JPEG.

    ``level`` is the JPEG quality 0-100 (default 95, matching
    ``imagecodecs.jpeg_encode`` so the file we emit is at least as
    high quality as ic's default — see docs/codec_api_conventions.md
    "Default settings: Pareto-better than the reference, no cheating").
    Ignored when ``lossless=True``.

    ``subsampling`` chooses the chroma subsampling for color JPEGs:
    "420" (default — same as imagecodecs / cjpeg / every JPEG tool),
    "422", "444", "440", "411". Higher ratios produce smaller files
    and encode/decode faster at a small chroma-resolution cost; "444"
    keeps full chroma. Pass ``"444"`` to match opencodecs's previous
    behavior. Ignored for grayscale input. Forced to ``"444"`` when
    ``lossless=True`` (lossless mode rejects chroma subsampling).

    ``iccprofile`` embeds an ICC color profile in an APP2 marker.
    libjpeg-turbo copies the bytes, so the caller can release them
    immediately after encode returns.

    ``lossless=True`` switches libjpeg-turbo into predictive lossless
    mode (PSV=1, point-transform=0). Output is bit-exact through the
    libjpeg-turbo decoder. ~3-5× larger than ``level=95`` baseline
    DCT on natural-image content. Niche — only useful when exact
    pixel preservation is the requirement (archival, scientific
    imaging). Note that lossless-mode JPEGs are JCS_RGB-tagged;
    some downstream tools (including libuhdr's ``uhdr_decode``)
    reject non-YCbCr/grayscale color spaces.
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
        const unsigned char[::1] icc_view

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
        # encode + decode and ~2x smaller files. Lossless mode
        # rejects chroma subsampling, so force 4:4:4 there.
        if lossless:
            subsamp = TJSAMP_444
        elif subsampling is None:
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

    quality = 95 if level is None else int(level)
    if quality < 1: quality = 1
    if quality > 100: quality = 100

    handle = tj3Init(TJINIT_COMPRESS)
    if handle == NULL:
        raise JpegError('tj3Init(COMPRESS) failed')
    try:
        # Lossless mode is set BEFORE quality / subsamp — libjpeg-turbo
        # tj3Set rejects subsamp values that lossless wouldn't accept,
        # so the order matters when we transition between modes on a
        # reused handle (not our case here, but the discipline is
        # cheap).
        if lossless:
            if tj3Set(handle, TJPARAM_LOSSLESS, 1) < 0:
                raise JpegError(
                    f'tj3Set(LOSSLESS): {tj3GetErrorStr(handle).decode()}')
            # PSV=1 is "predictor: pixel to the left" — the smallest
            # output for typical content. PSV 2-7 swap the predictor;
            # marginal differences on natural images. Point-transform
            # left at default 0 (no right-shift; full precision).
            if tj3Set(handle, TJPARAM_LOSSLESSPSV, 1) < 0:
                raise JpegError(
                    f'tj3Set(LOSSLESSPSV): {tj3GetErrorStr(handle).decode()}')
        if tj3Set(handle, TJPARAM_QUALITY, quality) < 0:
            raise JpegError(
                f'tj3Set(QUALITY): {tj3GetErrorStr(handle).decode()}')
        if tj3Set(handle, TJPARAM_SUBSAMP, subsamp) < 0:
            raise JpegError(
                f'tj3Set(SUBSAMP): {tj3GetErrorStr(handle).decode()}')
        if iccprofile is not None and len(iccprofile) > 0:
            icc_view = iccprofile
            rc = tj3SetICCProfile(
                handle, &icc_view[0], <size_t> icc_view.shape[0])
            if rc < 0:
                raise JpegError(
                    f'tj3SetICCProfile: {tj3GetErrorStr(handle).decode()}')
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


def supported_scaling_factors() -> list[tuple[int, int]]:
    """Return libjpeg-turbo's allowed decode-time scaling factors as
    ``(num, denom)`` pairs. Pass any of these to ``decode(..., scale=...)``
    or ``scale_num=/scale_denom=`` to decode the image at ``num/denom``
    of its stored size via the DCT-domain shortcut (skips most of the
    inverse-DCT work for high-frequency coefficients).

    Typical contents on libjpeg-turbo 3.x::

        [(2,1), (15,8), (7,4), (13,8), (3,2), (11,8), (5,4), (9,8),
         (1,1), (7,8), (3,4), (5,8), (1,2), (3,8), (1,4), (1,8)]
    """
    cdef int n = 0
    cdef tjscalingfactor* arr = tj3GetScalingFactors(&n)
    if arr == NULL or n <= 0:
        return []
    return [(int(arr[i].num), int(arr[i].denom)) for i in range(n)]


def _resolve_scale(scale, scale_num, scale_denom):
    """Coerce caller-supplied scale into a (num, denom) pair within
    libjpeg-turbo's supported set. Accepts:

    * ``None`` + ``scale_num=N, scale_denom=D`` — explicit ratio.
    * a single int ``N``: maps to ``(1, N)`` (the user thinks of
      "decode at 1/N size"; this is the common case).
    * a float in (0, 2]: snapped to the closest supported factor.
    * a tuple/list ``(num, denom)``: explicit ratio.

    Returns ``(num, denom)`` or ``(1, 1)`` if no downscale requested.
    Raises ``ValueError`` when the requested ratio isn't supported.
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
                f"jpeg decode: scale tuple must be (num, denom); "
                f"got {scale!r}")
        return (int(scale[0]), int(scale[1]))

    if isinstance(scale, int) and not isinstance(scale, bool):
        if scale < 1:
            raise ValueError(
                f"jpeg decode: scale int must be ≥ 1 (means '1/N'); "
                f"got {scale!r}")
        return (1, int(scale))

    # Float: snap to the closest supported factor.
    f = float(scale)
    if f <= 0:
        raise ValueError(f"jpeg decode: scale must be > 0; got {scale!r}")
    factors = supported_scaling_factors()
    if not factors:
        return (1, 1)
    best = min(factors, key=lambda nd: abs(nd[0] / nd[1] - f))
    return best


def decode(data, *, out=None, scale=None,
           scale_num=None, scale_denom=None) -> np.ndarray:
    """Decode JPEG bytes into a uint8 array.

    Parameters
    ----------
    data : bytes-like
        Encoded JPEG payload.
    out : ndarray, optional
        Preallocated output buffer. libjpeg-turbo's tj3Decompress8
        writes directly into the caller's buffer — true zero-alloc.
        Must match the post-scale output shape exactly. See
        ``_png.decode`` for the full contract.
    scale : int / float / (num, denom), optional
        Decode-time scaling factor via libjpeg-turbo's DCT-domain
        shortcut. Pass an integer ``N`` (interpreted as ``1/N``), a
        float (snapped to the closest supported factor), or an explicit
        ``(num, denom)`` pair. ``None`` (default) keeps the full
        resolution. ~N²× faster + memory than full-decode + post-resize.
    scale_num, scale_denom : int, optional
        Lower-level explicit ratio; takes precedence over ``scale``
        when ``scale`` is ``None``. Useful when threading a
        :func:`supported_scaling_factors` pick through code.

    Output shape is ``(TJSCALED(H, factor), TJSCALED(W, factor)[, 3])``
    where ``TJSCALED(d, (n, m)) = (d * n + m - 1) // m``.
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        tjhandle handle = NULL
        int rc
        int width, height
        int pf
        int channels
        cnp.ndarray out_arr
        cnp.npy_intp shape[3]
        int ndim
        tuple expected_shape
        tjscalingfactor factor
        int s_num, s_den

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if srcsize < 3:
        raise JpegError('input too short to be JPEG')

    s_num, s_den = _resolve_scale(scale, scale_num, scale_denom)

    handle = tj3Init(TJINIT_DECOMPRESS)
    if handle == NULL:
        raise JpegError('tj3Init(DECOMPRESS) failed')
    try:
        rc = tj3DecompressHeader(handle, &src[0], srcsize)
        if rc < 0:
            raise JpegError(
                f'tj3DecompressHeader: {tj3GetErrorStr(handle).decode()}')
        # Apply DCT-domain decode scaling if the caller asked for it.
        # tj3SetScalingFactor rejects unsupported ratios; the error
        # surfaces with libjpeg-turbo's own diagnostic. Skip the call
        # at 1/1 (default) so we don't pay the per-decode round-trip
        # for full-resolution decodes.
        if (s_num, s_den) != (1, 1):
            factor.num = s_num
            factor.denom = s_den
            if tj3SetScalingFactor(handle, factor) < 0:
                raise JpegError(
                    f'tj3SetScalingFactor({s_num}/{s_den}): '
                    f'{tj3GetErrorStr(handle).decode()}. '
                    f'Supported factors: {supported_scaling_factors()}')
        # libjpeg-turbo reports the scaled output dimensions through
        # the same JPEGWIDTH/JPEGHEIGHT params after SetScalingFactor.
        # TJSCALED(d, factor) = (d * num + denom - 1) // denom.
        width = (tj3Get(handle, TJPARAM_JPEGWIDTH) * s_num + s_den - 1) // s_den
        height = (tj3Get(handle, TJPARAM_JPEGHEIGHT) * s_num + s_den - 1) // s_den
        # TJSAMP_GRAY indicates a single-component (grayscale) JPEG;
        # everything else we coerce to RGB.
        if tj3Get(handle, TJPARAM_SUBSAMP) == TJSAMP_GRAY:
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
                    f"jpeg decode: out= must be an ndarray, "
                    f"got {type(out).__name__}")
            if out.shape != expected_shape:
                raise ValueError(
                    f"jpeg decode: out= shape {out.shape} does not match "
                    f"expected {expected_shape}")
            if out.dtype != np.uint8:
                raise ValueError(
                    f"jpeg decode: out= dtype must be uint8, got {out.dtype}")
            if not out.flags['C_CONTIGUOUS']:
                raise ValueError("jpeg decode: out= must be C-contiguous")
            out_arr = out
        else:
            out_arr = cnp.PyArray_EMPTY(ndim, shape, cnp.NPY_UINT8, 0)
        with nogil:
            rc = tj3Decompress8(
                handle, &src[0], srcsize,
                <unsigned char*> cnp.PyArray_DATA(out_arr),
                width * channels, pf,
            )
        if rc < 0:
            raise JpegError(
                f'tj3Decompress8: {tj3GetErrorStr(handle).decode()}')
        return out_arr
    finally:
        tj3Destroy(handle)


def read_icc_profile(data) -> bytes | None:
    """Return the embedded ICC profile from a JPEG, or ``None``.

    Parses just the header chain looking for an ICC APP2 marker;
    doesn't touch pixel data.
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        tjhandle handle = NULL
        unsigned char* icc_ptr = NULL
        size_t icc_size = 0
        int rc
        bytes out

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if srcsize < 3:
        return None
    handle = tj3Init(TJINIT_DECOMPRESS)
    if handle == NULL:
        raise JpegError('tj3Init(DECOMPRESS) failed')
    try:
        rc = tj3DecompressHeader(handle, &src[0], srcsize)
        if rc < 0:
            # Not a parseable JPEG — no ICC by definition.
            return None
        rc = tj3GetICCProfile(handle, &icc_ptr, &icc_size)
        if rc < 0 or icc_ptr == NULL or icc_size == 0:
            return None
        try:
            out = PyBytes_FromStringAndSize(
                <char*> icc_ptr, <Py_ssize_t> icc_size)
            return out
        finally:
            tj3Free(icc_ptr)
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
