# opencodecs/codecs/_jpeg.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native JPEG codec via libjpeg-turbo (TurboJPEG API v3).

Encode: uint8 or uint16 arrays shaped (H, W), (H, W, 1), (H, W, 3) or
(H, W, 4). Lossy JPEG holds 8 or 12 bits per sample and lossless JPEG
2 to 16 (ITU-T T.81 B.2.2); four samples per pixel are stored as CMYK
(or YCCK) with the Adobe APP14 marker, unless ``colorspace`` names a
packed RGB order with padding.

Decode: (H, W) for grayscale, (H, W, 3) for RGB/YCbCr and (H, W, 4)
for CMYK/YCCK streams; uint8 for 8-bit (and narrower lossless) streams
and uint16 for 9- to 16-bit ones, the same shapes and dtypes
``imagecodecs.jpeg8_decode`` returns.

The parameter names and meanings follow ``imagecodecs.jpeg8_encode``
and ``imagecodecs.jpeg8_decode``. A parameter this codec cannot honor
raises instead of being dropped.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t

import numpy as np
cimport numpy as cnp

from turbojpeg cimport (
    tjhandle, tj3Init, tj3Destroy, tj3GetErrorStr,
    tj3Set, tj3Get, tj3Free,
    tj3Compress8, tj3Compress12, tj3Compress16,
    tj3DecompressHeader, tj3Decompress8, tj3Decompress12, tj3Decompress16,
    tj3SetICCProfile, tj3GetICCProfile,
    tj3SetScalingFactor, tj3GetScalingFactors, tjscalingfactor,
    TJINIT_COMPRESS, TJINIT_DECOMPRESS,
    TJPF_GRAY, TJPF_RGB, TJPF_BGR, TJPF_RGBX, TJPF_BGRX, TJPF_XBGR,
    TJPF_XRGB, TJPF_RGBA, TJPF_BGRA, TJPF_ABGR, TJPF_ARGB, TJPF_CMYK,
    TJCS_RGB, TJCS_YCbCr, TJCS_GRAY, TJCS_CMYK, TJCS_YCCK,
    TJSAMP_GRAY, TJSAMP_444, TJSAMP_422, TJSAMP_420, TJSAMP_440, TJSAMP_411,
    TJPARAM_QUALITY, TJPARAM_SUBSAMP,
    TJPARAM_JPEGWIDTH, TJPARAM_JPEGHEIGHT,
    TJPARAM_PRECISION, TJPARAM_COLORSPACE, TJPARAM_OPTIMIZE,
    TJPARAM_FASTUPSAMPLE,
    TJPARAM_LOSSLESS, TJPARAM_LOSSLESSPSV,
)

from opencodecs.codecs._jpeg_common import (
    CS_SAMPLES, colorspace_name, subsampling_name, splice_stream,
    frame_header, override_colorspace, decode_plan, DEFAULT_OUTPUT,
    reject_alpha, label_ycbcr,
)

cnp.import_array()


class JpegError(RuntimeError):
    """Raised on JPEG encode/decode failures."""


_TJSAMP = {
    "444": TJSAMP_444,
    "422": TJSAMP_422,
    "420": TJSAMP_420,
    "440": TJSAMP_440,
    "411": TJSAMP_411,
    "gray": TJSAMP_GRAY,
}

# Packed pixel layouts TurboJPEG reads and writes, by colorspace name.
_TJPF = {
    "gray": TJPF_GRAY, "rgb": TJPF_RGB, "bgr": TJPF_BGR,
    "rgbx": TJPF_RGBX, "bgrx": TJPF_BGRX, "xbgr": TJPF_XBGR,
    "xrgb": TJPF_XRGB, "rgba": TJPF_RGBA, "bgra": TJPF_BGRA,
    "abgr": TJPF_ABGR, "argb": TJPF_ARGB, "cmyk": TJPF_CMYK,
}

_TJCS_NAME = {
    TJCS_GRAY: "gray", TJCS_RGB: "rgb", TJCS_YCbCr: "ycbcr",
    TJCS_CMYK: "cmyk", TJCS_YCCK: "ycck",
}


cdef object _error(tjhandle handle, str what):
    return JpegError(f"{what}: {tj3GetErrorStr(handle).decode(errors='replace')}")


def _jpeg_layout(str in_cs, out_cs, sub, bint lossless):
    """Return (TJCS, TJSAMP) for an input layout and the requested JPEG
    colorspace and subsampling, raising on any request TurboJPEG would
    otherwise override without saying so.

    T.81 gives every component its own sampling factors (B.2.2), so an
    RGB or CMYK JPEG can be subsampled as a YCbCr or YCCK one is.
    TurboJPEG lays the factors out the same way for each colorspace: the
    first component (and the fourth, if any) takes the factors the
    subsampling names, and the second and third are sampled once per
    MCU. imagecodecs ignores ``subsampling`` for RGB and CMYK output and
    writes them unsubsampled, which is what the default here does too.
    """
    if sub == "gray" and out_cs not in (None, "gray"):
        raise ValueError(f"JPEG encode: subsampling='gray' contradicts "
                         f"outcolorspace={out_cs!r}")
    if in_cs == "gray":
        if out_cs not in (None, "gray"):
            raise ValueError(
                f"JPEG encode: grayscale input is stored as grayscale; "
                f"outcolorspace={out_cs!r} needs color input")
        # One component has no chroma, so there is nothing to subsample.
        return TJCS_GRAY, TJSAMP_GRAY
    if lossless and sub not in (None, "444"):
        # T.81 Annex H lets every lossless component carry its own
        # sampling factors; TurboJPEG's lossless mode always writes them
        # 1x1 (TJSAMP_444), and imagecodecs ignores the argument there.
        raise ValueError(
            f"JPEG encode: subsampling={sub!r} with lossless=True: "
            f"TurboJPEG's lossless mode does not subsample")
    if in_cs == "cmyk":
        if out_cs is None:
            out_cs = "cmyk"
        if out_cs == "cmyk":
            return TJCS_CMYK, _TJSAMP[sub or "444"]
        if out_cs == "ycck" and not lossless:
            # libjpeg's own YCCK default (jcparam.c) samples Y and K at
            # 2x2, which is 4:2:0; imagecodecs inherits it.
            return TJCS_YCCK, _TJSAMP[sub or "420"]
        raise ValueError(
            f"JPEG encode: cannot store CMYK pixels as {out_cs}"
            + (" in TurboJPEG's lossless mode" if lossless else ""))
    # RGB family (rgb, bgr, rgba, ...): the alpha or padding sample is
    # not stored; the caller named that layout explicitly.
    if out_cs is None:
        out_cs = "rgb" if lossless else ("gray" if sub == "gray" else "ycbcr")
    if lossless:
        # T.81 leaves the stored colorspace to the application, lossless
        # or not; TurboJPEG's lossless mode stores the samples it is
        # given, without color conversion, and imagecodecs ignores
        # outcolorspace there.
        if out_cs != "rgb":
            raise ValueError(
                f"JPEG encode: outcolorspace={out_cs!r} with lossless=True: "
                f"TurboJPEG's lossless mode stores RGB pixels as RGB")
        return TJCS_RGB, TJSAMP_444
    if out_cs == "ycbcr":
        return TJCS_YCbCr, _TJSAMP[sub or "420"]
    if out_cs == "rgb":
        return TJCS_RGB, _TJSAMP[sub or "444"]
    if out_cs == "gray":
        return TJCS_GRAY, TJSAMP_GRAY
    raise ValueError(f"JPEG encode: cannot store RGB pixels as {out_cs}")


def encode(data, level=None, *,
           colorspace=None,
           outcolorspace=None,
           subsampling=None,
           optimize=None,
           smoothing=None,
           lossless=None,
           predictor=None,
           bitspersample=None,
           validate=None,
           iccprofile: bytes | None = None) -> bytes:
    """Encode an image as JPEG.

    Parameters follow ``imagecodecs.jpeg8_encode``.

    ``data`` is uint8 or uint16, shaped (H, W) or (H, W, 1) for
    grayscale, (H, W, 3) for RGB, or (H, W, 4) for CMYK.

    ``level`` is the JPEG quality 0-100 (default 95, matching
    imagecodecs). Not used when ``lossless=True``.

    ``colorspace`` names the input pixels: ``"gray"`` (also
    ``"minisblack"``), ``"rgb"``, the other packed orders (``"bgr"``,
    ``"rgbx"``, ``"bgrx"``, ``"xrgb"``, ``"xbgr"``), ``"ycbcr"``, or
    ``"cmyk"`` (``"separated"``); libjpeg ``J_COLOR_SPACE`` integers are
    accepted too. The default follows the sample count: 1 gray, 3 RGB,
    4 CMYK. The packed orders are libjpeg-turbo's ``JCS_EXT_*``
    layouts: the color samples are read in the order named and stored
    as YCbCr, RGB or grayscale, and the padding sample (X) of a
    four-sample layout is not stored. The orders with alpha
    (``"rgba"``, ``"bgra"``, ``"argb"``, ``"abgr"``) raise, because a
    JPEG has no alpha channel and the alpha samples would be lost; store
    all four samples as CMYK to keep them. imagecodecs raises for
    ``"rgba"`` and for the ``J_COLOR_SPACE`` integers of the other
    orders, and reads their names as an unknown colorspace, storing
    every sample unconverted as its own component. ``"ycbcr"`` pixels
    are stored unconverted as a YCbCr JPEG with a JFIF APP0 marker and
    component ids 1, 2, 3, as imagecodecs stores them, but with the
    luminance quantization and Huffman tables for all three components;
    ``"ycck"`` input raises: TurboJPEG takes RGB, YCbCr, grayscale or
    CMYK pixels.

    ``outcolorspace`` is the colorspace stored in the JPEG: RGB input
    is stored as ``"ycbcr"`` (default), ``"rgb"`` or ``"gray"``; YCbCr
    input as ``"ycbcr"``; CMYK input as ``"cmyk"`` (default) or
    ``"ycck"``. TurboJPEG's lossless mode converts no colors, so with
    ``lossless=True`` RGB is stored as ``"rgb"``, YCbCr as ``"ycbcr"``
    and CMYK as ``"cmyk"``, and another value raises.

    ``subsampling`` is the chroma subsampling of a YCbCr or YCCK JPEG:
    ``"420"`` (default), ``"422"``, ``"444"``, ``"440"``, ``"411"``, or
    the luma sampling factors as a tuple, ``(2, 2)`` for 4:2:0. An RGB
    or CMYK JPEG is not subsampled by default; given a subsampling, R
    (or C and K) keeps the sampling factors it names and the other
    components are subsampled, as T.81 allows. imagecodecs ignores the
    argument there. It has no meaning for a grayscale JPEG. With
    ``lossless=True`` any subsampling other than 4:4:4 raises: T.81
    allows a subsampled lossless JPEG, but TurboJPEG's lossless mode
    does not write one (imagecodecs ignores the argument there too).

    ``optimize`` computes optimal Huffman tables for lossy JPEG.
    12-bit and lossless JPEG always use optimal tables, so
    ``optimize=True`` changes nothing there and ``optimize=False``
    raises. ``smoothing`` other than 0 raises: TurboJPEG has no
    input smoothing filter.

    ``lossless=True`` writes a lossless JPEG (T.81 Annex H, SOF3) with
    predictor ``predictor`` (1-7, default 1) and point transform 0. The
    decoded samples equal the input exactly.

    ``bitspersample`` is the stored precision: 8 (uint8) or 12 (uint16)
    for lossy JPEG, 2-8 (uint8) or 9-16 (uint16) for lossless JPEG. The
    default is 8 for uint8 and 12 for uint16, except that lossless
    uint16 data with values above 4095 is stored at 16 bits, so a
    lossless encode never loses data. Values that do not fit the
    precision raise.

    ``validate`` is accepted for compatibility, as it is by
    ``imagecodecs.jpeg8_encode``, where it has no effect either.

    ``iccprofile`` embeds an ICC color profile in APP2 markers.
    """
    cdef:
        cnp.ndarray arr
        tjhandle handle = NULL
        unsigned char* out_ptr = NULL
        size_t out_size = 0
        int rc
        int pf
        int tjcs
        int subsamp
        int quality
        int height, width
        int bps
        int psv = 0
        int samples
        bint is_lossless = bool(lossless)
        bytes out
        const unsigned char[::1] icc_view
        void* src_ptr

    a = data if isinstance(data, np.ndarray) else np.asarray(data)
    if a.dtype != np.uint8 and a.dtype != np.uint16:
        raise JpegError(f'JPEG encode: unsupported dtype {a.dtype}; '
                        'expected uint8 or uint16')
    if a.ndim == 3 and a.shape[2] == 1:
        # A trailing singleton channel is grayscale, as in the PNG
        # codec and in imagecodecs.
        a = a[:, :, 0]
    if a.ndim == 2:
        samples = 1
    elif a.ndim == 3 and a.shape[2] in (3, 4):
        samples = <int> a.shape[2]
    else:
        raise JpegError(
            f'JPEG encode: unsupported shape {a.shape}; expected '
            '(H, W) or (H, W, 1) grayscale, (H, W, 3) RGB or (H, W, 4) CMYK')
    arr = np.ascontiguousarray(a)
    height = <int> arr.shape[0]
    width = <int> arr.shape[1]

    # -- pixel layout and stored colorspace --------------------------------
    in_cs = colorspace_name(colorspace, "colorspace")
    if in_cs is None:
        in_cs = {1: "gray", 3: "rgb", 4: "cmyk"}[samples]
    if in_cs not in _TJPF and in_cs != "ycbcr":
        raise NotImplementedError(
            f"JPEG encode: colorspace={colorspace!r}: TurboJPEG takes RGB, "
            "YCbCr, grayscale or CMYK pixels")
    if CS_SAMPLES[in_cs] != samples:
        raise ValueError(
            f"JPEG encode: colorspace={colorspace!r} has "
            f"{CS_SAMPLES[in_cs]} samples per pixel; the array has {samples}")
    reject_alpha(in_cs, colorspace, "JPEG encode")
    family = in_cs if in_cs in ("gray", "cmyk") else "rgb"
    if is_lossless and family == "rgb" and in_cs not in ("rgb", "ycbcr"):
        raise ValueError(
            f"JPEG encode: lossless encode takes grayscale, RGB, YCbCr or "
            f"CMYK pixels, not colorspace={colorspace!r}")
    out_name = colorspace_name(outcolorspace, "outcolorspace")
    sub_name = subsampling_name(subsampling)
    if in_cs == "ycbcr":
        # YCbCr samples are stored as they are. TurboJPEG converts only
        # from RGB, so the samples go in as RGB with RGB storage, which
        # copies them through, at the sampling factors a YCbCr JPEG has
        # (Y full, Cb and Cr subsampled); label_ycbcr then gives the
        # components the ids 1, 2, 3 and the JFIF marker of a YCbCr
        # JPEG.
        if out_name not in (None, "ycbcr"):
            raise NotImplementedError(
                f"JPEG encode: colorspace={colorspace!r} with "
                f"outcolorspace={outcolorspace!r}: TurboJPEG converts no "
                f"colors from YCbCr pixels, which are stored as YCbCr")
        if sub_name == "gray":
            raise ValueError(
                "JPEG encode: subsampling='gray' contradicts YCbCr pixels")
        if sub_name is None and not is_lossless:
            sub_name = "420"
        out_name = "rgb"
        pf = TJPF_RGB
    else:
        pf = _TJPF[in_cs]
    tjcs, subsamp = _jpeg_layout(family, out_name, sub_name, is_lossless)

    # -- lossless predictor --------------------------------------------------
    if predictor is not None and not is_lossless:
        raise ValueError("JPEG encode: predictor applies only with "
                         "lossless=True")
    if is_lossless:
        psv = 1 if predictor is None else int(predictor)
        if not 1 <= psv <= 7:
            raise ValueError(
                f"JPEG encode: predictor must be 1 to 7, got {predictor!r}")

    # -- precision -----------------------------------------------------------
    vmax = -1
    if bitspersample is not None:
        bps = int(bitspersample)
        if not 2 <= bps <= 16 or (bps + 7) // 8 != arr.dtype.itemsize:
            raise ValueError(
                f"JPEG encode: bitspersample={bitspersample!r} does not "
                f"fit {arr.dtype} (uint8 holds 2-8 bits, uint16 9-16)")
    elif arr.dtype == np.uint8:
        bps = 8
    else:
        bps = 12
        if is_lossless and arr.size:
            vmax = int(arr.max())
            if vmax > 4095:
                bps = 16
    if not is_lossless and bps != 8 and bps != 12:
        raise ValueError(
            f"JPEG encode: lossy JPEG stores 8 or 12 bits per sample, "
            f"not {bps}; pass lossless=True for other precisions")
    if bps < 8 * arr.dtype.itemsize and arr.size:
        if vmax < 0:
            vmax = int(arr.max())
        if vmax >= (1 << bps):
            raise ValueError(
                f"JPEG encode: the largest value, {vmax}, does not fit in "
                f"{bps} bits per sample"
                + ("; pass lossless=True with bitspersample=16"
                   if not is_lossless else ""))

    # -- other options -------------------------------------------------------
    if smoothing is not None and smoothing is not False and smoothing != 0:
        raise NotImplementedError(
            f"JPEG encode: smoothing={smoothing!r}: TurboJPEG has no input "
            "smoothing filter")
    if optimize is not None:
        if is_lossless and not optimize:
            raise ValueError(
                "JPEG encode: TurboJPEG's lossless mode always computes "
                "optimal Huffman tables; optimize=False cannot be honored")
        if bps == 12 and not optimize:
            raise ValueError(
                "JPEG encode: 12-bit JPEG always uses optimized Huffman "
                "tables; optimize=False cannot be honored")

    quality = 95 if level is None else int(level)
    if quality < 1: quality = 1
    if quality > 100: quality = 100

    handle = tj3Init(TJINIT_COMPRESS)
    if handle == NULL:
        raise JpegError('tj3Init(COMPRESS) failed')
    try:
        # Lossless mode is set before quality and subsampling: tj3Set
        # validates subsampling against the current mode.
        if is_lossless:
            if tj3Set(handle, TJPARAM_LOSSLESS, 1) < 0:
                raise _error(handle, 'tj3Set(LOSSLESS)')
            if tj3Set(handle, TJPARAM_LOSSLESSPSV, psv) < 0:
                raise _error(handle, 'tj3Set(LOSSLESSPSV)')
            if bps != 8 and bps != 12 and bps != 16:
                if tj3Set(handle, TJPARAM_PRECISION, bps) < 0:
                    raise _error(handle, 'tj3Set(PRECISION)')
        if tj3Set(handle, TJPARAM_QUALITY, quality) < 0:
            raise _error(handle, 'tj3Set(QUALITY)')
        if tj3Set(handle, TJPARAM_SUBSAMP, subsamp) < 0:
            raise _error(handle, 'tj3Set(SUBSAMP)')
        if tj3Set(handle, TJPARAM_COLORSPACE, tjcs) < 0:
            raise _error(handle, 'tj3Set(COLORSPACE)')
        if optimize is not None and bps == 8 and not is_lossless:
            if tj3Set(handle, TJPARAM_OPTIMIZE, 1 if optimize else 0) < 0:
                raise _error(handle, 'tj3Set(OPTIMIZE)')
        if iccprofile is not None and len(iccprofile) > 0:
            icc_view = iccprofile
            rc = tj3SetICCProfile(
                handle, &icc_view[0], <size_t> icc_view.shape[0])
            if rc < 0:
                raise _error(handle, 'tj3SetICCProfile')
        src_ptr = cnp.PyArray_DATA(arr)
        # pitch 0: rows are packed, width * samples apart.
        with nogil:
            if bps <= 8:
                rc = tj3Compress8(
                    handle, <const unsigned char*> src_ptr,
                    width, 0, height, pf, &out_ptr, &out_size)
            elif bps <= 12:
                rc = tj3Compress12(
                    handle, <const short*> src_ptr,
                    width, 0, height, pf, &out_ptr, &out_size)
            else:
                rc = tj3Compress16(
                    handle, <const unsigned short*> src_ptr,
                    width, 0, height, pf, &out_ptr, &out_size)
        if rc < 0:
            raise _error(handle, f'tj3Compress{8 if bps <= 8 else 12 if bps <= 12 else 16}')
        try:
            out = PyBytes_FromStringAndSize(
                <char*> out_ptr, <Py_ssize_t> out_size)
        finally:
            tj3Free(out_ptr)
    finally:
        tj3Destroy(handle)
    if in_cs == "ycbcr":
        out = label_ycbcr(out)
    if bps != 8 and bps != 12 and bps != 16:
        # libjpeg-turbo before 3.1 has no lossless precisions other
        # than 8, 12 and 16 and may write one of those instead.
        fh = frame_header(out)
        if fh is None or fh.precision != bps:
            raise JpegError(
                f"JPEG encode: this libjpeg-turbo build cannot write "
                f"{bps}-bit lossless JPEG (3.1 or newer is required)")
    return out


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


cdef class DecoderContext:
    """Explicit owned decoder handle, serialized across callers."""
    cdef tjhandle handle
    cdef object lock

    def __cinit__(self):
        import threading
        self.lock = threading.RLock()
        self.handle = tj3Init(TJINIT_DECOMPRESS)
        if self.handle == NULL:
            raise JpegError("decoder context initialization failed")

    def decode(self, data, **options):
        with self.lock:
            if self.handle == NULL:
                raise ValueError("decoder context is closed")
            return decode(data, _context=self, **options)

    def close(self):
        with self.lock:
            if self.handle != NULL:
                tj3Destroy(self.handle)
                self.handle = NULL

    def __dealloc__(self):
        if self.handle != NULL:
            tj3Destroy(self.handle)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def decode(data, *, tables=None, header=None, colorspace=None,
           outcolorspace=None, fancyupsampling=None, shape=None,
           bitspersample=None, out=None, scale=None,
           scale_num=None, scale_denom=None,
           DecoderContext _context=None) -> np.ndarray:
    """Decode a JPEG stream.

    Parameters follow ``imagecodecs.jpeg_decode``; ``scale`` is an
    opencodecs extension.

    Parameters
    ----------
    data : bytes-like
        Encoded JPEG stream.
    tables : bytes-like, optional
        Abbreviated table-specification stream (SOI, DQT/DHT, EOI) for
        an abbreviated image stream, such as TIFF's JPEGTables.
    header : bytes-like, optional
        Leading markers stored apart from the image data; the stream
        decoded is ``header + data + EOI``.
    colorspace : str or int, optional
        The colorspace the components are stored in, when the caller
        knows it better than the stream's markers say: ``"rgb"`` or
        ``"ycbcr"`` for three components, ``"cmyk"`` or ``"ycck"`` for
        four. Default: what the JFIF/Adobe markers declare. As in
        libjpeg and imagecodecs, giving ``colorspace`` without
        ``outcolorspace`` outputs that colorspace unconverted.
    outcolorspace : str or int, optional
        Output pixels: ``"gray"``, ``"rgb"``, ``"rgba"`` (alpha 255),
        ``"cmyk"``, another packed order (``"bgr"``, ``"rgbx"``, ...),
        or the stored ``"ycbcr"``/``"ycck"`` components unconverted.
        The packed orders are libjpeg-turbo's ``JCS_EXT_*`` layouts, so
        ``"bgr"`` returns blue, green, red: what imagecodecs returns for
        the ``J_COLOR_SPACE`` integer (8 for BGR). imagecodecs reads the
        names other than ``"rgba"`` as an unknown colorspace and returns
        its default, RGB.
        Default: grayscale for a grayscale stream, RGB for an RGB or
        YCbCr stream, CMYK for a CMYK or YCCK stream. For either
        argument, a TIFF photometric name that is no JPEG colorspace
        (``"CFA"``, ``"LINEAR_RAW"``, ``"CIELAB"``, ...), which tifffile
        passes, means the default, as in imagecodecs; any other unknown
        name raises ``ValueError``.
    fancyupsampling : bool, optional
        ``False`` replicates chroma samples instead of interpolating
        them (libjpeg-turbo's fast upsampling). Default ``True``.
    shape : tuple, optional
        Image (height, width, ...). As in imagecodecs it is needed only
        for images over 65500 pixels per side, which libjpeg-turbo
        cannot decode, so such a shape raises ``NotImplementedError``;
        otherwise the frame header gives the size.
    bitspersample : int, optional
        The container's sample width; raises if the decoded dtype
        cannot hold it (8 for uint8, 9-16 for uint16).
    out : ndarray, optional
        Preallocated C-contiguous output of the exact shape and dtype.
    scale : int / float / (num, denom), optional
        Decode-time scaling via libjpeg-turbo's DCT-domain shortcut: an
        integer ``N`` means 1/N, a float snaps to the closest supported
        factor, a tuple is explicit. Lossless JPEG has no DCT and raises
        for any factor other than 1.
    scale_num, scale_denom : int, optional
        The ratio spelled out; used when ``scale`` is ``None``.

    The output is uint8 for streams of 8 or fewer bits per sample and
    uint16 above that, shaped ``(TJSCALED(H), TJSCALED(W)[, samples])``
    where ``TJSCALED(d, (n, m)) = (d * n + m - 1) // m``.
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        tjhandle handle = NULL
        int rc
        int width, height
        int full_width, full_height
        int precision
        int pf
        int channels
        cnp.ndarray out_arr
        cnp.npy_intp shape_c[3]
        int ndim
        int typenum
        tuple expected_shape
        tjscalingfactor factor
        int s_num, s_den
        bint is_lossless
        void* dst_ptr

    # The common call, a whole stream and no options, skips the Python
    # helpers: a 16x16 tile decodes in a few microseconds, so their
    # calls would be a measurable share of it.
    if tables is None and header is None:
        stream = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    else:
        stream = splice_stream(data, tables, header)
    src = stream
    srcsize = <size_t> src.shape[0]
    if srcsize < 3:
        raise JpegError('input too short to be JPEG')

    want_cs = (None if colorspace is None
               else colorspace_name(colorspace, "colorspace", decode=True))
    out_cs = (None if outcolorspace is None
              else colorspace_name(outcolorspace, "outcolorspace", decode=True))
    if shape is not None and max(int(v) for v in tuple(shape)[:2]) >= 65500:
        # imagecodecs uses shape only for this case: a patched libjpeg
        # can decode images whose sides exceed the 65500 pixels stock
        # libjpeg-turbo allows. Below that the frame header is
        # authoritative and shape is not needed.
        raise NotImplementedError(
            f"jpeg decode: shape={tuple(shape)!r}: libjpeg-turbo decodes "
            "images up to 65500 pixels per side")
    if scale is None and scale_num is None and scale_denom is None:
        s_num = s_den = 1
    else:
        s_num, s_den = _resolve_scale(scale, scale_num, scale_denom)

    handle = tj3Init(TJINIT_DECOMPRESS) if _context is None else _context.handle
    if handle == NULL:
        raise JpegError('tj3Init(DECOMPRESS) failed')
    try:
        rc = tj3DecompressHeader(handle, &src[0], srcsize)
        if rc < 0:
            raise _error(handle, 'tj3DecompressHeader')
        stream_cs = _TJCS_NAME.get(tj3Get(handle, TJPARAM_COLORSPACE))
        if want_cs is None and out_cs is None and stream_cs is not None:
            declared = stream_cs
            out_cs = DEFAULT_OUTPUT[stream_cs]
        else:
            declared, out_cs = decode_plan(stream_cs, want_cs, out_cs, _TJPF)
        if declared != stream_cs:
            # The caller (a container such as TIFF) knows the colorspace
            # better than the stream's markers do, or wants the stored
            # components unconverted: restate the colorspace in an Adobe
            # marker, which is how libjpeg learns it.
            stream = override_colorspace(stream, declared)
            src = stream
            srcsize = <size_t> src.shape[0]
            rc = tj3DecompressHeader(handle, &src[0], srcsize)
            if rc < 0:
                raise _error(handle, 'tj3DecompressHeader')
            if _TJCS_NAME.get(tj3Get(handle, TJPARAM_COLORSPACE)) != declared:
                raise NotImplementedError(
                    f"jpeg decode: cannot decode this stream as "
                    f"colorspace={colorspace!r}")
        full_width = tj3Get(handle, TJPARAM_JPEGWIDTH)
        full_height = tj3Get(handle, TJPARAM_JPEGHEIGHT)
        precision = tj3Get(handle, TJPARAM_PRECISION)
        is_lossless = tj3Get(handle, TJPARAM_LOSSLESS) == 1
        pf = _TJPF[out_cs]
        channels = CS_SAMPLES[out_cs]

        if precision <= 8:
            typenum = cnp.NPY_UINT8
            out_dtype = np.uint8
        else:
            typenum = cnp.NPY_UINT16
            out_dtype = np.uint16
        if bitspersample is not None and \
                (int(bitspersample) + 7) // 8 != np.dtype(out_dtype).itemsize:
            raise ValueError(
                f"jpeg decode: bitspersample={bitspersample!r} does not "
                f"match the stream's {precision}-bit samples")
        if is_lossless and (s_num, s_den) != (1, 1):
            # Lossless JPEG has no DCT to scale, and libjpeg-turbo
            # decodes it at full size whatever factor is set.
            raise ValueError(
                f"jpeg decode: lossless JPEG cannot be decoded at scale "
                f"{s_num}/{s_den}")

        # Apply DCT-domain decode scaling if the caller asked for it.
        # tj3SetScalingFactor rejects unsupported ratios; the error
        # surfaces with libjpeg-turbo's own diagnostic. A context's
        # handle keeps its settings, so they are reset on every call.
        if _context is not None or (s_num, s_den) != (1, 1):
            factor.num = s_num
            factor.denom = s_den
            if tj3SetScalingFactor(handle, factor) < 0:
                raise JpegError(
                    f'tj3SetScalingFactor({s_num}/{s_den}): '
                    f'{tj3GetErrorStr(handle).decode(errors="replace")}. '
                    f'Supported factors: {supported_scaling_factors()}')
        if _context is not None or fancyupsampling is not None:
            if tj3Set(handle, TJPARAM_FASTUPSAMPLE,
                      0 if fancyupsampling is None or fancyupsampling
                      else 1) < 0:
                raise _error(handle, 'tj3Set(FASTUPSAMPLE)')
        # TJSCALED(d, factor) = (d * num + denom - 1) // denom.
        width = (full_width * s_num + s_den - 1) // s_den
        height = (full_height * s_num + s_den - 1) // s_den

        shape_c[0] = height
        shape_c[1] = width
        if channels == 1:
            ndim = 2
            expected_shape = (height, width)
        else:
            ndim = 3
            shape_c[2] = channels
            expected_shape = (height, width, channels)
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError(
                    f"jpeg decode: out= must be an ndarray, "
                    f"got {type(out).__name__}")
            if out.shape != expected_shape:
                raise ValueError(
                    f"jpeg decode: out= shape {out.shape} does not match "
                    f"expected {expected_shape}")
            if out.dtype != out_dtype:
                raise ValueError(
                    f"jpeg decode: out= dtype must be "
                    f"{np.dtype(out_dtype).name}, got {out.dtype}")
            if not out.flags['C_CONTIGUOUS'] or not out.flags.writeable:
                raise ValueError("jpeg decode: out= must be C-contiguous")
            out_arr = out
        else:
            out_arr = cnp.PyArray_EMPTY(ndim, shape_c, typenum, 0)
        dst_ptr = cnp.PyArray_DATA(out_arr)
        # pitch 0: rows packed, width * samples apart.
        with nogil:
            if precision <= 8:
                rc = tj3Decompress8(
                    handle, &src[0], srcsize,
                    <unsigned char*> dst_ptr, 0, pf)
            elif precision <= 12:
                rc = tj3Decompress12(
                    handle, &src[0], srcsize,
                    <short*> dst_ptr, 0, pf)
            else:
                rc = tj3Decompress16(
                    handle, &src[0], srcsize,
                    <unsigned short*> dst_ptr, 0, pf)
        if rc < 0:
            raise _error(
                handle, f'tj3Decompress'
                f'{8 if precision <= 8 else 12 if precision <= 12 else 16}')
        return out_arr
    finally:
        if _context is None:
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
