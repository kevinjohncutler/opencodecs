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
bitstreams are interoperable JPEGs). Streams MozJPEG's TurboJPEG v2
API cannot decode (12-bit DCT, lossless) go to the ``jpeg`` codec.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t

import numpy as np
cimport numpy as cnp

from mozjpeg cimport (
    tjhandle, tjInitCompress, tjInitDecompress, tjDestroy,
    tjGetErrorStr2, tjCompress2, tjDecompressHeader3, tjDecompress2,
    tjFree, tjscalingfactor, tjGetScalingFactors,
    TJPF_GRAY, TJPF_RGB, TJPF_BGR, TJPF_RGBX, TJPF_BGRX, TJPF_XBGR,
    TJPF_XRGB, TJPF_RGBA, TJPF_BGRA, TJPF_ABGR, TJPF_ARGB, TJPF_CMYK,
    TJCS_RGB, TJCS_YCbCr, TJCS_GRAY, TJCS_CMYK, TJCS_YCCK,
    TJSAMP_GRAY, TJSAMP_444, TJSAMP_422, TJSAMP_420, TJSAMP_440, TJSAMP_411,
    TJFLAG_FASTUPSAMPLE, TJFLAG_ACCURATEDCT, TJFLAG_PROGRESSIVE,
)

from opencodecs.codecs._jpeg_common import (
    CS_SAMPLES, colorspace_name, subsampling_name, splice_stream,
    override_colorspace, decode_plan, DEFAULT_OUTPUT, reject_alpha,
)

cnp.import_array()


class MozJpegError(RuntimeError):
    """Raised on MozJPEG encode/decode failures."""


_TJSAMP = {
    "444": TJSAMP_444,
    "422": TJSAMP_422,
    "420": TJSAMP_420,
    "440": TJSAMP_440,
    "411": TJSAMP_411,
    "gray": TJSAMP_GRAY,
}

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


def encode(data, level=None, *,
           colorspace=None,
           outcolorspace=None,
           subsampling=None,
           optimize=None,
           smoothing=None,
           notrellis=None,
           quanttable=None,
           progressive=None) -> bytes:
    """Encode a uint8 image as JPEG via MozJPEG.

    Parameters follow ``imagecodecs.mozjpeg_encode``. MozJPEG exposes
    only the TurboJPEG v2 API, which fixes several of them; a value it
    cannot honor raises ``NotImplementedError`` instead of being
    dropped.

    Parameters
    ----------
    data
        uint8 shaped (H, W) or (H, W, 1) for grayscale, (H, W, 3) RGB.
    level
        Quality 0-100 (default 95, matching imagecodecs).
    colorspace
        The input pixels: ``"gray"``, ``"rgb"``, ``"bgr"``, or a
        four-sample RGB order with padding (``"rgbx"``, ``"bgrx"``,
        ``"xrgb"``, ``"xbgr"``), whose padding sample is not stored. The
        orders with alpha (``"rgba"``, ...) raise: JPEG has no alpha
        channel. CMYK input is not supported by MozJPEG's TurboJPEG API.
    outcolorspace
        ``"ycbcr"`` (default for color) or ``"gray"``.
    subsampling
        ``"420"`` (default), ``"422"``, ``"444"``, ``"440"``, ``"411"``
        or a luma sampling tuple such as ``(2, 2)``. Not used for a
        grayscale JPEG.
    optimize
        MozJPEG always computes optimal Huffman tables; ``False`` raises.
    smoothing, notrellis, quanttable
        MozJPEG's defaults (no smoothing, trellis quantization, its
        default quantization tables) are fixed through TurboJPEG; any
        other value raises.
    progressive
        Default ``True``: MozJPEG's progressive encode is the source of
        most of its size advantage over libjpeg-turbo. ``False`` raises
        ``NotImplementedError``: MozJPEG's TurboJPEG API always writes
        progressive scans (the ``jpeg`` codec writes sequential JPEG).
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
        int samples
        bytes out

    a = data if isinstance(data, np.ndarray) else np.asarray(data)
    if a.dtype != np.uint8:
        raise MozJpegError(
            f'MozJPEG encode: unsupported dtype {a.dtype}; MozJPEG '
            'writes 8-bit JPEG only')
    if a.ndim == 3 and a.shape[2] == 1:
        # A trailing singleton channel is grayscale, as in the jpeg and
        # PNG codecs and in imagecodecs.
        a = a[:, :, 0]
    if a.ndim == 2:
        samples = 1
    elif a.ndim == 3 and a.shape[2] in (3, 4):
        samples = <int> a.shape[2]
    else:
        raise MozJpegError(
            f'MozJPEG encode: unsupported shape {a.shape}; expected '
            '(H, W) or (H, W, 1) grayscale or (H, W, 3) RGB')
    arr = np.ascontiguousarray(a)
    height = <int> arr.shape[0]
    width = <int> arr.shape[1]

    in_cs = colorspace_name(colorspace, "colorspace")
    if in_cs is None:
        in_cs = {1: "gray", 3: "rgb", 4: "cmyk"}[samples]
    if in_cs not in _TJPF or in_cs == "cmyk":
        raise NotImplementedError(
            f"MozJPEG encode: {in_cs} input is not supported through "
            "MozJPEG's TurboJPEG API"
            + ("; the 'jpeg' codec writes it" if in_cs in ("ycbcr", "cmyk")
               else ""))
    if CS_SAMPLES[in_cs] != samples:
        raise ValueError(
            f"MozJPEG encode: colorspace={colorspace!r} has "
            f"{CS_SAMPLES[in_cs]} samples per pixel; the array has {samples}")
    reject_alpha(in_cs, colorspace, "MozJPEG encode")
    pf = _TJPF[in_cs]
    out_cs = colorspace_name(outcolorspace, "outcolorspace")
    sub = subsampling_name(subsampling)
    if in_cs == "gray":
        if out_cs not in (None, "gray"):
            raise ValueError(
                f"MozJPEG encode: grayscale input is stored as grayscale; "
                f"outcolorspace={out_cs!r} needs color input")
        subsamp = TJSAMP_GRAY
    elif out_cs == "gray":
        subsamp = TJSAMP_GRAY
    elif out_cs in (None, "ycbcr"):
        if out_cs == "ycbcr" and sub == "gray":
            raise ValueError("MozJPEG encode: subsampling='gray' "
                             "contradicts outcolorspace='ycbcr'")
        subsamp = _TJSAMP[sub or "420"]
    else:
        raise NotImplementedError(
            f"MozJPEG encode: outcolorspace={outcolorspace!r}: MozJPEG's "
            "TurboJPEG API stores color as YCbCr")

    if optimize is not None and not optimize:
        raise NotImplementedError(
            "MozJPEG encode: optimize=False: MozJPEG always computes "
            "optimal Huffman tables")
    if smoothing is not None and smoothing is not False and smoothing != 0:
        raise NotImplementedError(
            f"MozJPEG encode: smoothing={smoothing!r}: not available "
            "through MozJPEG's TurboJPEG API")
    if notrellis:
        raise NotImplementedError(
            "MozJPEG encode: notrellis=True: trellis quantization cannot "
            "be disabled through MozJPEG's TurboJPEG API")
    if quanttable is not None:
        raise NotImplementedError(
            f"MozJPEG encode: quanttable={quanttable!r}: the quantization "
            "tables cannot be chosen through MozJPEG's TurboJPEG API")
    if progressive is not None and not progressive:
        # MozJPEG's default compression profile writes progressive scans
        # whatever TurboJPEG's flags say; the v2 API has no way to ask
        # for a sequential (baseline) JPEG.
        raise NotImplementedError(
            "MozJPEG encode: progressive=False: MozJPEG's TurboJPEG API "
            "always writes a progressive JPEG; the 'jpeg' codec writes "
            "sequential JPEG")

    # Default quality 95 -- matches imagecodecs.mozjpeg_encode and our
    # own _jpeg.pyx default, per the Pareto-better-or-equal policy in
    # docs/codec_api_conventions.md "Default settings".
    quality = 95 if level is None else int(level)
    if quality < 1: quality = 1
    if quality > 100: quality = 100

    # Progressive is where MozJPEG's main quality/size advantage comes
    # from; accurate DCT keeps decoded pixels closer to source.
    flags = TJFLAG_ACCURATEDCT | TJFLAG_PROGRESSIVE
    pitch = width * samples

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


cdef int _frame_precision(const uint8_t[::1] src, bint* lossless) noexcept:
    """Return P from the first frame header, or -1 when there is none.

    The C form of ``_jpeg_common.frame_header``. Every decode reads the
    frame header to hand 12-bit and lossless streams to the ``jpeg``
    codec, so the walk is kept out of Python. It walks the marker segments from SOI (T.81 B.1.1.4: every marker but
    SOI, EOI, RSTn and TEM carries a two-byte length) and stops at the
    first SOFn or at SOS, never touching entropy-coded data. Sets
    ``lossless`` for SOF3, SOF7, SOF11 and SOF15.
    """
    cdef Py_ssize_t n = src.shape[0]
    cdef Py_ssize_t i = 2
    cdef int marker
    lossless[0] = False
    if n < 4 or src[0] != 0xFF or src[1] != 0xD8:
        return -1
    while i + 3 < n:
        if src[i] != 0xFF:
            return -1
        marker = src[i + 1]
        if marker == 0xFF:          # fill byte before a marker
            i += 1
            continue
        if marker == 0xD8 or marker == 0x01 or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xD9 or marker == 0xDA:  # EOI or SOS before a frame
            return -1
        if 0xC0 <= marker <= 0xCF and marker != 0xC4 and marker != 0xC8 \
                and marker != 0xCC:
            if i + 9 >= n:
                return -1
            lossless[0] = (marker == 0xC3 or marker == 0xC7
                           or marker == 0xCB or marker == 0xCF)
            return src[i + 4]
        i += 2 + ((<Py_ssize_t> src[i + 2] << 8) | src[i + 3])
    return -1


cdef class DecoderContext:
    """Explicit owned decoder handle, serialized across callers."""
    cdef tjhandle handle
    cdef object lock

    def __cinit__(self):
        import threading
        self.lock = threading.RLock()
        self.handle = tjInitDecompress()
        if self.handle == NULL:
            raise MozJpegError("decoder context initialization failed")

    def decode(self, data, **options):
        with self.lock:
            if self.handle == NULL:
                raise ValueError("decoder context is closed")
            return decode(data, _context=self, **options)

    def close(self):
        with self.lock:
            if self.handle != NULL:
                tjDestroy(self.handle)
                self.handle = NULL

    def __dealloc__(self):
        if self.handle != NULL:
            tjDestroy(self.handle)

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

    Decode is standard JPEG: the output does not depend on which
    library encoded the input. The parameters are those of
    :func:`opencodecs.codecs._jpeg.decode` (``imagecodecs.jpeg_decode``
    plus ``scale``), with the same meanings and output shapes.

    MozJPEG's TurboJPEG v2 API decodes 8-bit DCT JPEG only. Streams with
    other precisions (12-bit DCT) or the lossless process (SOF3) are
    decoded by the ``jpeg`` codec (libjpeg-turbo 3), so this function
    accepts every stream that one does.

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
        int width, height, subsamp, jcs
        int full_width, full_height
        int s_num, s_den
        int pf
        int flags = 0
        int channels
        cnp.ndarray out_arr
        cnp.npy_intp shape_c[3]
        int ndim
        tuple expected_shape
        unsigned char* dst_ptr
        int pitch
        int precision
        bint sof_lossless = False

    # The common call, a whole stream and no options, skips the Python
    # helpers: a 16x16 tile decodes in a few microseconds, so their
    # calls would be a measurable share of it.
    if tables is None and header is None:
        stream = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    else:
        stream = splice_stream(data, tables, header)
    src = stream
    precision = _frame_precision(src, &sof_lossless)
    if precision != -1 and (precision != 8 or sof_lossless):
        try:
            from opencodecs.codecs import _jpeg
        except ImportError as exc:
            raise MozJpegError(
                f"MozJPEG decodes 8-bit DCT JPEG only; this "
                f"{precision}-bit{' lossless' if sof_lossless else ''} "
                f"stream needs the 'jpeg' codec, which is not built") from exc
        return _jpeg.decode(
            stream, colorspace=colorspace, outcolorspace=outcolorspace,
            fancyupsampling=fancyupsampling, shape=shape,
            bitspersample=bitspersample, out=out, scale=scale,
            scale_num=scale_num, scale_denom=scale_denom)

    want_cs = (None if colorspace is None
               else colorspace_name(colorspace, "colorspace", decode=True))
    out_cs = (None if outcolorspace is None
              else colorspace_name(outcolorspace, "outcolorspace", decode=True))
    if shape is not None and max(int(v) for v in tuple(shape)[:2]) >= 65500:
        raise NotImplementedError(
            f"mozjpeg decode: shape={tuple(shape)!r}: libjpeg-turbo decodes "
            "images up to 65500 pixels per side")
    if scale is None and scale_num is None and scale_denom is None:
        s_num = s_den = 1
    else:
        s_num, s_den = _resolve_scale(scale, scale_num, scale_denom)
    if s_num != 1 or s_den != 1:
        if (s_num, s_den) not in supported_scaling_factors():
            raise ValueError(
                f"mozjpeg decode: scaling factor {s_num}/{s_den} is not "
                f"supported; available: {supported_scaling_factors()}")
    if fancyupsampling is not None and not fancyupsampling:
        flags |= TJFLAG_FASTUPSAMPLE

    srcsize = <unsigned long> src.shape[0]
    if srcsize < 3:
        raise MozJpegError('input too short to be JPEG')

    handle = tjInitDecompress() if _context is None else _context.handle
    if handle == NULL:
        raise MozJpegError('tjInitDecompress failed')
    try:
        with nogil:
            rc = tjDecompressHeader3(
                handle, &src[0], srcsize,
                &full_width, &full_height, &subsamp, &jcs)
        if rc < 0:
            err = tjGetErrorStr2(handle).decode('ascii', errors='replace')
            raise MozJpegError(f'tjDecompressHeader3: {err}')
        stream_cs = _TJCS_NAME.get(jcs)
        if want_cs is None and out_cs is None and stream_cs is not None:
            declared = stream_cs
            out_cs = DEFAULT_OUTPUT[stream_cs]
        else:
            declared, out_cs = decode_plan(stream_cs, want_cs, out_cs, _TJPF)
        if declared != stream_cs:
            # Restate the colorspace in an Adobe marker, as the jpeg
            # codec does.
            stream = override_colorspace(stream, declared)
            src = stream
            srcsize = <unsigned long> src.shape[0]
            with nogil:
                rc = tjDecompressHeader3(
                    handle, &src[0], srcsize,
                    &full_width, &full_height, &subsamp, &jcs)
            if rc < 0:
                err = tjGetErrorStr2(handle).decode('ascii', errors='replace')
                raise MozJpegError(f'tjDecompressHeader3: {err}')
            if _TJCS_NAME.get(jcs) != declared:
                raise NotImplementedError(
                    f"mozjpeg decode: cannot decode this stream as "
                    f"colorspace={colorspace!r}")
        pf = _TJPF[out_cs]
        channels = CS_SAMPLES[out_cs]
        if bitspersample is not None and (int(bitspersample) + 7) // 8 != 1:
            raise ValueError(
                f"mozjpeg decode: bitspersample={bitspersample!r} does not "
                f"match the stream's 8-bit samples")

        # TurboJPEG v2 has no SetScalingFactor: tjDecompress2 picks the
        # scaling factor from the destination size it is given, so the
        # scaled extent has to be computed here. TJSCALED(d, f) is
        # (d * num + denom - 1) // denom.
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
            if not out.flags['C_CONTIGUOUS'] or not out.flags.writeable:
                raise ValueError("mozjpeg decode: out= must be C-contiguous")
            out_arr = out
        else:
            out_arr = cnp.PyArray_EMPTY(ndim, shape_c, cnp.NPY_UINT8, 0)
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
                width, pitch, height, pf, flags)
        if rc < 0:
            err = tjGetErrorStr2(handle).decode('ascii', errors='replace')
            raise MozJpegError(f'tjDecompress2: {err}')
        return out_arr
    finally:
        if _context is None:
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
