# opencodecs/codecs/_webp.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native WebP codec via libwebp.

Encode: 2D uint8 (gray, expanded to RGB), (H, W, 3) uint8 RGB,
        (H, W, 4) uint8 RGBA. Lossless and exact by default, as
        ``imagecodecs.webp_encode``; ``lossless=False`` for lossy.
Decode: returns (H, W, 3) RGB or (H, W, 4) RGBA; ``hasalpha`` forces
        one or the other, as in ``imagecodecs.webp_decode``.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdint cimport uint8_t

import numpy as np
cimport numpy as cnp

from webp cimport (
    oc_webp_encode, oc_webp_free, WebPGetEncoderVersion,
    WebPGetFeatures, WebPBitstreamFeatures,
    WebPDecodeRGBInto, WebPDecodeRGBAInto,
    VP8_STATUS_OK,
    WebPData, WebPAnimInfo, WebPAnimDecoder, WebPAnimDecoderOptions,
    WebPAnimDecoderOptionsInit, WebPAnimDecoderNew, WebPAnimDecoderGetInfo,
    WebPAnimDecoderGetNext, WebPAnimDecoderReset, WebPAnimDecoderDelete,
    MODE_RGBA,
)
from libc.string cimport memcpy

cnp.import_array()


class WebpError(RuntimeError):
    """Raised on WebP encode/decode failures."""


def version() -> str:
    """Return the linked libwebp encoder version, e.g. ``'libwebp 1.6.0'``.

    Same format as ``imagecodecs.webp_version()``.
    """
    cdef int v = WebPGetEncoderVersion()
    return f'libwebp {(v >> 16) & 0xff}.{(v >> 8) & 0xff}.{v & 0xff}'


def encode(data, *, level=None, lossless=None, numthreads=None,
           method=None) -> bytes:
    """Encode a uint8 image as WebP.

    The parameters mean what they mean in ``imagecodecs.webp_encode``.
    For an ``(H, W, 3)`` or ``(H, W, 4)`` array that imagecodecs also
    accepts, the same arguments give the same bytes when both link the
    same libwebp version.

    The arguments are deliberately not annotated: Cython enforces an
    annotation as an exact type, which would refuse ``lossless=1`` or a
    NumPy bool and truncate a fractional ``level``.

    Parameters
    ----------
    level : float, optional
        ``WebPConfig.quality``, 0-100, default 75, clamped to 100. For
        lossy encoding it is the quality factor; for lossless encoding
        libwebp defines it as the compression effort (0 fastest, 100
        smallest). Fractions are kept. A negative level selects lossless
        encoding at the default effort, as in imagecodecs.
    lossless : bool, optional
        ``None`` (default) encodes losslessly. Any other value is read
        as imagecodecs reads it, ``int(lossless)``, nonzero meaning
        lossless: ``True``, ``1`` and NumPy bools select lossless,
        ``False``, ``0`` and ``0.5`` lossy, and a string that is not an
        integer, such as ``'no'``, raises ``ValueError``. Lossless output is exact: RGB
        values under fully transparent pixels are kept
        (``WebPConfig.exact=1``).
    numthreads : int, optional
        ``None`` or ``<=0`` uses the libwebp default (single-threaded).
        Any positive value enables libwebp's worker thread for entropy
        coding (``WebPConfig.thread_level=1``). libwebp's threading model
        is a binary on/off, not an N-way pool — additional workers don't
        help. Typical speedup: 1.3-1.8× on lossy RGB encode.
    method : int, optional
        libwebp speed/size tradeoff 0..6, clamped to that range, so
        ``-1`` means 0, as in imagecodecs. ``None`` (default) is
        libwebp's default, 4.
    """
    cdef:
        cnp.ndarray arr
        const uint8_t* src_ptr
        uint8_t* shim_ptr = NULL
        size_t out_size = 0
        int width, height, stride
        float quality
        int thread_level
        int has_alpha_c = 0
        int lossless_c
        int method_c
        int rc
        bytes out

    if not isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data, dtype=np.uint8)
    else:
        if data.dtype != np.uint8:
            raise WebpError(f'WebP encode: unsupported dtype {data.dtype}')
        arr = np.ascontiguousarray(data)

    if arr.ndim == 2:
        # Promote grayscale to RGB so libwebp accepts it.
        arr = np.stack([arr] * 3, axis=-1)
        arr = np.ascontiguousarray(arr)
    elif arr.ndim == 3:
        if arr.shape[2] == 3:
            has_alpha_c = 0
        elif arr.shape[2] == 4:
            has_alpha_c = 1
        else:
            raise WebpError(
                f'WebP encode: unsupported channel count {arr.shape[2]}')
    else:
        raise WebpError(f'WebP encode: unsupported ndim {arr.ndim}')

    height = <int> arr.shape[0]
    width = <int> arr.shape[1]
    stride = <int> arr.strides[0]
    src_ptr = <const uint8_t*> cnp.PyArray_DATA(arr)

    # imagecodecs: level None -> 75, clamped to [-1, 100]; a negative
    # level means lossless at the default effort.
    quality = 75.0 if level is None else min(float(level), 100.0)
    # imagecodecs computes int(lossless is None or lossless or level < 0),
    # so a value int() cannot read, such as the string 'no', raises.
    lossless_v = lossless is None or lossless or quality < 0
    try:
        lossless_c = 1 if int(lossless_v) else 0
    except ValueError:
        raise ValueError(
            f'WebP encode: lossless={lossless!r} is not a truth value or '
            f'integer') from None
    except TypeError:
        raise TypeError(
            f'WebP encode: lossless must be a truth value or integer, '
            f'got {type(lossless).__name__}') from None
    if quality < 0: quality = 75.0

    # imagecodecs: method None -> 4, otherwise clamped to [0, 6].
    method_c = 4 if method is None else min(max(int(method), 0), 6)

    if numthreads is None or int(numthreads) <= 0:
        thread_level = 0
    else:
        thread_level = 1

    # Always the advanced API: the simple WebPEncode* helpers cannot set
    # the lossless effort or exact=1, and taking them for some argument
    # combinations made the bytes depend on numthreads and method.
    with nogil:
        rc = oc_webp_encode(
            src_ptr, width, height, stride,
            has_alpha_c, lossless_c, quality,
            thread_level, method_c,
            &shim_ptr, &out_size,
        )
    if rc != 0 or shim_ptr == NULL or out_size == 0:
        if shim_ptr != NULL:
            oc_webp_free(shim_ptr)
        raise WebpError(f'WebP encode failed (rc={rc})')
    try:
        out = PyBytes_FromStringAndSize(
            <char*> shim_ptr, <Py_ssize_t> out_size)
        return out
    finally:
        oc_webp_free(shim_ptr)


def decode(data, *, hasalpha=None, out=None) -> np.ndarray:
    """Decode WebP bytes into a uint8 array.

    ``hasalpha`` is imagecodecs' parameter: ``None`` (default) returns
    ``(H, W, 4)`` RGBA when the bitstream has alpha and ``(H, W, 3)`` RGB
    otherwise, a true value always returns RGBA (alpha 255 where the
    image has none), and a false value always returns RGB.

    ``out=`` is a preallocated ``(H, W, 3) | (H, W, 4) uint8`` ndarray
    matching that shape. See ``_png.decode`` for the full contract.
    WebPDecode{RGB,RGBA}Into write directly into the buffer so this is
    a true zero-alloc fast path.
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        WebPBitstreamFeatures features
        int rc
        int width = 0, height = 0
        uint8_t* dec_ptr
        cnp.ndarray out_arr
        cnp.npy_intp shape[3]
        int channels
        size_t out_size
        int out_stride
        tuple expected_shape

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if srcsize < 12:
        raise WebpError('input too short to be WebP')

    rc = WebPGetFeatures(&src[0], srcsize, &features)
    if rc != VP8_STATUS_OK:
        raise WebpError(f'WebPGetFeatures failed: status {rc}')

    width = features.width
    height = features.height
    if hasalpha is None:
        has_alpha = bool(features.has_alpha)
    else:
        has_alpha = bool(hasalpha)
    channels = 4 if has_alpha else 3
    shape[0] = height
    shape[1] = width
    shape[2] = channels
    expected_shape = (height, width, channels)

    if out is not None:
        if not isinstance(out, np.ndarray):
            raise TypeError(
                f"webp decode: out= must be an ndarray, "
                f"got {type(out).__name__}")
        if out.shape != expected_shape:
            raise ValueError(
                f"webp decode: out= shape {out.shape} does not match "
                f"expected {expected_shape}")
        if out.dtype != np.uint8:
            raise ValueError(
                f"webp decode: out= dtype must be uint8, got {out.dtype}")
        if not out.flags['C_CONTIGUOUS']:
            raise ValueError("webp decode: out= must be C-contiguous")
        out_arr = out
    else:
        out_arr = cnp.PyArray_EMPTY(3, shape, cnp.NPY_UINT8, 0)
    out_stride = width * channels
    out_size = <size_t> (out_stride * height)

    # Decode straight into the numpy array's buffer — skips the
    # malloc+memcpy step the WebPDecode{RGB,RGBA} variants would do.
    if has_alpha:
        with nogil:
            dec_ptr = WebPDecodeRGBAInto(
                &src[0], srcsize,
                <uint8_t*> cnp.PyArray_DATA(out_arr), out_size, out_stride,
            )
    else:
        with nogil:
            dec_ptr = WebPDecodeRGBInto(
                &src[0], srcsize,
                <uint8_t*> cnp.PyArray_DATA(out_arr), out_size, out_stride,
            )
    if dec_ptr == NULL:
        raise WebpError('WebP decode failed')
    return out_arr


def frame_count(data) -> int:
    """How many frames an animated WebP holds; 1 for a still.

    Reads the animation header only. A still has no ANIM chunk, so
    WebPAnimDecoderNew refuses it and the answer is 1 without decoding
    anything.
    """
    cdef:
        const uint8_t[::1] src
        WebPData wd
        WebPAnimDecoder* dec = NULL
        WebPAnimInfo info

    src = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    wd.bytes = &src[0]
    wd.size = <size_t> src.shape[0]
    dec = WebPAnimDecoderNew(&wd, NULL)
    if dec == NULL:
        return 1                       # not an animation
    try:
        if not WebPAnimDecoderGetInfo(dec, &info):
            return 1
        return int(info.frame_count)
    finally:
        WebPAnimDecoderDelete(dec)


def decode_animation(data, *, numthreads: int | None = None):
    """Every frame of an animated WebP, as a list of RGBA arrays.

    Frames are stored as sub-rectangles with disposal and blending
    rules, so frame N genuinely depends on the frames before it.
    WebPAnimDecoderGetNext reflects that: it hands back a fully
    reconstructed canvas and only moves forward. Decoding them all in
    one forward pass is therefore the cheap way to get any of them, and
    is why this returns the sequence rather than offering random
    access -- see the note on webp in capabilities.toml.

    Returns ``(frames, timestamps_ms, loop_count)``. Timestamps are the
    END of each frame's display, which is what libwebp reports.
    """
    cdef:
        const uint8_t[::1] src
        WebPData wd
        WebPAnimDecoderOptions opts
        WebPAnimDecoder* dec = NULL
        WebPAnimInfo info
        uint8_t* buf = NULL
        int timestamp = 0
        cnp.ndarray frame
        cnp.npy_intp shape[3]
        size_t nbytes

    src = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    wd.bytes = &src[0]
    wd.size = <size_t> src.shape[0]

    if not WebPAnimDecoderOptionsInit(&opts):
        raise WebpError('WebPAnimDecoderOptionsInit failed (ABI mismatch)')
    opts.color_mode = MODE_RGBA
    opts.use_threads = 0 if (numthreads is not None and numthreads <= 1) else 1

    dec = WebPAnimDecoderNew(&wd, &opts)
    if dec == NULL:
        raise WebpError(
            'WebPAnimDecoderNew failed; this is not an animated WebP')
    try:
        if not WebPAnimDecoderGetInfo(dec, &info):
            raise WebpError('WebPAnimDecoderGetInfo failed')
        shape[0] = <cnp.npy_intp> info.canvas_height
        shape[1] = <cnp.npy_intp> info.canvas_width
        shape[2] = 4
        nbytes = <size_t> info.canvas_width * info.canvas_height * 4
        frames = []
        stamps = []
        while WebPAnimDecoderGetNext(dec, &buf, &timestamp):
            # buf is owned by the decoder and is overwritten by the
            # next GetNext, so each frame is copied out here.
            frame = cnp.PyArray_EMPTY(3, shape, cnp.NPY_UINT8, 0)
            memcpy(cnp.PyArray_DATA(frame), buf, nbytes)
            frames.append(frame)
            stamps.append(int(timestamp))
        if len(frames) != info.frame_count:
            raise WebpError(
                f'animated WebP: header says {info.frame_count} frames, '
                f'decoded {len(frames)}')
        return frames, stamps, int(info.loop_count)
    finally:
        WebPAnimDecoderDelete(dec)


def check_signature(data) -> bool:
    """True if `data` is a RIFF/WEBP container."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:12])
    else:
        try:
            head = bytes(data)[:12]
        except Exception:
            return False
    return len(head) >= 12 and head[:4] == b'RIFF' and head[8:12] == b'WEBP'
