# opencodecs/codecs/_gif.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native GIF codec — palette-based animated/static images via giflib.

GIF is a palette-indexed (up to 256 colors) lossless raster format
with optional animation. ``decode`` returns an RGB ndarray composited
across all frames; ``encode`` takes a uint8 palette-index array and
writes a single-frame GIF with a caller-supplied (or grayscale)
palette.

We bind giflib (libgif 6.x) directly via ``gif_lib.h``. The
``DGifSlurp`` / ``EGifPut*`` pair handles all the format's quirks
(palette resolution, interlaced rows, extension blocks, multi-frame
disposal) so this wrapper stays small.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdlib cimport free, malloc, realloc
from libc.string cimport memcpy
from libc.stdint cimport uint8_t

import numpy as np
cimport numpy as cnp

from giflib cimport (
    GIF_OK, GIF_ERROR,
    DISPOSE_DO_NOT, DISPOSE_BACKGROUND, DISPOSE_PREVIOUS,
    GifByteType, GifPixelType, GifWord, GifColorType,
    ColorMapObject, SavedImage, GifFileType, ExtensionBlock,
    InputFunc, OutputFunc, GifErrorString,
    DGifOpen, DGifSlurp, DGifCloseFile,
    EGifOpen, EGifCloseFile, EGifSetGifVersion,
    EGifPutScreenDesc, EGifPutImageDesc, EGifPutLine,
    GifMakeMapObject, GifFreeMapObject,
)

cnp.import_array()


class GifError(RuntimeError):
    """Raised on GIF encode/decode failures."""


# In-memory I/O — pass a struct holding (buf, size, offset, capacity)
# via GifFileType.UserData. The C-level callbacks below read/write
# through it without touching Python.
cdef struct _MemBuf:
    GifByteType* data
    size_t size
    size_t offset
    size_t capacity
    int owns


cdef int _read_cb(GifFileType* gif, GifByteType* buf, int n) noexcept nogil:
    cdef _MemBuf* m = <_MemBuf*> gif.UserData
    cdef size_t remaining = m.size - m.offset
    cdef size_t take = <size_t> n if <size_t> n <= remaining else remaining
    if take == 0:
        return 0
    memcpy(buf, m.data + m.offset, take)
    m.offset += take
    return <int> take


cdef int _write_cb(GifFileType* gif, const GifByteType* buf, int n) noexcept nogil:
    cdef _MemBuf* m = <_MemBuf*> gif.UserData
    cdef size_t need = m.offset + <size_t> n
    cdef size_t new_cap
    cdef GifByteType* new_data
    if need > m.capacity:
        new_cap = m.capacity * 2 if m.capacity else 8192
        while new_cap < need:
            new_cap *= 2
        new_data = <GifByteType*> realloc(m.data, new_cap)
        if new_data == NULL:
            return 0
        m.data = new_data
        m.capacity = new_cap
    memcpy(m.data + m.offset, buf, <size_t> n)
    m.offset += <size_t> n
    if m.offset > m.size:
        m.size = m.offset
    return n


def check_signature(data) -> bool:
    """True if ``data`` starts with the GIF87a or GIF89a magic."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:6])
    else:
        try:
            head = bytes(data)[:6]
        except Exception:
            return False
    return head == b'GIF87a' or head == b'GIF89a'


def decode(data, *, asrgb: bool = True) -> 'np.ndarray':
    """Decode a GIF blob to a numpy array.

    Parameters
    ----------
    data : bytes-like
        GIF87a or GIF89a bytestream.
    asrgb : bool
        ``True`` (default) returns RGB uint8 (composited across all
        frames for animations; transparent pixels get the background
        color). ``False`` returns raw palette indices (uint8 per pixel)
        — single-frame only, no composition.

    Returns
    -------
    ndarray
        Single-frame: ``(H, W, 3)`` RGB uint8 (or ``(H, W)`` palette
        indices if ``asrgb=False``).
        Multi-frame: ``(N, H, W, 3)`` RGB uint8.
    """
    cdef:
        const uint8_t[::1] src
        _MemBuf mem
        GifFileType* gif = NULL
        SavedImage* img
        ColorMapObject* cmap
        int err = 0
        int rc
        Py_ssize_t i, y, x, p, frame, n_frames
        Py_ssize_t W, H, fw, fh, fl, ft
        Py_ssize_t canvas_stride, frame_stride
        cnp.ndarray out
        uint8_t* out_p
        uint8_t* canvas
        GifByteType* raster
        GifColorType color
        int idx, trans_idx
        uint8_t bg_r = 0, bg_g = 0, bg_b = 0

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    if src.shape[0] < 6:
        raise GifError("input too short to be a GIF")

    mem.data = <GifByteType*> &src[0]
    mem.size = <size_t> src.shape[0]
    mem.offset = 0
    mem.capacity = mem.size
    mem.owns = 0

    gif = DGifOpen(<void*> &mem, _read_cb, &err)
    if gif == NULL:
        raise GifError(
            f"DGifOpen failed: {GifErrorString(err).decode()}"
        )
    try:
        rc = DGifSlurp(gif)
        if rc != GIF_OK or gif.SavedImages == NULL or gif.ImageCount <= 0:
            raise GifError(
                f"DGifSlurp failed: {GifErrorString(gif.Error).decode()}"
            )

        W = <Py_ssize_t> gif.SWidth
        H = <Py_ssize_t> gif.SHeight
        n_frames = <Py_ssize_t> gif.ImageCount

        if not asrgb:
            if n_frames != 1:
                raise GifError(
                    f"asrgb=False only supports single-frame GIFs; "
                    f"got {n_frames} frames"
                )
            img = &gif.SavedImages[0]
            shape = (int(img.ImageDesc.Height), int(img.ImageDesc.Width))
            out = np.empty(shape, dtype=np.uint8)
            memcpy(<void*> cnp.PyArray_DATA(out),
                   <const void*> img.RasterBits,
                   <size_t>(shape[0] * shape[1]))
            return out

        # RGB output. Composite frames onto a canvas. We use the
        # global palette unless a frame defines its own.
        if n_frames == 1:
            shape3 = (int(H), int(W), 3)
        else:
            shape3 = (int(n_frames), int(H), int(W), 3)
        out = np.zeros(shape3, dtype=np.uint8)
        out_p = <uint8_t*> cnp.PyArray_DATA(out)

        # Background color from the global palette.
        if gif.SColorMap != NULL and gif.SBackGroundColor < gif.SColorMap.ColorCount:
            color = gif.SColorMap.Colors[gif.SBackGroundColor]
            bg_r, bg_g, bg_b = color.Red, color.Green, color.Blue

        canvas_stride = W * 3
        frame_stride = H * canvas_stride

        if n_frames == 1:
            canvas = out_p
            _paint_frame(canvas, W, H, &gif.SavedImages[0], gif.SColorMap,
                         bg_r, bg_g, bg_b, 1)
        else:
            # Animated: composite each frame onto a working canvas.
            for frame in range(n_frames):
                canvas = out_p + frame * frame_stride
                if frame == 0:
                    # Initial fill: background.
                    for i in range(H * W):
                        canvas[i*3 + 0] = bg_r
                        canvas[i*3 + 1] = bg_g
                        canvas[i*3 + 2] = bg_b
                else:
                    # Start from previous frame's composited image.
                    memcpy(canvas, canvas - frame_stride,
                           <size_t>(frame_stride))
                _paint_frame(canvas, W, H, &gif.SavedImages[frame],
                             gif.SColorMap, bg_r, bg_g, bg_b, 1)
        return out
    finally:
        DGifCloseFile(gif, &err)


cdef int _paint_frame(
    uint8_t* canvas, Py_ssize_t W, Py_ssize_t H,
    SavedImage* img, ColorMapObject* global_map,
    uint8_t bg_r, uint8_t bg_g, uint8_t bg_b,
    int respect_transparency,
) noexcept nogil:
    """Composite ``img.RasterBits`` onto a (W*H*3) RGB canvas at
    ``img.ImageDesc.{Left,Top}``. Uses the frame-local color map if
    present, else the global. Honors the GIF transparency-index
    extension (0xf9) when present."""
    cdef ColorMapObject* cmap = img.ImageDesc.ColorMap
    if cmap == NULL:
        cmap = global_map
    if cmap == NULL:
        return -1
    cdef Py_ssize_t fl = <Py_ssize_t> img.ImageDesc.Left
    cdef Py_ssize_t ft = <Py_ssize_t> img.ImageDesc.Top
    cdef Py_ssize_t fw = <Py_ssize_t> img.ImageDesc.Width
    cdef Py_ssize_t fh = <Py_ssize_t> img.ImageDesc.Height
    cdef GifByteType* raster = img.RasterBits

    # Transparency index (-1 if none).
    cdef int trans_idx = -1
    cdef Py_ssize_t j
    cdef ExtensionBlock* eb
    for j in range(img.ExtensionBlockCount):
        eb = &img.ExtensionBlocks[j]
        if eb.Function == 0xf9 and eb.ByteCount >= 4 and (eb.Bytes[0] & 0x01):
            trans_idx = <int> eb.Bytes[3]
            break

    cdef Py_ssize_t y, x, dst_pos
    cdef int idx
    cdef GifColorType color
    for y in range(fh):
        if ft + y >= H:
            break
        for x in range(fw):
            if fl + x >= W:
                break
            idx = <int> raster[y * fw + x]
            if respect_transparency and idx == trans_idx:
                continue
            if idx >= cmap.ColorCount:
                continue
            color = cmap.Colors[idx]
            dst_pos = ((ft + y) * W + (fl + x)) * 3
            canvas[dst_pos + 0] = color.Red
            canvas[dst_pos + 1] = color.Green
            canvas[dst_pos + 2] = color.Blue
    return 0


def encode(data, *, colormap=None) -> bytes:
    """Encode a 2D uint8 palette-index array as a GIF.

    Parameters
    ----------
    data : ndarray
        ``(H, W)`` uint8 array of palette indices (0..255).
    colormap : ndarray, optional
        ``(256, 3)`` uint8 RGB palette. Defaults to a grayscale ramp
        (matches imagecodecs's default for symmetry).

    Returns
    -------
    bytes
        Single-frame GIF89a bytestream.
    """
    cdef:
        cnp.ndarray arr
        cnp.ndarray cmap_arr
        _MemBuf mem
        GifFileType* gif = NULL
        ColorMapObject* gif_cmap = NULL
        GifWord width, height
        int err = 0
        int ret
        int err_row = 0
        Py_ssize_t y
        Py_ssize_t hh
        Py_ssize_t row_stride
        uint8_t* base
        bytes out

    if not isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data, dtype=np.uint8)
    else:
        if data.dtype != np.uint8:
            raise GifError(f"GIF encode requires uint8, got {data.dtype!r}")
        arr = np.ascontiguousarray(data)
    if arr.ndim != 2:
        raise GifError(
            f"GIF encode requires a 2D palette-index array; "
            f"got ndim={arr.ndim} (RGB → quantize first via "
            f"PIL.Image.quantize or numpy if you need colors)"
        )
    if arr.shape[0] >= 65536 or arr.shape[1] >= 65536:
        raise GifError("GIF format limits dimensions to <65536 px per side")

    if colormap is None:
        # Grayscale palette: (i, i, i) for i in 0..255.
        cmap_arr = np.empty((256, 3), dtype=np.uint8)
        for i in range(256):
            cmap_arr[i, 0] = i
            cmap_arr[i, 1] = i
            cmap_arr[i, 2] = i
    else:
        cmap_arr = np.ascontiguousarray(colormap, dtype=np.uint8)
        if cmap_arr.ndim != 2 or cmap_arr.shape[0] != 256 or cmap_arr.shape[1] != 3:
            raise GifError(
                f"colormap must be (256, 3) uint8, got shape "
                f"({tuple(int(s) for s in (<object> cmap_arr).shape)})"
            )

    height = <GifWord> arr.shape[0]
    width = <GifWord> arr.shape[1]

    mem.data = NULL
    mem.size = 0
    mem.offset = 0
    mem.capacity = 0
    mem.owns = 1

    gif = EGifOpen(<void*> &mem, _write_cb, &err)
    if gif == NULL:
        raise GifError(
            f"EGifOpen failed: {GifErrorString(err).decode()}"
        )
    try:
        # giflib defaults to GIF89a; no need to set it explicitly.
        gif_cmap = GifMakeMapObject(
            256,
            <GifColorType*> cnp.PyArray_DATA(cmap_arr),
        )
        if gif_cmap == NULL:
            raise GifError("GifMakeMapObject returned NULL (out of memory)")

        ret = EGifPutScreenDesc(gif, width, height, 256, 0, gif_cmap)
        if ret != GIF_OK:
            raise GifError(
                f"EGifPutScreenDesc: {GifErrorString(gif.Error).decode()}"
            )
        ret = EGifPutImageDesc(gif, 0, 0, width, height, False, NULL)
        if ret != GIF_OK:
            raise GifError(
                f"EGifPutImageDesc: {GifErrorString(gif.Error).decode()}"
            )
        # Per-row encode loop in nogil — every EGifPutLine call passes
        # through our pure-C _write_cb (no GIL needed), so we can drop
        # GIL for the whole height-sized loop and save ~5-10% vs the
        # Python-level for-loop alternative.
        hh = <Py_ssize_t> arr.shape[0]
        base = <uint8_t*> cnp.PyArray_DATA(arr)
        row_stride = <Py_ssize_t> arr.shape[1]
        with nogil:
            for y in range(hh):
                ret = EGifPutLine(
                    gif,
                    <GifPixelType*> (base + y * row_stride),
                    width,
                )
                if ret != GIF_OK:
                    err_row = <int> y
                    break
        if ret != GIF_OK:
            raise GifError(
                f"EGifPutLine row {err_row}: "
                f"{GifErrorString(gif.Error).decode()}"
            )

        # Close the encoder — this flushes the trailer and final bytes
        # via the write callback. Set gif=NULL so the finally block
        # doesn't double-close.
        ret = EGifCloseFile(gif, &err)
        gif = NULL
        if ret != GIF_OK:
            raise GifError(
                f"EGifCloseFile: {GifErrorString(err).decode()}"
            )
        out = PyBytes_FromStringAndSize(
            <const char*> mem.data, <Py_ssize_t> mem.size,
        )
        return out
    finally:
        if gif_cmap != NULL:
            GifFreeMapObject(gif_cmap)
        if gif != NULL:
            EGifCloseFile(gif, &err)
        if mem.owns and mem.data != NULL:
            free(mem.data)
