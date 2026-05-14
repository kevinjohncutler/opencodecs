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
    GifRecordType, IMAGE_DESC_RECORD_TYPE, EXTENSION_RECORD_TYPE,
    TERMINATE_RECORD_TYPE, SCREEN_DESC_RECORD_TYPE, UNDEFINED_RECORD_TYPE,
    ColorMapObject, SavedImage, GifFileType, ExtensionBlock,
    InputFunc, OutputFunc, GifErrorString,
    DGifOpen, DGifSlurp, DGifCloseFile,
    DGifGetRecordType, DGifGetImageDesc,
    DGifGetCode, DGifGetCodeNext,
    DGifGetExtension, DGifGetExtensionNext,
    EGifOpen, EGifCloseFile, EGifSetGifVersion,
    EGifPutScreenDesc, EGifPutImageDesc, EGifPutLine,
    GifMakeMapObject, GifFreeMapObject,
    oc_giflzw_decode,
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


def decode_fast(data, *, asrgb: bool = True) -> 'np.ndarray':
    """Fast single-frame decode via custom LZW.

    Uses libgif for the outer record walk + image-descriptor parsing,
    but pulls raw LZW sub-blocks via ``DGifGetCode`` / ``DGifGetCodeNext``
    and runs them through opencodecs's own ``oc_giflzw_decode`` — which
    benchmarks ~30% faster than libgif's reference LZW (matches Pillow).

    Currently single-frame only; ``decode()`` is still the right entry
    point for animated GIFs. Multi-frame fast-path is a straightforward
    extension once we collect transparency / disposal info from
    extension blocks during the record walk.
    """
    cdef:
        const uint8_t[::1] src
        _MemBuf mem
        GifFileType* gif = NULL
        SavedImage* img
        int err = 0
        int rc
        GifRecordType rec_type
        int lzw_min_code_size = 0
        int blk_len
        Py_ssize_t fw, fh
        Py_ssize_t W, H
        GifByteType* code_block
        GifByteType* ext_block
        int code_byte = 0
        cnp.ndarray out
        cnp.ndarray palette_arr
        uint8_t* palette_indices_buf
        uint8_t* rgb_p
        size_t lzw_buf_cap = 0
        size_t lzw_buf_len = 0
        uint8_t* lzw_buf = NULL
        int got_image = 0
        Py_ssize_t pix_count
        GifColorType color
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
        # Walk records until we find an image (single-frame fast path).
        while True:
            if DGifGetRecordType(gif, &rec_type) != GIF_OK:
                raise GifError(
                    f"DGifGetRecordType: "
                    f"{GifErrorString(gif.Error).decode()}"
                )
            if rec_type == UNDEFINED_RECORD_TYPE:
                continue
            if rec_type == TERMINATE_RECORD_TYPE:
                break
            if rec_type == SCREEN_DESC_RECORD_TYPE:
                continue
            if rec_type == EXTENSION_RECORD_TYPE:
                # Drain extension blocks without parsing — fast path is
                # single-frame, we don't need GCE delay/transparency.
                if DGifGetExtension(gif, &code_byte, &ext_block) != GIF_OK:
                    raise GifError("DGifGetExtension failed")
                while ext_block != NULL:
                    if DGifGetExtensionNext(gif, &ext_block) != GIF_OK:
                        raise GifError("DGifGetExtensionNext failed")
                continue
            if rec_type != IMAGE_DESC_RECORD_TYPE:
                continue

            # IMAGE_DESC: read geometry + local palette.
            if DGifGetImageDesc(gif) != GIF_OK:
                raise GifError(
                    f"DGifGetImageDesc: "
                    f"{GifErrorString(gif.Error).decode()}"
                )
            got_image = 1
            break

        if not got_image:
            raise GifError("no image found in GIF")

        # Geometry from the most recently parsed ImageDesc — giflib
        # stores it in gif.SavedImages[gif.ImageCount-1].
        img = &gif.SavedImages[gif.ImageCount - 1]
        fw = <Py_ssize_t> img.ImageDesc.Width
        fh = <Py_ssize_t> img.ImageDesc.Height
        W = <Py_ssize_t> gif.SWidth
        H = <Py_ssize_t> gif.SHeight

        # Accumulate raw LZW sub-blocks into a contiguous buffer.
        if DGifGetCode(gif, &lzw_min_code_size, &code_block) != GIF_OK:
            raise GifError(
                f"DGifGetCode: {GifErrorString(gif.Error).decode()}"
            )
        # Start with a reasonable capacity (most images <100 KB compressed).
        lzw_buf_cap = 65536
        lzw_buf = <uint8_t*> malloc(lzw_buf_cap)
        if lzw_buf == NULL:
            raise MemoryError("oom for LZW buffer")
        while code_block != NULL:
            # code_block[0] is the sub-block length byte; data follows.
            blk_len = <int> code_block[0]
            if lzw_buf_len + <size_t> blk_len > lzw_buf_cap:
                while lzw_buf_len + <size_t> blk_len > lzw_buf_cap:
                    lzw_buf_cap *= 2
                lzw_buf = <uint8_t*> realloc(lzw_buf, lzw_buf_cap)
                if lzw_buf == NULL:
                    raise MemoryError("oom growing LZW buffer")
            memcpy(lzw_buf + lzw_buf_len, code_block + 1, <size_t> blk_len)
            lzw_buf_len += <size_t> blk_len
            if DGifGetCodeNext(gif, &code_block) != GIF_OK:
                raise GifError(
                    f"DGifGetCodeNext: "
                    f"{GifErrorString(gif.Error).decode()}"
                )

        # Decode to palette indices using our custom LZW.
        pix_count = fw * fh
        palette_arr = np.empty(pix_count, dtype=np.uint8)
        palette_indices_buf = <uint8_t*> cnp.PyArray_DATA(palette_arr)
        with nogil:
            rc = oc_giflzw_decode(
                lzw_min_code_size,
                lzw_buf, lzw_buf_len,
                palette_indices_buf, <size_t> pix_count,
            )
        if rc != 0:
            raise GifError(f"oc_giflzw_decode failed: rc={rc}")

        # Stash the decoded raster into giflib's SavedImage so the
        # existing _paint_frame helper can run unchanged.
        if img.RasterBits == NULL:
            img.RasterBits = <GifByteType*> malloc(<size_t> pix_count)
            if img.RasterBits == NULL:
                raise MemoryError("oom for RasterBits")
        memcpy(img.RasterBits, palette_indices_buf, <size_t> pix_count)

        if not asrgb:
            out = palette_arr.reshape(<int>fh, <int>fw)
            return out

        # Composite to RGB on a global-screen-sized canvas.
        out = np.zeros((H, W, 3), dtype=np.uint8)
        rgb_p = <uint8_t*> cnp.PyArray_DATA(out)
        if gif.SColorMap != NULL and gif.SBackGroundColor < gif.SColorMap.ColorCount:
            color = gif.SColorMap.Colors[gif.SBackGroundColor]
            bg_r, bg_g, bg_b = color.Red, color.Green, color.Blue
        _paint_frame(rgb_p, W, H, img, gif.SColorMap,
                     bg_r, bg_g, bg_b, 1)
        return out
    finally:
        if lzw_buf != NULL:
            free(lzw_buf)
        DGifCloseFile(gif, &err)


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


# ---------------------------------------------------------------------------
# Streaming Reader / Writer
# ---------------------------------------------------------------------------
#
# Slurp the GIF once at open() time (libgif's DGifSlurp parses every frame
# into a palette-index raster). Compositing-to-RGB then happens lazily
# per-frame in iter_frames() / __getitem__. Memory savings vs. our
# decode()-everything-at-once function: 3x for RGB output (we hold N
# frames of u8 palette indices instead of N frames of u8 RGB).
#
# True record-by-record streaming (not even slurp the palette rasters)
# would require giflib's lower-level API; that's a bigger refactor and
# only matters for multi-GB animated GIFs, which are vanishingly rare.


cdef class GifReader:
    """Streaming GIF reader — yields one composited RGB frame at a time.

    Slurps frame indices on open (fast; just LZW decoding), then composites
    each frame to RGB on demand. ``iter_frames()`` yields ``(H, W, 3)``
    uint8 arrays; ``[i]`` random-access replays from frame 0 because GIF
    disposal modes make seek-O(1) impossible.
    """

    cdef GifFileType* _gif
    cdef _MemBuf _mem
    cdef bytes _src_bytes   # keep input alive for the duration of slurp
    cdef object _shape       # (n_frames, H, W, 3) or (H, W, 3) for single-frame
    cdef public object dtype
    cdef public int n_frames
    cdef public int width
    cdef public int height
    cdef uint8_t _bg_r, _bg_g, _bg_b

    def __cinit__(self, data):
        self._gif = NULL
        self._mem.data = NULL

    def __init__(self, data):
        cdef:
            int err = 0
            int rc
            GifColorType color
        # Keep a bytes ref so the read callback's pointer stays valid.
        if isinstance(data, (bytes, bytearray)):
            self._src_bytes = bytes(data)
        else:
            try:
                self._src_bytes = bytes(data)
            except Exception as e:
                raise GifError(f"unsupported input type: {e!r}")
        if len(self._src_bytes) < 6:
            raise GifError("input too short to be a GIF")

        self._mem.data = <GifByteType*> <const char*> self._src_bytes
        self._mem.size = <size_t> len(self._src_bytes)
        self._mem.offset = 0
        self._mem.capacity = self._mem.size
        self._mem.owns = 0

        self._gif = DGifOpen(<void*> &self._mem, _read_cb, &err)
        if self._gif == NULL:
            raise GifError(
                f"DGifOpen failed: {GifErrorString(err).decode()}"
            )
        rc = DGifSlurp(self._gif)
        if rc != GIF_OK or self._gif.SavedImages == NULL or \
                self._gif.ImageCount <= 0:
            raise GifError(
                f"DGifSlurp failed: {GifErrorString(self._gif.Error).decode()}"
            )

        self.n_frames = <int> self._gif.ImageCount
        self.width = <int> self._gif.SWidth
        self.height = <int> self._gif.SHeight
        self.dtype = np.uint8

        # Cache background color.
        self._bg_r = 0
        self._bg_g = 0
        self._bg_b = 0
        if self._gif.SColorMap != NULL and \
                self._gif.SBackGroundColor < self._gif.SColorMap.ColorCount:
            color = self._gif.SColorMap.Colors[self._gif.SBackGroundColor]
            self._bg_r = color.Red
            self._bg_g = color.Green
            self._bg_b = color.Blue

    def __dealloc__(self):
        cdef int err = 0
        if self._gif != NULL:
            DGifCloseFile(self._gif, &err)
            self._gif = NULL

    @property
    def shape(self):
        """``(H, W, 3)`` for single-frame; ``(n_frames, H, W, 3)`` for animated."""
        if self.n_frames == 1:
            return (self.height, self.width, 3)
        return (self.n_frames, self.height, self.width, 3)

    def close(self):
        cdef int err = 0
        if self._gif != NULL:
            DGifCloseFile(self._gif, &err)
            self._gif = NULL

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __len__(self):
        return self.n_frames

    def iter_frames(self):
        """Yield each frame composited to RGB ``(H, W, 3)`` uint8."""
        cdef:
            int i
            uint8_t* canvas
            cnp.ndarray prev = None
            cnp.ndarray fr
            Py_ssize_t H = self.height
            Py_ssize_t W = self.width
            Py_ssize_t frame_bytes = H * W * 3
            uint8_t bg_r = self._bg_r
            uint8_t bg_g = self._bg_g
            uint8_t bg_b = self._bg_b

        if self._gif == NULL:
            raise GifError("GifReader is closed")

        for i in range(self.n_frames):
            # Pre-fill via vectorised numpy (NOT a Python-level
            # element-by-element loop, which would be 25-30x slower on
            # a 2 MP frame). np.zeros is essentially free for the
            # common all-zero background; otherwise np.full broadcasts
            # the bg triplet in one pass.
            if bg_r == 0 and bg_g == 0 and bg_b == 0:
                fr = np.zeros((H, W, 3), dtype=np.uint8)
            else:
                fr = np.empty((H, W, 3), dtype=np.uint8)
                fr[..., 0] = bg_r
                fr[..., 1] = bg_g
                fr[..., 2] = bg_b
            canvas = <uint8_t*> cnp.PyArray_DATA(fr)
            if prev is not None:
                # Carry previous frame forward (DISPOSE_DO_NOT default
                # — what most decoders + browsers actually do).
                memcpy(canvas, <const void*> cnp.PyArray_DATA(prev),
                       <size_t> frame_bytes)
            _paint_frame(canvas, W, H, &self._gif.SavedImages[i],
                         self._gif.SColorMap,
                         bg_r, bg_g, bg_b, 1)
            prev = fr
            yield fr

    def __iter__(self):
        return self.iter_frames()

    def __getitem__(self, idx):
        """Random access. O(N) — replays frames 0..idx because GIF disposal
        chains forbid skipping."""
        if isinstance(idx, slice):
            return np.stack(list(self.iter_frames()), axis=0)[idx]
        cdef int i = int(idx)
        if i < 0:
            i += self.n_frames
        if i < 0 or i >= self.n_frames:
            raise IndexError(idx)
        cdef int seen = 0
        for fr in self.iter_frames():
            if seen == i:
                return fr
            seen += 1
        raise IndexError(idx)

    def read(self):
        """Return all frames stacked as ``(n_frames, H, W, 3)`` (or ``(H, W, 3)``
        for single-frame). Equivalent to ``np.stack(list(reader), axis=0)``
        with a fast-path for single-frame."""
        if self.n_frames == 1:
            return next(iter(self.iter_frames()))
        return np.stack(list(self.iter_frames()), axis=0)


cdef class GifWriter:
    """Streaming GIF writer — append frames one at a time.

    Single global colormap (caller-supplied or grayscale default).
    Each frame must be a uint8 palette-index ``(H, W)`` array of the
    declared screen dimensions. Optional per-frame delay (in
    centiseconds, GIF's native unit) and loop count via Netscape
    application extension are exposed through write_frame / __init__.
    """

    cdef GifFileType* _gif
    cdef _MemBuf _mem
    cdef ColorMapObject* _gcmap
    cdef int _width
    cdef int _height
    cdef int _loop
    cdef bint _header_written
    cdef bint _closed
    cdef bytes _last_bytes   # populated on close()

    def __cinit__(self, *args, **kwargs):
        self._gif = NULL
        self._mem.data = NULL
        self._gcmap = NULL

    def __init__(self, *, width: int, height: int,
                  colormap=None, loop: int = 0):
        """Create a streaming GIF writer.

        Parameters
        ----------
        width, height : int
            Screen (canvas) dimensions. All frames must match.
        colormap : (256, 3) uint8 ndarray, optional
            Global palette. Defaults to grayscale (i, i, i).
        loop : int
            0 = infinite looping (GIF's "Netscape 2.0" loop extension).
            >0 = play N times then stop. <0 = omit loop extension
            entirely (single-iteration playback).
        """
        cdef:
            int err = 0
            cnp.ndarray cmap_arr
            int ret

        if width <= 0 or height <= 0 or width >= 65536 or height >= 65536:
            raise GifError(
                f"GIF dimensions out of range: {width}x{height} "
                f"(must be 1..65535)"
            )

        self._width = width
        self._height = height
        self._loop = loop
        self._header_written = False
        self._closed = False

        # Build a copy of the colormap so it lives until close().
        if colormap is None:
            cmap_arr = np.empty((256, 3), dtype=np.uint8)
            for i in range(256):
                cmap_arr[i, 0] = i
                cmap_arr[i, 1] = i
                cmap_arr[i, 2] = i
        else:
            cmap_arr = np.ascontiguousarray(colormap, dtype=np.uint8)
            if (cmap_arr.ndim != 2 or cmap_arr.shape[0] != 256
                    or cmap_arr.shape[1] != 3):
                raise GifError(
                    f"colormap must be (256, 3) uint8, got shape "
                    f"({tuple(int(s) for s in (<object> cmap_arr).shape)})"
                )

        self._mem.data = NULL
        self._mem.size = 0
        self._mem.offset = 0
        self._mem.capacity = 0
        self._mem.owns = 1

        self._gif = EGifOpen(<void*> &self._mem, _write_cb, &err)
        if self._gif == NULL:
            raise GifError(
                f"EGifOpen failed: {GifErrorString(err).decode()}"
            )

        self._gcmap = GifMakeMapObject(
            256,
            <GifColorType*> cnp.PyArray_DATA(cmap_arr),
        )
        if self._gcmap == NULL:
            raise GifError("GifMakeMapObject returned NULL")

        ret = EGifPutScreenDesc(
            self._gif, self._width, self._height, 256, 0, self._gcmap,
        )
        if ret != GIF_OK:
            raise GifError(
                f"EGifPutScreenDesc: "
                f"{GifErrorString(self._gif.Error).decode()}"
            )

        # Netscape 2.0 looping extension — written before the first frame
        # so any standard viewer picks it up. Skip when loop < 0.
        if loop >= 0:
            self._write_netscape_loop(loop)

        self._header_written = True

    cdef _write_netscape_loop(self, int loop):
        """Emit the standard "NETSCAPE2.0" application extension that
        carries the loop count. ``loop=0`` means infinite."""
        cdef int ret
        # Application extension: 11-byte ID 'NETSCAPE2.0', then 3-byte
        # sub-block (0x03, lsb, msb of loop count). Use the legacy
        # ext-leader/block/trailer trio because giflib's
        # EGifPutExtension only handles single-block extensions.
        cdef GifByteType app_id[11]
        cdef GifByteType sub[3]
        cdef bytes name = b"NETSCAPE2.0"
        memcpy(app_id, <const char*> name, 11)
        sub[0] = 0x01
        sub[1] = <GifByteType> (loop & 0xff)
        sub[2] = <GifByteType> ((loop >> 8) & 0xff)
        # Emit the raw bytes via the write callback. _write_cb takes
        # (GifFileType*, const GifByteType*, int). 0xff = application
        # extension marker.
        cdef GifByteType hdr[14]
        # 0x21 = extension introducer, 0xff = application extension,
        # 0x0b = block size, then 11-byte NETSCAPE2.0.
        hdr[0] = 0x21
        hdr[1] = 0xff
        hdr[2] = 0x0b
        memcpy(&hdr[3], app_id, 11)
        _write_cb(self._gif, hdr, 14)
        # Then 0x03 = sub-block size, then 3 bytes payload, then 0x00 terminator.
        cdef GifByteType trailer[5]
        trailer[0] = 0x03
        trailer[1] = sub[0]
        trailer[2] = sub[1]
        trailer[3] = sub[2]
        trailer[4] = 0x00
        _write_cb(self._gif, trailer, 5)

    def write_frame(self, arr, *, delay_centiseconds: int = 0,
                     transparent_index: int = -1):
        """Append one frame.

        Parameters
        ----------
        arr : ndarray
            ``(H, W)`` uint8 palette indices matching the writer's
            declared width/height.
        delay_centiseconds : int
            Time to display this frame (1/100 sec units, GIF's native).
            ``0`` (default) = no GCE written (instant playback).
        transparent_index : int
            ``-1`` (default) = no transparent color. Otherwise the
            palette index to render as transparent. Requires
            ``delay_centiseconds > 0`` OR a non-default transparency to
            actually emit a Graphics Control Extension.
        """
        cdef:
            cnp.ndarray a
            int ret
            Py_ssize_t y
            uint8_t* base
            Py_ssize_t row_stride
            Py_ssize_t hh
            GifByteType gce_bytes[8]
            int has_gce

        if self._closed:
            raise GifError("write_frame on a closed GifWriter")
        if not self._header_written:
            raise GifError("internal: header not written before write_frame")

        if not isinstance(arr, np.ndarray):
            a = np.ascontiguousarray(arr, dtype=np.uint8)
        else:
            if arr.dtype != np.uint8:
                raise GifError(f"GIF write_frame: uint8 only, got {arr.dtype!r}")
            a = np.ascontiguousarray(arr)
        if a.ndim != 2:
            raise GifError(
                f"GIF write_frame: requires 2D palette-index array; "
                f"got ndim={a.ndim}"
            )
        if a.shape[0] != self._height or a.shape[1] != self._width:
            raise GifError(
                f"GIF write_frame: shape {tuple(int(s) for s in (<object> a).shape)} "
                f"doesn't match writer dimensions "
                f"({self._height}, {self._width})"
            )

        # Optional Graphics Control Extension (delay / transparency).
        has_gce = (delay_centiseconds > 0) or (transparent_index >= 0)
        if has_gce:
            # GCE block (8 bytes total): 0x21 0xf9 0x04 <pack> <dlow> <dhigh> <ti> 0x00
            gce_bytes[0] = 0x21
            gce_bytes[1] = 0xf9
            gce_bytes[2] = 0x04
            # Packed byte: bits 2..4 disposal=0 (none), bit 0 transparent flag.
            gce_bytes[3] = <GifByteType>(0x01 if transparent_index >= 0 else 0x00)
            gce_bytes[4] = <GifByteType>(delay_centiseconds & 0xff)
            gce_bytes[5] = <GifByteType>((delay_centiseconds >> 8) & 0xff)
            gce_bytes[6] = <GifByteType>(
                transparent_index if transparent_index >= 0 else 0
            )
            gce_bytes[7] = 0x00
            _write_cb(self._gif, gce_bytes, 8)

        ret = EGifPutImageDesc(
            self._gif, 0, 0, self._width, self._height, False, NULL,
        )
        if ret != GIF_OK:
            raise GifError(
                f"EGifPutImageDesc: "
                f"{GifErrorString(self._gif.Error).decode()}"
            )

        hh = <Py_ssize_t> a.shape[0]
        base = <uint8_t*> cnp.PyArray_DATA(a)
        row_stride = <Py_ssize_t> a.shape[1]
        with nogil:
            for y in range(hh):
                ret = EGifPutLine(
                    self._gif,
                    <GifPixelType*> (base + y * row_stride),
                    self._width,
                )
                if ret != GIF_OK:
                    break
        if ret != GIF_OK:
            raise GifError(
                f"EGifPutLine: "
                f"{GifErrorString(self._gif.Error).decode()}"
            )

    def close(self):
        """Finalize the stream and return the encoded bytes."""
        cdef int err = 0
        cdef int ret
        if self._closed:
            return self._last_bytes
        self._closed = True
        if self._gif != NULL:
            ret = EGifCloseFile(self._gif, &err)
            self._gif = NULL
            if ret != GIF_OK:
                if self._mem.owns and self._mem.data != NULL:
                    free(self._mem.data)
                    self._mem.data = NULL
                raise GifError(
                    f"EGifCloseFile: {GifErrorString(err).decode()}"
                )
        if self._gcmap != NULL:
            GifFreeMapObject(self._gcmap)
            self._gcmap = NULL
        if self._mem.data != NULL:
            self._last_bytes = PyBytes_FromStringAndSize(
                <const char*> self._mem.data, <Py_ssize_t> self._mem.size,
            )
            if self._mem.owns:
                free(self._mem.data)
            self._mem.data = NULL
        return self._last_bytes

    def __dealloc__(self):
        cdef int err = 0
        if self._gif != NULL:
            EGifCloseFile(self._gif, &err)
            self._gif = NULL
        if self._gcmap != NULL:
            GifFreeMapObject(self._gcmap)
            self._gcmap = NULL
        if self._mem.owns and self._mem.data != NULL:
            free(self._mem.data)
            self._mem.data = NULL

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
