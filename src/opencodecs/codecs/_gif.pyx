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

We bind giflib (libgif 6.x) directly via ``gif_lib.h`` for the record
walk and the encoder, and decode LZW with opencodecs's own
``oc_giflzw``. Frame compositing (background fill, transparency index,
disposal methods 2 and 3, deinterlacing) follows the GIF89a
specification and lives in :class:`GifReader`, which every decode path
shares, palette indices included.
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
    """Decode the first frame through opencodecs's own LZW decoder.

    Walks the records with libgif but pulls raw LZW sub-blocks via
    ``DGifGetCode`` / ``DGifGetCodeNext`` and runs them through
    ``oc_giflzw_decode`` (about 30% faster than libgif's reference LZW).
    This is the :class:`GifReader` walk stopped after one frame, so it
    honors the Graphic Control Extension transparency index and undoes
    interlacing exactly as the reader does.

    ``asrgb=True`` returns the ``(H, W, 3)`` RGB canvas (background
    color where the frame does not cover it). ``asrgb=False`` returns
    the ``(H, W)`` palette indices placed on a zero canvas, the same
    layout as :func:`decode` with ``asrgb=False``.
    """
    r = GifReader(data, _max_frames=1)
    try:
        if asrgb:
            return r._render_isolated(0)
        return r._indices(0)
    finally:
        r.close()


def decode(data, index=None, *, asrgb: bool = True) -> 'np.ndarray':
    """Decode a GIF blob to a numpy array.

    The signature follows imagecodecs ``gif_decode(data, index=None, *,
    asrgb=True, out=None)``: ``index`` may be given by position.

    Parameters
    ----------
    data : bytes-like
        GIF87a or GIF89a bytestream.
    index : int, optional
        Decode only this frame. As in imagecodecs, the frame is drawn
        on its own over the background (RGB) or zero (indices) canvas,
        without the frames before it and without disposal.
    asrgb : bool
        ``True`` (default) returns RGB uint8 frames composited per the
        GIF89a specification: the canvas starts as the logical screen
        background color, a frame's transparent index leaves the pixel
        underneath unchanged, and each frame's disposal method (restore
        to background, restore to previous) is applied before the next
        frame. ``False`` returns raw palette indices placed on a
        canvas-sized zero array, one plane per frame.

    Returns
    -------
    ndarray
        ``(H, W, 3)`` RGB uint8 for a single frame or ``index``, else
        ``(N, H, W, 3)``. With ``asrgb=False``: ``(H, W)`` or
        ``(N, H, W)`` uint8 palette indices. imagecodecs instead returns
        four channels, the fourth 255 everywhere, when the first frame
        uses its transparent index; this always returns three.
        imagecodecs 2026.8.16 also skips the restore of a disposal 3
        frame that follows a disposal 2 frame, which this applies. A
        frame that extends past the logical screen is clipped to it,
        the display area GIF89a positions images in, where imagecodecs
        enlarges the canvas to hold the frame, so its output can be
        larger. A frame of zero width or height draws nothing, where
        imagecodecs raises ``GifError``.
    """
    with GifReader(data) as r:
        if asrgb:
            if index is None:
                return r.read()
            return r._render_isolated(_frame_index(index, r.n_frames))
        if index is not None:
            return r._indices(_frame_index(index, r.n_frames))
        if r.n_frames == 1:
            return r._indices(0)
        return np.stack([r._indices(i) for i in range(r.n_frames)])


def _decode_indices_libgif(data) -> 'np.ndarray':
    """Every frame's palette indices through libgif's own reference LZW
    (``DGifSlurp``), placed as :func:`decode` places them with
    ``asrgb=False``. Kept as an independent decoder that the tests hold
    the fast path to.

    libgif writes past its raster for a frame of zero width or height,
    so such a file raises ``GifError`` here before libgif sees it.
    """
    cdef:
        const uint8_t[::1] src
        _MemBuf mem
        GifFileType* gif = NULL
        int err = 0
        int rc
        Py_ssize_t frame, n_frames
        Py_ssize_t W, H
        cnp.ndarray out

    with GifReader(data) as r:
        for frame in range(r.n_frames):
            rows, cols = r._rect(frame)
            if rows.stop == rows.start or cols.stop == cols.start:
                raise GifError(f"frame {frame} has zero width or height")
    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)

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
        out = np.zeros((n_frames, H, W), dtype=np.uint8)
        for frame in range(n_frames):
            _place_indices(
                <uint8_t*> cnp.PyArray_DATA(out) + frame * H * W,
                W, H, &gif.SavedImages[frame])
        return out[0] if n_frames == 1 else out
    finally:
        DGifCloseFile(gif, &err)


def _lzw_error(int rc) -> str:
    """Describe an ``oc_giflzw_decode`` failure in libgif's words."""
    if rc == -2:
        # A frame must code width * height pixels (GIF89a section 22);
        # libgif reports a short one with this message.
        return "Image EOF detected before image complete"
    if rc == -1:
        return "LZW minimum code size is not 2 to 8"
    return f"Image is defective, decoding aborted (LZW error {rc})"


def _frame_index(index, Py_ssize_t n_frames) -> int:
    """Validate a frame index (negative counts from the end)."""
    cdef Py_ssize_t i = int(index)
    if i < 0:
        i += n_frames
    if i < 0 or i >= n_frames:
        raise IndexError(
            f"GIF frame index {index} out of range for {n_frames} frames")
    return i


cdef void _place_indices(
    uint8_t* canvas, Py_ssize_t W, Py_ssize_t H, SavedImage* img,
) noexcept nogil:
    """Copy a frame's palette indices into a ``(H, W)`` canvas at the
    frame's position, clipped to the logical screen."""
    cdef Py_ssize_t fl = <Py_ssize_t> img.ImageDesc.Left
    cdef Py_ssize_t ft = <Py_ssize_t> img.ImageDesc.Top
    cdef Py_ssize_t fw = <Py_ssize_t> img.ImageDesc.Width
    cdef Py_ssize_t fh = <Py_ssize_t> img.ImageDesc.Height
    cdef Py_ssize_t y, cw
    if img.RasterBits == NULL or fl >= W or ft >= H:
        return
    cw = fw if fl + fw <= W else W - fl
    for y in range(fh):
        if ft + y >= H:
            break
        memcpy(canvas + (ft + y) * W + fl, img.RasterBits + y * fw,
               <size_t> cw)


cdef void _deinterlace(
    const uint8_t* src, uint8_t* dst, Py_ssize_t w, Py_ssize_t h,
) noexcept nogil:
    """Undo GIF interlacing (GIF89a section 20 and Appendix E).

    Rows are stored in four passes: every 8th row from 0, every 8th
    from 4, every 4th from 2, then every 2nd from 1.
    """
    cdef int[4] start = [0, 4, 2, 1]
    cdef int[4] step = [8, 8, 4, 2]
    cdef Py_ssize_t row = 0, y
    cdef int p
    for p in range(4):
        y = start[p]
        while y < h:
            memcpy(dst + y * w, src + row * w, <size_t> w)
            row += 1
            y += step[p]


cdef int _paint_frame(
    uint8_t* canvas, Py_ssize_t W, Py_ssize_t H,
    SavedImage* img, ColorMapObject* global_map, int trans_idx,
) noexcept nogil:
    """Composite ``img.RasterBits`` onto a (W*H*3) RGB canvas at
    ``img.ImageDesc.{Left,Top}``. Uses the frame-local color map if
    present, else the global. A pixel equal to ``trans_idx`` (the
    Graphic Control Extension transparency index, -1 for none) leaves
    the canvas unchanged, per GIF89a section 23."""
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
    cdef Py_ssize_t y, x, dst_pos
    cdef int idx
    cdef GifColorType color
    if raster == NULL:
        return -1
    for y in range(fh):
        if ft + y >= H:
            break
        for x in range(fw):
            if fl + x >= W:
                break
            idx = <int> raster[y * fw + x]
            if idx == trans_idx:
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
# Decode every frame's palette indices once at open() time (our own
# record walk plus oc_giflzw). Compositing to RGB then happens lazily
# per frame in iter_frames() / __getitem__, so the reader holds N frames
# of u8 palette indices instead of N frames of u8 RGB (3x less memory).


cdef class GifReader:
    """Streaming GIF reader — yields one composited RGB frame at a time.

    Decodes the frame indices on open (fast; just LZW decoding), then
    composites each frame to RGB on demand. ``iter_frames()`` yields
    ``(H, W, 3)`` uint8 arrays; ``[i]`` random-access replays from frame
    0 because GIF disposal modes make seek-O(1) impossible.

    Compositing follows the GIF89a specification: the canvas starts as
    the logical screen background color (section 18); each frame's
    Graphic Control Extension (section 23) supplies a transparency index
    whose pixels leave the canvas unchanged and a disposal method, where
    2 restores the frame's rectangle to the background color and 3
    restores what was there before the frame was drawn. Interlaced
    frames are deinterlaced (section 20, Appendix E).
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
    cdef list _disposal      # per-frame GCE disposal method (0..7)
    cdef list _transparent   # per-frame GCE transparency index, -1 if none

    def __cinit__(self, data, *args, **kwargs):
        self._gif = NULL
        self._mem.data = NULL

    def __init__(self, data, *, int _max_frames=0):
        cdef:
            int err = 0
            int rc
            int lzw_min_code_size = 0
            int blk_len
            int code_byte = 0
            int pending_disposal = 0
            int pending_trans = -1
            GifColorType color
            GifRecordType rec_type
            GifByteType* code_block
            GifByteType* ext_block
            SavedImage* img
            Py_ssize_t pix_count
            uint8_t* lzw_buf = NULL
            size_t lzw_buf_cap = 0
            size_t lzw_buf_len = 0
            uint8_t* raster
            uint8_t* flat
        self._disposal = []
        self._transparent = []
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

        # Replacement for DGifSlurp: walk records ourselves and run
        # raw LZW sub-blocks through oc_giflzw (1.5-1.6x faster than
        # libgif's reference LZW). For each frame we malloc a raster
        # buffer + decode into it + attach to giflib's SavedImage so
        # DGifCloseFile cleans it up via its standard FreeSavedImages
        # path (giflib free()s RasterBits, which matches our malloc).
        try:
            while True:
                if DGifGetRecordType(self._gif, &rec_type) != GIF_OK:
                    raise GifError(
                        f"DGifGetRecordType: "
                        f"{GifErrorString(self._gif.Error).decode()}"
                    )
                if rec_type == TERMINATE_RECORD_TYPE:
                    break
                if rec_type == UNDEFINED_RECORD_TYPE or \
                        rec_type == SCREEN_DESC_RECORD_TYPE:
                    continue
                if rec_type == EXTENSION_RECORD_TYPE:
                    if DGifGetExtension(
                        self._gif, &code_byte, &ext_block,
                    ) != GIF_OK:
                        raise GifError("DGifGetExtension failed")
                    # Graphic Control Extension (label 0xF9). giflib
                    # hands back the sub-block with its length byte
                    # first: [4, packed, delay lo, delay hi, index].
                    # Its scope is the next image descriptor only.
                    if code_byte == 0xF9 and ext_block != NULL \
                            and ext_block[0] >= 4:
                        pending_disposal = (ext_block[1] >> 2) & 0x07
                        if ext_block[1] & 0x01:
                            pending_trans = <int> ext_block[4]
                        else:
                            pending_trans = -1
                    while ext_block != NULL:
                        if DGifGetExtensionNext(
                            self._gif, &ext_block,
                        ) != GIF_OK:
                            raise GifError("DGifGetExtensionNext failed")
                    continue
                if rec_type != IMAGE_DESC_RECORD_TYPE:
                    continue

                # IMAGE_DESC: giflib parses geometry + local palette
                # into a new SavedImages[] entry.
                if DGifGetImageDesc(self._gif) != GIF_OK:
                    raise GifError(
                        f"DGifGetImageDesc: "
                        f"{GifErrorString(self._gif.Error).decode()}"
                    )
                img = &self._gif.SavedImages[self._gif.ImageCount - 1]
                pix_count = <Py_ssize_t> img.ImageDesc.Width * \
                            <Py_ssize_t> img.ImageDesc.Height
                self._disposal.append(pending_disposal)
                self._transparent.append(pending_trans)
                pending_disposal = 0
                pending_trans = -1

                # Pull raw LZW sub-blocks → flat buffer.
                if DGifGetCode(
                    self._gif, &lzw_min_code_size, &code_block,
                ) != GIF_OK:
                    raise GifError(
                        f"DGifGetCode: "
                        f"{GifErrorString(self._gif.Error).decode()}"
                    )
                lzw_buf_len = 0
                while code_block != NULL:
                    blk_len = <int> code_block[0]
                    if lzw_buf_len + <size_t> blk_len > lzw_buf_cap:
                        if lzw_buf_cap == 0:
                            lzw_buf_cap = 65536
                        while lzw_buf_len + <size_t> blk_len > lzw_buf_cap:
                            lzw_buf_cap *= 2
                        lzw_buf = <uint8_t*> realloc(lzw_buf, lzw_buf_cap)
                        if lzw_buf == NULL:
                            raise MemoryError("oom growing LZW buffer")
                    memcpy(
                        lzw_buf + lzw_buf_len,
                        code_block + 1,
                        <size_t> blk_len,
                    )
                    lzw_buf_len += <size_t> blk_len
                    if DGifGetCodeNext(self._gif, &code_block) != GIF_OK:
                        raise GifError(
                            f"DGifGetCodeNext: "
                            f"{GifErrorString(self._gif.Error).decode()}"
                        )

                # Decode into a fresh malloc'd raster — giflib's
                # cleanup will free() it via FreeSavedImages.
                raster = <uint8_t*> malloc(<size_t> pix_count + 1)
                if raster == NULL:
                    raise MemoryError("oom for frame raster")
                with nogil:
                    rc = oc_giflzw_decode(
                        lzw_min_code_size,
                        lzw_buf, lzw_buf_len,
                        raster, <size_t> pix_count,
                    )
                if rc != 0:
                    free(raster)
                    raise GifError(_lzw_error(rc))
                if img.ImageDesc.Interlace and img.ImageDesc.Height > 1:
                    flat = raster
                    raster = <uint8_t*> malloc(<size_t> pix_count + 1)
                    if raster == NULL:
                        free(flat)
                        raise MemoryError("oom for frame raster")
                    _deinterlace(flat, raster,
                                 <Py_ssize_t> img.ImageDesc.Width,
                                 <Py_ssize_t> img.ImageDesc.Height)
                    free(flat)
                # Hand ownership to giflib by assigning RasterBits.
                if img.RasterBits != NULL:
                    free(img.RasterBits)
                img.RasterBits = <GifByteType*> raster
                if _max_frames > 0 and self._gif.ImageCount >= _max_frames:
                    break
        finally:
            if lzw_buf != NULL:
                free(lzw_buf)

        if self._gif.SavedImages == NULL or self._gif.ImageCount <= 0:
            raise GifError("no frames found in GIF")

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

    @property
    def disposal(self):
        """Per-frame GIF89a disposal methods (0-3; 0 when no GCE)."""
        return tuple(self._disposal)

    @property
    def transparent_index(self):
        """Per-frame transparency index, ``-1`` where none is set."""
        return tuple(self._transparent)

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

    def _check_open(self):
        if self._gif == NULL:
            raise GifError("GifReader is closed")

    def _background(self):
        """A fresh ``(H, W, 3)`` canvas filled with the background color."""
        cdef cnp.ndarray fr
        if self._bg_r == 0 and self._bg_g == 0 and self._bg_b == 0:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        fr = np.empty((self.height, self.width, 3), dtype=np.uint8)
        fr[..., 0] = self._bg_r
        fr[..., 1] = self._bg_g
        fr[..., 2] = self._bg_b
        return fr

    def _rect(self, int i):
        """Frame ``i``'s rectangle as canvas slices (numpy clips them)."""
        cdef SavedImage* img = &self._gif.SavedImages[i]
        t = <Py_ssize_t> img.ImageDesc.Top
        l = <Py_ssize_t> img.ImageDesc.Left
        return (slice(t, t + <Py_ssize_t> img.ImageDesc.Height),
                slice(l, l + <Py_ssize_t> img.ImageDesc.Width))

    def _paint(self, cnp.ndarray canvas, int i):
        """Draw frame ``i`` onto ``canvas``; return the pre-draw
        snapshot that disposal method 3 needs (else None)."""
        saved = None
        if self._disposal[i] == 3:
            saved = canvas[self._rect(i)].copy()
        _paint_frame(<uint8_t*> cnp.PyArray_DATA(canvas),
                     self.width, self.height,
                     &self._gif.SavedImages[i], self._gif.SColorMap,
                     <int> self._transparent[i])
        return saved

    def _dispose(self, cnp.ndarray canvas, int i, saved):
        """Apply frame ``i``'s disposal method after it was shown."""
        cdef int d = self._disposal[i]
        if d == 2:
            rect = self._rect(i)
            canvas[rect + (0,)] = self._bg_r
            canvas[rect + (1,)] = self._bg_g
            canvas[rect + (2,)] = self._bg_b
        elif d == 3:
            canvas[self._rect(i)] = saved

    def _render_isolated(self, int i):
        """Frame ``i`` alone on the background, with no earlier frames
        and no disposal (the imagecodecs ``index=`` convention)."""
        self._check_open()
        canvas = self._background()
        _paint_frame(<uint8_t*> cnp.PyArray_DATA(canvas),
                     self.width, self.height,
                     &self._gif.SavedImages[i], self._gif.SColorMap,
                     <int> self._transparent[i])
        return canvas

    def _indices(self, int i):
        """Frame ``i``'s palette indices on a zero ``(H, W)`` canvas."""
        self._check_open()
        out = np.zeros((self.height, self.width), dtype=np.uint8)
        _place_indices(<uint8_t*> cnp.PyArray_DATA(out),
                       self.width, self.height, &self._gif.SavedImages[i])
        return out

    def iter_frames(self):
        """Yield each frame composited to RGB ``(H, W, 3)`` uint8."""
        cdef int i
        self._check_open()
        canvas = self._background()
        for i in range(self.n_frames):
            saved = self._paint(canvas, i)
            yield canvas.copy()
            self._dispose(canvas, i, saved)

    def __iter__(self):
        return self.iter_frames()

    def __getitem__(self, idx):
        """Random access. O(N) — replays frames 0..idx because GIF disposal
        chains forbid skipping."""
        if isinstance(idx, slice):
            return self.read_all()[idx]
        cdef int i = int(idx)
        cdef int k
        if i < 0:
            i += self.n_frames
        if i < 0 or i >= self.n_frames:
            raise IndexError(idx)
        self._check_open()
        canvas = self._background()
        for k in range(i):
            saved = self._paint(canvas, k)
            self._dispose(canvas, k, saved)
        self._paint(canvas, i)
        return canvas

    def read_all(self):
        """All frames as ``(n_frames, H, W, 3)``, even for one frame."""
        cdef int i
        self._check_open()
        out = np.empty((self.n_frames, self.height, self.width, 3),
                       dtype=np.uint8)
        canvas = self._background()
        for i in range(self.n_frames):
            saved = self._paint(canvas, i)
            out[i] = canvas
            self._dispose(canvas, i, saved)
        return out

    def read(self):
        """Return all frames stacked as ``(n_frames, H, W, 3)`` (or ``(H, W, 3)``
        for single-frame)."""
        if self.n_frames == 1:
            return self[0]
        return self.read_all()


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

    def drain(self):
        """Return completed output bytes while retaining the encoder state."""
        cdef bytes result
        if self._closed:
            raise GifError("drain on a closed GifWriter")
        result = PyBytes_FromStringAndSize(
            <const char*> self._mem.data, <Py_ssize_t> self._mem.size)
        self._mem.size = 0
        self._mem.offset = 0
        return result

    @property
    def output_buffer_capacity(self):
        """Native output allocation, reusable after each drain."""
        return self._mem.capacity

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
