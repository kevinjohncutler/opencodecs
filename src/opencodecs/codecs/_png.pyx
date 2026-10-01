# opencodecs/codecs/_png.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native PNG codec via libspng.

Decode keeps the PNG color type, the same mapping libpng gives with
png_set_expand (and the one imagecodecs returns):
  - grayscale             -> (H, W); 1/2/4-bit samples are scaled to 8-bit
  - grayscale+alpha       -> (H, W, 2)
  - RGB                   -> (H, W, 3)
  - RGBA                  -> (H, W, 4)
  - indexed               -> (H, W, 3) uint8, the palette applied
  - a tRNS chunk adds the alpha channel it defines (PNG specification,
    section 11.3.2.1): gray becomes (H, W, 2), RGB and indexed become
    (H, W, 4). Indexed alpha comes from the tRNS palette entries.
16-bit images decode to uint16 (host-endian), everything else to uint8.

Encode picks color type / bit depth from numpy shape and dtype:
  - 2D uint8 / uint16        -> grayscale
  - (H, W, 2) uint8/uint16   -> grayscale+alpha
  - (H, W, 3) uint8/uint16   -> RGB
  - (H, W, 4) uint8/uint16   -> RGBA
uint16 input may have either byte order; the sample values are kept.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.string cimport memcpy, memset
from libc.stdint cimport uint8_t, uint32_t
from libc.stdlib cimport free
from libc.stddef cimport size_t

import numpy as np
cimport numpy as cnp

from spng cimport (
    spng_ctx, spng_ctx_new, spng_ctx_free,
    spng_set_png_buffer, spng_get_png_buffer,
    spng_set_image_limits, spng_set_chunk_limits, spng_set_option,
    spng_decoded_image_size, spng_decode_image, spng_encode_image,
    spng_ihdr, spng_get_ihdr, spng_set_ihdr,
    spng_iccp, spng_get_iccp, spng_set_iccp,
    spng_trns, spng_get_trns,
    spng_strerror,
    SPNG_COLOR_TYPE_GRAYSCALE, SPNG_COLOR_TYPE_TRUECOLOR,
    SPNG_COLOR_TYPE_INDEXED, SPNG_COLOR_TYPE_GRAYSCALE_ALPHA,
    SPNG_COLOR_TYPE_TRUECOLOR_ALPHA,
    SPNG_FMT_RGBA8, SPNG_FMT_RGBA16, SPNG_FMT_RGB8,
    SPNG_FMT_G8, SPNG_FMT_GA8, SPNG_FMT_GA16, SPNG_FMT_PNG,
    SPNG_DECODE_TRNS, SPNG_ECHUNKAVAIL,
    SPNG_CTX_ENCODER, SPNG_ENCODE_FINALIZE,
    SPNG_IMG_COMPRESSION_LEVEL, SPNG_ENCODE_TO_BUFFER,
    SPNG_FILTER_CHOICE, SPNG_IMG_COMPRESSION_STRATEGY,
    SPNG_DISABLE_FILTERING,
    SPNG_FILTER_CHOICE_NONE, SPNG_FILTER_CHOICE_SUB,
    SPNG_FILTER_CHOICE_UP, SPNG_FILTER_CHOICE_AVG,
    SPNG_FILTER_CHOICE_PAETH, SPNG_FILTER_CHOICE_ALL,
)

cnp.import_array()


class PngError(RuntimeError):
    """Raised on PNG encode/decode failures."""


cdef inline _check(int rc, str where):
    if rc != 0:
        raise PngError(f'{where}: {spng_strerror(rc).decode()}')


cdef int _decode_layout(spng_ctx* ctx, spng_ihdr* ihdr, int* fmt, int* flags,
                        int* channels, bint* wide) except? -1:
    """Pick the libspng output format that keeps the PNG color type.

    Sets the format, the decode flags, the channel count and whether
    samples are 16-bit. Returns 0, or the libspng error from reading the
    chunks before the image data. The rule is libpng's png_set_expand plus
    png_set_tRNS_to_alpha, which is also what imagecodecs returns:
    gray stays gray (sub-byte depths scaled to 8 bits), a palette becomes
    RGB, and a tRNS chunk adds the alpha channel it describes.
    """
    cdef spng_trns trns
    cdef int rc
    cdef bint has_trns = False
    cdef int color_type = ihdr.color_type
    cdef int depth = ihdr.bit_depth
    flags[0] = 0
    wide[0] = depth == 16
    if color_type in (SPNG_COLOR_TYPE_GRAYSCALE, SPNG_COLOR_TYPE_TRUECOLOR,
                      SPNG_COLOR_TYPE_INDEXED):
        rc = spng_get_trns(ctx, &trns)
        if rc == 0:
            has_trns = True
        elif rc != SPNG_ECHUNKAVAIL:
            return rc
    if has_trns:
        flags[0] = SPNG_DECODE_TRNS
    if color_type == SPNG_COLOR_TYPE_INDEXED:
        if has_trns:
            channels[0] = 4
            fmt[0] = SPNG_FMT_RGBA8
            return 0
        channels[0] = 3
        fmt[0] = SPNG_FMT_RGB8
        return 0
    if color_type == SPNG_COLOR_TYPE_GRAYSCALE:
        if has_trns:
            channels[0] = 2
            fmt[0] = SPNG_FMT_GA16 if depth == 16 else SPNG_FMT_GA8
            return 0
        channels[0] = 1
        fmt[0] = SPNG_FMT_PNG if depth >= 8 else SPNG_FMT_G8
        return 0
    if color_type == SPNG_COLOR_TYPE_TRUECOLOR:
        if has_trns:
            channels[0] = 4
            fmt[0] = SPNG_FMT_RGBA16 if depth == 16 else SPNG_FMT_RGBA8
            return 0
        channels[0] = 3
        fmt[0] = SPNG_FMT_PNG
        return 0
    if color_type == SPNG_COLOR_TYPE_GRAYSCALE_ALPHA:
        channels[0] = 2
        fmt[0] = SPNG_FMT_PNG
        return 0
    if color_type == SPNG_COLOR_TYPE_TRUECOLOR_ALPHA:
        channels[0] = 4
        fmt[0] = SPNG_FMT_PNG
        return 0
    raise PngError(f'unsupported PNG color type {color_type}')


def decode(data, *, out=None):
    """Decode a PNG byte string to a numpy array.

    Parameters
    ----------
    out : np.ndarray | None, optional
        Preallocated output array. Matches imagecodecs's ``out=`` API.

        * ``None`` (default): allocate a fresh array sized + typed
          from the PNG header.
        * ``np.ndarray``: decode in-place into the provided array.
          Must be C-contiguous, the correct dtype (uint8 / uint16)
          for the PNG bit depth, and the correct shape — matching
          what the default path would have produced. Returns the
          same array. Zero-alloc fast path for tile / page reuse.
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        spng_ctx* ctx = NULL
        spng_ihdr ihdr
        int rc
        int fmt
        int flags = 0
        int channels = 0
        bint wide = False
        size_t out_size
        cnp.ndarray out_arr
        cnp.npy_intp shape[3]
        int ndim
        object dtype

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if srcsize < 8:
        raise PngError('input too short to be a PNG')

    ctx = spng_ctx_new(0)
    if ctx == NULL:
        raise PngError('spng_ctx_new failed')
    try:
        # Generous limits; libspng default is conservative.
        spng_set_image_limits(ctx, 200000, 200000)
        spng_set_chunk_limits(ctx, 64 * 1024 * 1024, 64 * 1024 * 1024)
        rc = spng_set_png_buffer(ctx, <const void*> &src[0], srcsize)
        _check(rc, 'spng_set_png_buffer')
        rc = spng_get_ihdr(ctx, &ihdr)
        _check(rc, 'spng_get_ihdr')

        # Pick the output format + numpy shape/dtype from the color
        # type, bit depth and tRNS (see _decode_layout). All formats
        # used return host byte order.
        rc = _decode_layout(ctx, &ihdr, &fmt, &flags, &channels, &wide)
        _check(rc, 'spng_get_trns')
        ndim = 2 if channels == 1 else 3
        shape[2] = channels
        dtype = np.uint16 if wide else np.uint8

        rc = spng_decoded_image_size(ctx, fmt, &out_size)
        _check(rc, 'spng_decoded_image_size')

        shape[0] = ihdr.height
        shape[1] = ihdr.width

        if out is not None:
            # Caller supplied a destination — validate it matches what
            # the default path would have produced and decode in-place.
            if not isinstance(out, np.ndarray):
                raise TypeError(
                    f"png decode: out= must be an ndarray, "
                    f"got {type(out).__name__}")
            expected_shape = (
                (int(shape[0]), int(shape[1]))
                if ndim == 2
                else (int(shape[0]), int(shape[1]), int(shape[2]))
            )
            if out.shape != expected_shape:
                raise ValueError(
                    f"png decode: out= shape {out.shape} does not match "
                    f"expected {expected_shape}")
            if out.dtype != dtype:
                raise ValueError(
                    f"png decode: out= dtype {out.dtype} does not match "
                    f"expected {np.dtype(dtype)}")
            if not out.flags['C_CONTIGUOUS']:
                raise ValueError(
                    "png decode: out= must be C-contiguous")
            out_arr = out
        else:
            out_arr = cnp.PyArray_EMPTY(
                ndim, shape,
                cnp.NPY_UINT16 if dtype is np.uint16 else cnp.NPY_UINT8,
                0,
            )

        if out_arr.nbytes != <Py_ssize_t> out_size:
            raise PngError(
                f'decoded image size mismatch: spng={out_size} '
                f'numpy={out_arr.nbytes}')

        with nogil:
            rc = spng_decode_image(
                ctx, cnp.PyArray_DATA(out_arr), out_size, fmt, flags,
            )
        _check(rc, 'spng_decode_image')

        return out_arr
    finally:
        spng_ctx_free(ctx)


def read_icc_profile(data) -> bytes | None:
    """Return the embedded ICC profile bytes from a PNG, or None.

    Reads only the IHDR + iCCP chunks from the header — fast even for
    large PNGs (libspng's chunk walker stops once iCCP has been seen
    or a non-ancillary chunk forces decoding to start).
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        spng_ctx* ctx = NULL
        spng_iccp iccp
        int rc

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if srcsize < 8:
        return None
    ctx = spng_ctx_new(0)
    if ctx == NULL:
        raise PngError('spng_ctx_new failed')
    try:
        rc = spng_set_png_buffer(ctx, <const void*> &src[0], srcsize)
        _check(rc, 'spng_set_png_buffer')
        rc = spng_get_iccp(ctx, &iccp)
        if rc != 0:
            # No iCCP chunk (rc=SPNG_ECHUNKAVAIL) or other error;
            # either way, no profile to return.
            return None
        if iccp.profile == NULL or iccp.profile_len == 0:
            return None
        # libspng owns the bytes; copy them out into a Python bytes
        # before the ctx is destroyed.
        return PyBytes_FromStringAndSize(
            iccp.profile, <Py_ssize_t> iccp.profile_len)
    finally:
        spng_ctx_free(ctx)


_FILTER_CHOICE_MAP = {
    # Friendly aliases for the SPNG_FILTER_CHOICE bitmask.
    "none":   SPNG_FILTER_CHOICE_NONE,
    "sub":    SPNG_FILTER_CHOICE_SUB,
    "up":     SPNG_FILTER_CHOICE_UP,
    "avg":    SPNG_FILTER_CHOICE_AVG,
    "paeth":  SPNG_FILTER_CHOICE_PAETH,
    "all":    SPNG_FILTER_CHOICE_ALL,
    # Useful presets (matches the heuristics libpng uses):
    "fast":   SPNG_FILTER_CHOICE_NONE | SPNG_FILTER_CHOICE_SUB
              | SPNG_FILTER_CHOICE_UP,
    "off":    SPNG_DISABLE_FILTERING,
    # imagecodecs' name for the same: PNG.FILTER.NO.
    "no":     SPNG_DISABLE_FILTERING,
}

# Marks a filter_choice the caller did not pass, so that the
# imagecodecs-style ``filter=`` alias can be told apart from it.
_UNSET = object()


class PngOptionError(PngError, ValueError):
    """An invalid PNG encode option.

    A ``PngError`` as the whole-image encoder raised before, and a
    ``ValueError`` as the row encoder raised, so either ``except`` works.
    """


# imagecodecs' PNG.STRATEGY names for the zlib strategies.
_STRATEGY_MAP = {
    "default": 0,
    "filtered": 1,
    "huffman_only": 2,
    "rle": 3,
    "fixed": 4,
}


def _strategy_option(strategy):
    """Resolve strategy= to a zlib strategy 0-4, or None.

    Accepts the numbers and imagecodecs' PNG.STRATEGY names; anything
    else raises, as in imagecodecs, instead of reaching zlib.
    """
    if strategy is None:
        return None
    if isinstance(strategy, str):
        key = strategy.lower().strip()
        if key not in _STRATEGY_MAP:
            raise PngOptionError(
                f'PNG encode: unknown strategy {strategy!r}'
                f'; expected one of {sorted(_STRATEGY_MAP)}')
        return _STRATEGY_MAP[key]
    value = int(strategy)
    if not 0 <= value <= 4:
        raise PngOptionError(
            f'PNG encode: strategy {value} is not a zlib strategy (0-4)')
    return value


def _filter_option(filter_choice, filter):
    """Resolve filter_choice / filter= to a libspng bitmask, or None.

    ``filter`` is the imagecodecs name for the same setting. Its
    PNG.FILTER values (NO=0, NONE=8, SUB=16, UP=32, AVG=64, PAETH=128,
    FAST=56, ALL=248) are the libpng PNG_FILTER_* bits, which are also
    the libspng SPNG_FILTER_CHOICE bits, so they pass straight through;
    0 writes every row unfiltered in both libraries. Its member names
    work as strings too. None means libspng's own default (try all
    five filters).
    """
    if filter is not None:
        if filter_choice is not _UNSET:
            raise TypeError("PNG encode: pass filter= or filter_choice=, not both")
        filter_choice = filter
    elif filter_choice is _UNSET:
        filter_choice = "fast"
    if filter_choice is None:
        return None
    if isinstance(filter_choice, str):
        key = filter_choice.lower().strip()
        if key not in _FILTER_CHOICE_MAP:
            raise PngOptionError(
                f'PNG encode: unknown filter_choice {filter_choice!r}'
                f'; expected one of {sorted(_FILTER_CHOICE_MAP)}')
        return _FILTER_CHOICE_MAP[key]
    choice = int(filter_choice)
    if choice < 0 or choice & ~SPNG_FILTER_CHOICE_ALL:
        raise PngOptionError(
            f'PNG encode: filter bitmask {choice} has bits outside '
            f'{int(SPNG_FILTER_CHOICE_ALL)} (NONE|SUB|UP|AVG|PAETH)')
    return choice


def _native_samples(arr):
    """Return ``arr`` with native byte order, keeping the sample values.

    A big-endian uint16 array holding 1 must be written as the PNG
    sample 1 (PNG specification section 7.1: samples are stored most
    significant byte first). Reinterpreting its bytes would write 256.
    """
    if arr.dtype.kind == 'u' and arr.dtype.itemsize > 1 and not arr.dtype.isnative:
        return arr.astype(arr.dtype.newbyteorder('='))
    return arr


def encode(data, *, level=None,
           filter_choice: object = _UNSET,
           strategy=None,
           iccprofile: bytes | None = None,
           iccprofile_name: str = "ICC profile",
           filter: object = None) -> bytes:
    """Encode a numpy array as a PNG byte string.

    uint16 input of either byte order is accepted; the sample values are
    what gets stored.

    ``filter_choice`` controls which of the 5 PNG row filters libspng
    considers when picking the best per scanline. Accepted values:

      * ``"fast"`` (default) — try NONE/SUB/UP (3 of 5). This is the
        same trade-off libpng's heuristic mode makes: within 1-5% of
        full-search file size on real-world content, ~1.5× faster
        encode. Brings opencodecs to parity with imagecodecs on the
        head-to-head bench.
      * ``"all"`` — try all 5 filters per row. Produces the smallest
        output (libspng's built-in default; was opencodecs's default
        before this change). Use when you care about file size and not
        about encode wall-clock.
      * ``"off"`` — disable filtering entirely (fastest; matches
        ``SPNG_DISABLE_FILTERING``). Sometimes the smallest output on
        very noisy data where filters hurt rather than help (e.g.
        Poisson-noise-dominated microscopy frames).
      * ``"none"`` / ``"sub"`` / ``"up"`` / ``"avg"`` / ``"paeth"`` —
        restrict to a single filter.
      * any int — passed through as the raw bitmask.
      * ``None``: libspng's own default, all five filters.

    ``filter`` is the imagecodecs spelling of the same setting and takes
    imagecodecs' ``PNG.FILTER`` values or names; pass one or the other.
    An invalid filter or strategy raises ``PngOptionError``, both a
    ``PngError`` and a ``ValueError``.

    ``strategy`` sets the zlib compression strategy
    (``Z_DEFAULT_STRATEGY=0``, ``Z_FILTERED=1``, ``Z_HUFFMAN_ONLY=2``,
    ``Z_RLE=3``, ``Z_FIXED=4``, or imagecodecs' ``PNG.STRATEGY`` names),
    as imagecodecs' ``strategy=`` does.
    ``None`` (default) compresses with the fastest backend the build
    has (libdeflate when available, which has no strategy setting);
    any explicit value compresses the image data through zlib with
    that strategy, so the setting is always honored.
    """
    cdef:
        spng_ctx* ctx = NULL
        spng_ihdr ihdr
        spng_iccp iccp
        bytes _icc_bytes_keep
        bytes _icc_name_keep
        int rc
        int fmt
        size_t img_len
        size_t buf_len = 0
        int err = 0
        void* png_buf
        cnp.ndarray arr
        bytes out
        int compression
        bint need_byteswap = False

    arr = _native_samples(np.ascontiguousarray(data))
    fc = _filter_option(filter_choice, filter)
    strategy = _strategy_option(strategy)

    if arr.dtype not in (np.uint8, np.uint16):
        raise PngError(f'PNG encode: unsupported dtype {arr.dtype}')

    # Determine color_type and channels.
    if arr.ndim == 2:
        color_type = SPNG_COLOR_TYPE_GRAYSCALE
        channels = 1
    elif arr.ndim == 3:
        c = arr.shape[2]
        if c == 1:
            color_type = SPNG_COLOR_TYPE_GRAYSCALE
            channels = 1
            arr = arr[:, :, 0]
        elif c == 2:
            color_type = SPNG_COLOR_TYPE_GRAYSCALE_ALPHA
            channels = 2
        elif c == 3:
            color_type = SPNG_COLOR_TYPE_TRUECOLOR
            channels = 3
        elif c == 4:
            color_type = SPNG_COLOR_TYPE_TRUECOLOR_ALPHA
            channels = 4
        else:
            raise PngError(
                f'PNG encode: unsupported number of channels {c}')
    else:
        raise PngError(
            f'PNG encode: unsupported array ndim {arr.ndim}')

    bit_depth = 16 if arr.dtype == np.uint16 else 8

    # spng_encode_image only accepts SPNG_FMT_PNG (host-endian, no
    # conversion) or SPNG_FMT_RAW (big-endian). Use SPNG_FMT_PNG; spng
    # converts to PNG file byte order (big-endian) internally.
    fmt = SPNG_FMT_PNG
    arr = np.ascontiguousarray(arr)

    ctx = spng_ctx_new(SPNG_CTX_ENCODER)
    if ctx == NULL:
        raise PngError('spng_ctx_new(ENCODER) failed')
    try:
        # Internal buffer mode (default; spng allocates and we read it back).
        ihdr.width = arr.shape[0] if arr.ndim == 1 else (
            <uint32_t> arr.shape[1])
        ihdr.height = <uint32_t> arr.shape[0]
        ihdr.bit_depth = <uint8_t> bit_depth
        ihdr.color_type = <uint8_t> color_type
        ihdr.compression_method = 0
        ihdr.filter_method = 0
        ihdr.interlace_method = 0
        rc = spng_set_ihdr(ctx, &ihdr)
        _check(rc, 'spng_set_ihdr')

        # Tell spng to allocate the output buffer internally; we fetch it
        # back via spng_get_png_buffer after encode.
        rc = spng_set_option(ctx, SPNG_ENCODE_TO_BUFFER, 1)
        _check(rc, 'spng_set_option(SPNG_ENCODE_TO_BUFFER)')

        if level is not None:
            compression = int(level)
            if compression < 0: compression = 0
            if compression > 9: compression = 9
            rc = spng_set_option(
                ctx, SPNG_IMG_COMPRESSION_LEVEL, compression)
            _check(rc, 'spng_set_option(compression_level)')

        # Filter-choice tuning. libspng's default (SPNG_FILTER_CHOICE_ALL)
        # tries all 5 PNG filters per scanline; that's the smallest
        # output but slowest encode. For real photographic / gradient
        # data the savings vs "fast" (NONE+SUB+UP) are tiny but the
        # speedup is ~1.5x. For incompressible (random RGB) data
        # disabling filtering entirely is ~2x faster with identical
        # output size.
        if fc is not None:
            rc = spng_set_option(ctx, SPNG_FILTER_CHOICE, <int> fc)
            _check(rc, 'spng_set_option(filter_choice)')

        if strategy is not None:
            rc = spng_set_option(
                ctx, SPNG_IMG_COMPRESSION_STRATEGY, strategy)
            _check(rc, 'spng_set_option(compression_strategy)')

        # Embed iCCP chunk if the caller provided an ICC profile.
        # libspng copies the bytes out of iccp.profile + the
        # null-terminated name out of iccp.profile_name on
        # spng_set_iccp, but we still hold _icc_bytes_keep alive for
        # the duration of the call so the borrowed pointers stay
        # valid even if Cython would otherwise GC them.
        if iccprofile is not None:
            _icc_bytes_keep = bytes(iccprofile)
            # PNG iCCP profile-name field is up to 79 bytes ASCII.
            _name = (iccprofile_name or "ICC profile")
            _icc_name_keep = _name.encode("ascii", errors="replace")[:79]
            # Zero the struct (Cython initializes but be explicit) and
            # populate the fields libspng expects.
            for i in range(80):
                iccp.profile_name[i] = 0
            for i in range(len(_icc_name_keep)):
                iccp.profile_name[i] = _icc_name_keep[i]
            iccp.profile_len = <size_t> len(_icc_bytes_keep)
            iccp.profile = <char*> _icc_bytes_keep
            rc = spng_set_iccp(ctx, &iccp)
            _check(rc, 'spng_set_iccp')

        img_len = <size_t> arr.nbytes
        with nogil:
            rc = spng_encode_image(
                ctx, cnp.PyArray_DATA(arr), img_len, fmt,
                SPNG_ENCODE_FINALIZE,
            )
        _check(rc, 'spng_encode_image')

        png_buf = spng_get_png_buffer(ctx, &buf_len, &err)
        if png_buf == NULL or err != 0:
            raise PngError(
                f'spng_get_png_buffer: {spng_strerror(err).decode()}')
        try:
            out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> buf_len)
            memcpy(<void*> PyBytes_AsString(out), png_buf, buf_len)
            return out
        finally:
            free(png_buf)
    finally:
        spng_ctx_free(ctx)


def check_signature(data) -> bool:
    """True if `data` starts with the 8-byte PNG signature."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:8])
    else:
        try:
            head = bytes(data)[:8]
        except Exception:
            return False
    return head == b'\x89PNG\r\n\x1a\n'


from spng cimport (spng_set_png_stream, spng_row_info, spng_get_row_info,
    spng_decode_scanline, spng_decode_chunks, spng_encode_row,
    SPNG_DECODE_PROGRESSIVE, SPNG_ENCODE_PROGRESSIVE, SPNG_EOI)


cdef int _row_read_callback(spng_ctx* ctx, void* user, void* dest,
                            size_t length) noexcept with gil:
    cdef object bridge = <object> user
    cdef bytes data
    try:
        data = bridge.read_exact(length)
        if len(data) != length:
            raise EOFError("truncated PNG input")
        memcpy(dest, PyBytes_AsString(data), length)
        return 0
    except BaseException as exc:
        bridge.error = exc
        return -1


cdef int _row_write_callback(spng_ctx* ctx, void* user, void* source,
                             size_t length) noexcept with gil:
    cdef object bridge = <object> user
    try:
        bridge.write(PyBytes_FromStringAndSize(<const char*> source, length))
        return 0
    except BaseException as exc:
        bridge.error = exc
        return -1


cdef class RowDecoder:
    """Decode one PNG scanline per call, including explicit Adam7 updates."""
    cdef spng_ctx* _ctx
    cdef object _bridge
    cdef object _dtype
    cdef unsigned _width
    cdef unsigned _height
    cdef int _channels
    cdef bint _interlaced
    cdef bint _finished

    def __init__(self, bridge):
        cdef spng_ihdr header
        cdef int status
        cdef int fmt = SPNG_FMT_PNG
        cdef int flags = 0
        cdef int channels = 0
        cdef bint wide = False
        self._bridge = bridge
        self._ctx = spng_ctx_new(0)
        if self._ctx == NULL:
            raise MemoryError()
        _check(spng_set_image_limits(self._ctx, 200000, 200000), "image limits")
        _check(spng_set_chunk_limits(self._ctx, 64 << 20, 64 << 20), "chunk limits")
        _check(spng_set_png_stream(self._ctx, _row_read_callback, <void*> bridge),
               "PNG read callback")
        with nogil:
            status = spng_get_ihdr(self._ctx, &header)
        self._check_status(status)
        self._width = header.width
        self._height = header.height
        self._interlaced = header.interlace_method != 0
        # Same color-type rule as decode(), so rows and whole images agree.
        status = _decode_layout(self._ctx, &header, &fmt, &flags, &channels, &wide)
        self._check_status(status)
        self._channels = channels
        self._dtype = np.dtype(np.uint16 if wide else np.uint8)
        with nogil:
            status = spng_decode_image(self._ctx, NULL, 0, fmt,
                                       flags | SPNG_DECODE_PROGRESSIVE)
        self._check_status(status)

    cdef _check_status(self, int status):
        if self._bridge.error is not None:
            raise self._bridge.error
        _check(status, "PNG row decode")

    @property
    def info(self):
        shape = (self._height, self._width)
        if self._channels != 1:
            shape += (self._channels,)
        return {"shape": shape, "dtype": self._dtype, "interlaced": bool(self._interlaced)}

    def read_row(self):
        cdef spng_row_info info
        cdef int status
        cdef int x_start = 0
        cdef int x_step = 1
        cdef Py_ssize_t width
        cdef cnp.ndarray row
        cdef size_t size
        cdef void* output
        if self._ctx == NULL:
            raise ValueError("PNG row decoder is closed")
        if self._finished:
            return None
        self._check_status(spng_get_row_info(self._ctx, &info))
        if self._interlaced:
            x_start = (0, 4, 0, 2, 0, 1, 0)[info.pass_num]
            x_step = (8, 8, 4, 4, 2, 2, 1)[info.pass_num]
        width = (self._width - x_start + x_step - 1) // x_step
        if self._channels == 1:
            shape = (width,)
        else:
            shape = (width, self._channels)
        row = np.empty(shape, dtype=self._dtype)
        output = <void*> row.data
        size = row.nbytes
        with nogil:
            status = spng_decode_scanline(self._ctx, output, size)
        if status == SPNG_EOI:
            self._finished = True
            # Validate the remainder, including the final chunk checksum.
            with nogil:
                status = spng_decode_chunks(self._ctx)
        self._check_status(status)
        return int(info.row_num), x_start, x_step, int(info.pass_num), row

    def close(self):
        if self._ctx != NULL:
            spng_ctx_free(self._ctx)
            self._ctx = NULL
        self._bridge = None

    def __dealloc__(self):
        if self._ctx != NULL:
            spng_ctx_free(self._ctx)


cdef class RowEncoder:
    """Encode sequential ordinary PNG rows directly to a destination callback."""
    cdef spng_ctx* _ctx
    cdef object _bridge
    cdef object _dtype
    cdef object _row_shape
    cdef unsigned _height
    cdef unsigned _written

    def __init__(self, bridge, shape, dtype, *, level=None, filter_choice=_UNSET,
                 strategy=None, iccprofile=None, iccprofile_name="ICC profile",
                 filter=None):
        cdef spng_ihdr header
        cdef spng_iccp profile
        cdef int channels
        cdef int status
        cdef bytes profile_data
        cdef bytes profile_name
        choice = _filter_option(filter_choice, filter)
        self._bridge = bridge
        # Rows of either byte order are accepted; samples are stored by value.
        self._dtype = np.dtype(dtype)
        if self._dtype.kind == 'u' and not self._dtype.isnative:
            self._dtype = self._dtype.newbyteorder('=')
        shape = tuple(shape)
        if len(shape) not in (2, 3) or any(int(n) != n or n <= 0 for n in shape):
            raise ValueError("PNG row shape must have positive height, width and optional channels")
        if shape[0] > 200000 or shape[1] > 200000:
            raise ValueError("PNG row shape exceeds image limits")
        channels = 1 if len(shape) == 2 else shape[2]
        if channels not in (1, 2, 3, 4):
            raise ValueError("PNG rows need 1, 2, 3 or 4 channels")
        if self._dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
            raise ValueError("PNG rows need uint8 or uint16 samples")
        self._height = shape[0]
        self._row_shape = shape[1:]
        self._ctx = spng_ctx_new(SPNG_CTX_ENCODER)
        if self._ctx == NULL:
            raise MemoryError()
        header.width = shape[1]
        header.height = shape[0]
        header.bit_depth = self._dtype.itemsize * 8
        header.color_type = (SPNG_COLOR_TYPE_GRAYSCALE,
                             SPNG_COLOR_TYPE_GRAYSCALE_ALPHA,
                             SPNG_COLOR_TYPE_TRUECOLOR,
                             SPNG_COLOR_TYPE_TRUECOLOR_ALPHA)[channels - 1]
        header.compression_method = 0
        header.filter_method = 0
        header.interlace_method = 0
        _check(spng_set_ihdr(self._ctx, &header), "PNG row header")
        _check(spng_set_png_stream(self._ctx, _row_write_callback, <void*> bridge),
               "PNG write callback")
        if level is not None:
            _check(spng_set_option(self._ctx, SPNG_IMG_COMPRESSION_LEVEL,
                                   max(0, min(9, int(level)))), "PNG compression level")
        if choice is not None:
            _check(spng_set_option(self._ctx, SPNG_FILTER_CHOICE, <int> choice),
                   "PNG filter choice")
        strategy = _strategy_option(strategy)
        if strategy is not None:
            _check(spng_set_option(self._ctx, SPNG_IMG_COMPRESSION_STRATEGY, strategy),
                   "PNG compression strategy")
        if iccprofile is not None:
            profile_data = bytes(iccprofile)
            profile_name = str(iccprofile_name).encode("ascii", "replace")[:79]
            memset(profile.profile_name, 0, 80)
            memcpy(profile.profile_name, PyBytes_AsString(profile_name), len(profile_name))
            profile.profile_len = len(profile_data)
            profile.profile = PyBytes_AsString(profile_data)
            _check(spng_set_iccp(self._ctx, &profile), "PNG color profile")
        with nogil:
            status = spng_encode_image(self._ctx, NULL, 0, SPNG_FMT_PNG,
                                       SPNG_ENCODE_PROGRESSIVE | SPNG_ENCODE_FINALIZE)
        self._check_status(status)

    cdef _check_status(self, int status):
        if self._bridge.error is not None:
            raise self._bridge.error
        _check(status, "PNG row encode")

    def write_row(self, row):
        cdef cnp.ndarray array
        cdef int status
        cdef const void* data
        cdef size_t size
        if self._ctx == NULL:
            raise ValueError("PNG row encoder is closed")
        if self._written >= self._height:
            raise ValueError("too many PNG rows")
        array = _native_samples(np.asarray(row))
        if tuple(array.shape[i] for i in range(array.ndim)) != self._row_shape:
            raise ValueError("PNG row has the wrong shape")
        if array.dtype != self._dtype:
            raise ValueError("PNG row has the wrong dtype")
        array = np.ascontiguousarray(array)
        data = <const void*> array.data
        size = array.nbytes
        with nogil:
            status = spng_encode_row(self._ctx, data, size)
        if status == SPNG_EOI:
            status = 0
        self._check_status(status)
        self._written += 1

    def finish(self):
        if self._written != self._height:
            raise ValueError("not enough PNG rows")

    def close(self):
        if self._ctx != NULL:
            spng_ctx_free(self._ctx)
            self._ctx = NULL
        self._bridge = None

    def __dealloc__(self):
        if self._ctx != NULL:
            spng_ctx_free(self._ctx)
