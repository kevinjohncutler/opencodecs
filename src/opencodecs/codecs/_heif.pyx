# opencodecs/codecs/_heif.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native HEIF / HEIC codec via libheif (system; depends on libde265 / x265)."""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memset
from libc.stdint cimport uint8_t, uint16_t, int64_t

import numpy as np
cimport numpy as cnp

from heif cimport (
    heif_init, heif_context, heif_context_alloc, heif_context_free,
    heif_error_code, heif_chroma, heif_colorspace,
    heif_context_read_from_memory_without_copy,
    heif_reader, heif_reader_grow_status,
    heif_reader_grow_status_size_reached, heif_reader_grow_status_size_beyond_eof,
    heif_context_read_from_reader,
    heif_context_get_primary_image_handle,
    heif_context_get_number_of_top_level_images,
    heif_context_get_list_of_top_level_image_IDs,
    heif_context_get_image_handle, heif_item_id,
    heif_image_handle, heif_image_handle_release,
    heif_image_handle_get_width, heif_image_handle_get_height,
    heif_image_handle_has_alpha_channel,
    heif_image_handle_get_luma_bits_per_pixel,
    heif_decode_image, heif_image, heif_image_release,
    heif_image_get_plane_readonly, heif_image_get_plane,
    heif_image_create, heif_image_add_plane,
    heif_colorspace_RGB, heif_colorspace_monochrome,
    heif_chroma_monochrome,
    heif_image_handle_get_preferred_decoding_colorspace,
    LIBHEIF_AUX_IMAGE_FILTER_OMIT_ALPHA,
    heif_image_handle_get_number_of_auxiliary_images,
    heif_image_handle_get_list_of_auxiliary_image_IDs,
    heif_image_handle_get_auxiliary_image_handle,
    heif_image_get_bits_per_pixel_range,
    heif_channel_Y, heif_channel_Alpha,
    heif_chroma_interleaved_RGB, heif_chroma_interleaved_RGBA,
    heif_chroma_interleaved_RRGGBB_LE, heif_chroma_interleaved_RRGGBBAA_LE,
    heif_channel_interleaved,
    heif_compression_HEVC,
    heif_context_get_encoder_for_format,
    heif_encoder, heif_encoder_release,
    heif_encoder_set_lossy_quality, heif_encoder_set_lossless,
    heif_encoder_set_parameter_string,
    heif_encoder_set_parameter_integer,
    heif_context_set_max_decoding_threads,
    heif_context_encode_image,
    heif_writer, heif_context_write,
    heif_error, heif_decoding_options,
    heif_color_profile_nclx,
    heif_nclx_color_profile_alloc, heif_nclx_color_profile_free,
    heif_nclx_color_profile_set_color_primaries,
    heif_nclx_color_profile_set_transfer_characteristics,
    heif_nclx_color_profile_set_matrix_coefficients,
    heif_image_set_nclx_color_profile,
    heif_image_handle_get_raw_color_profile_size,
    heif_image_handle_get_raw_color_profile,
    heif_image_set_raw_color_profile,
)

cnp.import_array()


class HeifError(RuntimeError):
    """Raised on HEIF/HEIC encode/decode failures."""


cdef bint _heif_initialized = False


cdef int64_t _source_position(void* userdata) noexcept with gil:
    return (<object>userdata).position


cdef int _source_seek(int64_t position, void* userdata) noexcept with gil:
    source = <object>userdata
    if position < 0 or position > source.size:
        return -1
    source.position = position
    return 0


cdef heif_reader_grow_status _source_size(int64_t size, void* userdata) noexcept with gil:
    return (heif_reader_grow_status_size_reached if size <= (<object>userdata).size
            else heif_reader_grow_status_size_beyond_eof)


cdef int _source_read(void* output, size_t size, void* userdata) noexcept with gil:
    source = <object>userdata
    cdef const uint8_t[::1] data
    cdef size_t offset = 0
    try:
        while offset < size:
            data = source.read_at(source.position, size - offset)
            if not data.shape[0]:
                raise EOFError("truncated HEIF source")
            memcpy(<uint8_t*>output + offset, &data[0], data.shape[0])
            offset += data.shape[0]
            source.position += data.shape[0]
        return 0
    except BaseException as error:
        source.error = error
        return -1


cdef heif_reader _source_reader
memset(&_source_reader, 0, sizeof(heif_reader))
_source_reader.reader_api_version = 1
_source_reader.get_position = _source_position
_source_reader.read = _source_read
_source_reader.seek = _source_seek
_source_reader.wait_for_file_size = _source_size


cdef _ensure_init():
    global _heif_initialized
    if not _heif_initialized:
        heif_init(NULL)
        _heif_initialized = True


def frame_count(data) -> int:
    """How many top-level images the file holds; 1 for a plain still.

    Reads the container's metadata only -- no image is decoded to
    answer this.
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        heif_context* ctx = NULL
        heif_error err
        int n

    _ensure_init()
    if hasattr(data, "read_at"):
        data.reset()
        src = b""
        srcsize = data.size
    else:
        if isinstance(data, (bytes, bytearray)):
            src = data
        else:
            src = bytes(data)
        srcsize = <size_t> src.shape[0]
    ctx = heif_context_alloc()
    if ctx == NULL:
        raise HeifError('heif_context_alloc failed')
    try:
        if hasattr(data, "read_at"):
            err = heif_context_read_from_reader(ctx, &_source_reader, <void*>data, NULL)
            data.raise_error()
        else:
            err = heif_context_read_from_memory_without_copy(
                ctx, &src[0], srcsize, NULL)
        if err.code != 0:
            raise HeifError(
                f'heif_context_read_from_memory: {err.message.decode()}')
        n = heif_context_get_number_of_top_level_images(ctx)
        return int(n)
    finally:
        heif_context_free(ctx)


_GRAY_NAMES = frozenset((
    'GRAY', 'BLACKISZERO', 'MINISBLACK', 'WHITEISZERO', 'MINISWHITE',
    'MONOCHROME'))


def _wants_monochrome(photometric) -> bool:
    """Read imagecodecs' ``photometric`` decode keyword.

    imagecodecs.heif_decode takes a libheif colorspace (0 YCbCr, 1 RGB,
    2 monochrome, 99 undefined) or a name, and returns one or two
    planes only for monochrome; every other accepted value means the
    default RGB(A) output. Anything else raises, as it does there.
    """
    if photometric is None:
        return False
    if isinstance(photometric, str):
        name = photometric.upper()
        if name[:3] == 'RGB' or name[:5] == 'YCBCR':
            return False
        if name in _GRAY_NAMES:
            return True
    elif not isinstance(photometric, bool):
        try:
            value = int(photometric)
        except (TypeError, ValueError):
            value = None
        if value is not None and value == photometric:
            if value in (0, 1, 99):
                return False
            if value == 2:
                return True
    raise ValueError(
        f'heif decode: photometric={photometric!r} is not supported; '
        f"use 'rgb', 'ycbcr' or 'monochrome' (or libheif's colorspace "
        f'numbers 0, 1, 2, 99)')


cdef int _alpha_bits(const heif_image_handle* handle) except -2:
    """Bit depth of the image's alpha plane, or -1 if none is found.

    HEIF stores alpha as an auxiliary image with a bit depth of its own,
    which need not equal the main image's. libheif lists it among the
    auxiliary images unless asked to omit alpha, so the alpha is the id
    present in the full list and missing from the filtered one.
    """
    cdef int n_all, n_rest, i, j, bits = -1
    cdef bint skip
    cdef heif_item_id* all_ids = NULL
    cdef heif_item_id* rest_ids = NULL
    cdef heif_image_handle* aux = NULL
    cdef heif_error err
    n_all = heif_image_handle_get_number_of_auxiliary_images(handle, 0)
    n_rest = heif_image_handle_get_number_of_auxiliary_images(
        handle, LIBHEIF_AUX_IMAGE_FILTER_OMIT_ALPHA)
    if n_all <= 0 or n_all <= n_rest:
        return -1
    all_ids = <heif_item_id*> malloc(n_all * sizeof(heif_item_id))
    rest_ids = <heif_item_id*> malloc((n_rest + 1) * sizeof(heif_item_id))
    try:
        if all_ids == NULL or rest_ids == NULL:
            raise MemoryError('heif: could not allocate the auxiliary id list')
        n_all = heif_image_handle_get_list_of_auxiliary_image_IDs(
            handle, 0, all_ids, n_all)
        if n_rest > 0:
            n_rest = heif_image_handle_get_list_of_auxiliary_image_IDs(
                handle, LIBHEIF_AUX_IMAGE_FILTER_OMIT_ALPHA, rest_ids, n_rest)
        for i in range(n_all):
            skip = False
            for j in range(n_rest):
                if rest_ids[j] == all_ids[i]:
                    skip = True
                    break
            if skip:
                continue
            err = heif_image_handle_get_auxiliary_image_handle(
                handle, all_ids[i], &aux)
            if err.code == 0 and aux != NULL:
                bits = heif_image_handle_get_luma_bits_per_pixel(aux)
                heif_image_handle_release(aux)
                aux = NULL
                break
    finally:
        free(all_ids)
        free(rest_ids)
    return bits


# libheif hands the HEVC decoder its own thread count only from 1.21
# (heif_decoding_options version 8, num_codec_threads); before that, and
# when this is left at 0, its libde265 plugin starts one worker thread, so
# a single coded image decoded on one core however many were asked for.
# libde265 threads over the wavefront rows that x265 writes by default
# (and over tiles, which x265 does not write): an untiled 4096 x 3072
# image went from 953 to 134 ms on 16 threads, pixel-identical. A grid
# image is many small coded images that libheif already decodes in
# parallel (heif_context_set_max_decoding_threads), so those keep one
# codec thread each rather than multiplying the two counts.
cdef extern from *:
    """
    #include <libheif/heif.h>
    #if LIBHEIF_NUMERIC_VERSION >= 0x01150000
    #include <libheif/heif_items.h>
    static heif_decoding_options* oc_heif_decoding_options(
            heif_context* ctx, const heif_image_handle* handle, int threads) {
        heif_decoding_options* options = heif_decoding_options_alloc();
        uint32_t type = heif_item_get_item_type(
            ctx, heif_image_handle_get_item_id(handle));
        if (options != NULL && options->version >= 8)
            options->num_codec_threads =
                type == (((uint32_t) 'g' << 24) | ((uint32_t) 'r' << 16)
                         | ((uint32_t) 'i' << 8) | (uint32_t) 'd') ? 1 : threads;
        return options;
    }
    static void oc_heif_decoding_options_free(heif_decoding_options* options) {
        if (options != NULL)
            heif_decoding_options_free(options);
    }
    #else
    static heif_decoding_options* oc_heif_decoding_options(
            heif_context* ctx, const heif_image_handle* handle, int threads) {
        (void) ctx; (void) handle; (void) threads;
        return NULL;
    }
    static void oc_heif_decoding_options_free(heif_decoding_options* options) {
        (void) options;
    }
    #endif
    """
    heif_decoding_options* oc_heif_decoding_options(
        heif_context* ctx, const heif_image_handle* handle, int threads)
    void oc_heif_decoding_options_free(heif_decoding_options* options)


def _decode_threads(numthreads):
    """Threads for one decode: as given (at least 1), or this call's share."""
    if numthreads is not None:
        return max(1, int(numthreads))
    from opencodecs.core.parallel import auto_threads
    return auto_threads(None)


def decode(data, *, numthreads: int | None = None, out=None,
           index=None, photometric=None, _info=False) -> np.ndarray:
    """Decode HEIF/HEIC bytes to a numpy array.

    Returns uint8 for 8-bit HEIFs, uint16 for 10/12-bit HEIFs (values in
    the low bits: for 10-bit the array contains values 0..1023, not
    shifted into the upper bits).

    The shape is (H, W, 3) for RGB and (H, W, 4) for RGBA, monochrome
    images included, as imagecodecs.heif_decode returns them (libheif
    copies the gray plane into all three channels). With
    ``photometric='monochrome'`` a monochrome image (HEVC
    chroma_format_idc 0) decodes to its gray plane, (H, W), or (H, W, 2)
    with alpha, the shape it was written from.

    An alpha plane coded at a different bit depth from the image cannot
    share one array with it, so such a file raises HeifError rather
    than returning rescaled or truncated alpha.

    Parameters
    ----------
    numthreads : int, optional
        Threads for the decode. ``None`` (default) takes this call's
        share of the CPU count, at most 16 (``auto_threads``). ``0`` or
        ``1`` decodes on one thread. A grid image (what cameras write)
        decodes its tiles on that many threads; a single coded image
        hands them to the HEVC decoder, which needs libheif 1.21 or
        newer to accept them and uses them only for what the bitstream
        allows to run in parallel (the wavefront rows x265 writes).
    out : np.ndarray | None, optional
        Preallocated output ndarray. See ``_png.decode`` for the full
        contract. libheif allocates its own RGB plane internally and
        we copy out of it row-by-row; out= just lets the caller
        provide the destination of that copy.
    photometric : str or int, optional
        imagecodecs' keyword. None, 'rgb', 'ycbcr' or libheif colorspace
        0, 1 or 99 return RGB(A). 'monochrome' (or 'gray', 'minisblack'
        and the other names imagecodecs accepts, or colorspace 2)
        returns the gray plane of a monochrome image. Asking for
        monochrome from a color image raises ValueError: imagecodecs
        hands back its red channel there, which is not a gray image.
    """
    from opencodecs.core.parallel import parallel_call
    # Registered for the whole call, so concurrent decodes size their
    # threads as shares of the machine (core.parallel.fair_share).
    with parallel_call():
        return _decode(data, numthreads, out, index, photometric, _info)


def _decode(data, numthreads, out, index, photometric, _info):
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        heif_context* ctx = NULL
        heif_image_handle* handle = NULL
        heif_image* img = NULL
        heif_error err
        int width, height, channels, has_alpha, stride
        int img_depth
        int alpha_depth
        int plane_depth
        bint want_monochrome
        int dtype_bytes
        heif_chroma chroma
        heif_colorspace colorspace
        heif_colorspace pref_colorspace
        heif_chroma pref_chroma
        bint monochrome
        const uint8_t* plane
        cnp.ndarray out_arr
        cnp.npy_intp shape[3]
        int y
        int _heif_n
        size_t row_bytes_out
        tuple expected_shape
        object expected_dtype
        int n_top
        int idx
        heif_item_id* ids = NULL
        heif_item_id wanted
        heif_decoding_options* options = NULL

    _ensure_init()
    want_monochrome = _wants_monochrome(photometric)

    if hasattr(data, "read_at"):
        data.reset()
        src = b""
        srcsize = data.size
    else:
        if isinstance(data, (bytes, bytearray)):
            src = data
        else:
            src = bytes(data)
        srcsize = <size_t> src.shape[0]

    ctx = heif_context_alloc()
    if ctx == NULL:
        raise HeifError('heif_context_alloc failed')
    try:
        _heif_n = _decode_threads(numthreads)
        heif_context_set_max_decoding_threads(ctx, _heif_n)
        if hasattr(data, "read_at"):
            err = heif_context_read_from_reader(ctx, &_source_reader, <void*>data, NULL)
            data.raise_error()
        else:
            err = heif_context_read_from_memory_without_copy(
                ctx, &src[0], srcsize, NULL)
        if err.code != 0:
            raise HeifError(
                f'heif_context_read_from_memory: {err.message.decode()}')

        if index is None:
            err = heif_context_get_primary_image_handle(ctx, &handle)
            if err.code != 0:
                raise HeifError(
                    f'get_primary_image_handle: {err.message.decode()}')
        else:
            # A HEIF holds a set of top-level images, one of which is
            # primary. A burst, a Live Photo's stills, a depth capture:
            # all put more than one there, and decoding only the
            # primary drops the rest silently. The IDs are item ids
            # rather than positions, so the list has to be fetched to
            # turn "image 2" into one.
            n_top = heif_context_get_number_of_top_level_images(ctx)
            idx = int(index)
            if idx < 0:
                idx += n_top
            if not 0 <= idx < n_top:
                raise IndexError(
                    f'heif: image {index} out of range for {n_top}')
            ids = <heif_item_id*> malloc(n_top * sizeof(heif_item_id))
            if ids == NULL:
                raise MemoryError('heif: could not allocate the image id list')
            try:
                heif_context_get_list_of_top_level_image_IDs(ctx, ids, n_top)
                wanted = ids[idx]
            finally:
                free(ids)
                ids = NULL
            err = heif_context_get_image_handle(ctx, wanted, &handle)
            if err.code != 0:
                raise HeifError(
                    f'get_image_handle({idx}): {err.message.decode()}')

        width = heif_image_handle_get_width(handle)
        height = heif_image_handle_get_height(handle)
        has_alpha = heif_image_handle_has_alpha_channel(handle)

        # A monochrome image (HEVC chroma_format_idc 0) has one plane.
        # By default it is returned as RGB(A), libheif copying the plane
        # into each channel, as imagecodecs returns it; with
        # photometric='monochrome' it is returned as gray, (H, W), or
        # gray plus alpha, (H, W, 2). Monochrome is the file's own
        # declaration, read from the coded configuration.
        err = heif_image_handle_get_preferred_decoding_colorspace(
            handle, &pref_colorspace, &pref_chroma)
        monochrome = (err.code == 0
                      and pref_colorspace == heif_colorspace_monochrome)
        if want_monochrome and not monochrome:
            raise ValueError(
                "heif decode: photometric='monochrome' asks for the gray "
                "plane, but this image is color; decode it without "
                "photometric to get RGB(A)")
        if monochrome and want_monochrome:
            channels = 2 if has_alpha else 1
        else:
            monochrome = False
            channels = 4 if has_alpha else 3

        # Probe luma bit depth (libheif 1.4+).
        img_depth = heif_image_handle_get_luma_bits_per_pixel(handle)
        if img_depth <= 0:
            img_depth = 8
        dtype_bytes = 1 if img_depth <= 8 else 2

        # Alpha is its own coded image with its own depth. One output
        # array has one dtype and one value range, and libheif's RGBA
        # conversion either refuses a mismatch or, for an alpha deeper
        # than an 8-bit image, keeps the wrong bits, so a mismatch is
        # refused here before anything is decoded.
        if has_alpha:
            alpha_depth = _alpha_bits(handle)
            if alpha_depth > 0 and alpha_depth != img_depth:
                raise HeifError(
                    f'heif decode: the alpha plane is {alpha_depth}-bit '
                    f'but the image is {img_depth}-bit; one array cannot '
                    f'hold both without rescaling one of them, which '
                    f'this decoder does not do')
        if channels == 1:
            expected_shape = (height, width)
        else:
            expected_shape = (height, width, channels)
        expected_dtype = np.uint8 if dtype_bytes == 1 else np.uint16
        if _info:
            return {"shape": expected_shape,
                    "dtype": np.dtype(expected_dtype)}

        if monochrome:
            colorspace = heif_colorspace_monochrome
            chroma = heif_chroma_monochrome
        elif dtype_bytes == 1:
            colorspace = heif_colorspace_RGB
            chroma = (heif_chroma_interleaved_RGBA if has_alpha
                      else heif_chroma_interleaved_RGB)
        else:
            colorspace = heif_colorspace_RGB
            chroma = (heif_chroma_interleaved_RRGGBBAA_LE if has_alpha
                      else heif_chroma_interleaved_RRGGBB_LE)

        options = oc_heif_decoding_options(ctx, handle, _heif_n)
        try:
            with nogil:
                err = heif_decode_image(
                    handle, &img, colorspace, chroma, options,
                )
        finally:
            oc_heif_decoding_options_free(options)
        if hasattr(data, "raise_error"):
            data.raise_error()
        if err.code != 0:
            raise HeifError(f'heif_decode_image: {err.message.decode()}')

        shape[0] = height
        shape[1] = width
        shape[2] = channels

        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError(
                    f"heif decode: out= must be an ndarray, "
                    f"got {type(out).__name__}")
            if out.shape != expected_shape:
                raise ValueError(
                    f"heif decode: out= shape {out.shape} does not match "
                    f"expected {expected_shape}")
            if out.dtype != expected_dtype:
                raise ValueError(
                    f"heif decode: out= dtype {out.dtype} does not match "
                    f"expected {np.dtype(expected_dtype)}")
            if not out.flags['C_CONTIGUOUS']:
                raise ValueError("heif decode: out= must be C-contiguous")
            out_arr = out
        else:
            out_arr = cnp.PyArray_EMPTY(
                2 if channels == 1 else 3, shape,
                cnp.NPY_UINT8 if dtype_bytes == 1 else cnp.NPY_UINT16, 0)

        if monochrome:
            # Each plane is read at its own depth and checked against
            # the output's, so a plane stored in a different word size
            # can never be reinterpreted as the output dtype.
            row_bytes_out = <size_t>(width * dtype_bytes)
            for c, channel in enumerate(
                    (heif_channel_Y, heif_channel_Alpha)[:channels]):
                plane = heif_image_get_plane_readonly(img, channel, &stride)
                if plane == NULL:
                    raise HeifError(
                        'heif_image_get_plane_readonly returned NULL')
                plane_depth = heif_image_get_bits_per_pixel_range(
                    img, channel)
                if plane_depth != img_depth:
                    raise HeifError(
                        f'heif decode: the {"alpha" if c else "gray"} '
                        f'plane decoded at {plane_depth} bits, but the '
                        f'image is {img_depth}-bit')
                if channels == 1:
                    for y in range(height):
                        memcpy(<uint8_t*> cnp.PyArray_DATA(out_arr)
                               + y * row_bytes_out,
                               plane + y * stride, row_bytes_out)
                else:
                    rows = np.frombuffer(
                        (<const char*> plane)[:<Py_ssize_t> stride * height],
                        dtype=np.uint8).reshape(height, stride)
                    out_arr[..., c] = np.ascontiguousarray(
                        rows[:, :row_bytes_out]).view(expected_dtype)
            return out_arr

        plane = heif_image_get_plane_readonly(
            img, heif_channel_interleaved, &stride)
        if plane == NULL:
            raise HeifError('heif_image_get_plane_readonly returned NULL')
        row_bytes_out = <size_t>(width * channels * dtype_bytes)
        for y in range(height):
            memcpy(<uint8_t*> cnp.PyArray_DATA(out_arr) + y * row_bytes_out,
                   plane + y * stride,
                   row_bytes_out)
        return out_arr
    finally:
        if img != NULL:
            heif_image_release(img)
        if handle != NULL:
            heif_image_handle_release(handle)
        heif_context_free(ctx)




# ---------------------------------------------------------------------------
# Encode (HEVC/HEIC). Captures the writer output via a small bytestream
# accumulator passed in via heif_writer.
# ---------------------------------------------------------------------------


def decode_info(data, *, index=None, photometric=None):
    """Read image geometry without decoding pixels."""
    return decode(data, index=index, photometric=photometric, _info=True)


cdef struct write_buffer:
    uint8_t* data
    size_t cap
    size_t size
    void* sink


# Static-storage empty/error strings for the writer callback's
# heif_error.message field. libheif >= 1.18 calls strlen() on the
# returned message even when code == 0 and rejects NULL with
# "heif_writer callback returned a null error text", so we must hand
# back a valid C string. Empty for success, descriptive for OOM.
cdef extern from *:
    """
    static const char _HEIF_WRITER_OK[] = "";
    static const char _HEIF_WRITER_OOM[] = "out of memory expanding heif write buffer";
    """
    const char* _HEIF_WRITER_OK
    const char* _HEIF_WRITER_OOM


cdef int _write_destination(void* sink, const void* data, size_t size) noexcept with gil:
    cdef object target = <object> sink
    cdef size_t offset = 0
    cdef size_t count
    try:
        while offset < size:
            count = min(size - offset, <size_t> 65536)
            target.write(PyBytes_FromStringAndSize(<const char*> data + offset, count))
            offset += count
        return 0
    except BaseException as exc:
        target.error = exc
        return -1


cdef heif_error _writer_cb(
    heif_context* ctx, const void* data, size_t size, void* userdata,
) noexcept nogil:
    cdef write_buffer* buf = <write_buffer*> userdata
    cdef heif_error err
    err.code = <heif_error_code> 0
    err.subcode = 0
    err.message = _HEIF_WRITER_OK
    cdef size_t new_cap
    cdef uint8_t* new_data
    if buf.sink != NULL:
        if _write_destination(buf.sink, data, size) != 0:
            err.code = <heif_error_code> 1
            err.message = _HEIF_WRITER_OOM
        return err
    if buf.size + size > buf.cap:
        new_cap = buf.cap * 2 if buf.cap else 65536
        while new_cap < buf.size + size:
            new_cap *= 2
        new_data = <uint8_t*> realloc(buf.data, new_cap)
        if new_data == NULL:
            err.code = <heif_error_code> 1  # libheif treats nonzero as failure
            err.message = _HEIF_WRITER_OOM
            return err
        buf.data = new_data
        buf.cap = new_cap
    memcpy(buf.data + buf.size, data, size)
    buf.size += size
    return err


def encode(data, *, level: int | None = None,
           lossless: bool | None = None, color=None,
           bit_depth: int | None = None,
           numthreads: int | None = None,
           iccprofile: bytes | None = None, destination=None):
    """Encode an array as HEIC.

    Parameters
    ----------
    data : ndarray
        (H, W) or (H, W, 1) gray, (H, W, 2) gray plus alpha, (H, W, 3)
        RGB or (H, W, 4) RGBA; uint8 or uint16. Gray is coded as HEVC
        monochrome (chroma_format_idc 0, libheif's heif_chroma_monochrome),
        the format's own representation. ``decode`` returns it as RGB(A),
        as imagecodecs does, and ``decode(..., photometric='monochrome')``
        returns the same shape it was written from.
    level : int, optional
        Quality, imagecodecs' meaning: with ``lossless`` left at None, no
        level or a level above 100 is lossless and 0-100 is lossy at
        that quality. With ``lossless=False`` and no level the quality
        is 50.
    lossless : bool, optional
        None (default) follows ``level`` as above. True forces lossless
        and raises if ``level`` asks for lossy output. False forces lossy.
        Color is coded 4:4:4 either way, as imagecodecs codes it.
    color : str or ColorSpec, optional
        Color-encoding spec. Same vocabulary as the JXL/AVIF codecs accept:
        'srgb', 'display-p3', 'rec2020-pq', 'rec2020-hlg', etc. Writes an
        NCLX colr box. If None, no NCLX is written (Apple typically defaults
        to sRGB).
    bit_depth : int, optional
        Coded bit depth, 8 for uint8 and 10 or 12 for uint16. For uint16
        with no ``bit_depth`` the smallest of 10 and 12 that holds the
        data's largest value is used. uint16 values are stored as they
        are, in the low bits (10-bit means 0..1023, not shifted into the
        upper bits). Data that does not fit raises HeifError rather than
        being clamped; the HEVC encoders libheif drives stop at 12 bits.
    numthreads : int, optional
        Worker threads for the HEVC encoder (``threads`` parameter on the
        x265 / kvazaar plugin). ``None`` (default) leaves the encoder
        plugin's own default. Typical 2-6× speedup on 4K+ encodes.
    """
    cdef:
        cnp.ndarray arr
        cnp.ndarray plane_src
        heif_context* ctx = NULL
        heif_encoder* enc = NULL
        heif_image* img = NULL
        heif_image_handle* handle = NULL
        heif_color_profile_nclx* nclx = NULL
        heif_error err
        write_buffer wbuf
        heif_writer wr
        int width, height
        int has_alpha, channels
        int monochrome
        int quality
        int dtype_bytes
        int actual_bit_depth
        uint8_t* plane
        int stride
        bytes out
        unsigned int y
        size_t row_bytes_in
        heif_chroma chroma
        heif_colorspace colorspace

    _ensure_init()
    from opencodecs._avif_heif_params import (
        resolve_bit_depth, resolve_lossless, gray_layout)

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    if data.dtype != np.uint8 and data.dtype != np.uint16:
        raise HeifError(
            f'HEIF: uint8 or uint16 input supported, got {data.dtype}')
    layout = gray_layout(data)
    if layout is None:
        raise HeifError(
            f'HEIF encode: unsupported shape {data.shape}; expected '
            f'(H, W), (H, W, 1), (H, W, 2), (H, W, 3) or (H, W, 4)')
    channels, has_alpha = layout
    monochrome = channels <= 2
    arr = np.ascontiguousarray(data)
    dtype_bytes = 1 if arr.dtype == np.uint8 else 2

    actual_bit_depth = resolve_bit_depth(
        arr, bit_depth, name='HEIF', error=HeifError)
    lossless, quality = resolve_lossless(
        level, lossless, name='heif', lossless_from=101, default_quality=50)

    height = <int> arr.shape[0]
    width = <int> arr.shape[1]

    # Chroma layout: monochrome planes for gray, 8-bit interleaved or
    # 16-bit little-endian interleaved for color.
    if monochrome:
        colorspace = heif_colorspace_monochrome
        chroma = heif_chroma_monochrome
    elif dtype_bytes == 1:
        colorspace = heif_colorspace_RGB
        chroma = (heif_chroma_interleaved_RGBA if has_alpha
                  else heif_chroma_interleaved_RGB)
    else:
        colorspace = heif_colorspace_RGB
        chroma = (heif_chroma_interleaved_RRGGBBAA_LE if has_alpha
                  else heif_chroma_interleaved_RRGGBB_LE)

    # Resolve color spec to CICP values for NCLX.
    cdef int cp = -1
    cdef int tc = -1
    cdef int mc = -1
    if color is not None:
        from opencodecs.core.color import parse_color
        spec = parse_color(color)
        # ColorSpec primaries/transfer enums are CICP-aligned.
        cp = int(spec.primaries)
        tc = int(spec.transfer)
        # Use BT.2020 NCL matrix for BT.2020 primaries; BT.709 for others.
        if cp == 9:
            mc = 9
        else:
            mc = 1
    if lossless and cp < 0 and not monochrome:
        # In lossless mode without an explicit color spec, force NCLX
        # with matrix_coefficients=0 (identity / "GBR"). Without this,
        # libheif's default BT.709 matrix triggers an RGB→YUV→RGB
        # transform whose integer rounding introduces ±1 LSB errors
        # in 30%+ of pixels even with chroma=4:4:4. Identity matrix
        # stores R/G/B directly into the YUV planes — true lossless.
        # Gray needs none of this: its one plane is stored as is, and
        # HEVC allows the identity matrix only with 4:4:4 chroma.
        cp = 1     # sRGB primaries (any value works; identity matrix
                   # bypasses chromaticity transforms)
        tc = 13    # sRGB transfer (likewise — purely a tag)
        mc = 0     # IDENTITY — the bit that actually makes it lossless

    ctx = heif_context_alloc()
    if ctx == NULL:
        raise HeifError('heif_context_alloc failed')
    try:
        err = heif_context_get_encoder_for_format(
            ctx, heif_compression_HEVC, &enc)
        if err.code != 0:
            raise HeifError(
                f'get_encoder_for_format(HEVC): {err.message.decode()}')

        if numthreads is not None and int(numthreads) > 0:
            # x265 / kvazaar plugins accept a `threads` int parameter.
            # The set_parameter call returns an error if the plugin
            # doesn't expose this knob — ignore it (default behavior wins).
            heif_encoder_set_parameter_integer(
                enc, b'threads', int(numthreads))

        if not monochrome:
            # x265 defaults to 4:2:0 chroma subsampling, even in
            # lossless mode. Color is coded 4:4:4 instead, as imagecodecs
            # codes it, lossy included: lossless needs it to keep every
            # channel byte-exact, and lossy keeps chroma at full
            # resolution, where 4:2:0 smears sharp color edges.
            # Monochrome has no chroma to subsample.
            heif_encoder_set_parameter_string(enc, b'chroma', b'444')
        if lossless:
            heif_encoder_set_lossless(enc, 1)
        else:
            heif_encoder_set_lossy_quality(enc, quality)

        err = heif_image_create(
            width, height, colorspace, chroma, &img)
        if err.code != 0:
            raise HeifError(f'heif_image_create: {err.message.decode()}')

        if monochrome:
            err = heif_image_add_plane(
                img, heif_channel_Y, width, height, actual_bit_depth)
            if err.code == 0 and has_alpha:
                err = heif_image_add_plane(
                    img, heif_channel_Alpha, width, height, actual_bit_depth)
        else:
            err = heif_image_add_plane(
                img, heif_channel_interleaved, width, height, actual_bit_depth)
        if err.code != 0:
            raise HeifError(f'heif_image_add_plane: {err.message.decode()}')

        # Attach NCLX color profile (writes a colr box to the HEIF container).
        if cp >= 0:
            nclx = heif_nclx_color_profile_alloc()
            if nclx == NULL:
                raise HeifError('heif_nclx_color_profile_alloc failed')
            err = heif_nclx_color_profile_set_color_primaries(
                nclx, <uint16_t> cp)
            if err.code != 0:
                raise HeifError(
                    f'nclx set_color_primaries({cp}): {err.message.decode()}')
            err = heif_nclx_color_profile_set_transfer_characteristics(
                nclx, <uint16_t> tc)
            if err.code != 0:
                raise HeifError(
                    f'nclx set_transfer({tc}): {err.message.decode()}')
            err = heif_nclx_color_profile_set_matrix_coefficients(
                nclx, <uint16_t> mc)
            if err.code != 0:
                raise HeifError(
                    f'nclx set_matrix({mc}): {err.message.decode()}')
            nclx.full_range_flag = 1
            err = heif_image_set_nclx_color_profile(img, nclx)
            if err.code != 0:
                raise HeifError(
                    f'heif_image_set_nclx_color_profile: {err.message.decode()}')

        # Attach raw ICC profile if the caller supplied one. ICC and
        # NCLX can coexist on a HEIF (NCLX describes the working color
        # space; ICC overrides for downstream renderers) but in
        # practice most consumers honour whichever was set last, so
        # ICC after NCLX is the right order.
        if iccprofile is not None and len(iccprofile) > 0:
            _icc_bytes = bytes(iccprofile)
            err = heif_image_set_raw_color_profile(
                img, b'prof',
                <const void*> <const char*> _icc_bytes,
                <size_t> len(_icc_bytes),
            )
            if err.code != 0:
                raise HeifError(
                    f'heif_image_set_raw_color_profile: '
                    f'{err.message.decode()}')

        if monochrome:
            # One plane per sample: gray into Y, alpha into Alpha.
            row_bytes_in = <size_t>(width * dtype_bytes)
            for c, channel in enumerate(
                    (heif_channel_Y, heif_channel_Alpha)[:channels]):
                plane = heif_image_get_plane(img, channel, &stride)
                if plane == NULL:
                    raise HeifError('heif_image_get_plane returned NULL')
                plane_src = np.ascontiguousarray(
                    arr if arr.ndim == 2 else arr[..., c])
                for y in range(height):
                    memcpy(plane + y * stride,
                           <const uint8_t*> cnp.PyArray_DATA(plane_src)
                           + y * row_bytes_in,
                           row_bytes_in)
        else:
            plane = heif_image_get_plane(img, heif_channel_interleaved, &stride)
            if plane == NULL:
                raise HeifError('heif_image_get_plane returned NULL')
            row_bytes_in = <size_t>(width * channels * dtype_bytes)
            for y in range(height):
                memcpy(plane + y * stride,
                       <const uint8_t*> cnp.PyArray_DATA(arr) + y * row_bytes_in,
                       row_bytes_in)

        with nogil:
            err = heif_context_encode_image(ctx, img, enc, NULL, &handle)
        if err.code != 0:
            raise HeifError(
                f'heif_context_encode_image: {err.message.decode()}')

        wbuf.data = NULL
        wbuf.cap = 0
        wbuf.size = 0
        wbuf.sink = NULL if destination is None else <void*> destination
        wr.writer_api_version = 1
        wr.write = _writer_cb

        err = heif_context_write(ctx, &wr, &wbuf)
        if destination is not None and destination.error is not None:
            raise destination.error
        if err.code != 0:
            free(wbuf.data)
            raise HeifError(f'heif_context_write: {err.message.decode()}')

        if destination is not None:
            return None
        try:
            out = PyBytes_FromStringAndSize(<char*> wbuf.data,
                                            <Py_ssize_t> wbuf.size)
            return out
        finally:
            free(wbuf.data)
    finally:
        if nclx != NULL:
            heif_nclx_color_profile_free(nclx)
        if handle != NULL:
            heif_image_handle_release(handle)
        if img != NULL:
            heif_image_release(img)
        if enc != NULL:
            heif_encoder_release(enc)
        heif_context_free(ctx)


def read_icc_profile(data) -> bytes | None:
    """Return the embedded ICC profile bytes from a HEIF/HEIC, or ``None``.

    Reads only the container + handle metadata; doesn't decode any
    HEVC frames.
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        heif_context* ctx = NULL
        heif_image_handle* handle = NULL
        heif_error err
        size_t icc_size
        bytes out_bytes

    _ensure_init()
    if hasattr(data, "read_at"):
        data.reset()
        src = b""
        srcsize = data.size
    else:
        if isinstance(data, (bytes, bytearray)):
            src = data
        else:
            src = bytes(data)
        srcsize = <size_t> src.shape[0]
    if srcsize < 12:
        return None

    ctx = heif_context_alloc()
    if ctx == NULL:
        raise HeifError('heif_context_alloc failed')
    try:
        if hasattr(data, "read_at"):
            err = heif_context_read_from_reader(ctx, &_source_reader, <void*>data, NULL)
            data.raise_error()
        else:
            err = heif_context_read_from_memory_without_copy(
                ctx, &src[0], srcsize, NULL)
        if err.code != 0:
            return None
        err = heif_context_get_primary_image_handle(ctx, &handle)
        if err.code != 0:
            return None
        icc_size = heif_image_handle_get_raw_color_profile_size(handle)
        if icc_size == 0:
            return None
        out_bytes = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> icc_size)
        err = heif_image_handle_get_raw_color_profile(
            handle, <void*> PyBytes_AsString(out_bytes))
        if err.code != 0:
            return None
        return out_bytes
    finally:
        if handle != NULL:
            heif_image_handle_release(handle)
        heif_context_free(ctx)


def check_signature(data) -> bool:
    """True if `data` looks like HEIF (ftyp box with heic/heix/mif1/msf1)."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:32])
    else:
        try:
            head = bytes(data)[:32]
        except Exception:
            return False
    if len(head) < 12 or head[4:8] != b'ftyp':
        return False
    brands = head[8:32]
    for b in (b'heic', b'heix', b'heim', b'heis', b'hevc',
              b'mif1', b'msf1'):
        if b in brands:
            return True
    return False
