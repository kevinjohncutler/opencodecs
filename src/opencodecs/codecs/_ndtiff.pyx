# opencodecs/codecs/_ndtiff.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native Cython parser for NDTiff.index files (Micro-Manager / Pycro-Manager).

The reference Python parser (`ndstorage.ndtiff_index.read_ndtiff_index`)
walks every record with three ``struct.unpack`` calls, two UTF-8
decodes, a ``json.loads``, and a Python class construction — totalling
about 10 µs per record. For lab-scale acquisitions of 100K+ frames
(stitched timelapses) this is the dominant open() cost.

This parser does the same walk in a nogil C loop over the underlying
buffer, returning a flat list of dicts the Python layer can wrap.
The only Python work per record is:

  * One bytes slice for the axes JSON (~10 B)
  * One bytes slice for the filename (~15-25 B)
  * Building an 8-int Python tuple

Axes-JSON parsing is *deferred* to the caller — most Pycro-Manager
records have a stable axes ordering and the same key set, so the
caller can intern them or batch-decode once.

Returned tuple per record (positional):
    (axes_json_bytes, filename_str, pixel_offset, image_width,
     image_height, pixel_type, pixel_compression,
     metadata_offset, metadata_length, metadata_compression)
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.list cimport PyList_Append
from libc.stdint cimport uint8_t, uint32_t


cdef inline uint32_t _read_u32_le(const uint8_t* p) noexcept nogil:
    return (
        <uint32_t>p[0]
        | (<uint32_t>p[1] << 8)
        | (<uint32_t>p[2] << 16)
        | (<uint32_t>p[3] << 24)
    )


class NDTiffIndexError(RuntimeError):
    """Raised on malformed NDTiff.index input."""


def parse_ndtiff_index(data):
    """Parse an NDTiff.index byte buffer into a list of records.

    Each record is a 10-tuple:
        (axes_json_bytes, filename_str, pixel_offset, image_width,
         image_height, pixel_type, pixel_compression,
         metadata_offset, metadata_length, metadata_compression)

    The axes JSON is returned as raw bytes so the caller can either
    json.loads() once per unique blob (interning is common) or skip
    parsing entirely for index-only operations.

    ``data`` accepts any buffer-protocol object: bytes, bytearray,
    memoryview, or a mmap.
    """
    cdef:
        const uint8_t[::1] buf
        const uint8_t* p
        Py_ssize_t n
        Py_ssize_t pos = 0
        uint32_t axes_len
        uint32_t fn_len
        uint32_t pixel_offset, image_width, image_height
        uint32_t pixel_type, pixel_compression
        uint32_t metadata_offset, metadata_length, metadata_compression
        bytes axes_blob
        bytes filename_blob

    try:
        buf = data
    except (TypeError, ValueError, BufferError):
        buf = bytes(data)
    n = buf.shape[0]
    if n == 0:
        return []
    p = &buf[0]

    out = []
    while pos < n:
        if pos + 4 > n:
            raise NDTiffIndexError(
                f"truncated NDTiff index at offset {pos}: "
                f"need 4 more bytes for axes_len"
            )
        axes_len = _read_u32_le(p + pos)
        if axes_len > <uint32_t>0x7fffffff:
            # The spec sentinel for a properly-closed dataset is a
            # negative axes_len (0xFFFFFFFF). ndstorage just stops on
            # it; we do the same.
            break
        if <Py_ssize_t>(pos + 4 + axes_len + 4) > n:
            raise NDTiffIndexError(
                f"truncated NDTiff index at offset {pos}: "
                f"declared axes_len={axes_len} runs past EOF ({n})"
            )
        # Copy the axes JSON bytes once (Cython slice copy is one
        # memcpy via PyBytes_FromStringAndSize).
        axes_blob = PyBytes_FromStringAndSize(
            <const char*>(p + pos + 4), <Py_ssize_t>axes_len,
        )
        pos += 4 + <Py_ssize_t>axes_len

        fn_len = _read_u32_le(p + pos)
        if <Py_ssize_t>(pos + 4 + fn_len + 32) > n:
            raise NDTiffIndexError(
                f"truncated NDTiff index at offset {pos}: "
                f"declared filename_len={fn_len} runs past EOF"
            )
        filename_blob = PyBytes_FromStringAndSize(
            <const char*>(p + pos + 4), <Py_ssize_t>fn_len,
        )
        pos += 4 + <Py_ssize_t>fn_len

        # 32-byte fixed struct: 8 × uint32 LE.
        pixel_offset          = _read_u32_le(p + pos)
        image_width           = _read_u32_le(p + pos + 4)
        image_height          = _read_u32_le(p + pos + 8)
        pixel_type            = _read_u32_le(p + pos + 12)
        pixel_compression     = _read_u32_le(p + pos + 16)
        metadata_offset       = _read_u32_le(p + pos + 20)
        metadata_length       = _read_u32_le(p + pos + 24)
        metadata_compression  = _read_u32_le(p + pos + 28)
        pos += 32

        PyList_Append(out, (
            axes_blob,
            filename_blob.decode("utf-8"),
            int(pixel_offset),
            int(image_width),
            int(image_height),
            int(pixel_type),
            int(pixel_compression),
            int(metadata_offset),
            int(metadata_length),
            int(metadata_compression),
        ))
    return out


# ---------------------------------------------------------------------------
# NDTiff pixel-type → (numpy dtype string, bit-depth) table
# ---------------------------------------------------------------------------
#
# From ndstorage.NDTiffIndexEntry constants. The bit depth informs
# downstream metadata; the storage dtype is always uint8 or uint16.

PIXEL_TYPE_EIGHT_BIT      = 0
PIXEL_TYPE_SIXTEEN_BIT    = 1
PIXEL_TYPE_EIGHT_BIT_RGB  = 2
PIXEL_TYPE_TEN_BIT        = 3
PIXEL_TYPE_TWELVE_BIT     = 4
PIXEL_TYPE_FOURTEEN_BIT   = 5
PIXEL_TYPE_ELEVEN_BIT     = 6
