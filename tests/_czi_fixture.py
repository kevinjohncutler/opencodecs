"""Synthesize tiny but valid Zeiss ZISRAW (.czi) files for tests.

CZI is decode-only in opencodecs (the writer is non-trivial: Zen-side
metadata, scene/mosaic indexing, etc.). For tests we don't need any of
that — just enough header + directory + sub-blocks to exercise the
opencodecs CziReader code paths on every platform without depending on
a real lab CZI mount.

Layout we emit (see CziReader for the full spec):

    [ZISRAWFILE header @ 0]
    [ZISRAWMETADATA segment]    minimal XML, no attachments
    [ZISRAWSUBBLOCK segments]   one per tile in the array
    [ZISRAWDIRECTORY segment]   directory walks all sub-blocks

The synthesizer accepts a ``(T, Y, X[, S])``-shaped uint8/uint16 ndarray
where T is the sub-block count (== n_frames for opencodecs's reader).
Compression types 0 (uncompressed), 5 (raw zstd), and 6 (ZSTDHDR with
optional byte-plane shuffle) are supported — matching the three branches
opencodecs decodes.

Usage in tests::

    from opencodecs.tests._czi_fixture import write_czi, czi_bytes

    p = tmp_path / "test.czi"
    write_czi(p, np.random.randint(0, 256, (4, 16, 16), dtype=np.uint8))

    raw = czi_bytes(arr, compression=6, hilo=True)
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


_FILE_MAGIC = b"ZISRAWFILE"
_DIR_MAGIC = b"ZISRAWDIRECTORY"
_META_MAGIC = b"ZISRAWMETADATA"
_SUB_MAGIC = b"ZISRAWSUBBLOCK"


_DTYPE_TO_PIXELTYPE = {
    np.dtype("u1"): (0, 1),     # GRAY8
    np.dtype("u2"): (1, 1),     # GRAY16
    np.dtype("f4"): (2, 1),     # GRAY32_FLOAT
}

# (dtype, samples per pixel) -> CZI pixel type. The reader derives the sample
# axis from the pixel type, so a color fixture has to name a color type: a
# 3-sample array declared GRAY8 produces a sub-block whose stored payload is
# three times the tile its own directory entry describes.
_PIXELTYPE_BY_DTYPE_SAMPLES = {
    (np.dtype("u1"), 1): 0,     # GRAY8
    (np.dtype("u2"), 1): 1,     # GRAY16
    (np.dtype("f4"), 1): 2,     # GRAY32_FLOAT
    (np.dtype("u1"), 3): 3,     # BGR24
    (np.dtype("u2"), 3): 4,     # BGR48
    (np.dtype("f4"), 3): 8,     # BGR96_FLOAT
    (np.dtype("u1"), 4): 9,     # BGRA32
}


#: Must equal the writer's default. libCZI refuses metadata not rooted at
#: ImageDocument, so a bare <Metadata/> produces files czicompress and ZEN
#: cannot open; the writer wraps it, and this serializer has to agree or the
#: byte-for-byte comparison in test_czi_segment_parts fails. Deliberately a
#: LITERAL rather than an import from the writer: this file is an independent
#: implementation of the layout, and that test is only meaningful while it
#: stays independent.
DEFAULT_METADATA = b"<ImageDocument><Metadata/></ImageDocument>"



def _pixel_type_for(array: np.ndarray, samples: int) -> int:
    """Pick the CZI pixel type matching this array's dtype and sample count."""
    try:
        return _PIXELTYPE_BY_DTYPE_SAMPLES[(array.dtype, samples)]
    except KeyError:
        raise ValueError(
            f"no CZI pixel type for dtype {array.dtype} with {samples} "
            f"sample(s) per pixel") from None


def _pad_segment(payload: bytes, payload_alloc: int | None = None) -> bytes:
    """Wrap ``payload`` in a 32-byte segment header (sid + alloc + used).

    The segment-id is the first 16 bytes of payload (caller must include it).
    `alloc` and `used` are 8-byte little-endian ints. We pad to a multiple of
    32 bytes (CZI segments are always 32-byte aligned).
    """
    used = len(payload) - 16  # bytes after the 16-byte sid
    alloc = payload_alloc if payload_alloc is not None else used
    sid = payload[:16]
    body = payload[16:]
    out = struct.pack("<16sqq", sid, alloc, used) + body
    pad = (-len(out)) % 32
    return out + b"\x00" * pad


def _build_metadata_segment(xml: bytes = DEFAULT_METADATA) -> bytes:
    """Build a minimal ZISRAWMETADATA segment.

    Layout post-segment-header (offset 32):
      int xml_size, int attachment_size, 248 bytes filler, xml_size bytes.
    """
    sid = _META_MAGIC + b"\x00\x00"  # 14+2 = 16
    payload_after_sid = (
        struct.pack("<ii", len(xml), 0)  # xml_size, attachment_size
        + b"\x00" * 248                   # filler to reach offset 256 from sub-block start
        + xml
    )
    raw = sid + payload_after_sid
    return _pad_segment(raw)


def _byteshuffle_encode(natural: bytes, itemsize: int) -> bytes:
    """Forward byte-plane shuffle (test fixtures only — pure Python is fine).

    Inverse of opencodecs.codecs._bytetools.byteshuffle_decode.
    """
    if itemsize == 1:
        return natural
    n = len(natural) // itemsize
    arr = np.frombuffer(natural, dtype=np.uint8).reshape(n, itemsize)
    # Transpose: rows = byte-planes, cols = elements; flatten C-order.
    return arr.T.tobytes()


def _zsthdr_compress(pixel_bytes: bytes, itemsize: int, hilo: bool) -> bytes:
    """Encode pixel bytes as a ZSTDHDR (compression=6) payload.

    Layout: 1-byte header_size, optional chunks (we emit chunk_type=1 with
    a 1-byte hilo flag), then a zstd stream over the (optionally shuffled)
    pixels.
    """
    from opencodecs.codecs._zstd import encode as zstd_encode
    if hilo and itemsize > 1:
        zstd_payload = zstd_encode(_byteshuffle_encode(pixel_bytes, itemsize), level=3)
    else:
        zstd_payload = zstd_encode(pixel_bytes, level=3)

    # Header: header_size byte then chunk_type=1 + flag byte (hilo bit)
    header = struct.pack("<BBB", 3, 1, 1 if hilo else 0)
    return header + zstd_payload


def _build_subblock(
    array: np.ndarray,
    pixel_type: int,
    compression: int,
    hilo: bool,
    file_position: int,
    *,
    # Pyramid + multi-tile parameters. By default a sub-block describes
    # its own grid at its stored size — the standard non-pyramid layout.
    # For pyramids: set ``logical_shape=(H, W)`` larger than ``array.shape``
    # so the DirectoryEntryDV records ``size=H`` ``stored=h``. For mosaic
    # acquisitions: ``location=(start_y, start_x)`` shifts where this
    # sub-block sits in the parent coordinate space.
    logical_shape: tuple[int, int] | None = None,
    location: tuple[int, int] = (0, 0),
    pyramid_type: int = 0,
    extra_dims: list[tuple[bytes, int]] | None = None,
) -> tuple[bytes, int]:
    """Build one ZISRAWSUBBLOCK segment payload + return (bytes, storage_size).

    The ``storage_size`` is the in-segment footprint of the inline
    DirectoryEntryDV (used by the reader to find pixel data).
    """
    h, w = array.shape[:2]
    samples = array.shape[2] if array.ndim == 3 else 1

    # 32-byte directory-entry header (DV schema)
    #   2s schema, int pixel_type, q file_position, int file_part,
    #   int compression, B pyramid, B reserved1, 4s reserved2, int dims_count
    if logical_shape is None:
        logical_h, logical_w = h, w
    else:
        logical_h, logical_w = logical_shape
    start_y, start_x = location

    dims = [
        (b"X", start_x, logical_w, 0.0, w),
        (b"Y", start_y, logical_h, 0.0, h),
    ]
    # Extra dimensions such as C or Z with a single index: size 1, stored 1.
    # These put the tile on a distinct plane without changing its pixels.
    for name, start in (extra_dims or []):
        dims.append((name, start, 1, 0.0, 1))

    de_header = struct.pack(
        "<2siqiiBB4si",
        b"DV",                  # schema
        pixel_type,             # pixel_type
        file_position,          # file_position (filled by caller's caller)
        0,                      # file_part
        compression,            # compression
        pyramid_type, 0, b"\x00\x00\x00\x00",  # pyramid + reserved
        len(dims),              # dims count
    )
    de_dims = b""
    for dim_b, start, size, coord, stored in dims:
        de_dims += struct.pack("<4siifi", dim_b, start, size, coord, stored)
    storage_size = len(de_header) + len(de_dims)
    pad = max(240 - storage_size, 0)

    # Pixel payload (encoded per compression type)
    pixel_bytes = np.ascontiguousarray(array).tobytes()
    itemsize = array.dtype.itemsize
    if compression == 0:
        data = pixel_bytes
    elif compression == 5:
        from opencodecs.codecs._zstd import encode as zstd_encode
        data = zstd_encode(pixel_bytes, level=3)
    elif compression == 6:
        data = _zsthdr_compress(pixel_bytes, itemsize, hilo)
    elif compression == 4:
        # JPEG XR, written by imagecodecs (the reference binding), lossless.
        # Grayscale only: which channel order a color stream carries is a
        # property of the stream, tested on the codec itself.
        import imagecodecs
        if array.ndim != 2:
            raise ValueError("test JPEG XR sub-blocks are grayscale")
        data = imagecodecs.jpegxr_encode(np.ascontiguousarray(array))
    else:
        raise ValueError(f"unsupported test compression {compression}")

    # Sub-block-header (16 bytes after the 32-byte segment header):
    #   int meta_size, int attachment_size, q data_size
    sub_header = struct.pack("<iiq", 0, 0, len(data))

    # Segment payload AFTER the segment header is:
    #   16-byte sub-block-header + DirectoryEntryDV + pad-to-256 + meta + data
    sid = _SUB_MAGIC + b"\x00\x00"  # 14 + 2 = 16
    body = sub_header + de_header + de_dims + b"\x00" * pad + data
    raw = sid + body
    return _pad_segment(raw), storage_size


def _build_directory_segment(
    entries: list[dict],
) -> bytes:
    """Build a ZISRAWDIRECTORY segment from a list of entry dicts.

    Each entry dict carries: ``file_position``, ``pixel_type``,
    ``compression``, ``stored_w``, ``stored_h``, ``logical_w``,
    ``logical_h``, ``start_x``, ``start_y``, ``pyramid_type``.

    The directory entry must match the inline DirectoryEntryDV in the
    sub-block segment exactly — pyramid info, mosaic positions, and
    stored vs logical sizes all need to round-trip through both.
    """
    sid = _DIR_MAGIC + b"\x00"  # 15 + 1 = 16
    # entry_count + 124 reserved bytes
    body = struct.pack("<I", len(entries)) + b"\x00" * 124
    for e in entries:
        extra = e.get("extra_dims") or []
        body += struct.pack(
            "<2siqiiBB4si",
            b"DV", e["pixel_type"], e["file_position"], 0,
            e["compression"], e.get("pyramid_type", 0),
            0, b"\x00\x00\x00\x00", 2 + len(extra),
        )
        body += struct.pack(
            "<4siifi", b"X",
            e.get("start_x", 0), e["logical_w"], 0.0, e["stored_w"],
        )
        body += struct.pack(
            "<4siifi", b"Y",
            e.get("start_y", 0), e["logical_h"], 0.0, e["stored_h"],
        )
        # Must match the inline entry exactly; the reader parses this copy.
        for name, start in extra:
            body += struct.pack("<4siifi", name, start, 1, 0.0, 1)
    return _pad_segment(sid + body)


def _build_file_header(directory_position: int, metadata_position: int,
                       file_size: int) -> bytes:
    """ZISRAWFILE header at offset 0.

    Layout post-segment-header (offset 32):
      uint32 major, uint32 minor, uint32 r1, uint32 r2,
      16 primary_guid, 16 file_guid, uint32 file_part,
      int64 directory_position, int64 metadata_position,
      uint32 update_pending, int64 attachment_directory_position
    """
    sid = _FILE_MAGIC + b"\x00" * 6  # 10 + 6 = 16
    payload = (
        struct.pack("<II", 1, 2)
        + b"\x00" * 8
        + b"\x00" * 32
        + struct.pack("<I", 0)
        + struct.pack("<q", directory_position)
        + struct.pack("<q", metadata_position)
        + struct.pack("<I", 0)
        + struct.pack("<q", 0)
    )
    raw = sid + payload
    return _pad_segment(raw, payload_alloc=file_size)


def czi_bytes(
    array: np.ndarray,
    *,
    compression: int = 0,
    hilo: bool = False,
    metadata_xml: bytes = DEFAULT_METADATA,
) -> bytes:
    """Serialize ``array`` as a tiny CZI buffer that opencodecs.CziReader
    can decode.

    Parameters
    ----------
    array : ndarray
        Either a single 2-D / 3-D image (one sub-block) or a stack
        ``(T, Y, X[, S])`` (one sub-block per leading T).
    compression : int
        0 (uncompressed), 5 (raw zstd), or 6 (ZSTDHDR).
    hilo : bool
        For ``compression=6`` only — apply the byte-plane shuffle.
    metadata_xml : bytes
        Minimal file-level XML metadata segment payload.

    Returns
    -------
    bytes
        The complete CZI file as a single bytes blob.
    """
    if array.ndim == 2:
        frames = [array]
    elif array.ndim == 3:
        # Heuristic: treat 3-channel as samples (single sub-block);
        # else treat first axis as T (multiple sub-blocks).
        if array.shape[2] in (1, 3, 4):
            frames = [array]
        else:
            frames = [array[i] for i in range(array.shape[0])]
    else:
        raise ValueError(f"unsupported test shape {array.shape}")

    samples = frames[0].shape[2] if frames[0].ndim == 3 else 1
    pixel_type = _pixel_type_for(array, samples)

    # Build segments in order to compute their offsets.
    # Order: file_header @ 0 → metadata_segment → sub-blocks → directory.
    file_header_size = 32 + 88  # sid(16) + alloc(8) + used(8) + payload(88)
    file_header_size = (file_header_size + 31) // 32 * 32  # round up to 32

    metadata_segment = _build_metadata_segment(metadata_xml)
    metadata_position = file_header_size

    cur_offset = metadata_position + len(metadata_segment)
    sub_segments = []
    sub_meta = []  # tuples (file_pos, pixel_type, compression, dims, w, h)

    for fr in frames:
        seg, _storage = _build_subblock(
            fr, pixel_type, compression, hilo, cur_offset,
        )
        sub_segments.append(seg)
        h, w = fr.shape[:2]
        sub_meta.append({
            "file_position": cur_offset, "pixel_type": pixel_type,
            "compression": compression,
            "stored_w": w, "stored_h": h,
            "logical_w": w, "logical_h": h,
            "start_x": 0, "start_y": 0, "pyramid_type": 0,
        })
        cur_offset += len(seg)

    directory_position = cur_offset
    directory_segment = _build_directory_segment(sub_meta)
    cur_offset += len(directory_segment)

    # Now we know total file_size; build a fresh file header with directory
    # and metadata positions.
    file_size = cur_offset
    file_header = _build_file_header(
        directory_position=directory_position,
        metadata_position=metadata_position,
        file_size=file_size,
    )

    # Assemble.
    out = bytearray()
    out += file_header
    out += b"\x00" * (metadata_position - len(file_header))
    out += metadata_segment
    for seg in sub_segments:
        out += seg
    out += directory_segment
    return bytes(out)


def pyramid_czi_bytes(
    levels: list[np.ndarray],
    *,
    compression: int = 0,
    hilo: bool = False,
    metadata_xml: bytes = DEFAULT_METADATA,
) -> bytes:
    """Serialize a multi-resolution CZI fixture.

    ``levels[0]`` is the full-resolution image. Each subsequent entry
    is a downscaled version; its logical extent equals level 0's
    shape, but its stored grid is the array's actual shape. That's
    exactly the on-disk encoding ZEN uses for pyramid CZIs.

    For example::

        base = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        half = base[::2, ::2]
        quarter = base[::4, ::4]
        data = pyramid_czi_bytes([base, half, quarter])

    produces a CZI with 3 sub-blocks: one at stored (256,256) +
    logical (256,256) [pyramid_type=0]; one at stored (128,128) +
    logical (256,256) [pyramid_type=2]; one at stored (64,64) +
    logical (256,256) [pyramid_type=2].
    """
    if not levels:
        raise ValueError("pyramid_czi_bytes: levels must be non-empty")
    base = levels[0]
    if base.ndim != 2:
        raise ValueError("pyramid_czi_bytes: levels must be 2-D")
    logical_h, logical_w = base.shape
    pixel_type = _pixel_type_for(
        base, base.shape[2] if base.ndim == 3 else 1)

    file_header_size = 32 + 88
    file_header_size = (file_header_size + 31) // 32 * 32

    metadata_segment = _build_metadata_segment(metadata_xml)
    metadata_position = file_header_size
    cur_offset = metadata_position + len(metadata_segment)
    sub_segments = []
    sub_meta = []

    for i, lvl in enumerate(levels):
        h, w = lvl.shape[:2]
        # ZEN convention: level 0 has pyramid_type=0; later levels =2.
        ptype = 0 if i == 0 else 2
        seg, _storage = _build_subblock(
            lvl, pixel_type, compression, hilo, cur_offset,
            logical_shape=(logical_h, logical_w),
            location=(0, 0),
            pyramid_type=ptype,
        )
        sub_segments.append(seg)
        sub_meta.append({
            "file_position": cur_offset, "pixel_type": pixel_type,
            "compression": compression,
            "stored_w": w, "stored_h": h,
            "logical_w": logical_w, "logical_h": logical_h,
            "start_x": 0, "start_y": 0, "pyramid_type": ptype,
        })
        cur_offset += len(seg)

    directory_position = cur_offset
    directory_segment = _build_directory_segment(sub_meta)
    cur_offset += len(directory_segment)

    file_size = cur_offset
    file_header = _build_file_header(
        directory_position=directory_position,
        metadata_position=metadata_position,
        file_size=file_size,
    )
    out = bytearray()
    out += file_header
    out += b"\x00" * (metadata_position - len(file_header))
    out += metadata_segment
    for seg in sub_segments:
        out += seg
    out += directory_segment
    return bytes(out)


def mosaic_czi_bytes(
    tiles: list,
    *,
    compression: int = 0,
    hilo: bool = False,
    metadata_xml: bytes = DEFAULT_METADATA,
) -> bytes:
    """Serialize a single-level mosaic CZI.

    ``tiles`` is a list of ``(array, (start_y, start_x))`` or
    ``(array, (start_y, start_x), extra_dims)`` where ``extra_dims`` is a
    list of ``(b"C", index)`` style pairs placing the tile on another plane.
    Tiles are written in the given order, which is also directory order,
    so overlap resolution can be tested deterministically. Every tile's
    logical extent equals its stored extent (no pyramid).
    """
    if not tiles:
        raise ValueError("mosaic_czi_bytes: tiles must be non-empty")
    first = tiles[0][0]
    samples = first.shape[2] if first.ndim == 3 else 1
    pixel_type = _pixel_type_for(first, samples)

    file_header_size = (32 + 88 + 31) // 32 * 32
    metadata_segment = _build_metadata_segment(metadata_xml)
    metadata_position = file_header_size
    cur_offset = metadata_position + len(metadata_segment)
    sub_segments, sub_meta = [], []

    for spec in tiles:
        arr, (sy, sx) = spec[0], spec[1]
        extra = list(spec[2]) if len(spec) > 2 else []
        if arr.dtype != first.dtype or arr.shape[2:] != first.shape[2:]:
            raise ValueError("mosaic tiles must share dtype and samples")
        h, w = arr.shape[:2]
        seg, _storage = _build_subblock(
            arr, pixel_type, compression, hilo, cur_offset,
            logical_shape=(h, w), location=(sy, sx), pyramid_type=0,
            extra_dims=extra,
        )
        sub_segments.append(seg)
        sub_meta.append({
            "file_position": cur_offset, "pixel_type": pixel_type,
            "compression": compression,
            "stored_w": w, "stored_h": h, "logical_w": w, "logical_h": h,
            "start_x": sx, "start_y": sy, "pyramid_type": 0,
            "extra_dims": extra,
        })
        cur_offset += len(seg)

    directory_position = cur_offset
    directory_segment = _build_directory_segment(sub_meta)
    cur_offset += len(directory_segment)
    file_header = _build_file_header(
        directory_position=directory_position,
        metadata_position=metadata_position,
        file_size=cur_offset,
    )
    out = bytearray()
    out += file_header
    out += b"\x00" * (metadata_position - len(file_header))
    out += metadata_segment
    for seg in sub_segments:
        out += seg
    out += directory_segment
    return bytes(out)


def write_czi(path: str | Path, array: np.ndarray, **kwargs) -> None:
    """Convenience: synthesize CZI bytes and write them to disk."""
    Path(path).write_bytes(czi_bytes(array, **kwargs))
