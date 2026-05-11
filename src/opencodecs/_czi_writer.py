"""CZI writer — produce Zeiss ZISRAW (.czi) files from numpy arrays.

The companion to :class:`opencodecs._czi_reader.CziReader` /
:class:`CziPyramidReader`. Produces files that round-trip cleanly
through both opencodecs's own reader and the reference implementations
(``czifile`` and the libCZI-backed ``pylibCZIrw``).

Two public classes:

* :class:`CziWriter` — single-resolution write. Equivalent to
  ``czifile`` / ``pylibCZIrw``'s basic write, with a smaller surface
  area focused on the common case.

* :class:`CziPyramidWriter` — multi-resolution write. Takes a list of
  arrays where ``levels[0]`` is full-resolution and each subsequent
  level is a downscaled version of the same scene. The writer
  records the right ``stored_shape`` vs ``shape`` per sub-block so
  pyramid-aware readers (including ours) can navigate the levels.
  This is the format ZEN produces; *no Python library other than
  opencodecs currently exposes this write API* — Zeiss's
  pylibCZIrw, czifile, and aicspylibczi are all read-only on pyramid
  metadata.

Scope of v1:

* uint8 / uint16 / float32 grayscale; mosaic / multi-channel deferred
* Compression: ``"none"``, ``"zstd"`` (raw stream — CZI compression=5),
  ``"zstdhdr"`` (ZSTDHDR / CZI compression=6, with optional hi-lo
  byte-plane shuffle as ZEN does)
* Single scene, single channel per page
* Minimal XML metadata; callers can pass a richer XML payload
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable

import numpy as np


# CZI segment ID magic strings (libCZI ABNF — fixed 16-byte ASCII).
_FILE_MAGIC = b"ZISRAWFILE"
_DIR_MAGIC = b"ZISRAWDIRECTORY"
_META_MAGIC = b"ZISRAWMETADATA"
_SUB_MAGIC = b"ZISRAWSUBBLOCK"


# Pixel-type encoding (subset of CZI's full table; covers all
# scientific-imaging dtypes commonly seen in microscopy).
_DTYPE_TO_PIXELTYPE = {
    np.dtype("u1"): 0,     # GRAY8
    np.dtype("u2"): 1,     # GRAY16
    np.dtype("f4"): 2,     # GRAY32_FLOAT
}


# Compression-name → CZI compression-code mapping.
_CMP_NAME_TO_CODE = {
    "none":    0,
    "raw":     0,
    "zstd":    5,    # raw zstd stream
    "zstdhdr": 6,    # CZI ZSTDHDR (with optional byte-plane shuffle header)
}


class CziWriterError(RuntimeError):
    """Raised on writer state-machine violations."""


# ---------------------------------------------------------------------------
# Low-level segment builders
# ---------------------------------------------------------------------------


def _pad_segment(payload: bytes, payload_alloc: int | None = None) -> bytes:
    """Wrap ``payload`` (which already starts with a 16-byte SID) in a
    standard 32-byte segment header (SID + alloc + used) and pad up to
    a 32-byte boundary."""
    used = len(payload) - 16
    alloc = payload_alloc if payload_alloc is not None else used
    sid = payload[:16]
    body = payload[16:]
    out = struct.pack("<16sqq", sid, alloc, used) + body
    pad = (-len(out)) % 32
    return out + b"\x00" * pad


def _build_metadata_segment(xml: bytes) -> bytes:
    sid = _META_MAGIC + b"\x00\x00"
    payload = struct.pack("<ii", len(xml), 0) + b"\x00" * 248 + xml
    return _pad_segment(sid + payload)


def _byteshuffle_encode(natural: bytes, itemsize: int) -> bytes:
    """ZEN's per-byte-plane shuffle: rearrange so all high bytes come
    first, then mid, then low. Compresses better when high bytes are
    correlated (typical for 12/14/16-bit microscopy)."""
    if itemsize == 1:
        return natural
    n = len(natural) // itemsize
    arr = np.frombuffer(natural, dtype=np.uint8).reshape(n, itemsize)
    return arr.T.tobytes()


def _zstdhdr_encode(pixel_bytes: bytes, itemsize: int, hilo: bool) -> bytes:
    """Encode as CZI ZSTDHDR (compression=6).

    Layout: 1-byte header_size, chunk_type=1 + hilo flag byte, then a
    zstd stream over the (optionally byte-shuffled) pixels.
    """
    from .codecs._zstd import encode as zstd_encode
    if hilo and itemsize > 1:
        payload = zstd_encode(_byteshuffle_encode(pixel_bytes, itemsize), level=3)
    else:
        payload = zstd_encode(pixel_bytes, level=3)
    header = struct.pack("<BBB", 3, 1, 1 if hilo else 0)
    return header + payload


def _build_subblock(
    array: np.ndarray,
    *,
    pixel_type: int,
    compression_code: int,
    hilo: bool,
    file_position: int,
    logical_shape: tuple[int, int],
    location: tuple[int, int] = (0, 0),
    pyramid_type: int = 0,
) -> tuple[bytes, dict]:
    """Build one ZISRAWSUBBLOCK segment payload.

    Returns ``(segment_bytes, directory_entry_dict)`` — the matching
    directory-entry dict is consumed later by
    :func:`_build_directory_segment`.
    """
    h, w = array.shape[:2]
    logical_h, logical_w = logical_shape
    start_y, start_x = location

    dims = [
        (b"X", start_x, logical_w, 0.0, w),
        (b"Y", start_y, logical_h, 0.0, h),
    ]
    de_header = struct.pack(
        "<2siqiiBB4si",
        b"DV", pixel_type, file_position, 0,
        compression_code, pyramid_type, 0, b"\x00\x00\x00\x00",
        len(dims),
    )
    de_dims = b"".join(
        struct.pack("<4siifi", d, st, sz, co, stored)
        for d, st, sz, co, stored in dims
    )
    storage_size = len(de_header) + len(de_dims)
    pad = max(240 - storage_size, 0)

    pixel_bytes = np.ascontiguousarray(array).tobytes()
    itemsize = array.dtype.itemsize
    if compression_code == 0:
        data = pixel_bytes
    elif compression_code == 5:
        from .codecs._zstd import encode as zstd_encode
        data = zstd_encode(pixel_bytes, level=3)
    elif compression_code == 6:
        data = _zstdhdr_encode(pixel_bytes, itemsize, hilo)
    else:
        raise CziWriterError(
            f"unsupported compression code {compression_code} "
            f"(expected 0, 5, or 6)"
        )

    sub_header = struct.pack("<iiq", 0, 0, len(data))
    sid = _SUB_MAGIC + b"\x00\x00"
    body = sub_header + de_header + de_dims + b"\x00" * pad + data
    seg = _pad_segment(sid + body)

    de_dict = {
        "file_position": file_position, "pixel_type": pixel_type,
        "compression": compression_code, "pyramid_type": pyramid_type,
        "stored_w": w, "stored_h": h,
        "logical_w": logical_w, "logical_h": logical_h,
        "start_x": start_x, "start_y": start_y,
    }
    return seg, de_dict


def _build_directory_segment(entries: list[dict]) -> bytes:
    sid = _DIR_MAGIC + b"\x00"
    body = struct.pack("<I", len(entries)) + b"\x00" * 124
    for e in entries:
        body += struct.pack(
            "<2siqiiBB4si",
            b"DV", e["pixel_type"], e["file_position"], 0,
            e["compression"], e.get("pyramid_type", 0),
            0, b"\x00\x00\x00\x00", 2,
        )
        body += struct.pack(
            "<4siifi", b"X",
            e.get("start_x", 0), e["logical_w"], 0.0, e["stored_w"],
        )
        body += struct.pack(
            "<4siifi", b"Y",
            e.get("start_y", 0), e["logical_h"], 0.0, e["stored_h"],
        )
    return _pad_segment(sid + body)


def _build_file_header(
    directory_position: int, metadata_position: int, file_size: int,
) -> bytes:
    sid = _FILE_MAGIC + b"\x00" * 6
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
    return _pad_segment(sid + payload, payload_alloc=file_size)


# ---------------------------------------------------------------------------
# Public assembly
# ---------------------------------------------------------------------------


def _assemble(
    sub_segments: list[bytes],
    sub_dir_entries: list[dict],
    metadata_xml: bytes,
) -> bytes:
    """Bundle the file header + metadata + sub-blocks + directory into
    one valid CZI byte stream."""
    file_header_size = 32 + 88
    file_header_size = (file_header_size + 31) // 32 * 32   # round up

    metadata_segment = _build_metadata_segment(metadata_xml)
    metadata_position = file_header_size
    cur_offset = metadata_position + len(metadata_segment)

    # Patch each sub-block's directory entry with its on-disk offset.
    # We've already laid them out left-to-right so file_position values
    # are correct; nothing to patch here.
    for seg in sub_segments:
        cur_offset += len(seg)
    directory_position = cur_offset

    directory_segment = _build_directory_segment(sub_dir_entries)
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


# ---------------------------------------------------------------------------
# CziWriter — single-resolution write
# ---------------------------------------------------------------------------


class CziWriter:
    """Write a single-resolution CZI file.

    Usage::

        with CziWriter("out.czi", compression="zstdhdr") as w:
            w.write(image)
    """

    def __init__(
        self,
        path: str | Path,
        *,
        compression: str = "none",
        hilo: bool = True,
        metadata_xml: bytes | str = b"<Metadata/>",
    ):
        self._path = Path(path)
        if compression not in _CMP_NAME_TO_CODE:
            raise CziWriterError(
                f"unknown compression {compression!r}; expected one of "
                f"{sorted(_CMP_NAME_TO_CODE)}"
            )
        self._cmp_code = _CMP_NAME_TO_CODE[compression]
        self._hilo = bool(hilo)
        self._metadata_xml = (
            metadata_xml.encode("utf-8") if isinstance(metadata_xml, str)
            else metadata_xml
        )
        self._closed = False
        self._frames: list[np.ndarray] = []

    def write(self, array: np.ndarray) -> None:
        """Add one 2D frame to the CZI.

        For now each call appends one sub-block; multiple calls produce
        a multi-frame CZI (one sub-block per call).
        """
        if self._closed:
            raise CziWriterError("CziWriter is closed")
        if array.ndim != 2:
            raise CziWriterError(
                f"CziWriter expects 2D ndarray; got shape={array.shape}"
            )
        if array.dtype not in _DTYPE_TO_PIXELTYPE:
            raise CziWriterError(
                f"unsupported dtype {array.dtype}; expected one of "
                f"{sorted(d.name for d in _DTYPE_TO_PIXELTYPE)}"
            )
        self._frames.append(array)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._frames:
            raise CziWriterError(
                "CziWriter.close(): no frames written — call .write(arr) first"
            )

        # Lay out sub-blocks at their final offsets.
        file_header_size = (32 + 88 + 31) // 32 * 32
        metadata_segment = _build_metadata_segment(self._metadata_xml)
        cur = file_header_size + len(metadata_segment)
        sub_segments: list[bytes] = []
        sub_dir_entries: list[dict] = []
        for fr in self._frames:
            pixel_type = _DTYPE_TO_PIXELTYPE[fr.dtype]
            seg, de = _build_subblock(
                fr,
                pixel_type=pixel_type,
                compression_code=self._cmp_code,
                hilo=self._hilo,
                file_position=cur,
                logical_shape=fr.shape,
            )
            sub_segments.append(seg)
            sub_dir_entries.append(de)
            cur += len(seg)
        raw = _assemble(sub_segments, sub_dir_entries, self._metadata_xml)
        self._path.write_bytes(raw)

    def __enter__(self) -> "CziWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


# ---------------------------------------------------------------------------
# CziPyramidWriter — multi-resolution write
# ---------------------------------------------------------------------------


class CziPyramidWriter:
    """Write a multi-resolution CZI file (the format ZEN produces for
    multiscale acquisitions).

    Usage::

        # Caller provides the downscaled levels (we don't downscale).
        base = my_image
        half = base[::2, ::2]
        quarter = base[::4, ::4]
        with CziPyramidWriter("pyramid.czi") as w:
            w.write_pyramid([base, half, quarter])

    The on-disk layout records ``shape`` (the level-0 logical extent)
    and ``stored_shape`` (the actual pixel grid stored on disk) per
    sub-block. The ratio is the pyramid level's scale factor — exactly
    the encoding ZEN uses and the format
    :class:`opencodecs.CziPyramidReader` reads.

    Validation
    ----------
    The output is byte-compatible with the reference Python CZI readers
    (``czifile`` and ``pylibCZIrw``). Tests
    (``tests/test_czi_pyramid.py``) cross-validate every fixture against
    both readers — when those agree with our own reader on
    ``shape`` / ``stored_shape`` / ``pyramid_type`` / ``is_pyramid``
    for every sub-block, the bytes are correct.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        compression: str = "none",
        hilo: bool = True,
        metadata_xml: bytes | str = b"<Metadata/>",
    ):
        self._path = Path(path)
        if compression not in _CMP_NAME_TO_CODE:
            raise CziWriterError(
                f"unknown compression {compression!r}; expected one of "
                f"{sorted(_CMP_NAME_TO_CODE)}"
            )
        self._cmp_code = _CMP_NAME_TO_CODE[compression]
        self._hilo = bool(hilo)
        self._metadata_xml = (
            metadata_xml.encode("utf-8") if isinstance(metadata_xml, str)
            else metadata_xml
        )
        self._closed = False
        self._levels: list[np.ndarray] = []

    def write_level(self, array: np.ndarray) -> None:
        """Append one resolution level. The first call is level 0
        (full resolution); each subsequent call is a downscaled
        version of the same scene."""
        if self._closed:
            raise CziWriterError("CziPyramidWriter is closed")
        if array.ndim != 2:
            raise CziWriterError(
                f"CziPyramidWriter expects 2D ndarray; got shape={array.shape}"
            )
        if array.dtype not in _DTYPE_TO_PIXELTYPE:
            raise CziWriterError(
                f"unsupported dtype {array.dtype}; expected one of "
                f"{sorted(d.name for d in _DTYPE_TO_PIXELTYPE)}"
            )
        self._levels.append(array)

    def write_pyramid(self, levels: Iterable[np.ndarray]) -> None:
        """Convenience: append all levels at once."""
        for lvl in levels:
            self.write_level(lvl)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._levels:
            raise CziWriterError(
                "CziPyramidWriter.close(): no levels written"
            )
        base = self._levels[0]
        logical_shape = base.shape
        pixel_type = _DTYPE_TO_PIXELTYPE[base.dtype]
        # All levels must share dtype.
        for lvl in self._levels[1:]:
            if lvl.dtype != base.dtype:
                raise CziWriterError(
                    f"all pyramid levels must share dtype; level 0 is "
                    f"{base.dtype}, another level is {lvl.dtype}"
                )

        file_header_size = (32 + 88 + 31) // 32 * 32
        metadata_segment = _build_metadata_segment(self._metadata_xml)
        cur = file_header_size + len(metadata_segment)
        sub_segments: list[bytes] = []
        sub_dir_entries: list[dict] = []
        for i, lvl in enumerate(self._levels):
            # ZEN convention: level 0 sub-blocks carry pyramid_type=0
            # (non-pyramid header) — they are themselves full-res. All
            # downscaled levels carry pyramid_type=2 (MULTI_SUBBLOCK).
            ptype = 0 if i == 0 else 2
            seg, de = _build_subblock(
                lvl,
                pixel_type=pixel_type,
                compression_code=self._cmp_code,
                hilo=self._hilo,
                file_position=cur,
                logical_shape=logical_shape,
                location=(0, 0),
                pyramid_type=ptype,
            )
            sub_segments.append(seg)
            sub_dir_entries.append(de)
            cur += len(seg)

        raw = _assemble(sub_segments, sub_dir_entries, self._metadata_xml)
        self._path.write_bytes(raw)

    def __enter__(self) -> "CziPyramidWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


__all__ = ["CziWriter", "CziPyramidWriter", "CziWriterError"]
