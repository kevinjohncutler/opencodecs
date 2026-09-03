"""DICOM file reader (.dcm).

We could already talk to a DICOMweb server and decode DICOM's RLE, but
not open a file on disk, which is a strange hole: the hard part, routing
a transfer syntax to the right codec, already existed in
:mod:`opencodecs._dicomweb`. This adds the dataset parser in front of it,
so ``decode_frame`` is shared rather than reimplemented.

What a DICOM file is, structurally:

* an optional 128-byte preamble followed by ``DICM``
* the File Meta group (0002), *always* explicit VR little-endian
  whatever the rest of the file uses, which is what makes the file
  self-describing: it carries the TransferSyntaxUID
* the dataset itself, in that transfer syntax

Pixel data comes two ways. **Native** is a plain buffer of samples.
**Encapsulated** wraps each frame in fragments delimited by item tags,
optionally preceded by a Basic Offset Table, and each frame's bytes are
then a complete JPEG / JPEG-LS / JPEG 2000 / HTJ2K stream.

Reference: PS3.5 (Data Structures and Encoding) and PS3.10 (Media
Storage). Transcribed from the standard.
"""

from __future__ import annotations

import os
import struct
from typing import Any, Iterator

import numpy as np

from .core._io_helpers import open_read_at as _open_read_at
from .core._io_helpers import read_src as _read_src

from ._dicomweb import (TS_EXPLICIT_VR_LE, TS_IMPLICIT_VR_LE, decode_frame)

# Explicit-VR little-endian, but big-endian on the wire. Retired, still
# found in older archives.
TS_EXPLICIT_VR_BE = "1.2.840.10008.1.2.2"
TS_DEFLATED = "1.2.840.10008.1.2.1.99"

_ITEM = 0xFFFE, 0xE000
_ITEM_DELIM = 0xFFFE, 0xE00D
_SEQ_DELIM = 0xFFFE, 0xE0DD

# VRs whose explicit-VR header carries a 4-byte length after two
# reserved bytes, rather than a 2-byte length.
_LONG_VRS = frozenset({b"OB", b"OW", b"OF", b"OD", b"OL", b"OV",
                       b"SQ", b"UT", b"UR", b"UC", b"UN"})

_ALL_VRS = frozenset({
    b"AE", b"AS", b"AT", b"CS", b"DA", b"DS", b"DT", b"FL", b"FD", b"IS",
    b"LO", b"LT", b"PN", b"SH", b"SL", b"SS", b"ST", b"TM", b"UI", b"UL",
    b"US", b"UR", b"UC", b"UT", b"UN", b"OB", b"OW", b"OF", b"OD", b"OL",
    b"OV", b"SQ", b"SV", b"UV",
})

# Tags we interpret. Everything else is kept as raw bytes.
TAG_TRANSFER_SYNTAX = (0x0002, 0x0010)
TAG_SAMPLES_PER_PIXEL = (0x0028, 0x0002)
TAG_PHOTOMETRIC = (0x0028, 0x0004)
TAG_PLANAR_CONFIG = (0x0028, 0x0006)
TAG_NUMBER_OF_FRAMES = (0x0028, 0x0008)
TAG_ROWS = (0x0028, 0x0010)
TAG_COLUMNS = (0x0028, 0x0011)
TAG_BITS_ALLOCATED = (0x0028, 0x0100)
TAG_BITS_STORED = (0x0028, 0x0101)
TAG_PIXEL_REPRESENTATION = (0x0028, 0x0103)
TAG_RESCALE_INTERCEPT = (0x0028, 0x1052)
TAG_RESCALE_SLOPE = (0x0028, 0x1053)
TAG_PIXEL_DATA = (0x7FE0, 0x0010)


class DicomError(Exception):
    """Raised for files that are not DICOM, or that we cannot decode."""


class _Element:
    __slots__ = ("tag", "vr", "value")

    def __init__(self, tag, vr, value):
        self.tag, self.vr, self.value = tag, vr, value


def _read_tag(buf: bytes, off: int, byteorder: str) -> tuple[int, int]:
    g, e = struct.unpack_from(byteorder + "HH", buf, off)
    return g, e


class DicomFile:
    """Reader for one DICOM file."""

    # One 64 KiB block, which is the HTTP source's own unit and far more
    # than a DICOM header usually needs. Bigger would still be correct
    # and would defeat the point: the header parse is supposed to be
    # cheap enough that opening a series does not read its pixels. When
    # a header really is larger, the prefix grows.
    _PREFIX = 1 << 16

    def __init__(self, src: Any):
        self._read_at, self._close = _open_read_at(src)
        self._meta: dict[tuple[int, int], _Element] = {}
        self._ds: dict[tuple[int, int], _Element] = {}
        self._pixel_offset: int | None = None
        self._pixel_length: int | None = None
        self._encapsulated = False

        # Read a prefix and parse it. Everything before Pixel Data is
        # header, so this reads kilobytes of a file that may be
        # gigabytes; the voxels are fetched later, by offset.
        prefix = self._PREFIX
        while True:
            self._raw = self._read_at(0, prefix)
            short = len(self._raw) < prefix          # hit the end of the file
            self._meta.clear()
            self._ds.clear()
            self._pixel_offset = None
            self._encapsulated = False
            self._parse()
            if self._pixel_offset is not None or short:
                break
            prefix *= 8
        self._prefix_len = len(self._raw)

    # -- parsing -----------------------------------------------------

    def _parse(self) -> None:
        raw = self._raw
        off = 0
        if len(raw) >= 132 and raw[128:132] == b"DICM":
            off = 132
        elif raw[:4] == b"DICM":
            off = 4
        else:
            # No preamble. Legal for datasets extracted from a network
            # stream, but only if it starts with a plausible element.
            if len(raw) < 8:
                raise DicomError("dicom: file is too short to be DICOM")
            group = struct.unpack_from("<H", raw, 0)[0]
            if group not in (0x0002, 0x0008, 0x0028):
                raise DicomError(
                    "dicom: no DICM magic and the file does not start with a "
                    "plausible element; this is not a DICOM file")

        # File Meta is always explicit VR little-endian.
        meta_end = self._parse_group2(off)
        had_meta = bool(self._meta)
        off = meta_end
        ts = self.transfer_syntax
        if not had_meta:
            # No File Meta group, so nothing declared the encoding. The
            # standard's default is implicit VR little-endian, but a
            # dataset lifted off a network stream is just as often
            # explicit, so look at the bytes instead of assuming: in
            # explicit VR the two bytes after the tag are a known VR.
            explicit = self._looks_explicit(off)
            self._parse_dataset(off, explicit, "<")
            return
        if ts == TS_DEFLATED:
            import zlib
            self._raw = raw[:off] + zlib.decompress(raw[off:], -zlib.MAX_WBITS)
            explicit, byteorder = True, "<"
            self._parse_dataset(off, explicit, byteorder)
            return
        if ts == TS_IMPLICIT_VR_LE:
            explicit, byteorder = False, "<"
        elif ts == TS_EXPLICIT_VR_BE:
            explicit, byteorder = True, ">"
        else:
            # Every remaining syntax, including all the encapsulated
            # ones, uses explicit VR little-endian for the dataset.
            explicit, byteorder = True, "<"
        self._parse_dataset(off, explicit, byteorder)

    def _looks_explicit(self, off: int) -> bool:
        raw = self._raw
        return len(raw) >= off + 6 and raw[off + 4:off + 6] in _ALL_VRS

    def _parse_group2(self, off: int) -> int:
        raw = self._raw
        end = len(raw)
        while off + 8 <= end:
            g, e = _read_tag(raw, off, "<")
            if g != 0x0002:
                break
            off, elem = self._read_element(off, True, "<")
            if elem is None:
                break
            self._meta[(g, e)] = elem
        return off

    def _read_element(self, off: int, explicit: bool, bo: str):
        """Read one element, returning (next_offset, element_or_None)."""
        raw = self._raw
        if off + 8 > len(raw):
            return len(raw), None
        g, e = _read_tag(raw, off, bo)
        off += 4
        if explicit:
            vr = raw[off:off + 2]
            off += 2
            if vr in _LONG_VRS:
                off += 2                                   # reserved
                (length,) = struct.unpack_from(bo + "I", raw, off)
                off += 4
            else:
                (length,) = struct.unpack_from(bo + "H", raw, off)
                off += 2
        else:
            vr = b""
            (length,) = struct.unpack_from(bo + "I", raw, off)
            off += 4

        if (g, e) == TAG_PIXEL_DATA:
            if length == 0xFFFFFFFF:
                self._encapsulated = True
                self._pixel_offset = off
                self._pixel_length = None
            else:
                self._pixel_offset = off
                self._pixel_length = length
                off += length
            if self._encapsulated:
                off = self._skip_encapsulated(off)
            return off, _Element((g, e), vr or b"OB", b"")

        if length == 0xFFFFFFFF:
            # Undefined-length sequence or item: walk to its delimiter
            # rather than guessing, so the following elements stay
            # aligned.
            off = self._skip_undefined_length(off, explicit, bo)
            return off, _Element((g, e), vr or b"SQ", b"")

        value = raw[off:off + length]
        off += length
        return off, _Element((g, e), vr, value)

    def _skip_undefined_length(self, off: int, explicit: bool, bo: str) -> int:
        raw = self._raw
        depth = 1
        while off + 8 <= len(raw) and depth:
            g, e = _read_tag(raw, off, bo)
            (length,) = struct.unpack_from(bo + "I", raw, off + 4)
            off += 8
            if (g, e) == _SEQ_DELIM:
                depth -= 1
            elif (g, e) == _ITEM and length == 0xFFFFFFFF:
                depth += 1
            elif (g, e) == _ITEM_DELIM:
                pass
            elif length != 0xFFFFFFFF:
                off += length
        return off

    def _skip_encapsulated(self, off: int) -> int:
        raw = self._raw
        while off + 8 <= len(raw):
            g, e = _read_tag(raw, off, "<")
            (length,) = struct.unpack_from("<I", raw, off + 4)
            off += 8
            if (g, e) == _SEQ_DELIM:
                break
            off += length
        return off

    def _parse_dataset(self, off: int, explicit: bool, bo: str) -> None:
        end = len(self._raw)
        while off < end:
            new_off, elem = self._read_element(off, explicit, bo)
            if elem is None or new_off <= off:
                break
            self._ds[elem.tag] = elem
            off = new_off

    # -- attributes --------------------------------------------------

    def _int(self, tag, default=None):
        el = self._ds.get(tag)
        if el is None or not el.value:
            return default
        v = el.value
        if el.vr in (b"US", b"SS"):
            return int(struct.unpack("<H" if el.vr == b"US" else "<h", v[:2])[0])
        if el.vr in (b"UL", b"SL"):
            return int(struct.unpack("<I" if el.vr == b"UL" else "<i", v[:4])[0])
        try:
            return int(float(v.decode("ascii", "replace").strip().rstrip("\x00")))
        except ValueError:
            # Implicit VR gives no type, so a 2-byte value is almost
            # certainly US and anything else is a decimal string.
            if len(v) == 2:
                return int(struct.unpack("<H", v)[0])
            return default

    def _str(self, tag, default=None):
        el = self._ds.get(tag) or self._meta.get(tag)
        if el is None:
            return default
        return el.value.decode("ascii", "replace").strip().rstrip("\x00") or default

    def _float(self, tag, default=None):
        s = self._str(tag)
        try:
            return float(s) if s is not None else default
        except ValueError:
            return default

    @property
    def transfer_syntax(self) -> str:
        el = self._meta.get(TAG_TRANSFER_SYNTAX)
        if el is None:
            # No File Meta group at all: the standard's default.
            return TS_IMPLICIT_VR_LE
        return el.value.decode("ascii", "replace").strip().rstrip("\x00")

    @property
    def rows(self) -> int | None:
        return self._int(TAG_ROWS)

    @property
    def columns(self) -> int | None:
        return self._int(TAG_COLUMNS)

    @property
    def samples_per_pixel(self) -> int:
        return self._int(TAG_SAMPLES_PER_PIXEL, 1) or 1

    @property
    def bits_allocated(self) -> int:
        return self._int(TAG_BITS_ALLOCATED, 8) or 8

    @property
    def pixel_representation(self) -> int:
        return self._int(TAG_PIXEL_REPRESENTATION, 0) or 0

    @property
    def photometric_interpretation(self) -> str | None:
        return self._str(TAG_PHOTOMETRIC)

    @property
    def n_frames(self) -> int:
        return max(1, self._int(TAG_NUMBER_OF_FRAMES, 1) or 1)

    @property
    def rescale(self) -> tuple[float, float]:
        return (self._float(TAG_RESCALE_SLOPE, 1.0) or 1.0,
                self._float(TAG_RESCALE_INTERCEPT, 0.0) or 0.0)

    @property
    def is_encapsulated(self) -> bool:
        return self._encapsulated

    @property
    def dtype(self) -> np.dtype:
        bits = self.bits_allocated
        signed = self.pixel_representation == 1
        if bits <= 8:
            return np.dtype("i1" if signed else "u1")
        if bits <= 16:
            return np.dtype("i2" if signed else "u2")
        return np.dtype("i4" if signed else "u4")

    @property
    def shape(self) -> tuple[int, ...]:
        r, c = self.rows, self.columns
        if r is None or c is None:
            raise DicomError("dicom: no Rows/Columns; this file has no image")
        spp = self.samples_per_pixel
        base = (r, c) if spp == 1 else (r, c, spp)
        return base if self.n_frames == 1 else (self.n_frames, *base)

    # -- pixels ------------------------------------------------------

    def _fragment_table(self) -> list[tuple[int, int]]:
        """(offset, length) of each encapsulated fragment.

        Walks only the 8-byte item headers, so locating frame 400 reads
        a few kilobytes rather than every fragment before it.
        """
        off = self._pixel_offset or 0
        table: list[tuple[int, int]] = []
        while True:
            head = self._read_at(off, 8)
            if len(head) < 8:
                break
            g, e = _read_tag(head, 0, "<")
            (length,) = struct.unpack_from("<I", head, 4)
            off += 8
            if (g, e) == _SEQ_DELIM or (g, e) != _ITEM:
                break
            table.append((off, length))
            off += length
        return table

    def _fragments(self) -> list[bytes]:
        return [self._read_at(o, n) for o, n in self._fragment_table()]

    def frame(self, index: int = 0) -> np.ndarray:
        """Decode one frame."""
        n = self.n_frames
        if not 0 <= index < n:
            raise IndexError(f"frame {index} out of range (0..{n - 1})")
        r, c = self.rows, self.columns
        spp = self.samples_per_pixel

        if not self._encapsulated:
            if self._pixel_offset is None:
                raise DicomError("dicom: file has no Pixel Data element")
            per = r * c * spp * (self.bits_allocated // 8)
            start = self._pixel_offset + index * per
            # Straight to the frame. This is the whole point of native
            # Pixel Data being a plain buffer at a known offset: frame
            # 400 of a series costs one read, not the file.
            buf = self._read_at(start, per)
            if len(buf) < per:
                raise DicomError(
                    f"dicom: truncated Pixel Data; frame {index} needs {per} "
                    f"bytes, {len(buf)} available")
            arr = np.frombuffer(buf, dtype=self.dtype)
            return arr.reshape((r, c) if spp == 1 else (r, c, spp))

        table = self._fragment_table()
        if not table:
            raise DicomError("dicom: encapsulated Pixel Data has no fragments")
        # The first item is the Basic Offset Table, which is empty when
        # the writer did not bother. With one fragment per frame the
        # mapping is direct; otherwise fall back to concatenating what
        # is left, which is right for the single-frame multi-fragment
        # case and the only sane guess without a BOT.
        body = table[1:] if len(table) > n else table
        if len(body) == n:
            off, length = body[index]
            data = self._read_at(off, length)
        elif n == 1:
            data = b"".join(self._read_at(o, ln) for o, ln in body)
        else:
            raise DicomError(
                f"dicom: {len(body)} fragments for {n} frames and no usable "
                f"basic offset table; cannot map fragments to frames")
        return decode_frame(
            data, self.transfer_syntax,
            rows=r, columns=c,
            bits_allocated=self.bits_allocated,
            samples_per_pixel=spp,
            pixel_representation=self.pixel_representation,
        )

    def asarray(self, *, rescale: bool = False) -> np.ndarray:
        """Decode every frame.

        ``rescale`` applies Rescale Slope and Intercept, which is what
        turns stored values into Hounsfield units on a CT. Off by
        default because it changes the dtype to float and most callers
        of a codec want the stored values.
        """
        n = self.n_frames
        if n == 1:
            out = self.frame(0)
        else:
            out = np.stack([self.frame(i) for i in range(n)])
        if rescale:
            slope, inter = self.rescale
            if slope != 1.0 or inter != 0.0:
                out = out * np.float32(slope) + np.float32(inter)
        return out

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(self.n_frames):
            yield self.frame(i)

    def close(self) -> None:
        self._raw = b""
        closer = getattr(self, "_close", None)
        if closer is not None:
            closer()
            self._close = None

    def __enter__(self) -> "DicomFile":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        try:
            shape = self.shape
        except DicomError:
            shape = None
        return (f"<DicomFile shape={shape} dtype={self.dtype.str} "
                f"ts={self.transfer_syntax} "
                f"encapsulated={self._encapsulated}>")


__all__ = ["DicomFile", "DicomError"]
