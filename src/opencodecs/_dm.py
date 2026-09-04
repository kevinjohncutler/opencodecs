"""Gatan Digital Micrograph reader (.dm3 / .dm4).

DM is what Gatan's software writes, which means most TEM imaging and
everything off a K2 or K3 detector before it is converted. Alongside EER
and MRC it completes the electron-microscopy set here: EER is what the
newest detectors emit, MRC is what pipelines convert to, and DM is what
the microscope's own software saved.

The file is a tree of tags rather than a header. A group holds entries;
an entry is either another group or a typed value, and the image is
found by walking to ``ImageList/<n>/ImageData``. Three things make it
awkward:

* **The tag structure is big-endian; the data usually is not.** The
  tree's counts and lengths are always big-endian, while a byte-order
  field in the header says how to read the samples themselves.
* **dm3 and dm4 differ in integer width.** dm4 widened tag counts and
  lengths from 32-bit to 64-bit and added a byte-count field per entry.
  Otherwise the formats are the same shape.
* **The interesting arrays are typed at runtime.** An array entry
  encodes its element type in the same recursive type language used for
  structs, so the parser has to evaluate that rather than look it up.

Reference: the DM3/DM4 tag structure as documented by the community
(Greg Jefferis's dm3 notes and the ImageJ DM3 reader description); no
code is derived from any implementation.
"""

from __future__ import annotations

import os
import struct
from typing import Any

import numpy as np

from .core._io_helpers import read_src as _read_src
from .core.codec import ArrayReader

# Simple tag types, from the DM type language.
_SIMPLE = {
    2: ("h", 2), 3: ("i", 4), 4: ("H", 2), 5: ("I", 4),
    6: ("f", 4), 7: ("d", 8), 8: ("b", 1), 9: ("b", 1),
    10: ("b", 1), 11: ("q", 8), 12: ("Q", 8),
}
_NUMPY_FOR = {
    2: "i2", 3: "i4", 4: "u2", 5: "u4", 6: "f4", 7: "f8",
    8: "u1", 9: "i1", 10: "u1", 11: "i8", 12: "u8",
}

T_STRUCT, T_STRING, T_ARRAY = 15, 18, 20

# Gatan's image data types, which are not the tag types above.
_IMAGE_DTYPES = {
    1: "i2", 2: "f4", 3: "c8", 6: "u1", 7: "i4", 8: "u4",
    9: "i1", 10: "u2", 11: "u4", 12: "f8", 13: "c16",
    14: "u1",          # binary, stored one byte per sample
    23: "u4",          # RGBA packed
}


class DmError(Exception):
    """Raised for malformed or unsupported Digital Micrograph files."""


class DmFile(ArrayReader):
    """Reader for one .dm3 / .dm4 file."""

    def __init__(self, src: Any):
        self._raw = _read_src(src)
        if len(self._raw) < 16:
            raise DmError("dm: file is too short to hold a header")
        self.version, self._little = self._parse_header()
        self._long = 8 if self.version == 4 else 4
        self.tags: dict[str, Any] = {}
        self._parse_group(self._header_size(), self.tags, "")

    def _header_size(self) -> int:
        # version (4) + root length + byte order (4). dm4 widened only
        # the root length, from 4 bytes to 8, so the header is 12 bytes
        # in dm3 and 16 in dm4.
        return 12 if self.version == 3 else 16

    def _parse_header(self) -> tuple[int, bool]:
        version = struct.unpack_from(">i", self._raw, 0)[0]
        if version not in (3, 4):
            raise DmError(
                f"dm: version {version} is not 3 or 4; this is not a "
                f"Digital Micrograph file")
        # The byte-order flag sits after version and root length, whose
        # width is what changed between the two versions.
        off = 8 if version == 3 else 12
        little = struct.unpack_from(">i", self._raw, off)[0] == 1
        return version, little

    # -- tag tree ----------------------------------------------------

    def _u(self, fmt: str, off: int):
        return struct.unpack_from(">" + fmt, self._raw, off)[0]

    def _count(self, off: int) -> tuple[int, int]:
        """Read a tag count or length, 32-bit in dm3 and 64-bit in dm4."""
        if self.version == 3:
            return self._u("i", off), off + 4
        return self._u("q", off), off + 8

    def _parse_group(self, off: int, out: dict, prefix: str) -> int:
        raw = self._raw
        off += 2                                  # sorted, open
        n, off = self._count(off)
        for _ in range(n):
            if off + 3 > len(raw):
                break
            kind = raw[off]
            off += 1
            label_len = self._u("H", off)
            off += 2
            label = raw[off:off + label_len].decode("latin-1")
            off += label_len
            if self.version == 4:
                off += 8                          # total bytes in this entry
            name = f"{prefix}/{label}" if label else f"{prefix}/[{_}]"
            if kind == 20:                        # nested group
                off = self._parse_group(off, out, name)
            elif kind == 21:                      # typed value
                off = self._parse_entry(off, out, name)
            else:
                break
        return off

    def _parse_entry(self, off: int, out: dict, name: str) -> int:
        raw = self._raw
        if raw[off:off + 4] != b"%%%%":
            raise DmError(f"dm: expected %%%% marker at offset {off}")
        off += 4
        ntypes, off = self._count(off)
        types = []
        for _ in range(ntypes):
            t, off = self._count(off)
            types.append(t)
        value, off = self._read_typed(types, 0, off, out, name)
        if value is not None:
            out[name] = value
        return off

    def _sizeof(self, types, i: int) -> int:
        """Byte size of the type at index i in a type list.

        The type language is recursive, so a struct's size is the sum of
        its fields'. Only used to step over values we do not read.
        """
        t = types[i]
        if t in _SIMPLE:
            return _SIMPLE[t][1]
        if t == T_STRUCT:
            nfields = types[i + 2]
            return sum(_SIMPLE[types[i + 4 + f * 2]][1]
                       for f in range(nfields))
        raise DmError(f"dm: cannot size type {t}")

    def _read_typed(self, types, i: int, off: int, out: dict, name: str):
        """Evaluate one entry of the type language, returning (value, offset)."""
        t = types[i]
        if t in _SIMPLE:
            fmt, size = _SIMPLE[t]
            endian = "<" if self._little else ">"
            v = struct.unpack_from(endian + fmt, self._raw, off)[0]
            return v, off + size
        if t == T_STRING:
            length = types[i + 1]
            s = self._raw[off:off + length * 2].decode("utf-16-le", "replace")
            return s, off + length * 2
        if t == T_STRUCT:
            # struct: namelen, nfields, then (fieldnamelen, fieldtype)*
            nfields = types[i + 2]
            values = []
            cursor = off
            for f in range(nfields):
                ftype = types[i + 4 + f * 2]
                v, cursor = self._read_typed([ftype], 0, cursor, out, name)
                values.append(v)
            return tuple(values), cursor
        if t == T_ARRAY:
            elem = types[i + 1]
            length = types[-1]
            if elem == T_STRUCT:
                # An array of structs. Nothing we read needs the values,
                # but the bytes must still be stepped over: returning
                # without advancing desynchronizes every offset after it,
                # which is how this parser first failed to find
                # ImageList at all.
                stride = self._sizeof(types, i + 1)
                return None, off + length * stride
            if elem not in _NUMPY_FOR:
                raise DmError(f"dm: array of unsupported element type {elem}")
            dt = np.dtype(_NUMPY_FOR[elem]).newbyteorder(
                "<" if self._little else ">")
            nbytes = length * dt.itemsize
            # Large arrays are the image itself; keep the location rather
            # than a copy so opening a file does not materialize it.
            if nbytes > 4096:
                return _ArrayRef(off, length, dt), off + nbytes
            arr = np.frombuffer(self._raw, dtype=dt, count=length, offset=off)
            return arr, off + nbytes
        raise DmError(f"dm: unsupported tag type {t}")

    # -- image -------------------------------------------------------

    def _image_prefix(self, index: int, include_thumbnails: bool) -> str:
        prefixes = self._all_image_prefixes
        if not prefixes:
            raise DmError("dm: no ImageList/.../ImageData/Data in the tag tree")
        order = list(range(len(prefixes))) if include_thumbnails \
            else self._data_indices()
        if not 0 <= index < len(order):
            raise IndexError(
                f"dm: image {index} out of range (0..{len(order) - 1})")
        return prefixes[order[index]]

    @property
    def thumbnail_indices(self) -> tuple[int, ...]:
        """Which ImageList entries are thumbnails.

        DM stores a preview alongside the real data and says so: the
        Thumbnails group carries an ImageIndex pointing at it. Reading
        that beats guessing "index 0", which is only conventionally true
        and would silently hand back a 24-bit RGBA preview in place of
        the acquisition.
        """
        out = []
        for key, value in self.tags.items():
            if key.startswith("/Thumbnails/") and key.endswith("/ImageIndex"):
                try:
                    out.append(int(value))
                except (TypeError, ValueError):
                    continue
        return tuple(sorted(set(out)))

    @property
    def _all_image_prefixes(self) -> list[str]:
        return sorted({k.split("/ImageData")[0] for k in self.tags
                       if "/ImageData/Data" in k})

    @property
    def n_images(self) -> int:
        """Number of real images, thumbnails excluded."""
        return len(self._data_indices())

    @property
    def n_images_including_thumbnails(self) -> int:
        return len(self._all_image_prefixes)

    def _data_indices(self) -> list[int]:
        thumbs = set(self.thumbnail_indices)
        total = len(self._all_image_prefixes)
        real = [i for i in range(total) if i not in thumbs]
        # A file that is nothing but a thumbnail is still readable.
        return real or list(range(total))

    def _image_tags(self, index: int, include_thumbnails: bool = False):
        prefix = self._image_prefix(index, include_thumbnails)
        data = None
        dims: list[int] = []
        dtype_code = None
        for key, value in self.tags.items():
            if not key.startswith(prefix):
                continue
            tail = key[len(prefix):]
            if tail.endswith("/ImageData/Data"):
                data = value
            elif "/ImageData/Dimensions/" in tail:
                dims.append(int(value))
            elif tail.endswith("/ImageData/DataType"):
                dtype_code = int(value)
        if data is None:
            raise DmError(f"dm: image {index} has no Data array")
        if not dims:
            raise DmError(f"dm: image {index} has no Dimensions")
        # Dimensions are listed fastest-axis-first.
        return data, tuple(reversed(dims)), dtype_code

    def asarray(self, index: int = 0, *,
                include_thumbnails: bool = False) -> np.ndarray:
        """Decode one image, skipping the embedded thumbnail by default.

        Every other DM reader does the same. Returning the preview as
        image 0 would hand back a 24-bit RGBA rendering where the caller
        asked for the acquisition.
        """
        data, shape, dtype_code = self._image_tags(index, include_thumbnails)
        count = 1
        for d in shape:
            count *= d
        if dtype_code is not None and dtype_code in _IMAGE_DTYPES:
            dt = np.dtype(_IMAGE_DTYPES[dtype_code]).newbyteorder(
                "<" if self._little else ">")
        elif isinstance(data, _ArrayRef):
            dt = data.dtype
        else:
            dt = data.dtype

        if isinstance(data, _ArrayRef):
            need = count * dt.itemsize
            if data.offset + need > len(self._raw):
                raise DmError(
                    f"dm: truncated image data; {shape} needs {need} bytes")
            arr = np.frombuffer(self._raw, dtype=dt, count=count,
                                offset=data.offset)
        else:
            arr = np.asarray(data)
            if arr.dtype != dt:
                arr = arr.view(dt) if arr.nbytes == count * dt.itemsize else arr
            if arr.size < count:
                raise DmError(
                    f"dm: image array holds {arr.size} samples, needs {count}")
            arr = arr[:count]
        return arr.reshape(shape)

    # -- Reader contract ---------------------------------------------
    #
    # A DM file is a collection of images rather than one volume, and
    # they need not share a shape, so a "frame" here is a whole image
    # and ``read()`` is the first one. That is already what
    # ``oc.read(path, format="dm")`` returns, so the two agree.

    is_chunked = True

    def _frame(self, index: int) -> np.ndarray:
        return self.asarray(index)

    @property
    def n_frames(self) -> int:
        return self.n_images

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of image 0. ``shape_at(i)`` for the others."""
        return self.shape_at(0)

    @property
    def dtype(self) -> np.dtype:
        return self.dtype_at(0)

    def shape_at(self, index: int = 0, *,
                 include_thumbnails: bool = False) -> tuple[int, ...]:
        return tuple(self._image_tags(index, include_thumbnails)[1])

    def dtype_at(self, index: int = 0, *,
                 include_thumbnails: bool = False) -> np.dtype:
        data, _, code = self._image_tags(index, include_thumbnails)
        if code is not None and code in _IMAGE_DTYPES:
            return np.dtype(_IMAGE_DTYPES[code]).newbyteorder(
                "<" if self._little else ">")
        return np.dtype(data.dtype)

    def close(self) -> None:
        self._raw = b""

    def __enter__(self) -> "DmFile":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"<DmFile version={self.version} images={self.n_images} "
                f"(+{len(self.thumbnail_indices)} thumbnail) "
                f"tags={len(self.tags)}>")


class _ArrayRef:
    """Where a large array lives, so opening a file does not copy it."""

    __slots__ = ("offset", "length", "dtype")

    def __init__(self, offset: int, length: int, dtype: np.dtype):
        self.offset, self.length, self.dtype = offset, length, dtype

    def __repr__(self) -> str:
        return f"<array {self.length} x {self.dtype.str} at {self.offset}>"


__all__ = ["DmFile", "DmError"]
