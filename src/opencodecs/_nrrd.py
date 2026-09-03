"""NRRD reader (Nearly Raw Raster Data).

NRRD is 3D Slicer's native format and turns up throughout medical image
computing, ITK and segmentation work. It is the friendliest format in
this package: a plain-text header of ``key: value`` lines, a blank line,
then the samples.

The parts that are not obvious:

* **The header is ASCII but the data may not follow it.** A ``data file``
  field detaches the samples into a separate file, which is how the
  ``.nhdr`` + ``.raw`` pair works.
* **Axis order is Fortran.** ``sizes: 3 256 256`` means the first axis
  varies fastest, so the numpy shape is reversed.
* **``encoding`` covers raw, gzip, bzip2, ascii and hex**, and gzip is
  the common case in the wild.

Reference: the NRRD file format definition published with teem
(teem.sourceforge.net/nrrd/format.html). Transcribed from it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from .core._io_helpers import open_read_at as _open_read_at
from .core._io_helpers import read_src as _read_src

_TYPES = {
    "signed char": "i1", "int8": "i1", "int8_t": "i1",
    "uchar": "u1", "unsigned char": "u1", "uint8": "u1", "uint8_t": "u1",
    "short": "i2", "short int": "i2", "signed short": "i2",
    "signed short int": "i2", "int16": "i2", "int16_t": "i2",
    "ushort": "u2", "unsigned short": "u2", "unsigned short int": "u2",
    "uint16": "u2", "uint16_t": "u2",
    "int": "i4", "signed int": "i4", "int32": "i4", "int32_t": "i4",
    "uint": "u4", "unsigned int": "u4", "uint32": "u4", "uint32_t": "u4",
    "longlong": "i8", "long long": "i8", "long long int": "i8",
    "int64": "i8", "int64_t": "i8",
    "ulonglong": "u8", "unsigned long long": "u8", "uint64": "u8",
    "uint64_t": "u8",
    "float": "f4", "double": "f8", "block": None,
}


class NrrdError(Exception):
    """Raised for malformed NRRD headers or unsupported encodings."""


class NrrdFile:
    """Reader for one NRRD file (attached or detached data)."""

    # The header is text and rarely long; read this much to start and
    # extend only if the blank line has not arrived yet.
    _HEADER_CHUNK = 1 << 16

    def __init__(self, src: Any):
        self._path = Path(src) if isinstance(src, (str, os.PathLike)) else None
        self._read_at, self._close = _open_read_at(src)
        head = self._read_at(0, self._HEADER_CHUNK)
        # A pathological header longer than the first read still works;
        # this just costs a second request rather than being an error.
        while (b"\n\n" not in head and b"\r\n\r\n" not in head
               and len(head) >= self._HEADER_CHUNK):
            more = self._read_at(len(head), self._HEADER_CHUNK)
            if not more:
                break
            head += more
        self.header, self._data_offset = self._parse_header(head)

    def _parse_header(self, raw: bytes):
        if not raw.startswith(b"NRRD"):
            raise NrrdError("nrrd: missing NRRD magic; this is not a NRRD file")
        # The header ends at the first blank line. Both line endings occur.
        for sep in (b"\r\n\r\n", b"\n\n"):
            idx = raw.find(sep)
            if idx != -1:
                head, offset = raw[:idx], idx + len(sep)
                break
        else:
            head, offset = raw, len(raw)

        header: dict[str, str] = {}
        for line in head.split(b"\n")[1:]:
            line = line.strip()
            if not line or line.startswith(b"#"):
                continue
            text = line.decode("utf-8", "replace")
            if ":=" in text:                       # key-value pair, not a field
                k, _, v = text.partition(":=")
                header[k.strip()] = v.strip()
            elif ":" in text:
                k, _, v = text.partition(":")
                header[k.strip().lower()] = v.strip()
        return header, offset

    # -- metadata ----------------------------------------------------

    @property
    def encoding(self) -> str:
        return self.header.get("encoding", "raw").lower()

    @property
    def dtype(self) -> np.dtype:
        t = self.header.get("type", "").strip().lower()
        code = _TYPES.get(t)
        if code is None:
            raise NrrdError(f"nrrd: unsupported type {t!r}")
        base = np.dtype(code)
        if base.itemsize == 1:
            return base
        endian = self.header.get("endian", "little").lower()
        return base.newbyteorder("<" if endian == "little" else ">")

    @property
    def shape(self) -> tuple[int, ...]:
        """Numpy shape, reversed from the file's fastest-first ``sizes``."""
        sizes = self.header.get("sizes")
        if not sizes:
            raise NrrdError("nrrd: header has no 'sizes' field")
        dims = tuple(int(s) for s in sizes.split())
        want = self.header.get("dimension")
        if want is not None and int(want) != len(dims):
            raise NrrdError(
                f"nrrd: dimension says {want} but sizes has {len(dims)} entries")
        return dims[::-1]

    @property
    def space_directions(self) -> str | None:
        return self.header.get("space directions")

    @property
    def detached_data_file(self) -> str | None:
        return self.header.get("data file") or self.header.get("datafile")

    # -- data --------------------------------------------------------

    def _payload(self, nbytes: int | None = None) -> bytes:
        """The encoded samples.

        ``nbytes`` is how much the caller actually needs, which for the
        raw encoding is exactly the volume: reading only that is what
        lets a large NRRD open over HTTP without pulling the file. The
        compressed encodings have no random access, so they still read
        to the end.
        """
        detached = self.detached_data_file
        if detached is None:
            if nbytes is None:
                out, off = b"", self._data_offset
                while True:
                    chunk = self._read_at(off, 1 << 22)
                    if not chunk:
                        break
                    out += chunk
                    off += len(chunk)
                return out
            return self._read_at(self._data_offset, nbytes)
        if self._path is None:
            raise NrrdError(
                f"nrrd: header points at a detached data file {detached!r}, "
                f"which cannot be resolved because the NRRD was read from "
                f"bytes rather than a path")
        target = (self._path.parent / detached).resolve()
        try:
            if nbytes is None:
                return target.read_bytes()
            with open(target, "rb") as fh:
                return fh.read(nbytes)
        except FileNotFoundError:
            raise NrrdError(
                f"nrrd: detached data file {detached!r} not found next to "
                f"the header") from None

    def _decode(self, payload: bytes) -> bytes:
        enc = self.encoding
        if enc in ("raw", ""):
            return payload
        if enc in ("gz", "gzip"):
            import gzip
            return gzip.decompress(payload)
        if enc in ("bz2", "bzip2"):
            import bz2
            return bz2.decompress(payload)
        raise NrrdError(f"nrrd: unsupported encoding {enc!r}")

    def asarray(self) -> np.ndarray:
        shape, dtype = self.shape, self.dtype
        count = 1
        for d in shape:
            count *= d
        enc = self.encoding

        if enc in ("txt", "text", "ascii"):
            text = self._payload().decode("ascii", "replace")
            arr = np.fromstring(text, dtype=dtype, sep=" ") \
                if hasattr(np, "fromstring") else None
            if arr is None or arr.size < count:
                arr = np.array(text.split()[:count], dtype=dtype)
        elif enc == "hex":
            cleaned = re.sub(rb"\s+", b"", self._payload())
            arr = np.frombuffer(bytes.fromhex(cleaned.decode("ascii")),
                                dtype=dtype, count=count)
        else:
            need = count * dtype.itemsize
            # Only the raw encoding has a computable length; gzip and
            # bzip2 must be read to the end before anything decompresses.
            wanted = need if enc in ("raw", "") else None
            data = self._decode(self._payload(wanted))
            if len(data) < need:
                raise NrrdError(
                    f"nrrd: truncated data; {shape} {dtype} needs {need} "
                    f"bytes, {len(data)} available")
            arr = np.frombuffer(data, dtype=dtype, count=count)
        # sizes lists the fastest axis first.
        return arr.reshape(shape[::-1]).T if len(shape) > 1 else arr.reshape(shape)

    def close(self) -> None:
        closer = getattr(self, "_close", None)
        if closer is not None:
            closer()
            self._close = None

    def __enter__(self) -> "NrrdFile":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        try:
            shape, dtype = self.shape, self.dtype.str
        except NrrdError:
            shape, dtype = None, "?"
        return f"<NrrdFile shape={shape} dtype={dtype} encoding={self.encoding}>"


__all__ = ["NrrdFile", "NrrdError"]
