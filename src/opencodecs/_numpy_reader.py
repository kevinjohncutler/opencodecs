"""A reader for .npy that fetches only the part of the array it needs.

A .npy file is a short ASCII header followed by the raw buffer in
C order, which makes plane N a computable offset and a length. Nothing
about the format requires reading the whole thing, and until now this
codec did: decode() handed the entire file to np.load.

That matters most where the file is not local. The header is under a
kilobyte, so opening a 4 GB array over HTTP costs one small range
request, and reading one plane out of it costs one more.

Fortran-ordered files are read whole and sliced, deliberately: in
column-major layout a leading-axis plane is strided across the entire
buffer, so "read just this plane" would be a read of everything with
extra steps.
"""

from __future__ import annotations

import ast
import re
from typing import Any

import numpy as np

from .core.codec import Reader
from .core.io import coerce_data_source

_MAGIC = b"\x93NUMPY"


class NpyFile(Reader):
    """Reader over one ``.npy`` file, plane by plane along axis 0."""

    is_chunked = True

    def __init__(self, src: Any):
        self._ds, self._owns, self._size = coerce_data_source(src)
        head = self._ds.read_at(0, 8)
        if len(head) < 8 or head[:6] != _MAGIC:
            raise ValueError(
                "npy: not a .npy file (missing the \\x93NUMPY magic)")
        major = head[6]
        # v1 stores a 2-byte header length, v2 and v3 store 4.
        len_width = 2 if major == 1 else 4
        raw_len = self._ds.read_at(8, len_width)
        if len(raw_len) < len_width:
            raise ValueError("npy: truncated before the header length")
        hlen = int.from_bytes(raw_len, "little")
        self.data_offset = 8 + len_width + hlen
        header = self._ds.read_at(8 + len_width, hlen)
        if len(header) < hlen:
            raise ValueError("npy: truncated inside the header")
        self._meta = self._parse_header(header)

        descr, fortran, shape = self._meta
        if not isinstance(descr, str):
            raise ValueError(
                f"npy: structured or object dtypes are not supported "
                f"here ({descr!r}); use decode() for those")
        self.dtype = np.dtype(descr)
        self.shape = tuple(int(s) for s in shape)
        self.fortran_order = bool(fortran)
        self.n_frames = self.shape[0] if len(self.shape) >= 3 else 1
        # A plane along axis 0 is only contiguous in C order.
        if self.fortran_order or len(self.shape) < 3:
            self.is_chunked = len(self.shape) >= 3 and not self.fortran_order

    @staticmethod
    def _parse_header(header: bytes):
        """Pull descr / fortran_order / shape out of the header dict.

        ``ast.literal_eval`` rather than ``eval``: the header is a
        Python dict literal by specification, and a .npy is untrusted
        input like any other file.
        """
        text = header.decode("latin-1").strip()
        try:
            meta = ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"npy: unreadable header {text[:80]!r}") from exc
        if not isinstance(meta, dict):
            raise ValueError(f"npy: header is not a dict: {text[:80]!r}")
        try:
            return meta["descr"], meta["fortran_order"], meta["shape"]
        except KeyError as exc:
            raise ValueError(f"npy: header is missing {exc}") from exc

    # ---- Reader ----------------------------------------------------

    def _plane_bytes(self) -> int:
        n = 1
        for s in self.shape[1:]:
            n *= s
        return n * self.dtype.itemsize

    def plane(self, index: int) -> np.ndarray:
        """One plane along axis 0, read at its own offset."""
        if not self.is_chunked:
            raise TypeError(
                "npy: this array is Fortran-ordered or fewer than 3 "
                "dimensions, so a plane is not contiguous; use read()")
        n = self.shape[0]
        if not -n <= index < n:
            raise IndexError(f"npy: plane {index} out of range for {n}")
        if index < 0:
            index += n
        size = self._plane_bytes()
        raw = self._ds.read_at(self.data_offset + index * size, size)
        if len(raw) < size:
            raise ValueError(
                f"npy: truncated file; plane {index} needs {size} bytes, "
                f"got {len(raw)}")
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.shape[1:])

    def __getitem__(self, idx) -> np.ndarray:
        if self.is_chunked:
            return self.plane(int(idx))
        return super().__getitem__(idx)

    def iter_frames(self):
        if self.is_chunked:
            for i in range(self.shape[0]):
                yield self.plane(i)
        else:
            yield self.read()

    def read(self) -> np.ndarray:
        total = int(np.prod(self.shape)) * self.dtype.itemsize
        raw = self._ds.read_at(self.data_offset, total)
        if len(raw) < total:
            raise ValueError(
                f"npy: truncated file; data needs {total} bytes, "
                f"got {len(raw)}")
        arr = np.frombuffer(raw, dtype=self.dtype)
        if self.fortran_order:
            return arr.reshape(self.shape[::-1]).T
        return arr.reshape(self.shape)

    def close(self) -> None:
        if self._owns and self._ds is not None:
            self._ds.close()
        self._ds = None

    def __enter__(self) -> "NpyFile":
        return self

    def __exit__(self, *_) -> bool:
        self.close()
        return False


__all__ = ["NpyFile"]
