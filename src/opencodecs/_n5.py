"""N5 reader (Janelia / Saalfeld lab chunked array format).

N5 is what large electron-microscopy volumes are stored in around FIJI,
BigDataViewer and Janelia's OpenOrganelle. It predates Zarr v3 and looks
superficially like Zarr v2 (a directory of JSON metadata plus chunk
files), which makes the differences dangerous rather than harmless:

* **Axis order is reversed.** ``dimensions`` and ``blockSize`` are
  column-major, the ImgLib2 convention N5 came from. A reader that takes
  them at face value gets a transposed volume of the right rank.
* **Every block carries a header**, unlike a Zarr chunk which is bare
  compressed bytes: mode, rank, and the block's own extents, all
  big-endian. Edge blocks are stored at their true size rather than
  padded, and only that header says so.
* **Block files are nested one directory per dimension**, in column-major
  order, so a chunk at C-order index ``(z, y, x)`` lives at ``x/y/z``.

Reference: the N5 specification at https://github.com/saalfeldlab/n5.
Transcribed from that document; no code is derived from any
implementation.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

# N5 dataType -> numpy. N5 stores data big-endian.
_DTYPES = {
    "uint8": "u1", "int8": "i1",
    "uint16": "u2", "int16": "i2",
    "uint32": "u4", "int32": "i4",
    "uint64": "u8", "int64": "i8",
    "float32": "f4", "float64": "f8",
}

_MODE_DEFAULT = 0
_MODE_VARLENGTH = 1


class N5Error(Exception):
    """Raised for malformed N5 metadata or blocks."""


class _NotFound(KeyError):
    """A key that is absent, which for N5 means an unwritten block."""


def _make_store(root: Any) -> Callable[[str], bytes]:
    """Return ``fetch(relative_key) -> bytes``, raising _NotFound if absent.

    Accepts a local directory, an http(s) base URL, or a callable, so an
    N5 on S3 reads through the same path as one on disk.
    """
    if callable(root):
        return root
    if isinstance(root, (str, os.PathLike)):
        text = str(root)
        if text.startswith(("http://", "https://")):
            base = text.rstrip("/")

            def fetch_http(key: str) -> bytes:
                import urllib.error
                import urllib.request
                try:
                    with urllib.request.urlopen(f"{base}/{key}", timeout=60) as r:
                        return r.read()
                except urllib.error.HTTPError as exc:
                    if exc.code in (403, 404):
                        raise _NotFound(key) from None
                    raise
            return fetch_http

        base_path = Path(text)

        def fetch_file(key: str) -> bytes:
            p = base_path / key
            try:
                return p.read_bytes()
            except FileNotFoundError:
                raise _NotFound(key) from None
        return fetch_file
    raise TypeError(
        f"N5: unsupported root {type(root).__name__}; pass a directory, an "
        f"http(s) URL, or a fetch callable")


class N5Array:
    """One N5 dataset (an array), addressed by its own directory."""

    def __init__(self, root: Any, path: str | None = None):
        self._fetch = _make_store(root)
        self._path = (path or "").strip("/")
        key = f"{self._path}/attributes.json" if self._path else "attributes.json"
        try:
            meta = json.loads(self._fetch(key).decode("utf-8"))
        except _NotFound:
            raise N5Error(f"N5: no attributes.json at {key!r}") from None
        except json.JSONDecodeError as exc:
            raise N5Error(f"N5: attributes.json at {key!r} is not JSON: {exc}")

        for field in ("dimensions", "blockSize", "dataType"):
            if field not in meta:
                raise N5Error(
                    f"N5: attributes.json at {key!r} has no {field!r}; this "
                    f"is a group, not a dataset")
        dt = meta["dataType"]
        if dt not in _DTYPES:
            raise N5Error(f"N5: unsupported dataType {dt!r}")

        # Column-major on disk, so reverse for numpy's C order. This is
        # the single most consequential line in the reader.
        self._shape = tuple(int(d) for d in reversed(meta["dimensions"]))
        self._chunks = tuple(int(b) for b in reversed(meta["blockSize"]))
        if len(self._shape) != len(self._chunks):
            raise N5Error(
                f"N5: rank mismatch, dimensions has {len(self._shape)} "
                f"entries and blockSize has {len(self._chunks)}")
        if any(c <= 0 for c in self._chunks) or any(d < 0 for d in self._shape):
            raise N5Error(f"N5: invalid shape {self._shape} / chunks {self._chunks}")

        self._dtype = np.dtype(">" + _DTYPES[dt])
        self._compression = meta.get("compression", {"type": "raw"})
        self._attrs = meta

    # -- metadata ----------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """C-order shape, reversed from the stored column-major dimensions."""
        return self._shape

    @property
    def chunks(self) -> tuple[int, ...]:
        return self._chunks

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def attrs(self) -> dict[str, Any]:
        return dict(self._attrs)

    @property
    def compression(self) -> str:
        c = self._compression
        return (c.get("type") if isinstance(c, dict) else str(c)) or "raw"

    @property
    def chunk_grid(self) -> tuple[int, ...]:
        return tuple(-(-s // c) for s, c in zip(self._shape, self._chunks))

    # -- blocks ------------------------------------------------------

    def _block_key(self, index: Iterable[int]) -> str:
        """C-order chunk index -> the nested column-major path N5 uses."""
        parts = [str(int(i)) for i in reversed(tuple(index))]
        return "/".join(([self._path] if self._path else []) + parts)

    def _decompress(self, raw: bytes) -> bytes:
        kind = self.compression
        if kind == "raw":
            return raw
        if kind == "gzip":
            import gzip
            import zlib
            # N5's "gzip" is zlib-with-gzip-wrapper by default, but the
            # useZlib flag switches it to a bare zlib stream.
            if isinstance(self._compression, dict) and \
                    self._compression.get("useZlib"):
                return zlib.decompress(raw)
            return gzip.decompress(raw)
        if kind == "bzip2":
            import bz2
            return bz2.decompress(raw)
        if kind == "xz":
            import lzma
            return lzma.decompress(raw)
        if kind in ("blosc", "lz4", "zstd"):
            # Route through the codecs we already ship rather than
            # duplicating a decompressor here.
            from . import get_codec
            name = {"blosc": "blosc2", "lz4": "lz4", "zstd": "zstd"}[kind]
            try:
                return bytes(get_codec(name).decode(raw))
            except Exception as exc:                     # noqa: BLE001
                raise N5Error(
                    f"N5: {kind} block failed to decompress: {exc}") from None
        raise N5Error(f"N5: unsupported compression {kind!r}")

    def read_block(self, index: Iterable[int]) -> np.ndarray | None:
        """Read one block, or None when it was never written.

        An absent block is normal in N5: sparse datasets simply do not
        write empty regions, and a reader that treated that as an error
        could not open half the volumes in the wild.
        """
        idx = tuple(int(i) for i in index)
        grid = self.chunk_grid
        if len(idx) != len(grid) or any(not 0 <= i < g for i, g in zip(idx, grid)):
            raise IndexError(f"block {idx} outside grid {grid}")
        try:
            raw = self._fetch(self._block_key(idx))
        except _NotFound:
            return None
        if len(raw) < 4:
            raise N5Error(f"N5: block {idx} is {len(raw)} bytes, too short "
                          f"for a header")

        mode, ndim = struct.unpack_from(">HH", raw, 0)
        if mode not in (_MODE_DEFAULT, _MODE_VARLENGTH):
            raise N5Error(f"N5: block {idx} has unknown mode {mode}")
        if ndim != len(grid):
            raise N5Error(
                f"N5: block {idx} declares {ndim} dimensions, dataset has "
                f"{len(grid)}")
        off = 4 + 4 * ndim
        if len(raw) < off:
            raise N5Error(f"N5: block {idx} header is truncated")
        # Stored column-major, like everything else in the header.
        block_shape = tuple(reversed(struct.unpack_from(f">{ndim}I", raw, 4)))
        if mode == _MODE_VARLENGTH:
            # Varlength blocks carry an element count before the data.
            if len(raw) < off + 4:
                raise N5Error(f"N5: block {idx} varlength header is truncated")
            off += 4

        data = self._decompress(raw[off:])
        count = 1
        for d in block_shape:
            count *= d
        need = count * self._dtype.itemsize
        if len(data) < need:
            raise N5Error(
                f"N5: block {idx} decompressed to {len(data)} bytes, needs "
                f"{need} for shape {block_shape}")
        return np.frombuffer(data[:need], dtype=self._dtype).reshape(block_shape)

    def asarray(self) -> np.ndarray:
        """Assemble the whole dataset.

        Unwritten blocks read as zeros, which is what every N5 reader
        does and what sparse datasets rely on.
        """
        out = np.zeros(self._shape, dtype=self._dtype)
        for idx in np.ndindex(*self.chunk_grid):
            block = self.read_block(idx)
            if block is None:
                continue
            sel = tuple(
                slice(i * c, i * c + b)
                for i, c, b in zip(idx, self._chunks, block.shape))
            # An edge block is stored at its true size, so trim if a
            # writer padded it anyway.
            trimmed = tuple(
                slice(0, min(s.stop, dim) - s.start)
                for s, dim in zip(sel, self._shape))
            sel = tuple(slice(s.start, s.start + t.stop)
                        for s, t in zip(sel, trimmed))
            out[sel] = block[trimmed]
        return out

    def __repr__(self) -> str:
        return (f"<N5Array {self._path or '/'} shape={self._shape} "
                f"chunks={self._chunks} dtype={self._dtype.str} "
                f"compression={self.compression}>")


__all__ = ["N5Array", "N5Error"]
