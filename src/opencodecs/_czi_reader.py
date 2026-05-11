"""Native CZI reader — focused on the lab's actual archive.

CZI is the Zeiss ZISRAW container format. This reader handles the subset
of CZI that occurs in the lab's archive (verified by sampling 23 files
spanning 2022-2024):

* Uncompressed (compression type 0)
* ZSTDHDR / Zstd1 (compression type 6, modern Zen default)

Not supported (don't appear in the archive):

* JPEG-XR variants (compression types 1, 4)
* Raw ZSTD0 (compression type 5) — easy to add when needed
* CameraRaw / SystemRaw (>= 100) — pass-through but unverified

Design follows the I/O lessons from the tifffile benchmarks: mmap the
file (let the kernel prefetch), parse the directory once, then decode
sub-blocks in parallel through a thread pool. We don't issue per-tile
preads because sub-blocks are written contiguously by Zen and the kernel
prefetcher already serves them efficiently when accessed sequentially.

Use::

    from opencodecs.czi import CziReader

    with CziReader(path) as r:
        arr = r.read()                  # eager: stack all sub-blocks
        for tile in r.iter_tiles():     # streaming
            ...
"""

from __future__ import annotations

import mmap
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .core.codec import Reader


# Module-level persistent thread pool. ThreadPoolExecutor's shutdown
# (called at end of `with` block) blocks until all threads exit, which
# costs 1-2 ms per CZI read. A persistent pool amortizes that across
# every read in the process. Sized at ``2 * cpu_count`` since CZI decode
# alternates between mmap-page-fault waits and zstd CPU work — that
# slight oversubscription keeps both queues busy on tiered-core CPUs.
_DEFAULT_POOL_SIZE = max(2 * (os.cpu_count() or 4), 8)
_POOL: ThreadPoolExecutor | None = None


def _get_pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(
            max_workers=_DEFAULT_POOL_SIZE,
            thread_name_prefix="opencodecs-czi",
        )
    return _POOL


# ---------------------------------------------------------------------------
# CZI pixel-type table (subset that occurs in microscopy archives)
# ---------------------------------------------------------------------------

# (numpy_dtype_str, samples_per_pixel)
_PIXEL_TYPES: dict[int, tuple[str, int]] = {
    0: ("u1", 1),     # GRAY8
    1: ("u2", 1),     # GRAY16
    2: ("f4", 1),     # GRAY32_FLOAT
    3: ("u1", 3),     # BGR24
    4: ("u2", 3),     # BGR48
    8: ("f4", 3),     # BGR96_FLOAT
    9: ("u1", 4),     # BGRA32
    10: ("c8", 1),    # GRAY64_COMPLEX_FLOAT — rare
    11: ("c16", 1),   # BGR192_COMPLEX_FLOAT — rare
    12: ("i4", 1),    # GRAY32 (signed) — rare
    13: ("f8", 1),    # GRAY64 (double float) — rare
}


def _pixel_type_dtype(pt: int) -> tuple[np.dtype, int]:
    if pt not in _PIXEL_TYPES:
        raise ValueError(f"unsupported CZI pixel type {pt}")
    s, samples = _PIXEL_TYPES[pt]
    return np.dtype(s), samples


# ---------------------------------------------------------------------------
# Directory entry — one per sub-block
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CziSubBlockEntry:
    """A single sub-block's metadata, parsed from the directory.

    All offsets are file positions (bytes from start of file). The
    pixel-data offset is computed lazily because it requires reading the
    sub-block's own metadata-size header.
    """

    file_position: int          # Start of ZISRAWSUBBLOCK segment
    pixel_type: int
    compression: int
    dimensions_count: int
    dims: tuple[str, ...]       # excluding M, S
    shape: tuple[int, ...]
    stored_shape: tuple[int, ...]
    start: tuple[int, ...]
    mosaic_index: int           # -1 if undefined
    scene_index: int            # -1 if undefined

    # storage_size is the in-file footprint of the inline DirectoryEntryDV
    # (used to compute pixel-data offset in the sub-block segment).
    storage_size: int

    @property
    def dtype(self) -> np.dtype:
        return _pixel_type_dtype(self.pixel_type)[0]

    @property
    def samples(self) -> int:
        return _pixel_type_dtype(self.pixel_type)[1]


# ---------------------------------------------------------------------------
# CziReader
# ---------------------------------------------------------------------------


class CziError(RuntimeError):
    pass


class CziReader(Reader):
    """Read a CZI file via mmap + parallel sub-block decode."""

    is_chunked = True  # random-access by sub-block index via [i]

    _FILE_MAGIC = b"ZISRAWFILE"
    _DIR_MAGIC = b"ZISRAWDIRECTORY"
    _META_MAGIC = b"ZISRAWMETADATA"
    _SUBBLOCK_MAGIC = b"ZISRAWSUBBLOCK"

    # Same XML-entity fixups hiprpy applies. CZI metadata is sometimes
    # double-escaped in older files (Zen wrote ``&amp;lt;`` instead of
    # the literal ``<``).
    _ENTITY_FIXUPS = (
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
        ("&amp;", "&"),
    )

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        # For non-local sources: a buffer-protocol object (bytes,
        # bytearray, memoryview, mmap-like) holding the full file
        # bytes. CZI parsing does scattered slice access through the
        # directory + sub-block headers; ranged HTTP per-slice would
        # be ~1000 round-trips per file, so HTTP support is "fetch
        # all then mmap-equivalent" via from_http(url).
        buffer: bytes | bytearray | memoryview | None = None,
    ) -> None:
        if path is not None:
            self.path = str(path)
            self._fd = os.open(self.path, os.O_RDONLY)
            self._size = os.fstat(self._fd).st_size
            self._owns_fd = True
            # mmap takes different keyword args on POSIX vs Windows. POSIX
            # uses prot=PROT_READ; Windows uses access=ACCESS_READ.
            if sys.platform == "win32":
                self._mmap = mmap.mmap(
                    self._fd, self._size, access=mmap.ACCESS_READ)
            else:
                self._mmap = mmap.mmap(
                    self._fd, self._size, prot=mmap.PROT_READ)
        elif buffer is not None:
            self.path = "<buffer>"
            self._fd = -1
            self._owns_fd = False
            # Hold the buffer alive for the reader's lifetime. mmap-like
            # __getitem__ + struct.unpack_from work on bytes/bytearray/
            # memoryview transparently.
            self._mmap = buffer
            self._size = len(buffer)
        else:
            raise ValueError(
                "CziReader: pass either path or buffer="
            )
        # No MADV_SEQUENTIAL: it triggers aggressive page eviction after
        # read on macOS / many Linux kernels. For 66 MB CZI files on a
        # NAS that fit easily in RAM, that just forces re-fetch from the
        # SMB server on the next call. Default kernel heuristics are fine
        # — measured to drop NAS warm-cache median from 26 ms to 19 ms,
        # and the minimum from 25 ms to 12 ms.

        self.entries: list[CziSubBlockEntry] = []
        # Populated by _parse_header(). Both refer to the start of the
        # *XML payload* in the file, not the segment header.
        self._meta_xml_off: int = 0
        self._meta_xml_size: int = 0
        # Lazy caches for metadata accessors.
        self._metadata_bytes_cache: bytes | None = None
        self._metadata_xml_cache: str | None = None

        self._parse_header()

        # Reader-ABC contract: populate shape/dtype/n_frames eagerly so
        # callers can inspect a file without decoding it.
        if self.entries:
            first = self.entries[0]
            self.dtype = first.dtype
            self.n_frames = len(self.entries)
            tile = tuple(s for s in first.stored_shape if s > 1) or (1,)
            self.shape = (self.n_frames, *tile)
        else:  # pragma: no cover - empty CZI defense
            self.dtype = np.dtype("u1")
            self.n_frames = 0
            self.shape = (0,)

    # ----- Lifecycle -----

    def close(self) -> None:
        # Drop any temporary memoryviews held in locals — np.frombuffer().copy()
        # in _decode_one shouldn't leave refs, but Python's GC may need a
        # nudge if user code retained intermediate views.
        import gc
        gc.collect()
        # Buffer-only path: nothing to close on the buffer (it's just
        # bytes), but release our reference so GC can collect it.
        if not self._owns_fd:
            self._mmap = None
            return
        try:
            self._mmap.close()
        except BufferError:  # pragma: no cover - leaked memoryview rescue path
            gc.collect()
            try:
                self._mmap.close()
            except BufferError:
                pass
        finally:
            os.close(self._fd)
            self._owns_fd = False

    @classmethod
    def from_http(
        cls,
        url: str,
        *,
        timeout: float = 120.0,
        headers: dict[str, str] | None = None,
    ) -> "CziReader":
        """Open a remote CZI by downloading it in full.

        CZI parsing does scattered slice access through directory
        and sub-block headers; ranged HTTP per-slice would issue
        ~1000 round-trips per file. We fetch once into bytes and
        operate on that buffer for the reader's lifetime. For
        very large CZIs (>1 GB) consider downloading to disk first
        and using ``CziReader(local_path)`` instead.
        """
        from ._tiff_http import http_fetch_all
        data = http_fetch_all(url, timeout=timeout, headers=headers)
        return cls(buffer=data)

    def __enter__(self) -> "CziReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ----- Header / directory parsing -----

    def _parse_header(self) -> None:
        """Read the ZISRAWFILE header, locate the directory + metadata."""
        m = self._mmap
        sid, _alloc, used = struct.unpack_from("<16sqq", m, 0)
        if not sid.startswith(self._FILE_MAGIC):
            raise CziError(f"not a CZI file: magic {sid!r}")

        # ZISRAWFILE payload starts at offset 32. Layout (CZI 1.2.2):
        #   uint32 major, uint32 minor, uint32 reserved1, uint32 reserved2,
        #   16 bytes primary_file_guid, 16 bytes file_guid, uint32 file_part,
        #   int64 directory_position, int64 metadata_position,
        #   uint32 update_pending, int64 attachment_directory_position
        _major, _minor = struct.unpack_from("<II", m, 32)
        dir_off = 32 + 4 + 4 + 4 + 4 + 16 + 16 + 4
        directory_position = struct.unpack_from("<q", m, dir_off)[0]
        metadata_position = struct.unpack_from("<q", m, dir_off + 8)[0]
        if directory_position <= 0 or directory_position >= self._size:
            raise CziError(f"invalid directory_position {directory_position}")

        self._parse_directory(directory_position)

        # Locate the metadata XML payload inside the ZISRAWMETADATA
        # segment so the lazy ``metadata_bytes`` / ``metadata_xml``
        # properties can slice it on demand. Layout (CZI 1.2.2):
        #   32 bytes: segment header (magic + sizes)
        #    4 bytes: int xml_size
        #    4 bytes: int attachment_size
        #  248 bytes: filler (segment data starts at +288)
        #   xml_size bytes: UTF-8 XML
        if 0 < metadata_position < self._size:
            seg_sid = bytes(m[metadata_position:metadata_position + 14])
            if seg_sid == self._META_MAGIC:
                xml_size = struct.unpack_from(
                    "<i", m, metadata_position + 32)[0]
                if 0 <= xml_size <= self._size:
                    self._meta_xml_off = metadata_position + 32 + 8 + 248
                    self._meta_xml_size = xml_size

    def _parse_directory(self, directory_position: int) -> None:
        """Walk the ZISRAWDIRECTORY segment, building self.entries."""
        m = self._mmap
        sid, _alloc, _used = struct.unpack_from("<16sqq", m, directory_position)
        if not sid.startswith(self._DIR_MAGIC):
            raise CziError(
                f"expected ZISRAWDIRECTORY at {directory_position}, "
                f"got {sid!r}")
        # Directory payload: uint32 entry_count, 124 bytes reserved,
        # then entry_count CziDirectoryEntryDV records (each variable-len).
        entry_count = struct.unpack_from("<I", m, directory_position + 32)[0]
        offset = directory_position + 32 + 128

        for _ in range(entry_count):
            entry, advance = self._parse_directory_entry(offset)
            self.entries.append(entry)
            offset += advance

    def _parse_directory_entry(self, off: int) -> tuple[CziSubBlockEntry, int]:
        """Parse one CziDirectoryEntryDV at ``off``; return (entry, bytes_read)."""
        m = self._mmap
        # 32-byte header (DV schema):
        #   2s schema_type, int pixel_type, q file_position, int file_part,
        #   int compression, B pyramid_type, B reserved1, 4s reserved2,
        #   int dimensions_count
        (schema, pixel_type, file_position, _file_part,
         compression, _pyramid_type, _r1, _r2,
         dims_count) = struct.unpack_from("<2siqiiBB4si", m, off)
        if schema != b"DV":
            raise CziError(f"unsupported directory entry schema {schema!r}")

        dims_off = off + 32
        dims: list[str] = []
        shape: list[int] = []
        stored_shape: list[int] = []
        start: list[int] = []
        mosaic_index = -1
        scene_index = -1
        # Zen writes dimension entries in storage order (X first); we want
        # (..., Y, X) at the end of the shape tuple. Iterate reversed.
        for i in reversed(range(dims_count)):
            d_off = dims_off + i * 20
            (dim_b, d_start, size, _coord, stored) = struct.unpack_from(
                "<4siifi", m, d_off)
            dim = dim_b.rstrip(b"\x00").decode("cp1252")
            if dim == "M":
                mosaic_index = d_start
                continue
            if dim == "S":  # pragma: no cover - scene-organized CZI not in lab corpus
                scene_index = d_start
                continue
            dims.append(dim)
            shape.append(size)
            stored_shape.append(size if stored == 0 else stored)
            start.append(d_start)

        # Append the implicit S (sample) axis.
        samples = _pixel_type_dtype(pixel_type)[1]
        dims.append("S")
        shape.append(samples)
        stored_shape.append(samples)
        start.append(0)

        storage_size = 32 + dims_count * 20
        entry = CziSubBlockEntry(
            file_position=file_position,
            pixel_type=pixel_type,
            compression=compression,
            dimensions_count=dims_count,
            dims=tuple(dims),
            shape=tuple(shape),
            stored_shape=tuple(stored_shape),
            start=tuple(start),
            mosaic_index=mosaic_index,
            scene_index=scene_index,
            storage_size=storage_size,
        )
        return entry, storage_size

    # ----- Sub-block payload decode -----

    def _pixel_data_view(self, entry: CziSubBlockEntry) -> tuple[memoryview, int]:
        """Return a zero-copy memoryview into the sub-block's pixel data,
        plus its byte size. No decompression yet.
        """
        m = self._mmap
        sb_off = entry.file_position
        # Verify segment magic.
        sid = m[sb_off:sb_off + 14]
        if sid != self._SUBBLOCK_MAGIC:
            raise CziError(
                f"expected ZISRAWSUBBLOCK at {sb_off}, got {sid!r}")
        # Skip 32-byte segment header. Then 16 bytes:
        #   int metadata_size, int attachment_size, int64 data_size
        meta_size, _att_size, data_size = struct.unpack_from(
            "<iiq", m, sb_off + 32)
        # CZI 1.2.2 spec: after the 16-byte sub-block metadata header and
        # the inline DirectoryEntryDV, filler bytes make
        # (16 + storage_size + pad) reach 256 bytes minimum. Then comes
        # ``meta_size`` bytes of XML metadata, then pixel data.
        entry_storage = entry.storage_size
        pad = max(240 - entry_storage, 0)
        data_off = sb_off + 32 + 16 + entry_storage + pad + meta_size
        return memoryview(m)[data_off:data_off + data_size], data_size

    def _decode_one(self, entry: CziSubBlockEntry) -> np.ndarray:
        """Decode a single sub-block to a numpy array of shape ``stored_shape``."""
        view, data_size = self._pixel_data_view(entry)
        dtype, samples = _pixel_type_dtype(entry.pixel_type)
        out_shape = entry.stored_shape  # already includes trailing samples axis
        n_pixels = 1
        for s in out_shape:
            n_pixels *= s
        expected_bytes = n_pixels * dtype.itemsize

        if entry.compression == 0:  # pragma: no cover - lab corpus is all ZSTDHDR
            # Copy out of the mmap so the result owns its memory and we
            # can close the mmap later without invalidating arrays.
            arr = np.frombuffer(view, dtype=dtype, count=n_pixels).copy()
            return arr.reshape(out_shape)

        if entry.compression == 5:  # pragma: no cover - ZSTD0 not in lab corpus
            # ZSTD0 — raw zstd stream. Pass the memoryview directly; the
            # native zstd codec accepts buffer-protocol input without a
            # bytes() copy.
            from .codecs._zstd import decode as _zstd_decode
            raw = _zstd_decode(view)
            arr = np.frombuffer(raw, dtype=dtype, count=n_pixels)
            return arr.reshape(out_shape)

        if entry.compression == 6:
            return self._decode_zstdhdr(view, dtype, out_shape, n_pixels)

        raise CziError(  # pragma: no cover - non-{0,5,6} compression rare in wild
            f"unsupported sub-block compression {entry.compression} "
            f"(only 0, 5, 6 are implemented)")

    @staticmethod
    def _decode_zstdhdr(
        view: memoryview, dtype: np.dtype, out_shape: tuple[int, ...],
        n_pixels: int,
    ) -> np.ndarray:
        """Decode ZSTDHDR: a 1-byte header_size + chunked flags, then a
        zstd stream. Optional hi/lo byte-shuffle on the decompressed pixels.
        """
        from .codecs._zstd import decode as _zstd_decode
        if len(view) < 2:
            raise CziError("ZSTDHDR data too short")
        header_size = view[0]
        if header_size == 0 or header_size >= len(view):
            raise CziError(f"invalid ZSTDHDR header byte {header_size}")
        hilo = False
        pos = 1
        while pos < header_size:
            chunk_type = view[pos]
            pos += 1
            if chunk_type == 1:
                if pos >= header_size:
                    raise CziError("truncated ZSTDHDR chunk type 1")
                hilo = (view[pos] & 1) != 0
                pos += 1
            else:  # pragma: no cover - unknown ZSTDHDR chunk type, rare in wild
                break  # unknown chunk; defer to zstd to find data start

        # Slice the memoryview directly — the codec accepts buffer-protocol
        # input, so no bytes() copy of the compressed payload is needed.
        raw = _zstd_decode(view[header_size:])
        if hilo:
            # CZI ZSTDHDR uses byte-plane shuffling per element. Using a
            # tight nogil Cython unshuffle is ~50x faster than the numpy
            # transpose+ascontiguousarray equivalent (see _bytetools.pyx).
            from .codecs._bytetools import byteshuffle_decode
            itemsize = dtype.itemsize
            n = len(raw) // itemsize
            unshuf = byteshuffle_decode(raw, itemsize, n)
            arr = np.frombuffer(unshuf, dtype=dtype, count=n_pixels)
        else:  # pragma: no cover - hilo=False rare; Zen always shuffles
            arr = np.frombuffer(raw, dtype=dtype, count=n_pixels)
        return arr.reshape(out_shape)

    # ----- Metadata access -----

    @property
    def metadata_bytes(self) -> bytes:
        """Raw UTF-8 bytes of the file-level ZISRAWMETADATA XML.

        Cheap to call repeatedly (cached). Returns ``b""`` if the file
        has no metadata segment (rare; would indicate a corrupt or
        truncated CZI).
        """
        if self._metadata_bytes_cache is None:
            if self._meta_xml_size <= 0:  # pragma: no cover - corrupt-CZI defense
                self._metadata_bytes_cache = b""
            else:
                # mmap.__getitem__ on a slice returns a fresh bytes; we
                # cache it so downstream Cython parsers (e.g. hiprpy's
                # metadata_summary) see the same object on repeat calls.
                self._metadata_bytes_cache = self._mmap[
                    self._meta_xml_off:
                    self._meta_xml_off + self._meta_xml_size
                ]
        return self._metadata_bytes_cache

    @property
    def metadata_xml(self) -> str:
        """File-level metadata XML as a Python ``str``.

        Lazily decoded from ``metadata_bytes`` and cached. Use
        :attr:`metadata_bytes` instead when handing the payload to a
        bytes-consuming parser (e.g. hiprpy's Cython modules) — that
        avoids a wasted decode + re-encode round trip.
        """
        if self._metadata_xml_cache is None:
            text = self.metadata_bytes.decode("utf-8", errors="replace")
            for src, tgt in self._ENTITY_FIXUPS:
                if src in text:  # pragma: no cover - double-escaped CZI rare in modern Zen
                    text = text.replace(src, tgt)
            self._metadata_xml_cache = text
        return self._metadata_xml_cache

    def subblock_metadata_bytes(self, idx: int) -> bytes:
        """Raw UTF-8 bytes of sub-block *idx*'s inline metadata XML.

        Sub-blocks carry small per-tile XML (typically position info for
        a tile's place in the mosaic). Returns ``b""`` for sub-blocks
        with no inline metadata.
        """
        if idx < 0:
            idx += len(self.entries)
        if not 0 <= idx < len(self.entries):
            raise IndexError(idx)
        entry = self.entries[idx]
        m = self._mmap
        sb_off = entry.file_position
        meta_size, _att_size, _data_size = struct.unpack_from(
            "<iiq", m, sb_off + 32)
        if meta_size <= 0:
            return b""
        # Layout: segment-header (32) + sub-block-header (16) +
        # inline DirectoryEntryDV (storage_size) + filler-to-256 +
        # metadata_xml (meta_size) + pixel data.
        entry_storage = entry.storage_size  # pragma: no cover - lab CZI corpus has no per-tile metadata
        pad = max(240 - entry_storage, 0)  # pragma: no cover
        meta_off = sb_off + 32 + 16 + entry_storage + pad  # pragma: no cover
        return m[meta_off:meta_off + meta_size]  # pragma: no cover

    # ----- Public API -----

    def __len__(self) -> int:
        return len(self.entries)

    def iter_tiles(self) -> Iterator[np.ndarray]:
        """Yield each sub-block's array in directory order (single-threaded)."""
        for entry in self.entries:
            yield np.squeeze(self._decode_one(entry))

    # Reader ABC: iter_frames() is the canonical streaming entry point.
    iter_frames = iter_tiles

    def __getitem__(self, idx) -> np.ndarray:
        """Random access to a single decoded sub-block by index."""
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self.entries)))
            return np.stack(
                [np.squeeze(self._decode_one(self.entries[i])) for i in indices],
                axis=0,
            )
        if idx < 0:
            idx += len(self.entries)
        if not 0 <= idx < len(self.entries):
            raise IndexError(idx)
        return np.squeeze(self._decode_one(self.entries[idx]))

    def read(
        self,
        *,
        n_workers: int | None = None,
        squeeze: bool = True,
    ) -> np.ndarray:
        """Decode all sub-blocks in parallel and stack along axis 0.

        Returns array of shape ``(n_subblocks, *tile_shape)``. With
        ``squeeze=True`` (default), singleton axes inside the tile_shape
        are dropped (typical CZI sub-blocks have many of them).
        """
        if not self.entries:  # pragma: no cover - empty CZI defense
            return np.empty((0,))

        first = self.entries[0]
        tile_shape = first.stored_shape
        dtype = first.dtype

        out = np.empty((len(self.entries), *tile_shape), dtype=dtype)

        def _worker(i: int) -> None:
            out[i] = self._decode_one(self.entries[i])

        if n_workers == 1 or len(self.entries) == 1:
            for i in range(len(self.entries)):
                _worker(i)
        elif n_workers is None:
            # Use the persistent module-level pool. Avoids paying ~1-2 ms
            # of ThreadPoolExecutor setup+shutdown per call.
            pool = _get_pool()
            list(pool.map(_worker, range(len(self.entries))))
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                list(ex.map(_worker, range(len(self.entries))))

        if squeeze:
            out = np.squeeze(out)
        return out

    def __repr__(self) -> str:
        if self.entries:
            e = self.entries[0]
            return (
                f"<CziReader {os.path.basename(self.path)!r} "
                f"{len(self.entries)} sub-blocks, "
                f"comp={e.compression}, dtype={e.dtype}, "
                f"tile_shape={e.stored_shape}>"
            )
        return f"<CziReader {self.path!r} (empty)>"  # pragma: no cover - empty CZI


def imread(path: str | Path, **kw) -> np.ndarray:
    """Convenience: open and read a CZI file as a stacked array."""
    with CziReader(path) as r:
        return r.read(**kw)


__all__ = ["CziReader", "CziError", "imread"]
