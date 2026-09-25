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

import math
import mmap
import operator
import os
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .core.codec import Reader
from .core.scratch import ScratchBuffer


# Module-level persistent thread pool. ThreadPoolExecutor's shutdown
# (called at end of `with` block) blocks until all threads exit, which
# costs 1-2 ms per CZI read. A persistent pool amortizes that across
# every read in the process. Sized at ``2 * cpu_count`` since CZI decode
# alternates between mmap-page-fault waits and zstd CPU work — that
# slight oversubscription keeps both queues busy on tiered-core CPUs.
_DEFAULT_POOL_SIZE = max(2 * (os.cpu_count() or 4), 8)
_POOL: ThreadPoolExecutor | None = None

#: Most concurrent workers a whole-stack read() will use when the caller
#: gives no count. Measured on a 20-core arm64 Mac and a 128-core x86-64 Linux host with
#: the batched persistent-pool dispatch below and a FRESH output each read
#: (a sweep that recycled the output favored 16, which does not survive
#: contact with real page faults): 8 and 12 tie on both machines, 16 doubles
#: the time on Linux.
_READ_MAX_WORKERS = 8


def _get_pool() -> ThreadPoolExecutor:
    # Shared with the other readers that do this. The local copy was
    # missing the double-checked lock, so a race on the first call
    # built two pools and leaked one.
    from .core.io import get_reader_pool
    return get_reader_pool("czi", _DEFAULT_POOL_SIZE)


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


# Pixel types that CZI stores in B-G-R channel order (non-alpha) or
# B-G-R-A (alpha at the end). When the caller passes ``as_rgb=True``,
# decoded arrays are re-ordered to R-G-B / R-G-B-A.
_BGR_PIXEL_TYPES = (3, 4, 8)
_BGRA_PIXEL_TYPES = (9,)


# Decode scratch is per thread, so a worker never shares the buffer with a
# concurrent decode and a view taken from it never outlives the call that took
# it. Every public result is written into storage the caller owns instead.
_TLS = threading.local()


def _decode_scratch() -> ScratchBuffer:
    """Return this thread's own reusable decode scratch."""
    scratch = getattr(_TLS, "czi_scratch", None)
    if scratch is None:
        scratch = ScratchBuffer()
        _TLS.czi_scratch = scratch
    return scratch


_ZSTD = None
_JPEGXR = None


def _load_zstd():
    """The native zstd module, imported once.

    A ``from ... import`` inside the per-tile path runs importlib's Python
    on every tile while holding the GIL, which is exactly what keeps
    decoding workers from overlapping.
    """
    global _ZSTD
    from .codecs import _zstd
    _ZSTD = _zstd
    return _zstd


def _decode_context():
    """Return this thread's own reusable zstd decompression context.

    ``ZSTD_decompress`` creates and frees a context per call; a region read
    decodes hundreds of small tiles, so each worker keeps one.
    """
    context = getattr(_TLS, "zstd_context", None)
    if context is None:
        context = (_ZSTD or _load_zstd()).DecodeContext()
        _TLS.zstd_context = context
    return context


def _payload_destination(dest, dtype, out_shape, n_pixels):
    """Validate a decode destination and return it with a flat byte view.

    ``dest=None`` allocates, so a standalone tile still returns an owned array
    with a stable lifetime rather than a view onto reusable scratch.
    """
    if dest is None:
        dest = np.empty(out_shape, dtype=dtype)
    else:
        if not isinstance(dest, np.ndarray):
            raise TypeError("CZI decode destination must be an ndarray")
        if not dest.flags.writeable:
            raise ValueError("CZI decode destination must be writable")
        if not dest.flags.c_contiguous:
            raise ValueError("CZI decode destination must be C-contiguous")
        if dest.dtype != dtype:
            raise ValueError(
                f"CZI decode destination dtype {dest.dtype} != {dtype}")
        if dest.size != n_pixels:
            raise ValueError(
                f"CZI decode destination holds {dest.size} elements, "
                f"expected {n_pixels}")
    if n_pixels == 0:  # pragma: no cover - empty tile defense
        return dest, None
    return dest, dest.reshape(-1).view(np.uint8)


#: Resolved once: ``entry.dtype`` and ``entry.samples`` are read per tile on
#: every decode and paste, and np.dtype() from a string is not free.
_PIXEL_TYPE_DTYPES: dict[int, tuple[np.dtype, int]] = {
    pt: (np.dtype(s), samples) for pt, (s, samples) in _PIXEL_TYPES.items()}


def _pixel_type_dtype(pt: int) -> tuple[np.dtype, int]:
    try:
        return _PIXEL_TYPE_DTYPES[pt]
    except KeyError:
        raise ValueError(f"unsupported CZI pixel type {pt}") from None


def _bgr_to_rgb(arr: np.ndarray, pixel_type: int) -> np.ndarray:
    """Reorder the channel axis from BGR(A) to RGB(A) in place where
    possible. Only acts on color CZI pixel types; pass-through for
    grayscale. The channel axis is always the last one in our decode
    layout (we keep CZI's storage order, so axis -1 = sample/channel).
    """
    if pixel_type in _BGR_PIXEL_TYPES:
        # 3-channel BGR -> RGB
        return arr[..., ::-1]
    if pixel_type in _BGRA_PIXEL_TYPES:
        # 4-channel BGRA -> RGBA (alpha stays at the end)
        return arr[..., [2, 1, 0, 3]]
    return arr


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

    # CZI 1.2.2 spec ``PyramidType`` (byte 25 of DirectoryEntryDV):
    #   0 = NONE         — full-resolution sub-block
    #   1 = SINGLE_SUBBLOCK — single sub-block per mosaic position, no pyramid
    #   2 = MULTI_SUBBLOCK  — pyramid level (downsampled)
    # Note that some producers leave this 0 even on downsampled
    # sub-blocks; ``is_pyramid`` covers that case by comparing the
    # logical shape with the stored shape.
    pyramid_type: int = 0

    @property
    def dtype(self) -> np.dtype:
        return _pixel_type_dtype(self.pixel_type)[0]

    @property
    def samples(self) -> int:
        return _pixel_type_dtype(self.pixel_type)[1]

    @property
    def is_pyramid(self) -> bool:
        """True if this sub-block stores a downsampled version (its
        logical extent in coordinate space is larger than the actual
        pixel grid stored on disk). Matches czifile's same-named field.
        """
        return self.shape != self.stored_shape

    @property
    def scale_factors(self) -> tuple[float, ...]:
        """Per-dimension downscale: ``shape[d] / stored_shape[d]``.

        Values > 1.0 mean this sub-block is downscaled on that axis.
        For full-res (non-pyramid) sub-blocks every value is 1.0.
        """
        return tuple(
            (float(s) / float(ss)) if ss > 0 else 1.0
            for s, ss in zip(self.shape, self.stored_shape)
        )


# ---------------------------------------------------------------------------
# CziReader
# ---------------------------------------------------------------------------


class CziError(RuntimeError):
    pass


class _RangeBuffer:
    """Selected source ranges with a persistent header and directory only."""

    def __init__(self, source, size):
        self.source = source
        self.size = size
        self.index_ranges = []
        # Physical request accounting, so a regional read can report how
        # many source calls and bytes it actually cost.
        self.requests = 0
        self.bytes_read = 0
        self.prefetch(0, min(size, 128))

    def prefetch(self, offset, length):
        data = self._read(offset, length)
        self.index_ranges.append((offset, offset + length, data))

    def _read(self, offset, length):
        if offset < 0 or length < 0 or offset + length > self.size:
            raise CziError("CZI range lies outside the source")
        data = self.source.read_at(offset, length)
        self.requests += 1
        self.bytes_read += len(data)
        if len(data) != length:
            raise CziError("truncated CZI source range")
        return data

    def __getitem__(self, selection):
        start, stop, step = selection.indices(self.size)
        if step != 1:
            raise ValueError("CZI source slices require unit stride")
        for lo, hi, data in self.index_ranges:
            if lo <= start and stop <= hi:
                return data[start - lo:stop - lo]
        return self._read(start, max(0, stop - start))


def _unpack_from(fmt, source, offset):
    if isinstance(source, _RangeBuffer):
        return struct.unpack(fmt, source[offset:offset + struct.calcsize(fmt)])
    return struct.unpack_from(fmt, source, offset)


_DE_HEADER = struct.Struct("<2siqiiBB4si")
_DE_DIMS: dict[int, struct.Struct] = {}
#: Directory dimension layouts seen so far, keyed by the dimension names in
#: storage order. A file repeats one or two layouts across every entry, so
#: the name decoding and M/S bookkeeping happen once per layout, not per entry.
_DE_LAYOUTS: dict[tuple[bytes, ...], tuple] = {}


def _tuple_getter(positions):
    """itemgetter that always returns a tuple, even for zero or one position."""
    if len(positions) == 1:
        (k,) = positions
        return lambda values: (values[k],)
    if not positions:
        return lambda values: ()
    return operator.itemgetter(*positions)


def _dims_layout(names: tuple[bytes, ...]):
    """How to read one dimension layout straight out of an unpacked entry.

    Returns ``(dims, shape getter, stored getter, start getter, M position,
    S position)``; the getters pick (reversed, M and S dropped) fields out of
    the flat ``4siifi`` values, and a position is -1 when that axis is absent.
    """
    layout = _DE_LAYOUTS.get(names)
    if layout is None:
        keep, dims = [], []
        m_pos = s_pos = -1
        for i in reversed(range(len(names))):
            dim = names[i].rstrip(b"\x00").decode("cp1252")
            if dim == "M":
                m_pos = 5 * i + 1
            elif dim == "S":  # pragma: no cover - scene-organized CZI not in lab corpus
                s_pos = 5 * i + 1
            else:
                keep.append(i)
                dims.append(dim)
        layout = (tuple(dims) + ("S",),
                  _tuple_getter([5 * i + 2 for i in keep]),
                  _tuple_getter([5 * i + 4 for i in keep]),
                  _tuple_getter([5 * i + 1 for i in keep]),
                  m_pos, s_pos)
        if len(_DE_LAYOUTS) < 256:
            _DE_LAYOUTS[names] = layout
    return layout


def _parse_directory_entries(m, offset: int, count: int) -> list:
    """Every CziDirectoryEntryDV from a local buffer, in directory order.

    Produces exactly what CziReader._parse_directory_entry does, one entry at
    a time, with one precompiled unpack for the header and one for all of an
    entry's dimensions. Whole-slide scans carry tens of thousands of entries,
    and the per-dimension unpack and name decode were most of opening one.
    """
    unpack_header = _DE_HEADER.unpack_from
    dim_structs = _DE_DIMS
    layouts = _DE_LAYOUTS
    samples: dict[int, tuple[int]] = {}
    entries = []
    append = entries.append
    make = CziSubBlockEntry
    for _ in range(count):
        (schema, pixel_type, file_position, _file_part, compression,
         pyramid_type, _r1, _r2, dims_count) = unpack_header(m, offset)
        if schema != b"DV":
            raise CziError(f"unsupported directory entry schema {schema!r}")
        st = dim_structs.get(dims_count)
        if st is None:
            if not 0 <= dims_count <= 64:
                raise CziError(f"implausible directory dimension count {dims_count}")
            st = dim_structs[dims_count] = struct.Struct("<" + "4siifi" * dims_count)
        values = st.unpack_from(m, offset + 32)
        names = values[0::5]
        layout = layouts.get(names) or _dims_layout(names)
        dims, get_shape, get_stored, get_start, m_pos, s_pos = layout
        sample = samples.get(pixel_type)
        if sample is None:
            sample = (_pixel_type_dtype(pixel_type)[1],)
            samples[pixel_type] = sample
        shape = get_shape(values)
        stored = get_stored(values)
        if 0 in stored:
            # A stored size of 0 means "same as the logical size".
            stored = tuple([ss or sz for ss, sz in zip(stored, shape)])
        storage_size = 32 + dims_count * 20
        append(make(
            file_position=file_position,
            pixel_type=pixel_type,
            compression=compression,
            dimensions_count=dims_count,
            dims=dims,
            shape=shape + sample,
            stored_shape=stored + sample,
            start=get_start(values) + (0,),
            mosaic_index=values[m_pos] if m_pos >= 0 else -1,
            scene_index=values[s_pos] if s_pos >= 0 else -1,
            storage_size=storage_size,
            pyramid_type=pyramid_type,
        ))
        offset += storage_size
    return entries


class CziReader(Reader):
    """Read a CZI file via mmap + parallel sub-block decode."""

    is_chunked = True  # random-access by sub-block index via [i]

    _FILE_MAGIC = b"ZISRAWFILE"
    _DIR_MAGIC = b"ZISRAWDIRECTORY"
    _META_MAGIC = b"ZISRAWMETADATA"
    _SUBBLOCK_MAGIC = b"ZISRAWSUBBLOCK"

    # XML-entity fixups. CZI metadata is sometimes double-escaped in
    # older files (Zen wrote ``&amp;lt;`` instead of the literal ``<``).
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
        # bytes. from_http(url) retains its whole-file default; explicit
        # range_reads=True uses indexed requests through data_source.
        buffer: bytes | bytearray | memoryview | None = None,
        data_source=None,
        size: int | None = None,
    ) -> None:
        self._owned_source = None
        self._unpack = _unpack_from if data_source is not None else struct.unpack_from
        if sum(v is not None for v in (path, buffer, data_source)) != 1:
            raise ValueError("CziReader requires exactly one input source")
        if data_source is not None:
            if size is None or size < 128:
                raise ValueError("CZI range sources require their size (at least 128 bytes)")
            self.path = "<range source>"
            self._fd = -1
            self._owns_fd = False
            self._size = size
            self._mmap = _RangeBuffer(data_source, size)
        elif path is not None:
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
        # Buffer-only path: nothing to close on the buffer (it's just
        # bytes), but release our reference so GC can collect it.
        if not self._owns_fd:
            self._mmap = None
            if self._owned_source is not None:
                self._owned_source.close()
                self._owned_source = None
            return
        try:
            self._mmap.close()
        except BufferError:  # pragma: no cover - leaked memoryview rescue path
            # Only when a consumer kept a live memoryview into the mmap
            # does mmap.close() refuse. Run gc here (not unconditionally
            # up top) to release just-out-of-scope views, then retry.
            # An unconditional gc.collect() in close() was observed to
            # SEGV during full-suite runs when traversing heap objects
            # left over from earlier tests.
            import gc
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
        max_workers: int = 1,
        chunk_bytes: int = 8 * 1024 * 1024,
        range_reads: bool = False,
    ) -> "CziReader":
        """Open a remote CZI, optionally fetching only indexed ranges.

        ``range_reads=True`` retains the header and directory, then fetches
        only selected sub-blocks and requested metadata. The default full
        download remains appropriate for sequential reads of the entire file.

        Full-download mode materializes the file before parsing. Range mode
        retains directory metadata and fetches selected sub-block headers and
        payloads on demand, avoiding unrelated tile data.

        ``max_workers`` (default ``1``) controls the download pattern:

        * ``max_workers=1`` — a single streaming GET. Matches the old
          behavior; lowest latency on LAN / loopback where TCP
          slow-start isn't the bottleneck.
        * ``max_workers>=2`` — pipeline the download as
          ``chunk_bytes``-sized parallel Range requests. Wins on
          high-bandwidth WAN connections where a single TCP stream
          can't saturate the link (typical S3 / GCS reads from
          outside the bucket region).

        For very large CZIs (> a few GB) prefer downloading to disk
        and using ``CziReader(local_path)``.
        """
        from ._tiff_http import HTTPDataSource, http_fetch_all

        if range_reads:
            src = HTTPDataSource(
                url, headers=headers, timeout=timeout, prefetch_bytes=0,
                max_workers=max_workers,
            )
            try:
                src.read_at(0, 128)
                total = src._total_size
                if total is None:
                    raise CziError("CZI range access requires a known source size")
                reader = cls(data_source=src, size=total)
                reader._owned_source = src
                return reader
            except BaseException:
                src.close()
                raise

        # Single-stream (default): no size probe, no fan-out — matches
        # the legacy http_fetch_all path exactly.
        if max_workers <= 1:
            data = http_fetch_all(url, timeout=timeout, headers=headers)
            return cls(buffer=data)

        src = HTTPDataSource(
            url,
            headers=headers,
            timeout=timeout,
            prefetch_bytes=0,
            cache_bytes=0,
            max_workers=max_workers,
            # Don't coalesce — we already chose chunk_bytes; combining
            # back into bigger fetches would defeat the parallelism.
            coalesce_gap=0,
        )
        try:
            src.read_at(0, 4096)  # discover total size
            total = src._total_size
            if total is None or total <= 0:
                # Server didn't return Content-Range — fall back to a
                # single-stream GET.
                data = http_fetch_all(url, timeout=timeout, headers=headers)
                return cls(buffer=data)

            ranges: list[tuple[int, int]] = []
            off = 0
            while off < total:
                n = min(chunk_bytes, total - off)
                ranges.append((off, n))
                off += n
            blobs = src.read_many(ranges)
            data = b"".join(blobs)
            if len(data) != total:  # pragma: no cover - server inconsistency
                raise CziError(
                    f"CZI download incomplete: got {len(data)} of {total} bytes"
                )
            return cls(buffer=data)
        finally:
            src.close()

    def __enter__(self) -> "CziReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ----- Header / directory parsing -----

    def _parse_header(self) -> None:
        """Read the ZISRAWFILE header, locate the directory + metadata."""
        m = self._mmap
        sid, _alloc, used = self._unpack("<16sqq", m, 0)
        if not sid.startswith(self._FILE_MAGIC):
            raise CziError(f"not a CZI file: magic {sid!r}")

        # ZISRAWFILE payload starts at offset 32. Layout (CZI 1.2.2):
        #   uint32 major, uint32 minor, uint32 reserved1, uint32 reserved2,
        #   16 bytes primary_file_guid, 16 bytes file_guid, uint32 file_part,
        #   int64 directory_position, int64 metadata_position,
        #   uint32 update_pending, int64 attachment_directory_position
        _major, _minor = self._unpack("<II", m, 32)
        dir_off = 32 + 4 + 4 + 4 + 4 + 16 + 16 + 4
        directory_position = self._unpack("<q", m, dir_off)[0]
        metadata_position = self._unpack("<q", m, dir_off + 8)[0]
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
                xml_size = self._unpack(
                    "<i", m, metadata_position + 32)[0]
                if 0 <= xml_size <= self._size:
                    self._meta_xml_off = metadata_position + 32 + 8 + 248
                    self._meta_xml_size = xml_size

    def _parse_directory(self, directory_position: int) -> None:
        """Walk the ZISRAWDIRECTORY segment, building self.entries."""
        m = self._mmap
        sid, _alloc, _used = self._unpack("<16sqq", m, directory_position)
        if not sid.startswith(self._DIR_MAGIC):
            raise CziError(
                f"expected ZISRAWDIRECTORY at {directory_position}, "
                f"got {sid!r}")
        if isinstance(m, _RangeBuffer):
            # One metadata transfer, independent of the number of sub-blocks.
            m.prefetch(directory_position, 32 + _used)
        # Directory payload: uint32 entry_count, 124 bytes reserved,
        # then entry_count CziDirectoryEntryDV records (each variable-len).
        entry_count = self._unpack("<I", m, directory_position + 32)[0]
        offset = directory_position + 32 + 128

        if isinstance(m, _RangeBuffer):
            for _ in range(entry_count):
                entry, advance = self._parse_directory_entry(offset)
                self.entries.append(entry)
                offset += advance
            return
        self.entries.extend(_parse_directory_entries(m, offset, entry_count))

    def _parse_directory_entry(self, off: int) -> tuple[CziSubBlockEntry, int]:
        """Parse one CziDirectoryEntryDV at ``off``; return (entry, bytes_read)."""
        m = self._mmap
        # 32-byte header (DV schema):
        #   2s schema_type, int pixel_type, q file_position, int file_part,
        #   int compression, B pyramid_type, B reserved1, 4s reserved2,
        #   int dimensions_count
        (schema, pixel_type, file_position, _file_part,
         compression, pyramid_type, _r1, _r2,
         dims_count) = self._unpack("<2siqiiBB4si", m, off)
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
            (dim_b, d_start, size, _coord, stored) = self._unpack(
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
            pyramid_type=int(pyramid_type),
        )
        return entry, storage_size

    # ----- Sub-block payload decode -----

    def _pixel_data_view(self, entry: CziSubBlockEntry) -> tuple[memoryview, int]:
        """Return a zero-copy memoryview into the sub-block's pixel data,
        plus its byte size. No decompression yet.
        """
        m = self._mmap
        sb_off = entry.file_position
        header = m[sb_off:sb_off + 48] if isinstance(m, _RangeBuffer) else None
        # Verify segment magic.
        sid = header[:14] if header is not None else m[sb_off:sb_off + 14]
        if sid != self._SUBBLOCK_MAGIC:
            raise CziError(
                f"expected ZISRAWSUBBLOCK at {sb_off}, got {sid!r}")
        # Skip 32-byte segment header. Then 16 bytes:
        #   int metadata_size, int attachment_size, int64 data_size
        meta_size, _att_size, data_size = (
            struct.unpack_from("<iiq", header, 32) if header is not None else
            struct.unpack_from("<iiq", m, sb_off + 32))
        # CZI 1.2.2 spec: after the 16-byte sub-block metadata header and
        # the inline DirectoryEntryDV, filler bytes make
        # (16 + storage_size + pad) reach 256 bytes minimum. Then comes
        # ``meta_size`` bytes of XML metadata, then pixel data.
        entry_storage = entry.storage_size
        pad = max(240 - entry_storage, 0)
        data_off = sb_off + 32 + 16 + entry_storage + pad + meta_size
        if meta_size < 0 or data_off < 0 or data_size < 0 or data_off + data_size > self._size:
            raise CziError("invalid CZI sub-block payload range")
        if isinstance(m, _RangeBuffer):
            return memoryview(m[data_off:data_off + data_size]), data_size
        return memoryview(m)[data_off:data_off + data_size], data_size

    def _decode_one(self, entry: CziSubBlockEntry, *, _view=None,
                    dest=None, scratch=None) -> np.ndarray:
        """Decode a single sub-block to a numpy array of shape ``stored_shape``.

        ``dest`` decodes straight into caller storage, which is how ``read``
        fills one slice of its final stack without a tile-sized intermediate.
        """
        if _view is None:
            view, data_size = self._pixel_data_view(entry)
        else:
            view, data_size = _view, len(_view)
        return self._decode_payload(view, entry.pixel_type, entry.compression,
                                    entry.stored_shape, dest=dest,
                                    scratch=scratch)

    @staticmethod
    def _check_decoded_size(actual, expected):
        """Reject a payload that does not decode to exactly one tile."""
        if actual != expected:
            raise CziError(
                f"CZI payload decoded to {actual} bytes, expected {expected}")

    @staticmethod
    def _zstd_into(payload, dest_bytes, expected_bytes):
        """Decompress one sub-block payload into exactly one tile.

        The destination is sized to the tile, so a frame that would expand
        past it is malformed for this sub-block and zstd reports a too-small
        buffer. Restating it keeps every malformed-payload path raising
        CziError. Decoding short is caught by the size check.
        """
        CziReader._zstd_unshuffle_into(payload, None, dest_bytes, 1, expected_bytes)

    @staticmethod
    def _zstd_unshuffle_into(payload, scratch_bytes, dest_bytes, itemsize,
                             expected_bytes):
        """Decompress one payload and undo its byte shuffle into dest_bytes.

        One native call on this thread's reusable context: decompression and
        the unshuffle share a single GIL release, which is what lets several
        workers decode small tiles at once (``itemsize=1`` means unshuffled,
        straight into the destination).
        """
        zstd = _ZSTD or _load_zstd()
        try:
            written = zstd.decode_unshuffle_into(payload, scratch_bytes, dest_bytes,
                                                 itemsize, _decode_context())
        except zstd.ZstdError as exc:
            raise CziError(
                f"CZI payload does not decode into one {expected_bytes}-byte "
                f"tile: {exc}") from exc
        CziReader._check_decoded_size(written, expected_bytes)

    @staticmethod
    def _decode_payload(view, pixel_type, compression, out_shape, *,
                        dest=None, scratch=None):
        """Decode stored pixel bytes, shared with optional writer verification.

        ``dest`` is an optional caller-owned contiguous array to decode into.
        ``scratch`` is a worker-owned :class:`ScratchBuffer`, used only by the
        shuffled variant, whose bytes have to be unshuffled before they mean
        anything. Without ``dest`` the result is freshly allocated, so a
        standalone tile keeps owning its memory.
        """
        dtype, samples = _pixel_type_dtype(pixel_type)
        n_pixels = 1
        for s in out_shape:
            n_pixels *= s
        expected_bytes = n_pixels * dtype.itemsize

        if compression == 6:
            return CziReader._decode_zstdhdr(
                view, dtype, out_shape, n_pixels, dest=dest, scratch=scratch)

        if compression == 4:
            return CziReader._decode_jpegxr(
                view, pixel_type, dtype, out_shape, n_pixels, dest=dest)

        if compression not in (0, 5):
            raise CziError(  # pragma: no cover - non-{0,4,5,6} compression rare
                f"unsupported sub-block compression {compression} "
                f"(only 0, 4, 5, 6 are implemented)")

        out, out_bytes = _payload_destination(dest, dtype, out_shape, n_pixels)

        if compression == 0:  # pragma: no cover - lab corpus is all ZSTDHDR
            # frombuffer checks the stored payload actually covers the tile,
            # then this copies once into owned storage, so closing the mmap
            # later cannot invalidate the result.
            src = np.frombuffer(view, dtype=dtype, count=n_pixels)
            if n_pixels:
                out.reshape(-1)[:] = src
            return out

        # ZSTD0 - raw zstd stream. The native codec takes buffer-protocol
        # input, so the compressed payload needs no bytes() copy, and the
        # decompressed pixels land straight in the destination.
        if n_pixels == 0:  # pragma: no cover - empty tile defense
            return out
        CziReader._zstd_into(view, out_bytes, expected_bytes)
        return out

    @staticmethod
    def _decode_jpegxr(view, pixel_type, dtype, out_shape, n_pixels, *, dest=None):
        """Decode a JPEG XR (compression 4) sub-block into its tile.

        Most Zeiss whole-slide scans store tiles this way. Pixels come back
        in CZI's stored order: blue first for the BGR pixel types, whichever
        order the stream itself uses (Zen may encode 48-bit color as RGB).
        The stream must decode to exactly this sub-block's pixel type and
        size; anything else is malformed for this sub-block.
        """
        global _JPEGXR
        if _JPEGXR is None:
            try:
                from .codecs import _jpegxr
            except ImportError as exc:
                raise CziError(
                    "this CZI stores JPEG XR sub-blocks (compression 4), which "
                    "need the optional _jpegxr extension (built when jxrlib's "
                    "headers are found)") from exc
            _JPEGXR = _jpegxr
        out, out_bytes = _payload_destination(dest, dtype, out_shape, n_pixels)
        if n_pixels == 0:  # pragma: no cover - empty tile defense
            return out
        bgr = True if pixel_type in _BGR_PIXEL_TYPES + _BGRA_PIXEL_TYPES else None
        try:
            header = _JPEGXR.info(view)
        except _JPEGXR.JpegXrError as exc:
            raise CziError(f"CZI JPEG XR payload is not decodable: {exc}") from exc
        itemsize = header["bits_per_pixel"] // 8
        if header["width"] * header["height"] * itemsize != out_bytes.nbytes:
            raise CziError(
                f"CZI JPEG XR payload is {header['width']}x{header['height']} "
                f"at {header['bits_per_pixel']} bits per pixel; the sub-block "
                f"holds {out_bytes.nbytes} bytes")
        try:
            result = _JPEGXR.decode(view, out=out_bytes, bgr=bgr)
        except (_JPEGXR.JpegXrError, ValueError) as exc:
            raise CziError(f"CZI JPEG XR payload is not decodable: {exc}") from exc
        if result.dtype != dtype:
            raise CziError(
                f"CZI JPEG XR payload decodes to {result.dtype}, but the "
                f"sub-block's pixel type is {dtype}")
        return out

    @staticmethod
    def _zstdhdr_payload(view):
        """Split a ZSTDHDR payload into ``(zstd stream, hilo)``.

        Layout: a 1-byte header size, then chunks; chunk type 1 carries the
        hi/lo byte-shuffle flag. The stream is a slice of the memoryview, so
        the codec reads it without a copy.
        """
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
        return view[header_size:], hilo

    def _decode_windows(self, entry, out2d, windows, y_i, x_i):
        """Decode a zstd sub-block once, straight into windows of an output.

        ``out2d`` is the region output as C-contiguous bytes, one row per
        image row; each ``(sy0, sy1, sx0, sx1, dy, dx)`` window puts tile
        rows ``sy0:sy1``, columns ``sx0:sx1`` at image row ``dy``, pixel
        column ``dx``. The caller has checked that the sub-block is
        zstd-compressed (5 or 6) and laid out Y, X, samples.
        """
        view, _ = self._pixel_data_view(entry)
        dtype, _samples = _pixel_type_dtype(entry.pixel_type)
        shape = entry.stored_shape
        h, w, samples = shape[y_i], shape[x_i], shape[-1]
        if entry.compression == 6:
            payload, hilo = CziReader._zstdhdr_payload(view)
        else:
            payload, hilo = view, False
        itemsize = dtype.itemsize
        total = h * w * samples * itemsize
        zstd = _ZSTD or _load_zstd()
        try:
            if hilo:
                written = zstd.decode_unshuffle_windows(
                    payload, _decode_scratch().bytes(total), out2d, itemsize,
                    samples, h, w, windows, _decode_context())
            else:
                written = zstd.decode_unshuffle_windows(
                    payload, _decode_scratch().bytes(total), out2d, 1,
                    samples * itemsize, h, w, windows, _decode_context())
        except zstd.ZstdError as exc:
            raise CziError(
                f"CZI payload does not decode into one {total}-byte "
                f"tile: {exc}") from exc
        CziReader._check_decoded_size(written, total)

    @staticmethod
    def _decode_zstdhdr(
        view: memoryview, dtype: np.dtype, out_shape: tuple[int, ...],
        n_pixels: int, *, dest=None, scratch=None,
    ) -> np.ndarray:
        """Decode ZSTDHDR: a 1-byte header_size + chunked flags, then a
        zstd stream. Optional hi/lo byte-shuffle on the decompressed pixels.

        Unshuffled payloads decompress straight into the destination.
        Shuffled payloads decompress into worker-owned scratch and are
        unshuffled into the destination, so neither variant allocates a
        tile-sized intermediate.
        """
        payload, hilo = CziReader._zstdhdr_payload(view)
        out, out_bytes = _payload_destination(dest, dtype, out_shape, n_pixels)
        expected_bytes = n_pixels * dtype.itemsize
        if n_pixels == 0:  # pragma: no cover - empty tile defense
            return out

        if hilo:
            # CZI ZSTDHDR uses byte-plane shuffling per element. The
            # shuffled bytes are only an intermediate, so they go to
            # reusable scratch and the native unshuffle, fused with the
            # decompression, writes the destination.
            if scratch is None:
                scratch = _decode_scratch()
            CziReader._zstd_unshuffle_into(
                payload, scratch.bytes(expected_bytes), out_bytes,
                dtype.itemsize, expected_bytes)
        else:  # pragma: no cover - hilo=False rare; Zen always shuffles
            CziReader._zstd_into(payload, out_bytes, expected_bytes)
        return out

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
                # cache it so downstream parsers see the same object on
                # repeat calls.
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
        bytes-consuming Cython parser — that avoids a wasted decode +
        re-encode round trip.
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
        meta_size, _att_size, _data_size = self._unpack(
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

    def iter_tiles(self, *, as_rgb: bool = False, n_workers: int | None = None,
                   max_pending_bytes: int | None = None,
                   worker_budget=None) -> Iterator[np.ndarray]:
        """Yield each sub-block's array in directory order.

        ``as_rgb=True`` reorders the channel axis from CZI's native BGR
        / BGRA storage to RGB / RGBA. Has no effect on grayscale tiles.

        Serial by default: for local mmap-backed tiles the scheduling costs
        more than it saves, so the fast path stays a plain loop. Passing a
        worker count, a byte budget or a worker budget switches to bounded
        prefetch and decode, which overlaps decoding with whatever the caller
        does per tile while capping how much decoded output is in flight.
        Either way each yielded array is freshly allocated, so it stays valid
        once iteration resumes.
        """
        if n_workers is None and max_pending_bytes is None and worker_budget is None:
            for entry in self.entries:
                arr = np.squeeze(self._decode_one(entry))
                yield _bgr_to_rgb(arr, entry.pixel_type) if as_rgb else arr
            return

        from contextlib import closing

        from .core.pipeline import map_bounded

        workers = (min(_DEFAULT_POOL_SIZE, 8) if n_workers is None
                   else max(1, int(n_workers)))

        def descriptors():
            for entry in self.entries:
                decoded = int(np.prod(entry.stored_shape)) * entry.dtype.itemsize
                # Reserve the decoded result the consumer will hold plus the
                # shuffle scratch a worker needs while producing it. The
                # borrowed mmap view and the reader's own caches are separate.
                yield entry, 2 * decoded

        def decode(item):
            entry, _ = item
            arr = np.squeeze(self._decode_one(entry, scratch=_decode_scratch()))
            return _bgr_to_rgb(arr, entry.pixel_type) if as_rgb else arr

        with closing(map_bounded(
                decode, descriptors(), workers,
                max_bytes=((64 << 20) if max_pending_bytes is None
                           else max_pending_bytes),
                size=lambda item: item[1], budget=worker_budget,
                executor=_get_pool(), name="czi-iter")) as results:
            yield from results

    # Reader ABC: iter_frames() is the canonical streaming entry point.
    iter_frames = iter_tiles

    def __getitem__(self, idx) -> np.ndarray:
        """Random access to a single decoded sub-block by index.

        For RGB-channel-ordered output use :meth:`read_tile` instead.
        """
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self.entries)))
            return self._stack_entries([self.entries[i] for i in indices])
        if idx < 0:
            idx += len(self.entries)
        if not 0 <= idx < len(self.entries):
            raise IndexError(idx)
        return np.squeeze(self._decode_one(self.entries[idx]))

    def _stack_entries(self, entries: list[CziSubBlockEntry]) -> np.ndarray:
        """Stack decoded sub-blocks, placing each tile directly when it can.

        ``np.stack`` over a list of decoded tiles holds every tile and the
        finished stack at the same time. When the selection is homogeneous the
        stack is allocated once and each tile decodes into its own row, so
        only the output is retained. A mixed selection keeps the old path,
        including the error ``np.stack`` raises for incompatible shapes.
        """
        if not entries:
            return np.stack([])  # same error as before for an empty selection
        if len({e.stored_shape for e in entries}) != 1 or \
                len({e.dtype for e in entries}) != 1:
            return np.stack(
                [np.squeeze(self._decode_one(e)) for e in entries], axis=0)

        first = entries[0]
        tile_shape = tuple(s for s in first.stored_shape if s != 1)
        out = np.empty((len(entries), *tile_shape), dtype=first.dtype)
        scratch = _decode_scratch()
        for i, entry in enumerate(entries):
            # A fully singleton tile squeezes to a scalar row, which is not an
            # ndarray destination; keep it a one-element array instead.
            dest = out[i] if tile_shape else out[i:i + 1]
            self._decode_one(entry, dest=dest, scratch=scratch)
        return out

    def read_tile(self, idx: int, *, as_rgb: bool = False) -> np.ndarray:
        """Read a single sub-block with optional channel re-ordering.

        Same as ``self[idx]`` plus ``as_rgb`` (re-orders BGR/BGRA color
        channels to RGB/RGBA when the sub-block is one of CZI's color
        pixel types — 3, 4, 8, 9). Grayscale tiles pass through
        unchanged.
        """
        if idx < 0:
            idx += len(self.entries)
        if not 0 <= idx < len(self.entries):
            raise IndexError(idx)
        entry = self.entries[idx]
        arr = np.squeeze(self._decode_one(entry))
        if as_rgb:
            arr = _bgr_to_rgb(arr, entry.pixel_type)
        return arr

    def read(
        self,
        *,
        n_workers: int | None = None,
        squeeze: bool = True,
        as_rgb: bool = False,
        max_pending_bytes: int | None = None,
        worker_budget=None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Decode all sub-blocks in parallel and stack along axis 0.

        Returns array of shape ``(n_subblocks, *tile_shape)``. With
        ``squeeze=True`` (default), singleton axes inside the tile_shape
        are dropped (typical CZI sub-blocks have many of them).

        ``as_rgb=True`` reorders the channel axis from CZI's native BGR
        / BGRA storage to RGB / RGBA. Matches ``czifile``'s default
        decoded-pixel convention; lets callers compare outputs directly
        without an explicit ``[..., ::-1]`` swap. Grayscale CZIs are
        unaffected.

        ``out`` is a caller-owned destination, for example a
        ``numpy.memmap`` when the stack should not live in the heap. It
        must be a writable C-contiguous ndarray of the file's dtype whose
        shape is ``(n_subblocks, *tile_shape)`` or that shape with its
        singleton axes removed. Every sub-block decodes straight into its
        row; the array returned is ``out`` itself (squeezed as a view when
        ``squeeze`` is set). No file is created here, and mapping the
        output does not by itself lower resident memory: page faults and
        the operating system cache still cost what they cost.
        ``as_rgb`` is refused with ``out`` because the stored order would
        differ from the returned view.
        """
        if not self.entries:  # pragma: no cover - empty CZI defense
            return np.empty((0,))

        first = self.entries[0]
        tile_shape = first.stored_shape
        dtype = first.dtype

        if out is None:
            out = np.empty((len(self.entries), *tile_shape), dtype=dtype)
        else:
            if as_rgb:
                raise ValueError(
                    "read(out=...) stores CZI's native channel order; "
                    "as_rgb cannot reorder a caller-owned destination")
            out = self._validate_stack_destination(
                out, len(self.entries), tile_shape, dtype)
        rows = out.reshape(len(self.entries), -1)

        def _worker(i: int) -> None:
            # Pre-touching the destination pages before decode was measured
            # on both platforms and changed nothing; the faults cost the
            # same wherever they are taken.
            self._decode_one(self.entries[i], dest=rows[i],
                             scratch=_decode_scratch())

        if max_pending_bytes is not None or worker_budget is not None:
            from contextlib import closing
            from .core.pipeline import map_bounded
            workers = (min(_DEFAULT_POOL_SIZE, 8) if n_workers is None else
                       max(1, int(n_workers)))
            source = self._mmap
            def descriptors():
                for i, entry in enumerate(self.entries):
                    header = source[entry.file_position:entry.file_position + 48]
                    if header[:14] != self._SUBBLOCK_MAGIC:
                        raise CziError("invalid CZI sub-block header")
                    meta, _, length = struct.unpack_from("<iiq", header, 32)
                    offset = entry.file_position + 48 + max(240, entry.storage_size) + meta
                    if meta < 0 or length < 0 or offset + length > self._size:
                        raise CziError("invalid CZI sub-block payload range")
                    decoded = int(np.prod(entry.stored_shape)) * entry.dtype.itemsize
                    yield i, entry, offset, length, length + 3 * decoded
            def decode_descriptor(item):
                i, entry, offset, length, _ = item
                view = (memoryview(source[offset:offset + length])
                        if isinstance(source, _RangeBuffer) else
                        memoryview(source)[offset:offset + length])
                self._decode_one(entry, _view=view, dest=rows[i],
                                 scratch=_decode_scratch())
            with closing(map_bounded(
                    decode_descriptor, descriptors(), workers,
                    max_bytes=(64 << 20) if max_pending_bytes is None else max_pending_bytes,
                    size=lambda item: item[4], budget=worker_budget,
                    executor=_get_pool(), name="czi-read")) as results:
                for _ in results:
                    pass
        else:
            # Always the persistent module-level pool, never a pool per
            # call: creating one costs 1-2 ms on macOS and several on a
            # 128-core Linux host, which is more than the read itself.
            #
            # And never one task per tile: with decode writing straight
            # into the fresh output, too many threads first-touching one
            # large allocation measured 44% slower on macOS than a cap of
            # eight; Linux prefers more. A thread count is a budget:
            # resolve it from the output size under the platform's
            # measured cap, and give each worker a contiguous run.
            from .core.parallel import resolve_workers

            n = len(self.entries)
            if n_workers is not None:
                workers = max(1, min(int(n_workers), n))
            else:
                workers = resolve_workers(
                    None, n, output_bytes=out.nbytes,
                    has_decode_work=any(e.compression != 0 for e in self.entries),
                    max_workers=_READ_MAX_WORKERS)
            if workers <= 1:
                for i in range(n):
                    _worker(i)
            else:
                step = (n + workers - 1) // workers
                batches = [range(i, min(i + step, n)) for i in range(0, n, step)]

                def _run_batch(batch):
                    for i in batch:
                        _worker(i)

                list(_get_pool().map(_run_batch, batches))

        if as_rgb:
            out = _bgr_to_rgb(out, first.pixel_type)
        if squeeze:
            out = np.squeeze(out)
        return out

    @staticmethod
    def _validate_stack_destination(out, n_entries, tile_shape, dtype):
        """Check a caller-owned stack destination without copying it.

        The frame count comes from the file, never from the array, so a
        destination with the wrong number of rows is a shape error rather
        than a row-size error discovered mid-decode.
        """
        if not isinstance(out, np.ndarray):
            raise TypeError("read(out=...) needs an ndarray (memmap is fine)")
        if not out.flags.writeable:
            raise ValueError("read(out=...) destination must be writable")
        if not out.flags.c_contiguous:
            raise ValueError("read(out=...) destination must be C-contiguous")
        if out.dtype != dtype:
            raise ValueError(
                f"read(out=...) destination dtype {out.dtype} != file dtype {dtype}")
        full = (n_entries,) + tuple(tile_shape)
        squeezed = (n_entries,) + tuple(d for d in tile_shape if d != 1)
        if out.shape != full and out.shape != squeezed:
            raise ValueError(
                f"read(out=...) destination shape {out.shape} must be "
                f"{full} or {squeezed}")
        return out

    # ----- Pyramid support -----

    @property
    def is_pyramidal(self) -> bool:
        """True if any sub-block stores a downscaled version, indicating
        the file contains multiple resolution levels."""
        return any(e.is_pyramid for e in self.entries)

    def scale_factors_per_level(
        self, axes: tuple[str, str] = ("Y", "X"),
    ) -> list[tuple[float, float]]:
        """Distinct (y, x) downscale factors present in the directory,
        sorted ascending (1.0, 1.0) first.

        Each unique pair corresponds to one pyramid level. The default
        ``axes=('Y', 'X')`` is the natural spatial pyramid axis pair;
        callers can pass a different pair if the CZI has unusual
        dimension naming.
        """
        # Called unbound so reader stand-ins that borrow these two public
        # methods (the pyramid tests do) need not also borrow the helper.
        groups = CziReader._level_groups(self, axes)
        return [] if groups is None else list(groups[0])

    def _level_groups(self, axes):
        """``(sorted scales, {scale: entries})`` from one pass, cached per axes.

        Each entry's two scale factors are computed once. The pyramid reader
        used to ask for the scale list, then filter the directory once per
        level, recomputing every entry's full scale tuple each time; on a
        12k-entry slide that was most of constructing the reader.
        ``None`` when there are no entries; a single (1, 1) level when the
        axes are absent.
        """
        cache = self.__dict__.setdefault("_level_groups_cache", {})
        key = tuple(axes)
        if key in cache:
            return cache[key]
        if not self.entries:
            result = None
        else:
            ref_dims = self.entries[0].dims
            try:
                y_i = ref_dims.index(axes[0])
                x_i = ref_dims.index(axes[1])
            except ValueError:
                result = ((1.0, 1.0),), None
            else:
                groups: dict[tuple[float, float], list] = {}
                need = max(y_i, x_i)
                for e in self.entries:
                    shape, stored = e.shape, e.stored_shape
                    if need >= len(shape):
                        continue
                    sy, ssy = shape[y_i], stored[y_i]
                    sx, ssx = shape[x_i], stored[x_i]
                    k = ((float(sy) / float(ssy)) if ssy > 0 else 1.0,
                         (float(sx) / float(ssx)) if ssx > 0 else 1.0)
                    bucket = groups.get(k)
                    if bucket is None:
                        groups[k] = [e]
                    else:
                        bucket.append(e)
                result = tuple(sorted(groups)), groups
        cache[key] = result
        return result

    def entries_at_level(
        self,
        level: int = 0,
        *,
        axes: tuple[str, str] = ("Y", "X"),
    ) -> list[CziSubBlockEntry]:
        """Sub-blocks belonging to one pyramid level.

        Level 0 is the full-resolution sub-blocks; subsequent levels are
        progressively downscaled. ``level=-1`` returns the lowest-res
        overview.
        """
        groups = CziReader._level_groups(self, axes)
        if groups is None:
            return list(self.entries)
        scales, by_scale = groups
        if level < 0:
            level += len(scales)
        if not (0 <= level < len(scales)):
            raise IndexError(
                f"pyramid level {level} out of range (have {len(scales)})"
            )
        if by_scale is None:
            # The spatial axes are absent, so there is no second level to
            # separate: the old filter indexed the missing axis and raised.
            raise ValueError(f"axes {tuple(axes)} are not all in {self.entries[0].dims}")
        return list(by_scale[scales[level]])

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


# ---------------------------------------------------------------------------
# CziPyramidReader — multi-resolution view over a CziReader
# ---------------------------------------------------------------------------


from .core.pyramid import PyramidLevel, PyramidReader, _normalize_axis


#: Decoded bytes per regional-read task. Small enough that a large crop
#: still splits across workers, large enough that scheduling is a rounding
#: error next to the decode it carries.
_REGION_BATCH_BYTES = 2 << 20


def _subtract_rect(rects, cut):
    """``rects`` minus rectangle ``cut``, as disjoint (y0, y1, x0, x1) pieces."""
    cy0, cy1, cx0, cx1 = cut
    out = []
    for r in rects:
        y0, y1, x0, x1 = r
        if cy0 >= y1 or cy1 <= y0 or cx0 >= x1 or cx1 <= x0:
            out.append(r)
            continue
        if y0 < cy0:
            out.append((y0, cy0, x0, x1))
        if cy1 < y1:
            out.append((cy1, y1, x0, x1))
        my0, my1 = max(y0, cy0), min(y1, cy1)
        if x0 < cx0:
            out.append((my0, my1, x0, cx0))
        if cx1 < x1:
            out.append((my0, my1, cx1, x1))
    return out


class _CziLevelIndex:
    """Immutable spatial index over one pyramid level's stored tile bounds.

    A uniform grid keyed on the median tile size. Zen mosaics are near-regular
    grids of equal tiles, so almost every tile lands in one cell and a query
    touches only the cells under its box; an irregular tile simply registers
    in every cell it spans, which keeps lookups exact for uneven layouts at
    the cost of a longer candidate list. Candidates are then bounds-checked,
    and hits come back in composition order so composition semantics do not
    change. The index never mixes planes: each tile carries a key of its
    non-spatial coordinates, and the reader refuses a request that touches
    more than one.

    The grid is built on the second query, not the first. A viewer that
    opens a slide to show one crop would otherwise pay for the whole grid
    before its first pixel; one vectorized scan of the bounds answers that
    crop in microseconds, and the grid pays off from the next query on.
    """

    __slots__ = ("n", "y0", "y1", "x0", "x1", "bounds", "plane_axes",
                 "_entries", "_plane_of", "_plane_keys", "_queries",
                 "cell_h", "cell_w", "_cells", "_cell_ranges", "_visible")

    #: Candidate sets at or below this size are bounds-checked in Python;
    #: numpy's per-call overhead exceeds a short loop by roughly 10x.
    SMALL = 32

    def __init__(self, entries, placement):
        n = len(entries)
        self.n = n
        # Level-pixel bounds from the reader's placement: stored extents at
        # positions already shifted to the slide origin and divided by the
        # level's scale.
        ty, tx, th, tw = placement
        self.y0, self.x0 = ty, tx
        self.y1 = ty + th
        self.x1 = tx + tw
        # Python ints for the small-set path and for pasting; the arrays
        # serve large candidate sets.
        self.bounds = list(zip(self.y0.tolist(), self.y1.tolist(),
                               self.x0.tolist(), self.x1.tolist()))
        ref_dims = entries[0].dims
        self.plane_axes = tuple(
            i for i, d in enumerate(ref_dims) if d not in ("Y", "X", "S"))
        self._entries = entries
        self._plane_of = _tuple_getter(self.plane_axes)
        self._plane_keys: dict[int, tuple] = {}
        self._queries = 0
        self._cells = None
        self._cell_ranges = None
        self._visible: dict[int, list] = {}

    def plane_key(self, i):
        """Tile ``i``'s non-spatial coordinates plus its scene, cached."""
        key = self._plane_keys.get(i)
        if key is None:
            e = self._entries[i]
            key = self._plane_of(e.start) + (e.scene_index,)
            self._plane_keys[i] = key
        return key

    @property
    def cells(self):
        if self._cells is None:
            self._build_cells()
        return self._cells

    @property
    def n_cells(self):
        return len(self.cells)

    @property
    def cell_ranges(self):
        if self._cells is None:
            self._build_cells()
        return self._cell_ranges

    def _build_cells(self):
        self.cell_h = max(1, int(np.median(self.y1 - self.y0))) if self.n else 1
        self.cell_w = max(1, int(np.median(self.x1 - self.x0))) if self.n else 1
        cells: dict[tuple[int, int], list[int]] = {}
        # Cell ranges for every tile at once, then plain ints in the loop:
        # indexing numpy scalars per tile cost more than the grid itself.
        cy_lo = (self.y0 // self.cell_h).tolist()
        cy_hi = ((self.y1 - 1) // self.cell_h).tolist()
        cx_lo = (self.x0 // self.cell_w).tolist()
        cx_hi = ((self.x1 - 1) // self.cell_w).tolist()
        empty = ((self.y1 <= self.y0) | (self.x1 <= self.x0)).tolist()
        get = cells.get
        for i, (ylo, yhi, xlo, xhi, skip) in enumerate(
                zip(cy_lo, cy_hi, cx_lo, cx_hi, empty)):
            if skip:
                continue  # empty tile; cannot intersect anything
            if ylo == yhi and xlo == xhi:
                key = (ylo, xlo)
                bucket = get(key)
                if bucket is None:
                    cells[key] = [i]
                else:
                    bucket.append(i)
                continue
            for cy in range(ylo, yhi + 1):
                for cx in range(xlo, xhi + 1):
                    cells.setdefault((cy, cx), []).append(i)
        # Per tile, the cells it registered in, for overlap checks.
        self._cell_ranges = list(zip(cy_lo, cy_hi, cx_lo, cx_hi))
        self._cells = cells

    def visible(self, i):
        """The parts of tile ``i`` that no later tile on its plane covers.

        Where tiles overlap, the later one in composition order owns the pixel,
        so each tile only needs to write what is left of it after removing
        every later same-plane tile. Those pieces are disjoint across tiles,
        which is what lets workers write any tile at any time. Rectangles in
        level pixels; an empty list means the tile is fully hidden. Only
        tiles sharing a cell can overlap, so only those are compared. Cached.
        """
        pieces = self._visible.get(i)
        if pieces is not None:
            return pieces
        cells = self.cells
        bounds = self.bounds
        mine = bounds[i]
        key = self.plane_key(i)
        ylo, yhi, xlo, xhi = self._cell_ranges[i]
        later = set()
        for cy in range(ylo, yhi + 1):
            for cx in range(xlo, xhi + 1):
                for j in cells.get((cy, cx), ()):
                    if j > i:
                        later.add(j)
        pieces = [mine]
        for j in sorted(later):
            b = bounds[j]
            if (b[0] >= mine[1] or b[1] <= mine[0] or b[2] >= mine[3]
                    or b[3] <= mine[2] or self.plane_key(j) != key):
                continue
            pieces = _subtract_rect(pieces, b)
            if not pieces:
                break
        self._visible[i] = pieces
        return pieces

    def query(self, y0, y1, x0, x1):
        """Return (hits in composition order, candidates examined)."""
        if y1 <= y0 or x1 <= x0 or not self.n:
            return [], 0
        if self.n <= self.SMALL:
            # Fewer tiles than a cell walk is worth: a straight scan of the
            # bounds list is the old loop minus its attribute lookups.
            hits = [i for i, b in enumerate(self.bounds)
                    if b[0] < y1 and b[1] > y0 and b[2] < x1 and b[3] > x0]
            return hits, self.n
        if self._cells is None and self._queries == 0:
            # First query: one vectorized scan instead of building the grid.
            self._queries = 1
            keep = ((self.y0 < y1) & (self.y1 > y0) &
                    (self.x0 < x1) & (self.x1 > x0))
            return np.flatnonzero(keep).tolist(), self.n
        cells = self.cells
        candidates: set[int] = set()
        for cy in range(int(y0 // self.cell_h), int((y1 - 1) // self.cell_h) + 1):
            for cx in range(int(x0 // self.cell_w), int((x1 - 1) // self.cell_w) + 1):
                bucket = cells.get((cy, cx))
                if bucket:
                    candidates.update(bucket)
        if not candidates:
            return [], 0
        if len(candidates) <= self.SMALL:
            bounds = self.bounds
            hits = sorted(
                i for i in candidates
                if (b := bounds[i])[0] < y1 and b[1] > y0
                and b[2] < x1 and b[3] > x0)
            return hits, len(candidates)
        idx = np.fromiter(candidates, np.int64, len(candidates))
        keep = ((self.y0[idx] < y1) & (self.y1[idx] > y0) &
                (self.x0[idx] < x1) & (self.x1[idx] > x0))
        hits = np.sort(idx[keep])
        return hits.tolist(), len(candidates)


class CziPyramidReader(PyramidReader):
    """Pyramid view of a CZI file with multiple resolution levels.

    CZI's pyramid model differs from TIFF / OME-Zarr: rather than
    separate IFDs or array groups, each sub-block carries a logical
    ``shape`` (the coordinate-space extent at full resolution) and a
    ``stored_shape`` (the actual pixel grid). Downscaled sub-blocks
    have ``stored_shape < shape`` and the ratio is the level's scale
    factor. Sub-blocks with the same scale factor form one pyramid
    level.

    Examples
    --------
    Open a Zeiss multiscale CZI and crop a region from a chosen level::

        with CziPyramidReader("acquisition.czi") as p:
            best = p.best_level_for(max_pixels_y=2048)
            crop = p.read_region(best, y=(1000, 2000), x=(2000, 3000))

    Note
    ----
    Not all CZI files are pyramidal. For non-pyramidal files this
    reader presents a single level. Construct via
    :func:`CziReader.pyramid` for the natural entry point.
    """

    def __init__(self, czi: "CziReader", *, decode_workers: int | None = None,
                 max_pending_bytes: int | None = None,
                 decoded_cache_bytes: int | None = None):
        self._czi = czi
        self._scales = czi.scale_factors_per_level()
        if not self._scales:
            raise CziError("CziPyramidReader: file has no sub-blocks")
        # Optional decoded-tile cache for repeated navigation. Off unless a
        # budget is given: it spends memory to save decompression, and
        # nothing here knows whether the workload revisits tiles. Keys carry
        # this reader's identity and the tile's directory index; the file
        # must not change during the session (call invalidate_cache after
        # any change you make yourself). Cached tiles are read-only.
        self.tile_cache = None
        if decoded_cache_bytes:
            from .core.tile_cache import DecodedTileCache
            self.tile_cache = DecodedTileCache(decoded_cache_bytes)
        # ``decode_workers=None`` lets resolve_workers decide per request;
        # an explicit count is honored (1 pins a serial read). The byte
        # budget bounds decoded tiles in flight on the parallel path.
        self._decode_workers = decode_workers
        self._max_pending_bytes = max_pending_bytes
        # Keyed by id(level): levels live as long as this reader, and a
        # list.index lookup would compare PyramidLevel fields, including the
        # whole entry list, on every regional read.
        self._indexes: dict[int, _CziLevelIndex] = {}
        self._window_layouts: dict[int, tuple | None] = {}
        #: Diagnostics for the most recent read_region call: candidate
        #: entries examined, tiles decoded, workers used, elapsed seconds,
        #: and physical source requests/bytes when the source is a range
        #: source. Informational; not part of the pixel contract.
        self.region_stats: dict = {}

        # Build a PyramidLevel per distinct scale factor. Each level lists
        # its tiles in composition order: where tiles overlap, the later one
        # wins. For mosaics that is ascending mosaic index (M), ties keeping
        # directory order, the rule libCZI and czifile apply (higher M on
        # top); directory order alone put the wrong tile on top wherever Zen
        # wrote tiles out of M order, which it does (on the Axioscan corpus
        # slide, 8.9 million overlap pixels at level 0). Every path below
        # composes in this list's order, so they all inherit the rule.
        self._level_entries: list[list[CziSubBlockEntry]] = [
            _composition_order(czi.entries_at_level(i))
            for i in range(len(self._scales))
        ]
        # Sub-block starts are full-resolution coordinates on the slide, not
        # pixels of any level: Zen scans start far from zero, often
        # negative, and a level-k tile's start is not divided by its scale.
        # Level pixels therefore count from the full-resolution origin (the
        # smallest level-0 start, as czifile's image ``start``) and divide by
        # the level's scale. Before this, a real Axioscan slide reported
        # every level as (N, 0) and read nothing. ``origin`` maps back:
        # absolute = origin + level_pixel * scale.
        base = self._level_entries[0]
        by_i, bx_i = self._spatial_axes(base)
        self.origin: tuple[int, int] = (
            min(e.start[by_i] for e in base), min(e.start[bx_i] for e in base))
        self._placements: list[tuple] = []
        self._level_numbers: dict[int, int] = {}
        self._levels: list[PyramidLevel] = []
        for i, (entries, scale) in enumerate(
            zip(self._level_entries, self._scales)
        ):
            # The level's (h, w) is the bounding box of its tiles' stored
            # extents at their level positions.
            placement = _level_placement(entries, self.origin, scale)
            self._placements.append(placement)
            ty, tx, th, tw = placement
            h = max(0, int((ty + th).max()))
            w = max(0, int((tx + tw).max()))
            sy, sx = scale
            self._levels.append(PyramidLevel(
                reader=entries,
                downscale=(int(round(sy)), int(round(sx))),
                shape=(h, w),
                dtype=entries[0].dtype,
            ))
            self._level_numbers[id(self._levels[-1])] = i

    @property
    def levels(self) -> list[PyramidLevel]:
        return self._levels

    def invalidate_cache(self) -> None:
        """Drop every cached decoded tile (after mutating the source)."""
        if self.tile_cache is not None:
            self.tile_cache.clear()

    def close(self) -> None:
        if self.tile_cache is not None:
            self.tile_cache.clear()
        self._czi.close()

    def _level_index(self, level) -> _CziLevelIndex:
        """Build the level's spatial index on first use; it never changes."""
        index = self._indexes.get(id(level))
        if index is None:
            number = self._level_numbers[id(level)]
            index = _CziLevelIndex(level.reader, self._placements[number])
            self._indexes[id(level)] = index
        return index

    @staticmethod
    def _spatial_axes(entries):
        ref_dims = entries[0].dims
        y_i = ref_dims.index("Y") if "Y" in ref_dims else len(ref_dims) - 2
        x_i = ref_dims.index("X") if "X" in ref_dims else len(ref_dims) - 1
        return y_i, x_i

    @staticmethod
    def _tile_plane(tile, e, y_i, x_i):
        """View a decoded tile as (Y, X) or (Y, X, samples).

        The decoded tile carries ``entry.stored_shape`` (e.g. (1, 32, 48, 1)
        for a 1-sample tile with a singleton C). When every axis other than
        Y, X and the trailing samples is a singleton and Y precedes X, the
        memory is already in that order and a reshape is enough; anything
        else goes through moveaxis, which is independent of axis order.
        """
        shape = e.stored_shape
        h, w, samples = shape[y_i], shape[x_i], shape[-1]
        if y_i < x_i and tile.size == h * w * samples:
            return tile.reshape((h, w, samples) if samples > 1 else (h, w))
        tile = np.moveaxis(tile, (y_i, x_i), (0, 1))
        if e.samples > 1:
            return tile.reshape(tile.shape[0], tile.shape[1], -1)[:, :, :e.samples]
        return tile.reshape(tile.shape[0], tile.shape[1])

    @classmethod
    def _paste_tile(cls, out, tile, e, y_i, x_i, bounds, y0, y1, x0, x1):
        """Place one decoded tile's intersection with the request into out.

        ``bounds`` is the tile's (y0, y1, x0, x1) in level pixels, from the
        level index.
        """
        tile_y, tile_y1, tile_x, tile_x1 = bounds
        iy0, iy1 = max(y0, tile_y), min(y1, tile_y1)
        ix0, ix1 = max(x0, tile_x), min(x1, tile_x1)
        if iy1 <= iy0 or ix1 <= ix0:
            return
        plane = cls._tile_plane(tile, e, y_i, x_i)
        out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = plane[
            iy0 - tile_y:iy1 - tile_y, ix0 - tile_x:ix1 - tile_x]

    # ----- Batching hooks shared with the PyramidReader planner -----

    def _select_tiles(self, level, y0, y1, x0, x1):
        """Intersecting sub-blocks in composition order, refusing mixed planes.

        The old scan silently overwrote one plane with another when tiles
        from different channel, time, depth or scene coordinates shared a
        place; that is not a region of anything, so it is an error here.
        """
        index = self._level_index(level)
        hits, examined = index.query(y0, y1, x0, x1)
        if hits:
            planes = {index.plane_key(i) for i in hits}
            if len(planes) > 1:
                ref_dims = level.reader[0].dims
                names = [ref_dims[a] for a in index.plane_axes] + ["scene"]
                raise CziError(
                    "read_region touches sub-blocks from "
                    f"{len(planes)} distinct planes along {names}; select "
                    "one plane, this reader will not composite them")
        return hits, examined, index

    def _region_tiles(self, level, y0, y1, x0, x1):
        return self._select_tiles(level, y0, y1, x0, x1)[0]

    def _region_tile_bytes(self, level, i):
        e = level.reader[i]
        return math.prod(e.stored_shape) * e.dtype.itemsize

    def _new_region_output(self, level, y0, y1, x0, x1):
        ref = level.reader[0]
        extra = (ref.samples,) if ref.samples > 1 else ()
        return np.zeros((y1 - y0, x1 - x0) + extra, dtype=level.dtype)

    def _paste_region_tile(self, level, i, tile, out, y0, y1, x0, x1):
        y_i, x_i = self._spatial_axes(level.reader)
        bounds = self._level_index(level).bounds[i]
        self._paste_tile(out, tile, level.reader[i], y_i, x_i, bounds,
                         y0, y1, x0, x1)

    def _decode_region_tiles(self, level, tile_ids, *, output_bytes=None):
        """Yield ``(i, tile)`` in the given order, from the cache when enabled.

        Ids arrive sorted (composition order), which is also the order claims
        are taken in, so two threads reading overlapping regions cannot wait
        on each other. Misses are decoded through the batched path in order
        and stored as they arrive; a failure releases every unstored claim
        so waiters decode for themselves.
        """
        cache = self.tile_cache
        if cache is None:
            yield from self._decode_uncached(level, tile_ids,
                                             output_bytes=output_bytes)
            return

        hits = list(tile_ids)
        source_id = id(self._czi)
        cached: dict = {}
        misses: list = []
        for i in hits:
            status, tile = cache.claim((source_id, i))
            if status == "hit":
                cached[i] = tile
            else:
                misses.append(i)
        self._last_cache = {"cache_hits": len(cached), "cache_misses": len(misses)}

        stored = 0
        try:
            miss_iter = (self._decode_uncached(level, misses, output_bytes=output_bytes)
                         if misses else iter(()))
            if not misses:
                self._last_decode = {"workers": 1, "batches": 0}
            for i in hits:
                tile = cached.get(i)
                if tile is None:
                    j, tile = next(miss_iter)
                    if j != i:  # pragma: no cover - ordering invariant
                        raise RuntimeError("cache miss decode order diverged")
                    tile = cache.store((source_id, i), tile)
                    stored += 1
                yield i, tile
        except BaseException:
            for i in misses[stored:]:
                cache.release((source_id, i))
            raise

    def _decode_uncached(self, level, tile_ids, *, output_bytes=None):
        """Yield ``(i, tile)`` in the given order, sharing decode when it pays.

        One task per tile does not pay: a 128 KB tile decodes in tens of
        microseconds, about what submitting it costs. Consecutive ids are
        grouped into tasks of roughly _REGION_BATCH_BYTES of decoded pixels
        and the worker count is resolved against the task count. Records
        the decision in ``self._last_decode``.
        """
        entries = level.reader
        hits = list(tile_ids)
        stored = [self._region_tile_bytes(level, i) for i in hits]

        batches: list[list[int]] = []
        current: list[int] = []
        current_bytes = 0
        for i, b in zip(hits, stored):
            if current and current_bytes + b > _REGION_BATCH_BYTES:
                batches.append(current)
                current, current_bytes = [], 0
            current.append(i)
            current_bytes += b
        if current:
            batches.append(current)

        if len(batches) < 2 or self._decode_workers == 1:
            workers = 1  # nothing to share, or pinned serial
        else:
            from .core.parallel import resolve_workers
            workers = resolve_workers(
                self._decode_workers, len(batches),
                output_bytes=sum(stored) if output_bytes is None else output_bytes,
                has_decode_work=any(entries[i].compression != 0 for i in hits),
                max_workers=8)
        self._last_decode = {"workers": workers, "batches": len(batches)}

        if workers == 1:
            for i in hits:
                yield i, self._czi._decode_one(entries[i])
            return

        from contextlib import closing

        from .core.pipeline import map_bounded

        def decode(batch):
            return [(i, self._czi._decode_one(entries[i])) for i in batch]

        size = dict(zip(hits, stored))
        # Reserve the decoded tiles plus the worker's shuffle scratch; the
        # output and the source cache are separate costs.
        with closing(map_bounded(
                decode, iter(batches), workers,
                max_bytes=((64 << 20) if self._max_pending_bytes is None
                           else self._max_pending_bytes),
                size=lambda batch: 2 * sum(size[i] for i in batch),
                executor=_get_pool(), name="czi-region")) as results:
            for tiles in results:
                yield from tiles

    def _window_layout(self, level):
        """``(y_i, x_i)`` when every tile of the level can decode into a window.

        That needs zstd tiles (ZSTD0 or ZSTDHDR) laid out Y, X, samples with
        every other axis a singleton. ``None`` sends the level through the
        owned-tile path. Cached per level.
        """
        key = id(level)
        if key in self._window_layouts:
            return self._window_layouts[key]
        entries = level.reader
        y_i, x_i = self._spatial_axes(entries)
        layout = (y_i, x_i)
        for e in entries:
            shape = e.stored_shape
            if (e.compression not in (5, 6) or not y_i < x_i
                    or math.prod(shape) != shape[y_i] * shape[x_i] * shape[-1]):
                layout = None
                break
        self._window_layouts[key] = layout
        return layout

    def _decode_windows(self, level, index, hits, out, y0, y1, x0, x1, layout):
        """Decode each hit straight into its window of ``out``.

        Serial requests go in composition order, each tile writing its whole
        intersection, so a later tile overwrites an earlier one. With enough
        work to share, each tile instead writes only the pieces that no later
        tile on its plane covers (see _CziLevelIndex.visible): those are
        disjoint, so workers write them in any order and the result is the
        same pixel for pixel, overlapping Zen mosaics included, and a tile
        hidden entirely is not decoded. Returns the decision for
        region_stats.
        """
        y_i, x_i = layout
        entries = level.reader
        bounds = index.bounds
        out2d = out.reshape(out.shape[0], -1).view(np.uint8)
        czi = self._czi

        def clipped(i, rects):
            ty0, _, tx0, _ = bounds[i]
            windows = []
            for ry0, ry1, rx0, rx1 in rects:
                iy0, iy1 = max(y0, ry0), min(y1, ry1)
                ix0, ix1 = max(x0, rx0), min(x1, rx1)
                if iy1 > iy0 and ix1 > ix0:
                    windows.append((iy0 - ty0, iy1 - ty0, ix0 - tx0, ix1 - tx0,
                                    iy0 - y0, ix0 - x0))
            return windows

        sizes = [self._region_tile_bytes(level, i) for i in hits]
        total = sum(sizes)
        n_batches = max(1, -(-total // _REGION_BATCH_BYTES))
        workers = 1
        if n_batches >= 2 and self._decode_workers != 1:
            from .core.parallel import resolve_workers
            workers = resolve_workers(self._decode_workers, n_batches,
                                      output_bytes=out.nbytes,
                                      has_decode_work=True, max_workers=8)
        if workers <= 1:
            for i in hits:
                windows = clipped(i, (bounds[i],))
                if windows:
                    czi._decode_windows(entries[i], out2d, windows, y_i, x_i)
            return {"workers": 1, "batches": 0, "tiles_hidden": 0}

        batches, current, current_bytes = [], [], 0
        for i, b in zip(hits, sizes):
            if current and current_bytes + b > _REGION_BATCH_BYTES:
                batches.append(current)
                current, current_bytes = [], 0
            current.append(i)
            current_bytes += b
        if current:
            batches.append(current)

        def run(batch):
            hidden = 0
            for i in batch:
                windows = clipped(i, index.visible(i))
                if windows:
                    czi._decode_windows(entries[i], out2d, windows, y_i, x_i)
                else:
                    hidden += 1
            return hidden

        from contextlib import closing

        from .core.pipeline import map_bounded
        # Nothing is retained between tasks (each worker decodes into its own
        # scratch and writes the output), so no byte reservation is needed.
        with closing(map_bounded(run, iter(batches), workers, size=lambda b: 0,
                                 executor=_get_pool(), name="czi-region")) as results:
            hidden = sum(results)
        return {"workers": workers, "batches": len(batches),
                "tiles_hidden": hidden}

    def _read_region(self, level, y0, y1, x0, x1):
        """Assemble a (y0:y1, x0:x1) region from a pyramid level.

        The level's spatial index yields only the sub-blocks whose stored
        bounds intersect the request. Zstd tiles decode straight into their
        window of the output (see _decode_windows); anything else, and any
        read with the tile cache on, decodes owned tiles serially or through
        the bounded pipeline and pastes them in composition order on this
        thread. Either way, where tiles overlap the later one wins exactly
        as before. Pixels no tile covers stay zero. A request touching more
        than one plane is refused (see _select_tiles).
        """
        t_start = time.perf_counter()
        entries = level.reader
        y_i, x_i = self._spatial_axes(entries)
        out = self._new_region_output(level, y0, y1, x0, x1)
        hits, examined, index = self._select_tiles(level, y0, y1, x0, x1)

        source = getattr(self._czi, "_mmap", None)
        counting = isinstance(source, _RangeBuffer)
        req0 = (source.requests, source.bytes_read) if counting else (0, 0)

        stats = {
            "level_entries": len(entries),
            "candidates_examined": examined,
            "tiles_intersecting": len(hits),
            "tiles_decoded": 0,
            "workers": 1,
            "batches": 0,
            "index_cells": 0 if index._cells is None else len(index._cells),
        }
        layout = (self._window_layout(level)
                  if hits and self.tile_cache is None else None)
        if layout is not None:
            stats.update(self._decode_windows(level, index, hits, out,
                                              y0, y1, x0, x1, layout))
            stats["tiles_decoded"] = len(hits)
        elif hits:
            self._last_cache = {}
            bounds = index.bounds
            for i, tile in self._decode_region_tiles(level, hits,
                                                     output_bytes=out.nbytes):
                self._paste_tile(out, tile, entries[i], y_i, x_i, bounds[i],
                                 y0, y1, x0, x1)
            stats.update(self._last_decode)
            stats.update(self._last_cache)
            stats["tiles_decoded"] = len(hits) - stats.get("cache_hits", 0)

        if counting:
            stats["source_requests"] = source.requests - req0[0]
            stats["source_bytes"] = source.bytes_read - req0[1]
        stats["elapsed_s"] = time.perf_counter() - t_start
        self.region_stats = stats
        return out


def _composition_order(entries):
    """Entries sorted so that a later one is drawn over an earlier one.

    Ascending mosaic index when any tile has one, stable so equal indices
    (and files without M) keep directory order.
    """
    if any(e.mosaic_index >= 0 for e in entries):
        return sorted(entries, key=lambda e: e.mosaic_index)
    return list(entries)


def _level_placement(entries, origin, scale):
    """Each tile's (y, x) position and stored (h, w) in one level's pixels.

    Positions are the full-resolution start relative to ``origin``, divided
    by the level's scale and floored, so a level's grid is the full slide
    shrunk by its scale. Returns four int64 arrays in the entries' order.
    """
    ref_dims = entries[0].dims
    y_i = ref_dims.index("Y") if "Y" in ref_dims else len(ref_dims) - 2
    x_i = ref_dims.index("X") if "X" in ref_dims else len(ref_dims) - 1
    n = len(entries)
    oy, ox = origin
    sy, sx = scale
    ys = np.fromiter((e.start[y_i] for e in entries), np.int64, n) - oy
    xs = np.fromiter((e.start[x_i] for e in entries), np.int64, n) - ox
    if sy != 1.0:
        ys = np.floor(ys / sy).astype(np.int64)
    if sx != 1.0:
        xs = np.floor(xs / sx).astype(np.int64)
    th = np.fromiter((e.stored_shape[y_i] for e in entries), np.int64, n)
    tw = np.fromiter((e.stored_shape[x_i] for e in entries), np.int64, n)
    return ys, xs, th, tw


__all__ = ["CziReader", "CziError", "CziPyramidReader", "imread"]
