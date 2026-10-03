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
import threading
from pathlib import Path
from typing import Iterable

import numpy as np

from .core._write_helpers import write_all
from .core.codec import Writer
from .core.scratch import ScratchBuffer


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


class CziCancelled(CziWriterError):
    """Raised when a caller's ``should_continue`` stops a rewrite.

    Distinct from :class:`CziWriterError` so a caller can tell "the user
    pressed stop" apart from "the file or the writer is broken" without
    matching on message text. The partial output is removed before this
    propagates: a truncated file that still parses as a CZI is worse than
    no file, because the next reader opens it without complaint.
    """


# ---------------------------------------------------------------------------
# Low-level segment builders
# ---------------------------------------------------------------------------


_ZERO_PAD = bytes(32)


class _SubBlockSegment:
    """One sub-block ready to emit, kept as parts until it is written.

    ``header`` is the 32-byte segment header plus the fixed sub-block
    prelude (sub-block header, directory entry, pad to 256); ``payload`` is
    the encoded or raw pixel bytes as a view; ``pad`` is the count of zero
    bytes that brings the segment to 32-byte alignment. ``keep`` retains
    whatever object backs ``payload`` when that is a borrowed buffer, so
    the pending segment stays valid while a verification still reads it.
    Positions are computed from ``size``, which is exact, so the file bytes
    and directory offsets are the same as when the segment was one string.
    """

    # A __slots__ class rather than a frozen dataclass: a sequential write
    # of 8 KB frames spends a measurable share of each frame on the
    # descriptor, and frozen-dataclass construction is several times the
    # cost of a plain instance.
    __slots__ = ("header", "payload", "pad", "keep")

    def __init__(self, header, payload, pad, keep=None):
        self.header = header
        self.payload = payload
        self.pad = pad
        self.keep = keep

    @property
    def size(self) -> int:
        return len(self.header) + len(self.payload) + self.pad

    #: Below this size the three part writes cost more than one copy: a
    #: sequential write of 8 KB frames measured about 5 us per frame slower
    #: as parts than as one string. Large payloads keep the copy-free path.
    _COALESCE_BELOW = 64 << 10

    def write_to(self, dest) -> None:
        """Write the segment through the short-write-safe helper.

        Small segments are written as one string; large ones as parts, so
        a big payload is never copied on its way to the destination.
        """
        if self.size <= self._COALESCE_BELOW:
            data = self.tobytes()
            written = dest.write(data)
            if written is not None and written != len(data):
                # A short write: hand the remainder to the checked loop.
                write_all(dest, memoryview(data)[written:])
            return
        write_all(dest, self.header)
        if len(self.payload):
            write_all(dest, self.payload)
        if self.pad:
            write_all(dest, _ZERO_PAD[:self.pad])

    def with_payload(self, payload) -> "_SubBlockSegment":
        return _SubBlockSegment(self.header, payload, self.pad, self.keep)

    #: Offset of the directory entry's file_position inside ``header``:
    #: 32-byte segment header, 16-byte sub-block header, then "DV" (2) and
    #: pixel_type (4) precede the int64 position.
    _POSITION_OFFSET = 32 + 16 + 6

    def with_position(self, file_position: int) -> "_SubBlockSegment":
        """Return the segment with its directory entry pointing at ``file_position``.

        Parallel compression cannot know a sub-block's final offset until it
        is emitted, so segments are built at position 0 and patched here.
        """
        header = bytearray(self.header)
        struct.pack_into("<q", header, self._POSITION_OFFSET, file_position)
        return _SubBlockSegment(bytes(header), self.payload, self.pad, self.keep)

    def tobytes(self) -> bytes:
        """The concatenated segment, for callers that want one string."""
        return self.header + bytes(self.payload) + bytes(self.pad)


def _segment_size(segment) -> int:
    return segment.size if isinstance(segment, _SubBlockSegment) else len(segment)


#: Pixel bytes per parallel-compression task. Small frames are grouped up to
#: this; a larger frame is its own task.
_WRITE_BATCH_BYTES = 2 << 20

# Per-thread shuffle scratch for parallel compression: workers cannot share
# the writer's single buffer, and a thread's buffer is never touched by
# another thread.
_TLS = threading.local()


def _worker_scratch() -> ScratchBuffer:
    scratch = getattr(_TLS, "czi_shuffle", None)
    if scratch is None:
        scratch = ScratchBuffer()
        _TLS.czi_shuffle = scratch
    return scratch


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


def _byteshuffle_encode(natural, itemsize: int, *, out=None):
    """ZEN's per-byte-plane shuffle: rearrange so all high bytes come
    first, then mid, then low. Compresses better when high bytes are
    correlated (typical for 12/14/16-bit microscopy).

    Runs the native tight-loop primitive rather than a NumPy transpose,
    which had to materialize a second full-size buffer per frame and fill
    it with a strided copy. ``out`` lets one allocation serve every frame;
    the result is a view of it, so the caller must finish with the bytes
    before the next shuffle. One-byte elements need no shuffle at all and
    pass through without copying.
    """
    if itemsize == 1:
        return natural
    from .codecs._bytetools import byteshuffle_encode as _native
    n = len(natural) // itemsize
    return _native(natural, itemsize, n, out=out)


def _zstdhdr_encode(pixel_bytes, itemsize: int, hilo: bool, *, scratch=None):
    """Encode as CZI ZSTDHDR (compression=6).

    Layout: 1-byte header_size, chunk_type=1 + hilo flag byte, then a
    zstd stream over the (optionally byte-shuffled) pixels.

    ``scratch`` holds the shuffled intermediate. Compression consumes it
    before this returns, and verification later reads the encoded payload
    and the caller's own pixel snapshot rather than this buffer, so the
    next frame may reuse it.
    """
    from .codecs._zstd import encode as zstd_encode
    if hilo and itemsize > 1:
        buffer = None if scratch is None else scratch.bytes(len(pixel_bytes))
        shuffled = _byteshuffle_encode(pixel_bytes, itemsize, out=buffer)
        payload = zstd_encode(shuffled, level=3)
    else:
        payload = zstd_encode(pixel_bytes, level=3)
    header = struct.pack("<BBB", 3, 1, 1 if hilo else 0)
    return header + payload


#: Dimensions CZI addresses with a single index rather than an extent. The
#: reader splits these out of its ``dims`` tuple into ``mosaic_index`` and
#: ``scene_index``, so a round-trip has to put them back.
_INDEX_DIMS = (b"M", b"S")


def _normalize_dims(dims):
    """Validate a caller's dimension list into the 5-tuples the format wants.

    Accepts ``(name, start, size, stored)`` or the full
    ``(name, start, size, coordinate, stored)``, with ``name`` as ``str`` or
    ``bytes``. A dimension name is one to four characters in the file, so it
    is padded rather than truncated silently: a wrong name here produces a
    file that reads back with dimensions nobody asked for.
    """
    out = []
    for d in dims:
        if len(d) == 4:
            name, start, size, stored = d
            coord = 0.0
        elif len(d) == 5:
            name, start, size, coord, stored = d
        else:
            raise CziWriterError(
                f"czi: a dimension is (name, start, size[, coordinate], "
                f"stored_size), got {len(d)} values: {d!r}")
        if isinstance(name, str):
            name = name.encode("ascii")
        if not isinstance(name, (bytes, bytearray)) or not 1 <= len(name) <= 4:
            raise CziWriterError(
                f"czi: dimension name must be 1-4 characters, got {name!r}")
        # ``start`` may be negative: a mosaic tile's X/Y start is a stage
        # coordinate, and a real slide scan puts tiles left of and above the
        # origin. Only the extents have to be non-negative.
        for label, value in (("size", size), ("stored_size", stored)):
            if int(value) < 0:
                raise CziWriterError(
                    f"czi: dimension {name!r} has negative {label} {value}")
        out.append((bytes(name).ljust(4, b"\x00"), int(start), int(size),
                    float(coord), int(stored)))
    if not out:
        raise CziWriterError("czi: a sub-block needs at least one dimension")
    return tuple(out)


def subblock_dims(entry):
    """The dimension list that rewrites ``entry`` faithfully.

    Three things have to be undone to get back to what the file stores.
    ``CziSubBlockEntry.dims`` is in REVERSE file order, because the reader
    presents axes slowest-first like an array shape. Its last element is a
    synthetic ``"S"`` standing for samples-per-pixel, which is not a file
    dimension at all: emitting it produces a file whose scene index reads
    back as 0 instead of absent. And the two index-only dimensions, M and S,
    are pulled out of ``dims`` into ``mosaic_index`` and ``scene_index``, so
    they have to be put back.

    M and S are appended rather than restored to their original positions,
    which the entry does not record. libCZI does not care about the order,
    and the reader drops them from ``dims`` again on the way back in, so a
    round-trip still compares equal.
    """
    n = len(entry.dims) - 1          # drop the trailing samples axis
    dims = [
        (entry.dims[i], entry.start[i], entry.shape[i], 0.0,
         entry.stored_shape[i])
        for i in reversed(range(n))  # reader order is reversed file order
    ]
    if entry.mosaic_index >= 0:
        dims.append((b"M", entry.mosaic_index, 1, 0.0, 1))
    if entry.scene_index >= 0:
        dims.append((b"S", entry.scene_index, 1, 0.0, 1))
    return _normalize_dims(dims)


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
    dims: tuple | None = None,
    verification_input: list | None = None,
    scratch=None,
) -> tuple[_SubBlockSegment, dict]:
    """Build one ZISRAWSUBBLOCK segment as parts.

    Returns ``(segment, directory_entry_dict)``; the segment is a
    :class:`_SubBlockSegment` whose byte layout matches the former single
    string exactly, and the entry dict is consumed later by
    :func:`_build_directory_segment`.

    With ``verification_input`` the pixels are snapshotted, because the
    check runs after this returns while the producer may already be
    refilling ``array``. Without it nothing here outlives the call, so the
    array's own buffer feeds the compressor or becomes the raw payload.
    """
    h, w = array.shape[:2]
    logical_h, logical_w = logical_shape
    start_y, start_x = location

    if dims is None:
        # The plain case: one plane, addressed only in X and Y.
        dims = (
            (b"X", start_x, logical_w, 0.0, w),
            (b"Y", start_y, logical_h, 0.0, h),
        )
    else:
        dims = _normalize_dims(dims)
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

    contiguous = np.ascontiguousarray(array)
    if verification_input is not None:
        # Owned snapshot: the deferred check views it after this call.
        pixel_bytes = contiguous.tobytes()
        verification_input.append(
            np.frombuffer(pixel_bytes, dtype=array.dtype).reshape(array.shape))
        pixels = memoryview(pixel_bytes)
        keep = pixel_bytes
    else:
        # Borrowed: consumed by the compressor, or written, before return.
        pixels = memoryview(contiguous).cast("B")
        keep = contiguous
    itemsize = array.dtype.itemsize
    if compression_code == 0:
        data = pixels
    elif compression_code == 5:
        from .codecs._zstd import encode as zstd_encode
        data = zstd_encode(pixels, level=3)
        keep = None
    elif compression_code == 6:
        data = _zstdhdr_encode(pixels, itemsize, hilo, scratch=scratch)
        keep = None
    else:
        raise CziWriterError(
            f"unsupported compression code {compression_code} "
            f"(expected 0, 5, or 6)"
        )

    # Same bytes _pad_segment produced, without concatenating the payload:
    # 32-byte segment header (sid, alloc, used), then the fixed prelude, then
    # the payload, then zero padding to a 32-byte boundary.
    sub_header = struct.pack("<iiq", 0, 0, len(data))
    sid = _SUB_MAGIC + b"\x00\x00"
    prelude = sub_header + de_header + de_dims + b"\x00" * pad
    used = len(prelude) + len(data)
    header = struct.pack("<16sqq", sid, used, used) + prelude
    seg = _SubBlockSegment(header=header, payload=data,
                           pad=(-(32 + used)) % 32, keep=keep)

    de_dict = {
        "file_position": file_position, "pixel_type": pixel_type,
        "compression": compression_code, "pyramid_type": pyramid_type,
        "stored_w": w, "stored_h": h,
        "logical_w": logical_w, "logical_h": logical_h,
        "start_x": start_x, "start_y": start_y,
        # The directory must describe the same dimensions as the sub-block
        # itself, so it is written from the same list rather than rebuilt.
        "dims": dims,
    }
    return seg, de_dict


def _build_directory_segment(entries: list[dict]) -> bytes:
    sid = _DIR_MAGIC + b"\x00"
    body = struct.pack("<I", len(entries)) + b"\x00" * 124
    for e in entries:
        dims = e.get("dims")
        if dims is None:
            dims = (
                (b"X", e.get("start_x", 0), e["logical_w"], 0.0, e["stored_w"]),
                (b"Y", e.get("start_y", 0), e["logical_h"], 0.0, e["stored_h"]),
            )
        body += struct.pack(
            "<2siqiiBB4si",
            b"DV", e["pixel_type"], e["file_position"], 0,
            e["compression"], e.get("pyramid_type", 0),
            0, b"\x00\x00\x00\x00", len(dims),
        )
        for name, start, size, coord, stored in dims:
            body += struct.pack("<4siifi", name, start, size, coord, stored)
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


#: Smallest frame, in bytes, whose compression write() moves to a
#: background thread. Below this the thread handoff costs about as much as
#: compressing (an 8 KB frame takes tens of microseconds either way).
_PIPELINE_MIN_BYTES = 1 << 20


class _CziStreamWriter(Writer):
    """Shared sub-block sink with optional bounded pixel verification."""

    def __init__(self, path: str | Path, *, compression: str = "none",
                 hilo: bool = True, metadata_xml: bytes | str = b"<Metadata/>",
                 verify: bool = False, background_encode: bool = False):
        if compression not in _CMP_NAME_TO_CODE:
            raise CziWriterError(f"unknown compression {compression!r}")
        self._path = Path(path)
        self._cmp_code = _CMP_NAME_TO_CODE[compression]
        self._hilo = bool(hilo)
        xml = metadata_xml.encode("utf-8") if isinstance(metadata_xml, str) else metadata_xml
        self._metadata_segment = _build_metadata_segment(xml)
        self._header_size = (32 + 88 + 31) // 32 * 32
        self._position = self._header_size + len(self._metadata_segment)
        self._entries: list[dict] = []
        self._file = None
        self._closed = False
        self._pending_verified = None
        # With background_encode, unverified compressed frames of at least
        # _PIPELINE_MIN_BYTES compress on a thread of this writer's own
        # while the caller prepares the next one; see _append.
        self._background_encode = bool(background_encode)
        self._pending_encode = None
        self._encode_pool = None
        # One shuffle buffer per writer. Each frame's shuffled bytes are
        # consumed by the compressor before _append returns, and a pending
        # verification reads the encoded payload plus its own pixel snapshot,
        # so reusing this across frames cannot race either of them. Separate
        # writers hold separate buffers, so concurrent writers stay isolated.
        self._shuffle_scratch = ScratchBuffer()
        self._verification = None
        if verify:
            from .core.verification import DeferredVerification
            self._verification = DeferredVerification()

    def _validate(self, array):
        if self._closed:
            raise CziWriterError(f"{type(self).__name__} is closed")
        if array.ndim != 2:
            raise CziWriterError(f"{type(self).__name__} expects 2D ndarray; got shape={array.shape}")
        if array.dtype not in _DTYPE_TO_PIXELTYPE:
            raise CziWriterError(f"unsupported dtype {array.dtype}")

    @staticmethod
    def _verify_subblock(original, segment, entry):
        from ._czi_reader import CziReader
        from .core.verification import verify_with_decoder
        if isinstance(segment, _SubBlockSegment):
            payload = segment.payload
        else:
            # A plain byte string: locate the payload the way the reader does.
            # This writer's two-dimensional directory entry fits the 256-byte
            # sub-block metadata area.
            metadata_size, _, data_size = struct.unpack_from("<iiq", segment, 32)
            offset = 32 + 256 + metadata_size
            payload = memoryview(segment)[offset:offset + data_size]
        verify_with_decoder(
            original, payload,
            lambda data: CziReader._decode_payload(
                data, entry["pixel_type"], entry["compression"], original.shape),
            codec="czi")

    def _finish_verified(self):
        if self._pending_verified is not None:
            self._verification.finish()
            segment, entry = self._pending_verified
            self._pending_verified = None
            self._emit_subblock(segment, entry)

    def _finish_encode(self):
        """Emit the frame compressing in the background, if there is one.

        Its position is assigned here, in emission order, and patched into
        the sub-block header, exactly as write_many does, so the file is
        byte-identical to compressing on the calling thread. A compression
        error surfaces here, one call after the frame that caused it.
        """
        pending = self._pending_encode
        if pending is None:
            return
        self._pending_encode = None
        segment, entry = pending.result()
        position = self._position
        entry["file_position"] = position
        self._emit_subblock(segment.with_position(position), entry)

    def _encode_in_background(self, array, logical_shape, pyramid_type, *, dims=None):
        """Snapshot the frame and compress it on this writer's own thread."""
        if self._encode_pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._encode_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="opencodecs-czi-encode")
        # The caller may refill its array as soon as write() returns, so the
        # frame is copied, into one of two buffers reused in turn: at most
        # one frame is in flight, and fresh 8 MB allocations cost page
        # faults on every frame.
        slots = self.__dict__.setdefault("_snapshot_slots", [None, None])
        turn = self.__dict__.get("_snapshot_turn", 0)
        self._snapshot_turn = 1 - turn
        pixels = slots[turn]
        if pixels is None or pixels.shape != array.shape or pixels.dtype != array.dtype:
            pixels = slots[turn] = np.empty(array.shape, array.dtype)
        np.copyto(pixels, array)

        def encode():
            return _build_subblock(
                pixels, pixel_type=_DTYPE_TO_PIXELTYPE[pixels.dtype],
                compression_code=self._cmp_code, hilo=self._hilo,
                file_position=0, logical_shape=logical_shape,
                pyramid_type=pyramid_type, dims=dims, verification_input=None,
                scratch=_worker_scratch())

        self._pending_encode = self._encode_pool.submit(encode)

    def _append(self, array, *, logical_shape, pyramid_type=0, dims=None):
        # With background_encode, large compressed frames without
        # verification compress in the background while the caller moves on;
        # at most one is in flight, and it is written out before anything
        # that follows it. It is opt-in because it only pays when the caller
        # has work to overlap: a caller spending about a frame's compression
        # time producing each frame wrote 12 frames of 2048 x 2048 1.44 to
        # 1.46x faster, but a caller with frames ready and nothing else to
        # do was 0.96x (macOS) to 0.87x (Linux), compressing on another
        # core than the one that just wrote the frame. Small frames would
        # spend more on the handoff than they save, and verified writes
        # already overlap verification with the next compression, so both
        # stay on this thread regardless.
        if (self._background_encode and self._verification is None
                and self._cmp_code != 0 and array.nbytes >= _PIPELINE_MIN_BYTES):
            try:
                self._finish_encode()
                self._encode_in_background(array, logical_shape, pyramid_type,
                                           dims=dims)
            except BaseException:
                self._abort()
                raise
            return
        try:
            self._finish_encode()
        except BaseException:
            self._abort()
            raise
        captured = [] if self._verification is not None else None
        offset = self._position
        if self._pending_verified is not None:
            offset += _segment_size(self._pending_verified[0])
        try:
            segment, entry = _build_subblock(
                array, pixel_type=_DTYPE_TO_PIXELTYPE[array.dtype],
                compression_code=self._cmp_code, hilo=self._hilo,
                file_position=offset, logical_shape=logical_shape,
                pyramid_type=pyramid_type, dims=dims, verification_input=captured,
                scratch=self._shuffle_scratch,
            )
            if self._verification is None:
                self._emit_subblock(segment, entry)
            else:
                # Compression of this segment overlaps checking the prior one.
                # Only verified segments may reach the destination or index.
                self._finish_verified()
                self._verification.submit(self._verify_subblock, captured[0], segment, entry)
                self._pending_verified = segment, entry
        except BaseException:
            self._abort()
            raise

    def _plan_frame(self, array):
        """Return ``(logical_shape, pyramid_type)`` for the next frame.

        Runs on the producer thread in submission order, so a subclass that
        derives one frame's plan from an earlier frame (pyramid levels) can
        keep that state here.
        """
        return array.shape, 0

    def write_many(self, frames, *, workers=None, max_pending_bytes=None,
                   copy_frames=True, worker_budget=None, batch_bytes=None):
        """Compress frames on workers and emit them in submission order.

        Each frame is validated and snapshotted on this thread before the
        source iterator advances, so producers that refill one buffer are
        safe with the default ``copy_frames=True``; ``copy_frames=False``
        skips the snapshot and requires every frame to stay unchanged until
        it has been emitted. Encoding runs on up to ``workers`` threads
        (default: decided from the CPU count, at most 8) under
        ``max_pending_bytes`` of retained input, encoded output and, with
        ``verify=True``, verification scratch. Sub-blocks are emitted in
        order and their file positions assigned then, so the file is
        byte-identical to sequential ``write`` calls. With ``verify=True``
        each task checks its own payload inside the same worker budget; no
        separate verification pool is created. The compressor itself is
        single-threaded, so outer workers are not multiplied by codec
        threads. Frames are grouped into tasks of about ``batch_bytes``
        (default 2 MiB) of pixels so that small frames do not spend more
        time being scheduled than encoded; a frame larger than that is its
        own task. A task that fails, including a failed verification, is
        not emitted at all, so a failure mid-task also withholds the good
        frames grouped before it; the writer is aborted either way.
        """
        from contextlib import closing

        from .core.parallel import resolve_workers
        from .core.pipeline import map_bounded
        from .core.verification import verification_reservation

        if self._closed:
            raise CziWriterError(f"{type(self).__name__} is closed")
        verify = self._verification is not None
        batch_limit = _WRITE_BATCH_BYTES if batch_bytes is None else max(1, int(batch_bytes))
        if workers is None:
            workers = resolve_workers(None, 1 << 30, output_bytes=None, max_workers=8)
        workers = max(1, int(workers))

        def snapshot(frame):
            """Own the frame's pixels before the producer can touch them.

            A frame is an array, ``(array, dims)`` to place the sub-block in
            more than X and Y, or ``(array, dims, pyramid_type)`` to also say
            it is a down-scaled level. Those forms are what make a faithful
            re-encode possible on this path as well as on ``write_frame``.
            """
            dims = override_pyramid = None
            if isinstance(frame, tuple):
                if len(frame) == 2:
                    frame, dims = frame
                elif len(frame) == 3:
                    frame, dims, override_pyramid = frame
                else:
                    raise CziWriterError(
                        "czi: a frame is an array, (array, dims) or "
                        f"(array, dims, pyramid_type), got a {len(frame)}-tuple")
                if dims is not None:
                    dims = _normalize_dims(dims)
            self._validate(frame)
            pixels = (np.array(frame, order="C", copy=True) if copy_frames
                      else np.ascontiguousarray(frame))
            logical_shape, pyramid_type = self._plan_frame(pixels)
            if override_pyramid is not None:
                pyramid_type = int(override_pyramid)
            if dims is not None:
                sizes = {d[0].rstrip(b"\x00"): d[2] for d in dims}
                if b"Y" in sizes and b"X" in sizes:
                    logical_shape = (sizes[b"Y"], sizes[b"X"])
            return pixels, logical_shape, pyramid_type, dims

        def batches():
            # One task per frame does not pay for small frames: an 8 KB frame
            # encodes in about the time it takes to schedule it. Snapshot
            # each frame as it is pulled, then hand workers groups worth
            # about _WRITE_BATCH_BYTES. A frame larger than that is its own
            # task, so large frames are scheduled exactly as before.
            batch, batch_bytes = [], 0
            for frame in frames:
                item = snapshot(frame)
                n = item[0].nbytes
                if batch and batch_bytes + n > batch_limit:
                    yield batch
                    batch, batch_bytes = [], 0
                batch.append(item)
                batch_bytes += n
            if batch:
                yield batch

        def size(batch):
            n = sum(item[0].nbytes for item in batch)
            # Snapshot + encoded output (bounded by the input for these
            # codecs) + decoded verification result and comparison scratch.
            return 2 * n + (verification_reservation(n) if verify else 0)

        def encode(batch):
            out = []
            scratch = _worker_scratch()
            for pixels, logical_shape, pyramid_type, dims in batch:
                captured = [] if verify else None
                segment, entry = _build_subblock(
                    pixels, pixel_type=_DTYPE_TO_PIXELTYPE[pixels.dtype],
                    compression_code=self._cmp_code, hilo=self._hilo,
                    file_position=0, logical_shape=logical_shape,
                    pyramid_type=pyramid_type, dims=dims,
                    verification_input=captured, scratch=scratch,
                )
                if verify:
                    self._verify_subblock(captured[0], segment, entry)
                out.append((segment, entry))
            return out

        try:
            # Nothing from write() may still be pending: positions are
            # assigned in emission order and this path emits directly.
            self._finish_encode()
            self._finish_verified()
            # A persistent pool: creating one per call costs milliseconds
            # on a many-core Linux host, more than a batch of small frames
            # takes to compress.
            from .core.io import get_reader_pool

            with closing(map_bounded(
                    encode, batches(), workers, size=size,
                    max_bytes=(64 << 20) if max_pending_bytes is None
                    else max_pending_bytes,
                    budget=worker_budget, name="czi-write",
                    executor=get_reader_pool("czi-write", 16))) as results:
                for encoded in results:
                    for segment, entry in encoded:
                        position = self._position
                        entry["file_position"] = position
                        self._emit_subblock(segment.with_position(position), entry)
        except BaseException:
            self._abort()
            raise

    def _emit_subblock(self, segment, entry):
        if self._file is None:
            self._file = self._path.open("w+b")
            self._file.write(b"\0" * self._header_size)
            self._file.write(self._metadata_segment)
        if isinstance(segment, _SubBlockSegment):
            segment.write_to(self._file)
        else:
            write_all(self._file, segment)
        self._file.flush()
        self._entries.append(entry)
        self._position += _segment_size(segment)

    def close(self):
        if self._closed:
            return
        try:
            self._finish_encode()
            self._finish_verified()
        except BaseException:
            self._abort()
            raise
        self._closed = True
        self._shutdown_encoder()
        if self._verification is not None:
            try:
                self._verification.close()
            except BaseException:
                self._abort()
                raise
        if not self._entries:
            raise CziWriterError(f"{type(self).__name__}.close(): no frames or levels written")
        try:
            directory = _build_directory_segment(self._entries)
            self._file.write(directory)
            header = _build_file_header(
                directory_position=self._position,
                metadata_position=self._header_size,
                file_size=self._position + len(directory),
            )
            self._file.seek(0)
            self._file.write(header)
        finally:
            self._file.close()
            self._file = None
            self._entries.clear()

    def _shutdown_encoder(self):
        pool, self._encode_pool = self._encode_pool, None
        if pool is not None:
            pool.shutdown(wait=True)

    def _abort(self):
        self._closed = True
        # Join a background compression before releasing anything it uses;
        # its result, or its error, is discarded along with the file.
        pending, self._pending_encode = self._pending_encode, None
        if pending is not None:
            try:
                pending.result()
            except BaseException:
                pass
        self._shutdown_encoder()
        if self._verification is not None:
            # Preserve the original encode/sink failure while still joining
            # the outstanding verification task before releasing its buffers.
            try:
                self._verification.close()
            except BaseException:
                pass
        self._pending_verified = None
        if self._file is not None:
            self._file.close()
            self._file = None
        self._entries.clear()

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self._abort()
        return False


class CziWriter(_CziStreamWriter):
    """Write each frame as a sub-block immediately; finalize the index on close.

    By default one encoded frame and the directory entries need memory. The caller
    may reuse its array after write() or write_frame() returns. With
    background_encode=True, compressed frames of 1 MiB or more are copied and
    compressed on a background thread while the caller prepares the next one:
    two frame copies are held, an error compressing a frame is raised by the
    following write() or close() (which abort the writer, as any write error
    does), and the file is byte-identical. That is faster only when the caller
    does work between frames (acquisition, computation); with frames ready and
    nothing to overlap it is slower, so it is off by default. With
    verify=True, one owned pending segment is decoded and compared while the
    next is encoded; close joins verification before writing the final
    segment and directory.
    """

    def write(self, array: np.ndarray) -> None:
        self._validate(array)
        self._append(array, logical_shape=array.shape)

    def write_frame(self, arr, *, dims=None, pyramid_type=0, **opts):
        """Write one sub-block, optionally saying where it sits.

        ``dims`` gives this sub-block's own dimensions as
        ``(name, start, size[, coordinate], stored_size)``, which is what
        lets a file with channels, z-planes, scenes, mosaic tiles or pyramid
        levels be written rather than a bare stack of planes. Without it the
        sub-block is addressed in X and Y only, exactly as ``write`` does.

        ``subblock_dims(entry)`` builds the list from a ``CziSubBlockEntry``,
        so re-encoding a file is a read, a transform and a write that keeps
        every coordinate it came with.

        ``pyramid_type`` is the CZI ``PyramidType`` byte: 0 for a
        full-resolution sub-block, 2 for a down-scaled pyramid level.
        """
        if opts:
            raise TypeError(f"czi: write_frame takes no options, got {sorted(opts)}")
        if dims is None and not pyramid_type:
            self.write(arr)
            return
        self._validate(arr)
        logical_shape = arr.shape
        if dims is not None:
            dims = _normalize_dims(dims)
            # The logical extent lives in the dimension list now; keep the
            # X/Y scalars consistent with it so the directory and the
            # sub-block cannot disagree.
            sizes = {d[0].rstrip(b"\x00"): (d[2], d[4]) for d in dims}
            if b"Y" in sizes and b"X" in sizes:
                logical_shape = (sizes[b"Y"][0], sizes[b"X"][0])
        self._append(arr, logical_shape=logical_shape,
                     pyramid_type=pyramid_type, dims=dims)


class CziPyramidWriter(_CziStreamWriter):
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

    def __init__(self, path: str | Path, **opts):
        super().__init__(path, **opts)
        self._base_shape = None
        self._base_dtype = None

    def _plan_frame(self, array):
        if self._base_dtype is not None and array.dtype != self._base_dtype:
            raise CziWriterError(
                f"all pyramid levels must share dtype; level 0 is "
                f"{self._base_dtype}, another level is {array.dtype}")
        logical_shape = self._base_shape if self._base_shape is not None else array.shape
        pyramid_type = 0 if self._base_shape is None else 2
        self._base_shape = logical_shape
        self._base_dtype = array.dtype
        return logical_shape, pyramid_type

    def write_level(self, array: np.ndarray) -> None:
        self._validate(array)
        logical_shape, pyramid_type = self._plan_frame(array)
        self._append(array, logical_shape=logical_shape, pyramid_type=pyramid_type)

    def write_pyramid(self, levels: Iterable[np.ndarray]) -> None:
        for level in levels:
            self.write_level(level)

    def write_frame(self, arr, **opts):
        """Each frame is one resolution level, starting at full resolution."""
        if opts:
            raise TypeError(f"czi: write_frame takes no options, got {sorted(opts)}")
        self.write_level(arr)


__all__ = ["CziWriter", "CziPyramidWriter", "CziWriterError"]


def czi_recompress(src, dst, *, compression: str = "zstdhdr", hilo: bool = True,
               workers=None, verify: bool = False,
               background_encode: bool = False,
               should_continue=None) -> dict:
    """Rewrite a CZI with different compression, keeping the container.

    Every sub-block keeps its dimension coordinates, its scene and mosaic
    position, its pyramid level and its pixels; only the encoding of the
    payload changes. The file's metadata XML is carried across unchanged.
    This is the operation ZEISS's ``czicompress`` performs, and the point of
    doing it here is that the pixels are already decoded in this process, so
    a caller can hash, verify or inspect them on the same pass instead of
    re-reading the file two or three more times.

    ``verify=True`` decodes each sub-block back from its encoded payload and
    compares it with the input before the sub-block is allowed into the
    file, which makes a lossless claim something the writer checked rather
    than something the format promises.

    ``should_continue`` is an optional zero-argument callable polled once per
    sub-block before it is read. When it returns false the rewrite raises
    :class:`CziCancelled` and ``dst`` is removed. It is a plain callable
    rather than a ``threading.Event`` so that a caller can just as well pass
    a deadline, a signal flag or a cancellation token; nothing here needs to
    know which. Encodes already dispatched still finish, so cancellation
    takes effect within roughly ``workers`` sub-blocks rather than instantly.

    Returns a summary dict: sub-block count, input and output sizes, and the
    compression that was applied.
    """
    import os

    from ._czi_reader import CziReader

    with CziReader(str(src)) as reader:
        n = len(reader.entries)
        if not n:
            raise CziWriterError(f"czi: {src} has no sub-blocks to recompress")
        metadata = reader.metadata_bytes or b"<Metadata/>"

        def frames():
            # One sub-block at a time: a slide scan does not fit in memory,
            # and write_many only needs the next frame, not all of them.
            for i in range(n):
                if should_continue is not None and not should_continue():
                    raise CziCancelled(
                        f"czi: recompress of {src} cancelled after "
                        f"{i} of {n} sub-blocks")
                entry = reader.entries[i]
                yield reader[i], subblock_dims(entry), entry.pyramid_type

        try:
            with CziWriter(str(dst), compression=compression, hilo=hilo,
                           metadata_xml=metadata, verify=verify,
                           background_encode=background_encode) as writer:
                writer.write_many(frames(), workers=workers, copy_frames=False)
        except BaseException:
            # An interrupted rewrite must not leave a file behind. A CZI
            # missing sub-blocks still has a valid header and directory, so
            # readers accept it and the loss is silent. Covers cancellation,
            # a failed verification and KeyboardInterrupt alike.
            try:
                os.remove(dst)
            except OSError:
                pass
            raise

    return {
        "subblocks": n,
        "src_bytes": os.path.getsize(src),
        "dst_bytes": os.path.getsize(dst),
        "compression": compression,
    }
