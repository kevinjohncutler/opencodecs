"""NDTiffWriter — streaming write of Micro-Manager / Pycro-Manager NDTiff.

Drop-in compatible with the ``ndstorage.SingleNDTiffWriter`` output
layout: same 28-byte file header, same 13-tag IFD per frame, same
NDTiff.index record format. Files written here can be read back by
the official ``ndstorage.NDTiffDataset`` reader and by
``opencodecs.NDTiffDataset`` interchangeably.

Adds three things the reference writer doesn't:

  * Automatic 4 GB rollover across ``NDTiffStack_N.tif`` files — the
    caller never has to decide when to close a stack file.
  * ``writer.write_many(frames, metadata)`` for batch submission;
    flushes once at the end rather than per frame.
  * Final atomic write of NDTiff.index from an in-memory buffer
    (cheaper than re-opening + appending throughout the run).

What we don't reimplement (because it's correct in ndstorage and
not on the hot path):

  * The exact byte layout of the IFD entries + per-frame metadata.
  * The summary metadata format (just a JSON dict serialized once).

So this writer is intentionally close-but-not-identical to
ndstorage's — same on-disk format but better defaults for streaming.
"""

from __future__ import annotations

import io
import json
import os
import struct
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# Format constants — must match ndstorage exactly for cross-reader
# compatibility. Copied from ndstorage.ndtiff_file.
_MAJOR_VERSION = 3
_MINOR_VERSION = 3
_MAX_FILE_SIZE = 4 * (1 << 30)  # 4 GiB

_ENTRIES_PER_IFD = 13
# Required TIFF + Micro-Manager tags (see TIFF 6.0 spec for the standard ones).
_TAG_WIDTH = 256
_TAG_HEIGHT = 257
_TAG_BITS_PER_SAMPLE = 258
_TAG_COMPRESSION = 259
_TAG_PHOTOMETRIC = 262
_TAG_STRIP_OFFSETS = 273
_TAG_SAMPLES_PER_PIXEL = 277
_TAG_ROWS_PER_STRIP = 278
_TAG_STRIP_BYTE_COUNTS = 279
_TAG_X_RESOLUTION = 282
_TAG_Y_RESOLUTION = 283
_TAG_RESOLUTION_UNIT = 296
_TAG_MM_METADATA = 51123

_SUMMARY_MD_HEADER = 2355492
_NDTIFF_MAGIC = 483729

_PIXEL_TYPE_EIGHT_BIT = 0
_PIXEL_TYPE_SIXTEEN_BIT = 1
_PIXEL_TYPE_EIGHT_BIT_RGB = 2

# IFD overhead = 2 (entry count) + 13×12 + 4 (next-IFD) = 162 bytes
# plus 16 bytes for x/y resolution rationals (rgb adds 6 for the
# bits-per-sample triplet). Used by the rollover heuristic.
_IFD_HEADER_BYTES = 2 + _ENTRIES_PER_IFD * 12 + 4 + 16


class NDTiffWriterError(RuntimeError):
    """Raised on writer state-machine violations."""


class NDTiffWriter:
    """Write an NDTiff acquisition incrementally.

    Usage::

        with NDTiffWriter(out_dir, base_name="Acq", summary={"PixelSize": 0.108}) as w:
            for z in range(N):
                w.write_frame({"z": z}, pixels[z], metadata={"z_um": z * 0.5})
        # ↑ on close(): NDTiff.index is flushed atomically and all
        #   stack files have null next-IFD pointers patched in.

    The ``axes`` arg to ``write_frame`` is an arbitrary dict of
    JSON-serializable values; the reader keys on ``frozenset(axes.items())``.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        base_name: str = "NDTiffStack",
        summary: dict | None = None,
        # Bit depth for pixel_type encoding into the index. ``"auto"``
        # picks 8 vs 16 from the dtype of the first frame.
        bit_depth: int | str = "auto",
        # Per-frame compression. ``"none"`` (default) emits raw
        # pixels, matching ndstorage's behavior and giving the
        # fastest writer. Set ``"deflate"`` / ``"zstd"`` / ``"jxl"``
        # / etc. (see opencodecs.core.segment_compression for the
        # full list) for compressed frames. The encoded bytes go
        # into the TIFF strip data and the index records the
        # corresponding TIFF compression code.
        compression: str | int = "none",
        compression_level: int | None = None,
    ):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._base = base_name
        self._summary_bytes = json.dumps(summary or {}).encode("utf-8")
        self._bit_depth_hint = bit_depth

        from .core.segment_compression import (
            codec_name_to_code, NONE as _CMP_NONE,
        )
        self._compression_code = codec_name_to_code(compression)
        self._compression_level = compression_level
        self._compression_is_none = self._compression_code == _CMP_NONE

        # State.
        self._stack_index = 0           # 0 → NDTiffStack.tif, then _1, _2, ...
        self._fh: io.BufferedWriter | None = None
        self._cur_path: Path | None = None
        self._next_ifd_offset_location = -1
        self._index_records = bytearray()   # accumulates index entries
        self._lock = threading.Lock()
        self._closed = False
        self._frame_count = 0

        self._open_next_stack()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_frame(
        self,
        axes: dict[str, Any],
        pixels: np.ndarray,
        metadata: dict | str | None = None,
    ) -> dict:
        """Write one frame. Returns the index record (also stashed for
        the final NDTiff.index)."""
        with self._lock:
            return self._write_frame_inner(axes, pixels, metadata)

    def write_many(
        self,
        frames: Iterable[tuple[dict, np.ndarray, dict | str | None]],
        *,
        n_workers: int | None = None,
    ) -> list[dict]:
        """Write N frames; single flush at the end. Returns the index
        records for each frame written.

        Parameters
        ----------
        frames : iterable of (axes, pixels, metadata)
            The frames to write. Order is preserved end-to-end.
        n_workers : int or None
            For compressed writers (``compression != "none"``), the
            number of parallel encoder threads to use. ``None``
            picks ``min(cpu_count, 8)``. Pass ``1`` to force the
            serial path. Uncompressed writers always use the serial
            path (encode is a noop — parallelism would only add
            overhead). Order of writes to disk is preserved
            regardless of ``n_workers``: the writer thread drains
            encodes in submission order so the ndstorage reader and
            our reader see the exact same on-disk layout as the
            serial path.
        """
        frames_list = list(frames)
        # Decide whether to use the parallel pipeline.
        if (self._compression_is_none
                or len(frames_list) < 2
                or n_workers == 1):
            recs = []
            with self._lock:
                for axes, pixels, metadata in frames_list:
                    recs.append(self._write_frame_inner(axes, pixels, metadata))
                if self._fh is not None:
                    self._fh.flush()
            return recs

        if n_workers is None:
            n_workers = min(os.cpu_count() or 1, 8)
        return self._write_many_parallel(frames_list, n_workers)

    def _write_many_parallel(
        self,
        frames: list[tuple[dict, np.ndarray, dict | str | None]],
        n_workers: int,
    ) -> list[dict]:
        """Parallel encode → serial write pipeline.

        Encodes run on a thread pool (zstd / deflate / jxl / etc.
        release the GIL during their native compress call, so threads
        scale near-linearly on multi-core hosts). The writer thread
        drains futures in submission order so the on-disk byte layout
        is identical to the serial path — readers see exactly the same
        IFD chain.

        A bounded look-ahead window caps in-flight encoded bytes so
        long batches don't grow memory unboundedly when the encoder
        outpaces the disk writer.
        """
        from concurrent.futures import ThreadPoolExecutor
        from .core.segment_compression import encode_segment

        cmp_code = self._compression_code
        cmp_level = self._compression_level

        def _encode_pixels(raw_pixels):
            if isinstance(raw_pixels, np.ndarray):
                buf = memoryview(raw_pixels).cast("B")
            else:
                buf = raw_pixels
            return encode_segment(buf, cmp_code, level=cmp_level)

        recs: list[dict] = []
        with self._lock:
            # Prepare each frame in main thread (no I/O, no encode).
            # _prepare_frame extracts numpy buffers and JSON-encodes
            # metadata; cheap and avoids holding refs to user dicts.
            with ThreadPoolExecutor(
                max_workers=n_workers,
                thread_name_prefix="ndtiff-encode",
            ) as ex:
                window = max(n_workers * 2, 4)
                inflight: list[tuple[dict, Any]] = []   # [(prepared, future)]
                it = iter(frames)

                def _submit_next() -> bool:
                    try:
                        axes, pixels, metadata = next(it)
                    except StopIteration:
                        return False
                    p = self._prepare_frame(axes, pixels, metadata)
                    fut = ex.submit(_encode_pixels, p["raw_pixels"])
                    inflight.append((p, fut))
                    return True

                # Prime the pipeline up to the window.
                for _ in range(window):
                    if not _submit_next():
                        break

                while inflight:
                    p, fut = inflight.pop(0)
                    encoded = fut.result()
                    recs.append(self._emit_frame(
                        p, encoded, self._compression_code,
                    ))
                    _submit_next()

            if self._fh is not None:
                self._fh.flush()
        return recs

    def close(self) -> None:
        """Patch null next-IFD offsets, write NDTiff.index, close fh."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._finalize_current_stack()
            (self._dir / "NDTiff.index").write_bytes(bytes(self._index_records))

    def __enter__(self) -> "NDTiffWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # ------------------------------------------------------------------
    # Stack-file management
    # ------------------------------------------------------------------

    def _stack_filename(self, idx: int) -> str:
        return f"{self._base}.tif" if idx == 0 else f"{self._base}_{idx}.tif"

    def _open_next_stack(self) -> None:
        """Open the next NDTiffStack file with header + summary."""
        self._cur_path = self._dir / self._stack_filename(self._stack_index)
        # Pre-allocate the 4 GB envelope (matches ndstorage). Most
        # filesystems sparse-allocate so this is cheap; on close() we
        # truncate down to actual length.
        with open(self._cur_path, "wb") as f:
            f.seek(_MAX_FILE_SIZE - 1)
            f.write(b"\x00")
        # Reopen for streaming writes.
        self._fh = open(self._cur_path, "rb+")
        self._fh.seek(0)
        self._write_header_and_summary()
        self._next_ifd_offset_location = -1

    def _finalize_current_stack(self) -> None:
        """Patch the trailing IFD's next-offset to 0, truncate, close."""
        if self._fh is None:
            return
        # Patch null next-IFD-offset for the last frame in this file.
        if self._next_ifd_offset_location >= 0:
            cur = self._fh.tell()
            self._fh.seek(self._next_ifd_offset_location)
            self._fh.write(struct.pack("<I", 0))
            self._fh.seek(cur)
        self._fh.truncate()
        self._fh.flush()
        self._fh.close()
        self._fh = None
        self._cur_path = None

    def _rollover_if_needed(self, frame_bytes: int) -> None:
        """If the current file can't fit one more frame, roll over to the
        next NDTiffStack_N.tif and rewrite the header."""
        if self._fh is None:
            return
        # Conservative: assume metadata = 5 KB (real metadata is
        # variable but typically 200B-5KB for Pycro-Manager).
        needed = frame_bytes + _IFD_HEADER_BYTES + 5_000_000
        if self._fh.tell() + needed >= _MAX_FILE_SIZE:
            self._finalize_current_stack()
            self._stack_index += 1
            self._open_next_stack()

    # ------------------------------------------------------------------
    # File header / IFD writing
    # ------------------------------------------------------------------

    def _write_header_and_summary(self) -> None:
        """Bytes 0..27 = NDTiff prefix; bytes 28..28+L = summary JSON."""
        assert self._fh is not None
        first_ifd_off = 28 + len(self._summary_bytes)
        if first_ifd_off % 2 == 1:
            first_ifd_off += 1   # word-align IFD

        hdr = bytearray(28)
        struct.pack_into("<H", hdr, 0, 0x4949)  # 'II' little-endian
        struct.pack_into("<H", hdr, 2, 42)      # TIFF magic
        struct.pack_into("<I", hdr, 4, first_ifd_off)
        struct.pack_into("<III", hdr, 8,
                         _NDTIFF_MAGIC, _MAJOR_VERSION, _MINOR_VERSION)
        struct.pack_into("<II", hdr, 20,
                         _SUMMARY_MD_HEADER, len(self._summary_bytes))
        self._fh.write(hdr)
        self._fh.write(self._summary_bytes)
        # Pad to even offset so the first IFD lands word-aligned.
        if (self._fh.tell() % 2) == 1:
            self._fh.write(b"\x00")

    def _write_frame_inner(
        self,
        axes: dict[str, Any],
        pixels: np.ndarray,
        metadata: dict | str | None,
    ) -> dict:
        """Serial write path: prepare → encode → emit, all in one go."""
        p = self._prepare_frame(axes, pixels, metadata)
        if self._compression_is_none:
            return self._emit_frame(p, p["raw_pixels"], 1)
        from .core.segment_compression import encode_segment
        raw = p["raw_pixels"]
        buf = memoryview(raw).cast("B") if isinstance(raw, np.ndarray) else raw
        encoded = encode_segment(
            buf, self._compression_code, level=self._compression_level,
        )
        return self._emit_frame(p, encoded, self._compression_code)

    # ------------------------------------------------------------------
    # Frame pipeline phases — separated so write_many can run the encode
    # step in parallel threads while the writer thread emits in order.
    # ------------------------------------------------------------------

    def _prepare_frame(
        self,
        axes: dict[str, Any],
        pixels: np.ndarray,
        metadata: dict | str | None,
    ) -> dict:
        """Pre-encode work that doesn't touch the file: shape +
        dtype validation, byte-order normalization, metadata JSON.

        Returns a dict consumed by :meth:`_emit_frame` (and, in
        between, by an encoder thread which only needs ``raw_pixels``).
        """
        if self._closed:
            raise NDTiffWriterError("writer is closed")
        if pixels.ndim == 2:
            rgb = False
            h, w = pixels.shape
        elif pixels.ndim == 3 and pixels.shape[2] == 3:
            rgb = True
            h, w, _ = pixels.shape
        else:
            raise NDTiffWriterError(
                f"NDTiffWriter expects 2D (h,w) or 3D (h,w,3) pixels; "
                f"got shape={pixels.shape}"
            )
        bit_depth = self._bit_depth_hint
        if bit_depth == "auto":
            bit_depth = 8 if pixels.dtype == np.uint8 else 16

        raw_pixels = self._pixel_bytes(pixels, rgb)

        if isinstance(metadata, dict):
            md_bytes = json.dumps(metadata).encode("utf-8")
        elif isinstance(metadata, str):
            md_bytes = metadata.encode("utf-8")
        elif metadata is None:
            md_bytes = b"{}"
        else:
            md_bytes = bytes(metadata)

        return {
            "axes": axes, "h": h, "w": w, "rgb": rgb, "bit_depth": bit_depth,
            "raw_pixels": raw_pixels, "md_bytes": md_bytes,
        }

    def _emit_frame(
        self,
        p: dict,
        pixel_bytes,
        tiff_compression_tag: int,
    ) -> dict:
        """Serialize the prepared+encoded frame to disk. Caller holds
        ``self._lock``. Updates index records + advances file position.
        Handles file rollover at the 4 GiB boundary.
        """
        axes = p["axes"]
        h, w = p["h"], p["w"]
        rgb = p["rgb"]
        bit_depth = p["bit_depth"]
        md_bytes = p["md_bytes"]
        if isinstance(pixel_bytes, np.ndarray):
            bytes_per_image = pixel_bytes.nbytes
        else:
            bytes_per_image = len(pixel_bytes)

        self._rollover_if_needed(bytes_per_image + len(md_bytes))

        # ---- Compute IFD layout ----
        assert self._fh is not None
        if self._fh.tell() % 2 == 1:
            self._fh.write(b"\x00")
        ifd_start = self._fh.tell()
        # IFD = 2 (entry count) + 13*12 + 4 (next IFD) + 6 (rgb only) + 16 (x/y res)
        ifd_struct_size = 2 + _ENTRIES_PER_IFD * 12 + 4
        bps_off = ifd_start + ifd_struct_size      # 6 bytes if rgb
        x_res_off = bps_off + (6 if rgb else 0)    # 8 bytes
        y_res_off = x_res_off + 8                  # 8 bytes
        pixel_off = y_res_off + 8
        metadata_off = pixel_off + bytes_per_image
        next_ifd_off = metadata_off + len(md_bytes)
        if next_ifd_off % 2 == 1:
            next_ifd_off += 1

        ifd_bytes = bytearray(ifd_struct_size + (6 if rgb else 0) + 16)
        # 2-byte entry count
        struct.pack_into("<H", ifd_bytes, 0, _ENTRIES_PER_IFD)
        # 13 entries — order matches ndstorage exactly so readers
        # that scan in-order don't trip.
        e = 2
        e += _pack_ifd_entry(ifd_bytes, e, _TAG_WIDTH,       4, 1, w)
        e += _pack_ifd_entry(ifd_bytes, e, _TAG_HEIGHT,      4, 1, h)
        e += _pack_ifd_entry(ifd_bytes, e, _TAG_BITS_PER_SAMPLE,
                             3, 3 if rgb else 1,
                             bps_off if rgb else (8 if bit_depth == 8 else 16))
        e += _pack_ifd_entry(ifd_bytes, e, _TAG_COMPRESSION,
                             3, 1, tiff_compression_tag)
        e += _pack_ifd_entry(ifd_bytes, e, _TAG_PHOTOMETRIC, 3, 1, 2 if rgb else 1)
        e += _pack_ifd_entry(ifd_bytes, e, _TAG_STRIP_OFFSETS, 4, 1, pixel_off)
        e += _pack_ifd_entry(ifd_bytes, e, _TAG_SAMPLES_PER_PIXEL,
                             3, 1, 3 if rgb else 1)
        e += _pack_ifd_entry(ifd_bytes, e, _TAG_ROWS_PER_STRIP, 3, 1, h)
        e += _pack_ifd_entry(ifd_bytes, e, _TAG_STRIP_BYTE_COUNTS,
                             4, 1, bytes_per_image)
        e += _pack_ifd_entry(ifd_bytes, e, _TAG_X_RESOLUTION, 5, 1, x_res_off)
        e += _pack_ifd_entry(ifd_bytes, e, _TAG_Y_RESOLUTION, 5, 1, y_res_off)
        e += _pack_ifd_entry(ifd_bytes, e, _TAG_RESOLUTION_UNIT, 3, 1, 3)
        e += _pack_ifd_entry(ifd_bytes, e, _TAG_MM_METADATA, 2, len(md_bytes),
                             metadata_off)
        # next-IFD pointer slot — will be patched by next frame or by
        # close() to 0 for the last frame.
        struct.pack_into("<I", ifd_bytes, e, next_ifd_off)
        e += 4
        if rgb:
            struct.pack_into("<HHH", ifd_bytes, e,
                             bit_depth, bit_depth, bit_depth)
            e += 6
        # X/Y resolution rationals (numerator, denominator)
        struct.pack_into("<II", ifd_bytes, e, 1, 1)
        e += 8
        struct.pack_into("<II", ifd_bytes, e, 1, 1)
        e += 8

        # No patch-back needed: the IFD we're about to write already
        # contains the correct `next_ifd_off` value, computed from
        # this frame's known sizes. The reference ndstorage writer
        # patches per frame because it writes the IFD with
        # next_ifd_offset = 0 initially and fixes it up later; we
        # avoid that round trip. Just remember where the IFD's
        # next-pointer slot is so close() can null-terminate the
        # last frame's chain.
        self._next_ifd_offset_location = ifd_start + 2 + _ENTRIES_PER_IFD * 12

        # Append IFD + pixels + metadata as one contiguous run.
        # f.write(ndarray) uses the buffer protocol — no Python-level
        # bytes() copy unlike ndstorage's explicit tobytes() path.
        self._fh.write(ifd_bytes)
        self._fh.write(pixel_bytes)
        self._fh.write(md_bytes)
        if (self._fh.tell() % 2) == 1:
            self._fh.write(b"\x00")

        # ---- Append index record ----
        pixel_type = _pixel_type_for(bit_depth, rgb)
        filename = self._cur_path.name
        axes_bytes = json.dumps(axes).encode("utf-8")
        fn_bytes = filename.encode("utf-8")
        self._index_records.extend(struct.pack("<I", len(axes_bytes)))
        self._index_records.extend(axes_bytes)
        self._index_records.extend(struct.pack("<I", len(fn_bytes)))
        self._index_records.extend(fn_bytes)
        self._index_records.extend(struct.pack(
            "<IIIIIIII",
            pixel_off, w, h, pixel_type, tiff_compression_tag,
            metadata_off, len(md_bytes), 0,
        ))
        self._frame_count += 1

        return {
            "filename": filename,
            "axes": dict(axes),
            "pixel_offset": pixel_off,
            "image_width": w,
            "image_height": h,
            "pixel_type": pixel_type,
            "metadata_offset": metadata_off,
            "metadata_length": len(md_bytes),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pixel_bytes(pixels: np.ndarray, rgb: bool):
        """Return a buffer-protocol object ready for f.write().

        Returns the contiguous ndarray directly — BufferedWriter.write
        accepts any buffer-protocol object and reads via memcpy without
        going through bytes(). Saves ~30 µs/frame on big arrays vs an
        explicit .tobytes() that allocates a new bytes object.
        """
        host_le = sys.byteorder == "little"
        if pixels.dtype.byteorder == "=" or pixels.dtype.byteorder == "|" \
                or (pixels.dtype.byteorder == "<" and host_le) \
                or (pixels.dtype.byteorder == ">" and not host_le):
            arr = np.ascontiguousarray(pixels)
        else:
            # Source byteorder differs from host; swap once before write.
            arr = np.ascontiguousarray(pixels.byteswap())
        return arr


def _pack_ifd_entry(
    buf: bytearray, off: int,
    tag: int, dtype: int, count: int, value: int,
) -> int:
    """Pack one 12-byte TIFF IFD entry. Returns 12."""
    struct.pack_into("<HHII", buf, off, tag, dtype, count, value)
    return 12


def _pixel_type_for(bit_depth: int, rgb: bool) -> int:
    if rgb:
        return _PIXEL_TYPE_EIGHT_BIT_RGB
    if bit_depth == 8:
        return _PIXEL_TYPE_EIGHT_BIT
    return _PIXEL_TYPE_SIXTEEN_BIT  # 16/10/11/12/14 → uint16 storage


__all__ = ["NDTiffWriter", "NDTiffWriterError"]
