"""TiffWriter — native classic TIFF 6.0 writer.

Companion to :class:`opencodecs._tiff_codec.TiffStream`. Writes
classic (32-bit offset) TIFF files that round-trip cleanly through
both our reader and ``tifffile``. Supports:

* All standard integer + float dtypes (uint/int 8/16/32/64, float
  16/32/64); grayscale + multi-channel chunky (contig) layout.
* Strip layout (default) or tile layout.
* Per-tile / per-strip compression via
  :mod:`opencodecs.core.segment_compression` —
  ``none / deflate / zstd / lzw / packbits / jpeg / jpeg2000 / jxl
  / webp / lerc``.
* Horizontal predictor (tag 317 = 2) on encode for the byte-stream
  codecs that benefit (deflate / zstd / lzw).
* Multi-page IFD chain so callers can write pyramidal COG-style TIFFs
  by issuing ``write_page`` once per resolution level.

Deferred to v2: BigTIFF (64-bit offsets), floating-point predictor
3, JPEG-with-shared-tables (tag 347) encode, SubIFDs (tag 330).
"""

from __future__ import annotations

import io
import os
import struct
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Tag + type constants — match _tiff_codec.py / codecs/_tiff.pyx
# ---------------------------------------------------------------------------

TAG_NEW_SUBFILE_TYPE   = 254
TAG_IMAGE_WIDTH        = 256
TAG_IMAGE_LENGTH       = 257
TAG_BITS_PER_SAMPLE    = 258
TAG_COMPRESSION        = 259
TAG_PHOTOMETRIC        = 262
TAG_IMAGE_DESCRIPTION  = 270
TAG_STRIP_OFFSETS      = 273
TAG_SAMPLES_PER_PIXEL  = 277
TAG_ROWS_PER_STRIP     = 278
TAG_STRIP_BYTE_COUNTS  = 279
TAG_X_RESOLUTION       = 282
TAG_Y_RESOLUTION       = 283
TAG_PLANAR_CONFIG      = 284
TAG_RESOLUTION_UNIT    = 296
TAG_SOFTWARE           = 305
TAG_PREDICTOR          = 317
TAG_TILE_WIDTH         = 322
TAG_TILE_LENGTH        = 323
TAG_TILE_OFFSETS       = 324
TAG_TILE_BYTE_COUNTS   = 325
TAG_SAMPLE_FORMAT      = 339

# TIFF entry type codes (matches the type-size table in _tiff.pyx).
T_BYTE      = 1
T_ASCII     = 2
T_SHORT     = 3
T_LONG      = 4
T_RATIONAL  = 5
T_FLOAT     = 11
T_DOUBLE    = 12
# BigTIFF-only types (TIFF Technical Note 2 / Adobe BigTIFF spec).
T_LONG8     = 16   # 8-byte unsigned int (== uint64)
T_SLONG8    = 17   # 8-byte signed int   (== int64)
T_IFD8      = 18   # 8-byte IFD offset

_TYPE_SIZE = {
    T_BYTE: 1, T_ASCII: 1, T_SHORT: 2, T_LONG: 4,
    T_RATIONAL: 8, T_FLOAT: 4, T_DOUBLE: 8,
    T_LONG8: 8, T_SLONG8: 8, T_IFD8: 8,
}

# Photometric interpretation codes (TIFF 6 §8).
PHOTOMETRIC_MINISWHITE = 0
PHOTOMETRIC_MINISBLACK = 1
PHOTOMETRIC_RGB        = 2
PHOTOMETRIC_PALETTE    = 3
PHOTOMETRIC_TRANSPARENCY = 4
PHOTOMETRIC_CMYK       = 5
PHOTOMETRIC_YCBCR      = 6
PHOTOMETRIC_CIELAB     = 8


class TiffWriterError(RuntimeError):
    """Raised on writer state-machine or argument errors."""


# ---------------------------------------------------------------------------
# dtype ↔ (bits_per_sample, sample_format) mapping
# ---------------------------------------------------------------------------

def _bps_and_sample_format(dtype: np.dtype) -> tuple[int, int]:
    """Return (BitsPerSample, SampleFormat) for a numpy dtype.

    SampleFormat codes (TIFF 6 §10):
      1 = unsigned int, 2 = signed int, 3 = IEEE float, 4 = undefined.
    """
    kind = dtype.kind
    bits = dtype.itemsize * 8
    if kind == "u":
        return bits, 1
    if kind == "i":
        return bits, 2
    if kind == "f":
        return bits, 3
    raise TiffWriterError(
        f"TiffWriter: unsupported dtype {dtype!r} (kind={kind!r}); "
        f"supported kinds are u/i/f"
    )


# ---------------------------------------------------------------------------
# Horizontal predictor (encode side)
# ---------------------------------------------------------------------------

def _apply_horizontal_predictor(seg: np.ndarray) -> np.ndarray:
    """In-place horizontal differencing along the column axis.

    Matches the inverse used in opencodecs/codecs/_tiff.pyx
    (undo_horizontal_*): each sample (after column 0) becomes
    ``sample - sample_to_left`` per channel, with modular wrap for
    integer dtypes. Predictor 2 applies row-by-row, so the operation
    is reversible regardless of byte order.

    ``seg`` must be C-contiguous (rows, cols, samples) or (rows, cols);
    a 2D input is treated as (rows, cols, 1).
    """
    if seg.ndim == 2:
        view = seg.reshape(seg.shape[0], seg.shape[1], 1)
    else:
        view = seg
    cols = view.shape[1]
    if cols < 2:
        return seg
    # Iterate columns from last to second so we never overwrite a value
    # we still need to read for the left-neighbor. Numpy handles the
    # row + channel broadcast in one call per column-pair, which is a
    # single SIMD-friendly subtract over an entire row of samples.
    for c in range(cols - 1, 0, -1):
        view[:, c] -= view[:, c - 1]
    return seg


# ---------------------------------------------------------------------------
# IFD entry packing
# ---------------------------------------------------------------------------

def _value_bytes_for(
    type_code: int, values: tuple, byte_order: str,
) -> bytes:
    """Pack a tag's payload to its on-disk bytes."""
    bo = byte_order
    if type_code == T_BYTE:
        return bytes(values)
    if type_code == T_ASCII:
        # values is a single bytes/str — count = len(payload) including \0.
        raw = values[0]
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        if not raw.endswith(b"\x00"):
            raw = raw + b"\x00"
        return raw
    if type_code == T_SHORT:
        return struct.pack(f"{bo}{len(values)}H", *values)
    if type_code == T_LONG:
        return struct.pack(f"{bo}{len(values)}I", *values)
    if type_code == T_RATIONAL:
        # values is a list of (num, den) pairs.
        flat = []
        for v in values:
            if isinstance(v, tuple):
                flat.extend(v)
            else:
                # Allow int → (v, 1).
                flat.extend((int(v), 1))
        return struct.pack(f"{bo}{len(flat)}I", *flat)
    if type_code == T_FLOAT:
        return struct.pack(f"{bo}{len(values)}f", *values)
    if type_code == T_DOUBLE:
        return struct.pack(f"{bo}{len(values)}d", *values)
    if type_code == T_LONG8 or type_code == T_IFD8:
        return struct.pack(f"{bo}{len(values)}Q", *values)
    if type_code == T_SLONG8:
        return struct.pack(f"{bo}{len(values)}q", *values)
    raise TiffWriterError(f"unsupported TIFF tag type {type_code}")


def _count_for(type_code: int, values: tuple) -> int:
    """TIFF 'count' field — number of items, NOT bytes."""
    if type_code == T_ASCII:
        raw = values[0]
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        n = len(raw)
        if not raw.endswith(b"\x00"):
            n += 1
        return n
    if type_code == T_RATIONAL:
        # 1 rational = 1 count (= 2 LONGs = 8 bytes)
        return len(values)
    return len(values)


class _IFDEntry:
    """A single tag, deferred until layout time so values can be
    placed either inline (≤4 bytes) or out-of-line (offset).

    Attributes
    ----------
    tag : int
    type_code : int
    payload : bytes
        Pre-packed bytes for the values.
    count : int
        TIFF count field.
    """

    __slots__ = ("tag", "type_code", "payload", "count")

    def __init__(self, tag, type_code, values, byte_order):
        self.tag = int(tag)
        self.type_code = int(type_code)
        self.count = _count_for(type_code, values)
        self.payload = _value_bytes_for(type_code, values, byte_order)


# ---------------------------------------------------------------------------
# TiffWriter
# ---------------------------------------------------------------------------

class TiffWriter:
    """Streaming classic-TIFF writer with multi-page support.

    Usage — one-shot single-page::

        with TiffWriter("out.tif") as w:
            w.write_page(image, compression="zstd")

    Multi-page (e.g. pyramidal COG)::

        with TiffWriter("out.tif") as w:
            w.write_page(
                full_res, tile=(256, 256), compression="zstd",
                photometric="minisblack",
            )
            w.write_page(
                half_res, tile=(256, 256), compression="zstd",
                subfiletype=1,                # mark as reduced-res IFD
            )
            ...

    The output is classic 32-bit-offset TIFF, little-endian. Use
    ``compression="none"`` for maximum write throughput (memcpy-only
    path) or any of the supported codecs for compressed output.
    """

    # 4 GiB cap for classic 32-bit TIFF offsets. Effectively unlimited
    # for BigTIFF (64-bit offsets cap at 16 EiB).
    _CLASSIC_MAX_OFFSET = (1 << 32) - 1
    _BIGTIFF_MAX_OFFSET = (1 << 64) - 1

    def __init__(
        self,
        dest: str | Path | io.BufferedWriter,
        *,
        byte_order: str = "<",
        streaming: bool = False,
        bigtiff: bool = False,
    ):
        if byte_order not in ("<", ">"):
            raise TiffWriterError(
                f"byte_order must be '<' or '>'; got {byte_order!r}"
            )
        self._byte_order = byte_order
        self._streaming = bool(streaming)
        self._bigtiff = bool(bigtiff)
        # Per-format structural constants. Layout differences between
        # classic TIFF and BigTIFF are encoded here so the rest of the
        # writer parameterizes on them.
        if self._bigtiff:
            # BigTIFF (Adobe BigTIFF spec / TIFF Tech Note 2):
            #   header = 16 bytes
            #     2B byte order, 2B magic=43, 2B byte-size-of-offsets=8,
            #     2B constant=0, 8B first-IFD offset
            #   entry = 20 bytes (was 12): tag(2) + type(2) + count(8) +
            #           value/offset(8)
            #   ifd  = 8B entry-count + 20*N entries + 8B next-IFD offset
            #   inline payload fits in the 8-byte value slot.
            self._magic = 43
            self._header_size = 16
            self._entry_count_size = 8
            self._entry_count_fmt = "Q"      # u64
            self._entry_size = 20
            self._offset_size = 8
            self._offset_fmt = "Q"           # u64
            self._inline_max = 8
            # Offsets and counts that could overflow 32-bit use T_LONG8.
            # ImageWidth/ImageLength fit easily in T_LONG even for
            # 4-billion-pixel images; the type only matters for
            # strip/tile offsets/counts.
            self._offset_type = T_LONG8
            self._max_offset = self._BIGTIFF_MAX_OFFSET
        else:
            self._magic = 42
            self._header_size = 8
            self._entry_count_size = 2
            self._entry_count_fmt = "H"      # u16
            self._entry_size = 12
            self._offset_size = 4
            self._offset_fmt = "I"           # u32
            self._inline_max = 4
            self._offset_type = T_LONG
            self._max_offset = self._CLASSIC_MAX_OFFSET
        # Two backing storage modes:
        #   1. path → raw fd (os.write / os.writev / os.pwrite) — bypasses
        #      Python BufferedWriter so per-segment writes hit the kernel
        #      directly in one syscall via scatter-gather. Measured ~30%
        #      faster on multi-tile TIFF write workloads.
        #   2. file-like dest → use the supplied object's write/seek API.
        #      We can't safely promote arbitrary file-likes to a raw fd.
        # ``self._pos`` is the manually tracked write cursor in BOTH
        # modes; ``self._fh.tell()`` calls (Python-level method dispatch)
        # are the second-biggest overhead after per-write GIL grabs.
        self._fd: int = -1
        self._fh = None
        self._pos: int = 0
        if isinstance(dest, (str, os.PathLike)):
            self._path = Path(dest)
            flags = (os.O_RDWR | os.O_CREAT | os.O_TRUNC
                     | getattr(os, "O_BINARY", 0))
            # In streaming mode an O_WRONLY open is sufficient and avoids
            # accidentally promising back-patch capabilities we shouldn't
            # exercise.
            if self._streaming:
                flags = (os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                         | getattr(os, "O_BINARY", 0))
            self._fd = os.open(str(self._path), flags, 0o644)
            self._owns_fh = True
        elif hasattr(dest, "write"):
            # Streaming mode tolerates dests that lack ``.seek`` (raw
            # pipes, sockets, HTTP request bodies, gzip streams).
            # write_page() still requires seek for the back-patch path.
            if not self._streaming and not hasattr(dest, "seek"):
                raise TypeError(
                    "TiffWriter dest must be a seekable writable file-like "
                    "for write_page(); pass streaming=True + write_stream() "
                    f"for non-seekable sinks. Got {type(dest).__name__}"
                )
            self._path = None
            self._fh = dest
            self._owns_fh = False
        else:
            raise TypeError(
                f"TiffWriter dest must be path or writable file-like; "
                f"got {type(dest).__name__}"
            )
        # First IFD offset slot is at bytes 4..7 of a classic-TIFF
        # header — patched once the first IFD is laid out (only in
        # non-streaming mode; streaming writes the header lazily once
        # the first page's layout is known and never patches).
        self._next_ifd_offset_slot: int | None = None
        self._wrote_header = False
        self._closed = False
        self._n_pages = 0
        if not self._streaming:
            self._write_header()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_page(
        self,
        arr: np.ndarray,
        *,
        tile: tuple[int, int] | None = None,
        rows_per_strip: int | None = None,
        compression: str | int = "none",
        compression_level: int | None = None,
        predictor: int = 1,
        subfiletype: int = 0,
        photometric: str | int = "auto",
        planar_config: int = 1,
        metadata: str | None = None,
        software: str | None = None,
        resolution: tuple[float, float] | None = None,
        extra_tags: list[tuple[int, int, tuple]] | None = None,
        n_workers: int | None = None,
        _in_chain: bool = True,
    ) -> dict:
        """Encode + emit one IFD for ``arr``.

        Parameters
        ----------
        arr : ndarray
            (H, W) or (H, W, C). dtype must be one of the
            uint/int/float standard sizes.
        tile : (tile_h, tile_w) or None
            If given, write a tiled IFD. Otherwise write strips.
        rows_per_strip : int or None
            Rows per strip (default: ~8 KiB worth of pixels, rounded).
        compression : name or int
            ``"none"``, ``"deflate"``, ``"zstd"``, ``"lzw"``,
            ``"packbits"``, ``"jpeg"``, ``"jpeg2000"``, ``"webp"``,
            ``"jxl"``, ``"lerc"``. LZW/packbits encode is not yet
            implemented (decoders exist; encoders need to be vendored).
        compression_level : int or None
            Passed through to deflate/zstd/jxl etc.
        predictor : 1 or 2
            Horizontal differencing (TIFF tag 317) on encode. Only
            applied for byte-stream codecs that benefit (deflate,
            zstd, lzw, packbits, none) — image-format codecs do their
            own internal prediction.
        subfiletype : int
            NewSubfileType (tag 254). Pass 1 to flag this IFD as a
            reduced-resolution version of another image (pyramid).
        photometric : "auto" | "minisblack" | "rgb" | int
            Photometric interpretation. ``"auto"`` picks MinIsBlack for
            single-channel and RGB for 3 channels.
        metadata : str or None
            Optional ImageDescription (tag 270). Free-form ASCII.
        software : str or None
            Optional Software (tag 305) ASCII tag.
        resolution : (x_res, y_res) or None
            Optional XResolution/YResolution (tags 282, 283) as
            float values; written as rationals with denominator 1000
            for sub-integer precision. Defaults to (1, 1) DPI.
        extra_tags : list of (tag, type_code, values) or None
            User-supplied raw tags appended after the standard set.
            ``values`` is a tuple matching the type_code (e.g.
            ``(123,)`` for one SHORT, ``("hello",)`` for ASCII).
            See module-level ``T_SHORT`` etc. for codes.
        n_workers : int or None
            Parallel encoder thread count for compressed writes. The
            on-disk byte layout is identical regardless of worker
            count — encodes run in parallel but the writer thread
            drains them in submission order. ``None`` picks
            ``min(cpu_count, 8)`` when there are at least 2 segments
            and compression is on; ``1`` forces the serial path.
            Uncompressed writes always use the serial path (no
            encode work to parallelize).

        Returns
        -------
        info : dict
            ``{ifd_offset, n_segments, encoded_bytes, shape, dtype}``
            — handy for callers building external indexes or pyramid
            manifests.
        """
        if self._closed:
            raise TiffWriterError("writer is closed")
        if self._streaming:
            raise TiffWriterError(
                "TiffWriter(streaming=True) only supports write_stream(); "
                "write_page requires a seekable sink for IFD back-patching"
            )
        if not isinstance(arr, np.ndarray):
            raise TiffWriterError(
                f"write_page expects an ndarray; got {type(arr).__name__}"
            )
        if arr.ndim not in (2, 3):
            raise TiffWriterError(
                f"write_page expects 2D (h,w) or 3D (h,w,c); got shape={arr.shape}"
            )

        # Resolve compression name → numeric code via the shared dispatcher.
        from .core.segment_compression import (
            codec_name_to_code, NONE as _CMP_NONE,
        )
        cmp_code = codec_name_to_code(compression)
        cmp_is_none = (cmp_code == _CMP_NONE)

        # Layout & dtype.
        if arr.ndim == 2:
            h, w = arr.shape
            samples_per_pixel = 1
        else:
            h, w, samples_per_pixel = arr.shape
        bps, sample_format = _bps_and_sample_format(arr.dtype)

        # Default photometric.
        if photometric == "auto":
            photometric_code = (
                PHOTOMETRIC_RGB if samples_per_pixel >= 3
                else PHOTOMETRIC_MINISBLACK
            )
        elif isinstance(photometric, str):
            photometric_code = _PHOTOMETRIC_NAMES[photometric.lower()]
        else:
            photometric_code = int(photometric)

        # Ensure native, contiguous, host-or-target-endian buffer.
        arr_le = self._coerce_byte_order(arr)
        arr_le = np.ascontiguousarray(arr_le)

        # Decide layout: tile vs strip.
        is_tiled = tile is not None
        if is_tiled:
            tile_h, tile_w = int(tile[0]), int(tile[1])
            if tile_h <= 0 or tile_w <= 0:
                raise TiffWriterError(f"tile must be positive; got {tile}")
            # TIFF 6 requires tile dims to be multiples of 16.
            if tile_h % 16 or tile_w % 16:
                raise TiffWriterError(
                    f"tile dims must be multiples of 16 (TIFF 6 §15); "
                    f"got tile={tile}"
                )
            n_tiles_y = (h + tile_h - 1) // tile_h
            n_tiles_x = (w + tile_w - 1) // tile_w
            segments = self._iter_tile_segments(
                arr_le, h, w, tile_h, tile_w, samples_per_pixel,
                n_tiles_y, n_tiles_x,
            )
            n_segments = n_tiles_y * n_tiles_x
            strip_h_for_tag = None
        else:
            tile_h = tile_w = 0  # unused
            if rows_per_strip is None:
                # Default: aim for ~8 KiB per strip (TIFF 6 spec
                # recommendation), but at least 1 row.
                bytes_per_row = w * samples_per_pixel * arr.dtype.itemsize
                rows_per_strip = max(1, 8192 // max(1, bytes_per_row))
                rows_per_strip = min(rows_per_strip, h)
            else:
                rows_per_strip = int(rows_per_strip)
                if rows_per_strip <= 0:
                    raise TiffWriterError(
                        f"rows_per_strip must be > 0; got {rows_per_strip}"
                    )
            n_strips = (h + rows_per_strip - 1) // rows_per_strip
            segments = self._iter_strip_segments(
                arr_le, h, w, rows_per_strip, samples_per_pixel, n_strips,
            )
            n_segments = n_strips
            strip_h_for_tag = rows_per_strip

        # Validate predictor + compression combo.
        if predictor not in (1, 2):
            raise TiffWriterError(
                f"writer supports predictor 1 (none) or 2 (horizontal); "
                f"got predictor={predictor}"
            )
        is_byte_stream = cmp_code in _BYTE_STREAM_CMP
        if predictor == 2 and not is_byte_stream:
            # Image-format codecs (jpeg, jpeg2000, webp, jxl, lerc) do
            # their own internal prediction; a TIFF predictor on top
            # would be incorrect/lossy. Silently downgrade to 1.
            predictor = 1

        # Pre-encode all segments. We need byte counts up-front to lay
        # out the IFD before writing data. For very large pages this
        # buffers everything in RAM; for streaming-friendly behaviour
        # callers should switch to ``start_page`` (v2; not yet wired).
        offsets: list[int] = []
        byte_counts: list[int] = []
        encoded_segments: list[bytes | memoryview | np.ndarray] = []

        if cmp_is_none and predictor == 1 and not is_tiled:
            # Fast path: strips are row-contiguous slices of the input
            # buffer; we can write them as memoryviews directly into
            # the file without copying or per-segment Python work.
            row_bytes = w * samples_per_pixel * arr.dtype.itemsize
            flat = arr_le.reshape(-1).view(np.uint8)
            for i in range(n_segments):
                y0 = i * rows_per_strip
                y1 = min(y0 + rows_per_strip, h)
                n = (y1 - y0) * row_bytes
                start = y0 * row_bytes
                encoded_segments.append(flat[start:start + n])
                byte_counts.append(n)
        elif (not cmp_is_none) and n_segments >= 2 and n_workers != 1:
            # Parallel encode path. Same pattern as NDTiffWriter:
            # submit segment encodes to a threadpool, drain in
            # submission order on the writer thread. Output bytes
            # are identical to the serial path (we use the same
            # encode function; only scheduling changes).
            if n_workers is None:
                _nw = min(os.cpu_count() or 1, 8)
            else:
                _nw = max(1, int(n_workers))
            seg_list = list(segments)

            def _encode_one(seg):
                if predictor == 2:
                    seg = np.ascontiguousarray(seg).copy()
                    _apply_horizontal_predictor(seg)
                return self._encode_segment_bytes(
                    seg, cmp_code, compression_level,
                )

            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(
                max_workers=_nw,
                thread_name_prefix="tiff-encode",
            ) as ex:
                futures = [ex.submit(_encode_one, seg) for seg in seg_list]
                for fut in futures:
                    encoded = fut.result()
                    encoded_segments.append(encoded)
                    byte_counts.append(len(encoded))
        else:
            for seg in segments:
                # seg arrives as a contiguous (h, w[, c]) ndarray.
                if predictor == 2:
                    # Predictor wants per-channel horizontal diffs;
                    # operate on a writable C-contiguous copy so we
                    # never mutate the caller's array.
                    seg = np.ascontiguousarray(seg).copy()
                    _apply_horizontal_predictor(seg)
                if cmp_is_none:
                    seg_c = np.ascontiguousarray(seg)
                    encoded_segments.append(seg_c)
                    byte_counts.append(seg_c.nbytes)
                else:
                    encoded = self._encode_segment_bytes(
                        seg, cmp_code, compression_level,
                    )
                    encoded_segments.append(encoded)
                    byte_counts.append(len(encoded))

        # ---- Write segments to disk, recording offsets ----
        # TIFF segment offsets are absolute file positions. We append
        # at the current cursor and advance — but instead of issuing
        # one syscall per segment, batch the entire encoded-segments
        # list into a single os.writev call. Offsets are predictable
        # from the running cursor + per-segment byte counts.
        if encoded_segments:
            total = sum(byte_counts)
            self._ensure_room(total)
            start = self._pos
            run = start
            for n in byte_counts:
                offsets.append(run)
                run += n
            self._writev(encoded_segments)

        total_data_bytes = sum(byte_counts)

        # ---- Build IFD entries ----
        bo = self._byte_order
        entries: list[_IFDEntry] = []
        def add(tag, tc, vals):
            entries.append(_IFDEntry(tag, tc, vals, bo))

        # Tag ordering: TIFF 6 §2 requires ascending tag numbers. We
        # build a sorted list at the end, but emit in canonical order
        # for readability.
        if subfiletype:
            add(TAG_NEW_SUBFILE_TYPE, T_LONG, (int(subfiletype),))
        add(TAG_IMAGE_WIDTH,       T_LONG,  (int(w),))
        add(TAG_IMAGE_LENGTH,      T_LONG,  (int(h),))
        bps_tuple = (bps,) * samples_per_pixel
        add(TAG_BITS_PER_SAMPLE,   T_SHORT, bps_tuple)
        add(TAG_COMPRESSION,       T_SHORT, (int(cmp_code),))
        add(TAG_PHOTOMETRIC,       T_SHORT, (int(photometric_code),))
        if metadata is not None:
            add(TAG_IMAGE_DESCRIPTION, T_ASCII, (metadata,))
        if not is_tiled:
            # Strip offsets / counts: T_LONG in classic, T_LONG8 in
            # BigTIFF (8-byte u64 per entry, supports >4 GiB files).
            add(TAG_STRIP_OFFSETS, self._offset_type, tuple(offsets))
        add(TAG_SAMPLES_PER_PIXEL, T_SHORT, (int(samples_per_pixel),))
        if not is_tiled:
            add(TAG_ROWS_PER_STRIP, T_LONG, (int(strip_h_for_tag),))
            add(TAG_STRIP_BYTE_COUNTS, self._offset_type, tuple(byte_counts))
        if resolution is not None:
            x_res, y_res = resolution
            add(TAG_X_RESOLUTION, T_RATIONAL,
                ((int(round(x_res * 1000)), 1000),))
            add(TAG_Y_RESOLUTION, T_RATIONAL,
                ((int(round(y_res * 1000)), 1000),))
            add(TAG_RESOLUTION_UNIT, T_SHORT, (2,))  # inch
        else:
            add(TAG_X_RESOLUTION, T_RATIONAL, ((1, 1),))
            add(TAG_Y_RESOLUTION, T_RATIONAL, ((1, 1),))
            add(TAG_RESOLUTION_UNIT, T_SHORT, (1,))  # no abs unit
        if samples_per_pixel > 1:
            add(TAG_PLANAR_CONFIG, T_SHORT, (int(planar_config),))
        if software is not None:
            add(TAG_SOFTWARE, T_ASCII, (software,))
        if predictor != 1:
            add(TAG_PREDICTOR, T_SHORT, (int(predictor),))
        if is_tiled:
            add(TAG_TILE_WIDTH,       T_LONG, (tile_w,))
            add(TAG_TILE_LENGTH,      T_LONG, (tile_h,))
            add(TAG_TILE_OFFSETS,     self._offset_type, tuple(offsets))
            add(TAG_TILE_BYTE_COUNTS, self._offset_type, tuple(byte_counts))
        # SampleFormat is required for non-uint8 dtypes. tifffile
        # always writes it; we follow suit for round-trip stability.
        sf_tuple = (sample_format,) * samples_per_pixel
        add(TAG_SAMPLE_FORMAT, T_SHORT, sf_tuple)
        if extra_tags:
            for tag_id, tc, vals in extra_tags:
                add(tag_id, tc, tuple(vals))

        # ---- Pack the IFD ----
        # For sub-IFDs, _in_chain=False: write the IFD struct at the
        # current file position but do NOT splice it into the top-
        # level IFD next-pointer chain. The caller (write_pyramid with
        # subifds=True) records this offset for the parent's
        # SubIFDs tag.
        if _in_chain:
            ifd_offset = self._write_ifd(entries)
            self._patch_next_ifd_offset(ifd_offset)
            self._n_pages += 1
        else:
            # Stash current next-slot before _write_ifd clobbers it
            # so chain-mode pages emitted later continue from the
            # right parent.
            saved_next_slot = self._next_ifd_offset_slot
            ifd_offset = self._write_ifd(entries)
            self._next_ifd_offset_slot = saved_next_slot

        return {
            "ifd_offset": ifd_offset,
            "n_segments": n_segments,
            "encoded_bytes": int(total_data_bytes),
            "shape": tuple(arr.shape),
            "dtype": str(arr.dtype),
        }

    def write_stream(
        self,
        pages: Iterable[np.ndarray],
        *,
        total_pages: int,
        tile: tuple[int, int] | None = None,
        rows_per_strip: int | None = None,
        compression: str | int = "none",
        compression_level: int | None = None,
        predictor: int = 1,
        photometric: str | int = "auto",
        planar_config: int = 1,
        subfiletype: int = 0,
        metadata: str | None = None,
        software: str | None = None,
        resolution: tuple[float, float] | None = None,
        extra_tags: list[tuple[int, int, tuple]] | None = None,
        n_workers: int | None = None,
    ) -> list[dict]:
        """Emit TIFF to an unseekable sink, COG-style.

        ``write_stream`` lays out each page as ``[IFD ⇒ pixels]`` so
        every strip/tile offset in the IFD points *forward* into the
        not-yet-written pixel region — predictable from the encoded
        byte counts, so no back-patch (and thus no seek) is needed.
        The next-IFD pointer is similarly computed up front from the
        running cursor + this page's known total bytes.

        Requirements:

        * Writer must have been constructed with ``streaming=True``.
        * Caller must declare ``total_pages`` so the last page can
          set next-IFD = 0.
        * All pages share the same per-page tag settings (tile,
          compression, etc.). For heterogeneous pages use a seekable
          sink and ``write_page``.

        Use cases: HTTP response bodies, S3 multipart upload streams,
        stdout pipelines, gzip wrappers — anything supporting
        ``.write()`` but not ``.seek()``.

        Returns one ``info`` dict per page (same shape as
        ``write_page``'s return).
        """
        if self._closed:
            raise TiffWriterError("writer is closed")
        if not self._streaming:
            raise TiffWriterError(
                "write_stream requires TiffWriter(streaming=True); "
                "construct a new writer in streaming mode"
            )
        if self._wrote_header:
            raise TiffWriterError(
                "write_stream is single-shot; construct a new writer "
                "for each output"
            )
        if int(total_pages) < 1:
            raise TiffWriterError(
                f"total_pages must be >= 1; got {total_pages}"
            )

        # Header: first IFD lives immediately after the header (offset
        # = ``self._header_size``, which is 8 for classic / 16 for
        # BigTIFF). We know that *before* writing any page bytes, so
        # we can emit the header up-front with the correct first-IFD
        # offset and never patch it.
        bo = self._byte_order
        bo_mark = b"II" if bo == "<" else b"MM"
        magic = struct.pack(bo + "H", self._magic)
        first_ifd_off_val = self._header_size
        if self._bigtiff:
            const_block = struct.pack(bo + "HH", 8, 0)  # offset-size, const
            first_ifd_off = struct.pack(bo + "Q", first_ifd_off_val)
            self._write(bo_mark + magic + const_block + first_ifd_off)
        else:
            first_ifd_off = struct.pack(bo + "I", first_ifd_off_val)
            self._write(bo_mark + magic + first_ifd_off)
        self._wrote_header = True

        infos: list[dict] = []
        pages_iter = iter(pages)
        for page_idx in range(int(total_pages)):
            try:
                arr = next(pages_iter)
            except StopIteration:
                raise TiffWriterError(
                    f"pages iterable exhausted at page {page_idx}; "
                    f"declared total_pages={total_pages}"
                )
            info = self._emit_streaming_page(
                arr,
                is_last=(page_idx == int(total_pages) - 1),
                tile=tile, rows_per_strip=rows_per_strip,
                compression=compression,
                compression_level=compression_level,
                predictor=predictor,
                photometric=photometric,
                planar_config=planar_config,
                subfiletype=subfiletype,
                metadata=metadata if page_idx == 0 else None,
                software=software if page_idx == 0 else None,
                resolution=resolution,
                extra_tags=extra_tags,
                n_workers=n_workers,
            )
            infos.append(info)
            self._n_pages += 1

        # Optional trailing-iterable sanity check: complain if caller
        # promised total_pages but actually has more.
        try:
            next(pages_iter)
        except StopIteration:
            pass
        else:
            raise TiffWriterError(
                f"pages iterable yielded more than declared "
                f"total_pages={total_pages}"
            )
        return infos

    def _emit_streaming_page(
        self,
        arr: np.ndarray,
        *,
        is_last: bool,
        tile: tuple[int, int] | None,
        rows_per_strip: int | None,
        compression: str | int,
        compression_level: int | None,
        predictor: int,
        photometric: str | int,
        planar_config: int,
        subfiletype: int,
        metadata: str | None,
        software: str | None,
        resolution: tuple[float, float] | None,
        extra_tags: list[tuple[int, int, tuple]] | None,
        n_workers: int | None,
    ) -> dict:
        """Encode + emit one streaming page in [IFD ⇒ pixels] order.

        Shares the segment-encoding helpers with ``write_page`` but
        uses a forward-only layout: we compute strip/tile offsets
        from the post-IFD cursor BEFORE writing the IFD, so the IFD
        carries correct values on its single trip through the sink.
        """
        if not isinstance(arr, np.ndarray):
            raise TiffWriterError(f"pages must be numpy arrays; got {type(arr).__name__}")
        if arr.ndim not in (2, 3):
            raise TiffWriterError(
                f"expected 2D (h,w) or 3D (h,w,c); got shape={arr.shape}"
            )

        from .core.segment_compression import (
            codec_name_to_code, NONE as _CMP_NONE,
        )
        cmp_code = codec_name_to_code(compression)
        cmp_is_none = (cmp_code == _CMP_NONE)

        if arr.ndim == 2:
            h, w = arr.shape
            samples_per_pixel = 1
        else:
            h, w, samples_per_pixel = arr.shape
        bps, sample_format = _bps_and_sample_format(arr.dtype)

        if photometric == "auto":
            photometric_code = (
                PHOTOMETRIC_RGB if samples_per_pixel >= 3
                else PHOTOMETRIC_MINISBLACK
            )
        elif isinstance(photometric, str):
            photometric_code = _PHOTOMETRIC_NAMES[photometric.lower()]
        else:
            photometric_code = int(photometric)

        arr_le = self._coerce_byte_order(arr)
        arr_le = np.ascontiguousarray(arr_le)

        is_tiled = tile is not None
        if is_tiled:
            tile_h, tile_w = int(tile[0]), int(tile[1])
            if tile_h <= 0 or tile_w <= 0 or tile_h % 16 or tile_w % 16:
                raise TiffWriterError(
                    f"tile dims must be positive multiples of 16; got {tile}"
                )
            n_tiles_y = (h + tile_h - 1) // tile_h
            n_tiles_x = (w + tile_w - 1) // tile_w
            segments = self._iter_tile_segments(
                arr_le, h, w, tile_h, tile_w, samples_per_pixel,
                n_tiles_y, n_tiles_x,
            )
            n_segments = n_tiles_y * n_tiles_x
            strip_h_for_tag = None
        else:
            tile_h = tile_w = 0
            if rows_per_strip is None:
                bytes_per_row = w * samples_per_pixel * arr.dtype.itemsize
                rows_per_strip = max(1, 8192 // max(1, bytes_per_row))
                rows_per_strip = min(rows_per_strip, h)
            else:
                rows_per_strip = int(rows_per_strip)
            n_strips = (h + rows_per_strip - 1) // rows_per_strip
            segments = self._iter_strip_segments(
                arr_le, h, w, rows_per_strip, samples_per_pixel, n_strips,
            )
            n_segments = n_strips
            strip_h_for_tag = rows_per_strip

        if predictor not in (1, 2):
            raise TiffWriterError(f"unsupported predictor {predictor}")
        is_byte_stream = cmp_code in _BYTE_STREAM_CMP
        if predictor == 2 and not is_byte_stream:
            predictor = 1

        # ---- Encode segments ----
        byte_counts: list[int] = []
        encoded_segments: list[bytes | memoryview | np.ndarray] = []
        if cmp_is_none and predictor == 1 and not is_tiled:
            row_bytes = w * samples_per_pixel * arr.dtype.itemsize
            flat = arr_le.reshape(-1).view(np.uint8)
            for i in range(n_segments):
                y0 = i * rows_per_strip
                y1 = min(y0 + rows_per_strip, h)
                n = (y1 - y0) * row_bytes
                start = y0 * row_bytes
                encoded_segments.append(flat[start:start + n])
                byte_counts.append(n)
        elif (not cmp_is_none) and n_segments >= 2 and n_workers != 1:
            if n_workers is None:
                _nw = min(os.cpu_count() or 1, 8)
            else:
                _nw = max(1, int(n_workers))
            seg_list = list(segments)

            def _encode_one(seg):
                if predictor == 2:
                    seg = np.ascontiguousarray(seg).copy()
                    _apply_horizontal_predictor(seg)
                return self._encode_segment_bytes(
                    seg, cmp_code, compression_level,
                )

            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(
                max_workers=_nw, thread_name_prefix="tiff-encode",
            ) as ex:
                futures = [ex.submit(_encode_one, seg) for seg in seg_list]
                for fut in futures:
                    encoded = fut.result()
                    encoded_segments.append(encoded)
                    byte_counts.append(len(encoded))
        else:
            for seg in segments:
                if predictor == 2:
                    seg = np.ascontiguousarray(seg).copy()
                    _apply_horizontal_predictor(seg)
                if cmp_is_none:
                    seg_c = np.ascontiguousarray(seg)
                    encoded_segments.append(seg_c)
                    byte_counts.append(seg_c.nbytes)
                else:
                    encoded = self._encode_segment_bytes(
                        seg, cmp_code, compression_level,
                    )
                    encoded_segments.append(encoded)
                    byte_counts.append(len(encoded))

        total_data_bytes = sum(byte_counts)
        self._ensure_room(total_data_bytes)

        # ---- Word-align IFD start ----
        if self._pos % 2:
            self._write(b"\x00")
        ifd_off = self._pos

        # ---- Build all IFD entries EXCEPT strip/tile offsets ----
        # Strip/tile offsets depend on where pixels land, which depends
        # on the IFD's own size. We build everything else first to
        # measure the IFD, then place strip offsets pointing into the
        # post-IFD pixel region.
        bo = self._byte_order
        entries: list[_IFDEntry] = []
        def add(tag, tc, vals):
            entries.append(_IFDEntry(tag, tc, vals, bo))

        if subfiletype:
            add(TAG_NEW_SUBFILE_TYPE, T_LONG, (int(subfiletype),))
        add(TAG_IMAGE_WIDTH,       T_LONG,  (int(w),))
        add(TAG_IMAGE_LENGTH,      T_LONG,  (int(h),))
        bps_tuple = (bps,) * samples_per_pixel
        add(TAG_BITS_PER_SAMPLE,   T_SHORT, bps_tuple)
        add(TAG_COMPRESSION,       T_SHORT, (int(cmp_code),))
        add(TAG_PHOTOMETRIC,       T_SHORT, (int(photometric_code),))
        if metadata is not None:
            add(TAG_IMAGE_DESCRIPTION, T_ASCII, (metadata,))
        add(TAG_SAMPLES_PER_PIXEL, T_SHORT, (int(samples_per_pixel),))
        if not is_tiled:
            add(TAG_ROWS_PER_STRIP, T_LONG, (int(strip_h_for_tag),))
        if resolution is not None:
            x_res, y_res = resolution
            add(TAG_X_RESOLUTION, T_RATIONAL,
                ((int(round(x_res * 1000)), 1000),))
            add(TAG_Y_RESOLUTION, T_RATIONAL,
                ((int(round(y_res * 1000)), 1000),))
            add(TAG_RESOLUTION_UNIT, T_SHORT, (2,))
        else:
            add(TAG_X_RESOLUTION, T_RATIONAL, ((1, 1),))
            add(TAG_Y_RESOLUTION, T_RATIONAL, ((1, 1),))
            add(TAG_RESOLUTION_UNIT, T_SHORT, (1,))
        if samples_per_pixel > 1:
            add(TAG_PLANAR_CONFIG, T_SHORT, (int(planar_config),))
        if software is not None:
            add(TAG_SOFTWARE, T_ASCII, (software,))
        if predictor != 1:
            add(TAG_PREDICTOR, T_SHORT, (int(predictor),))
        if is_tiled:
            add(TAG_TILE_WIDTH,  T_LONG, (tile_w,))
            add(TAG_TILE_LENGTH, T_LONG, (tile_h,))
        sf_tuple = (sample_format,) * samples_per_pixel
        add(TAG_SAMPLE_FORMAT, T_SHORT, sf_tuple)
        if extra_tags:
            for tag_id, tc, vals in extra_tags:
                add(tag_id, tc, tuple(vals))

        # ---- Add placeholder strip/tile entries (real offsets after layout) ----
        # We need their PAYLOAD LENGTHS now so we can compute total IFD
        # ext-blob size, but the actual offset values can stay 0 since
        # we'll overwrite them once pixel_base is known. Type is
        # T_LONG (4B) in classic, T_LONG8 (8B) in BigTIFF.
        offset_type = self._offset_type
        if is_tiled:
            sb_tag, bc_tag = TAG_TILE_OFFSETS, TAG_TILE_BYTE_COUNTS
        else:
            sb_tag, bc_tag = TAG_STRIP_OFFSETS, TAG_STRIP_BYTE_COUNTS
        add(sb_tag, offset_type, tuple([0] * n_segments))
        add(bc_tag, offset_type, tuple(byte_counts))

        # ---- Compute IFD layout + pixel base ----
        entries.sort(key=lambda e: e.tag)
        n_entries = len(entries)
        # Layout: entry_count + entry_size*N + offset_size.
        ifd_struct_size = (
            self._entry_count_size + self._entry_size * n_entries
            + self._offset_size
        )

        # Predict ext-blob layout: every entry whose payload exceeds
        # the inline slot width spills out-of-line, each blob word-
        # aligned.
        inline_max = self._inline_max
        ext_total = 0
        for ent in entries:
            if len(ent.payload) > inline_max:
                ext_total += len(ent.payload)
                if ext_total % 2:
                    ext_total += 1

        pixel_base = ifd_off + ifd_struct_size + ext_total

        # ---- Rebuild strip/tile-offsets entry with real offsets ----
        cursor = pixel_base
        real_offsets: list[int] = []
        for n in byte_counts:
            real_offsets.append(cursor)
            cursor += n
        # cursor is now where pixels END.
        if cursor > self._max_offset:
            raise TiffWriterError(
                f"page would push file past {'16 EiB' if self._bigtiff else '4 GiB'} "
                f"at offset {cursor}"
                + ("" if self._bigtiff
                   else "; pass bigtiff=True to enable 64-bit offsets")
            )

        # Swap in the real strip/tile-offsets entry.
        for i, ent in enumerate(entries):
            if ent.tag == sb_tag:
                entries[i] = _IFDEntry(
                    sb_tag, self._offset_type, tuple(real_offsets), bo,
                )
                break

        # ---- Compute next-IFD offset ----
        if is_last:
            next_ifd_off = 0
        else:
            next_ifd_off = cursor
            if next_ifd_off % 2:
                next_ifd_off += 1
            if next_ifd_off > self._max_offset:
                raise TiffWriterError(
                    "next IFD would land past the format's max offset"
                )

        # ---- Pack IFD struct + ext blobs ----
        entry_blocks = bytearray()
        ext_blobs: list[bytes] = []
        ext_cursor = ifd_off + ifd_struct_size
        offset_fmt = self._offset_fmt

        for ent in entries:
            payload = ent.payload
            if len(payload) <= inline_max:
                slot = payload + b"\x00" * (inline_max - len(payload))
            else:
                slot = struct.pack(bo + offset_fmt, ext_cursor)
                ext_blobs.append(payload)
                ext_cursor += len(payload)
                if ext_cursor % 2:
                    ext_blobs.append(b"\x00")
                    ext_cursor += 1
            # Entry header: tag(2) + type(2) + count(4 or 8) + slot(4 or 8)
            if self._bigtiff:
                entry_blocks.extend(struct.pack(
                    bo + "HHQ", ent.tag, ent.type_code, ent.count,
                ))
            else:
                entry_blocks.extend(struct.pack(
                    bo + "HHI", ent.tag, ent.type_code, ent.count,
                ))
            entry_blocks.extend(slot)

        # Sanity: predicted ext_total must match actual placement.
        actual_ext = ext_cursor - (ifd_off + ifd_struct_size)
        if actual_ext != ext_total:  # pragma: no cover - layout invariant
            raise TiffWriterError(
                f"streaming IFD layout invariant violated: predicted "
                f"ext_total={ext_total}, actual={actual_ext}"
            )

        ifd_struct = bytearray()
        ifd_struct.extend(struct.pack(bo + self._entry_count_fmt, n_entries))
        ifd_struct.extend(entry_blocks)
        ifd_struct.extend(struct.pack(bo + self._offset_fmt, next_ifd_off))

        # ---- Emit IFD + ext blobs + pixel segments (one scatter call) ----
        # Pad to align next-IFD start if needed (between this page's
        # pixels and the next page's IFD).
        emit_buffers: list = [bytes(ifd_struct)]
        emit_buffers.extend(ext_blobs)
        emit_buffers.extend(encoded_segments)
        if (not is_last) and (cursor % 2):
            # cursor is current pixel-end; next-IFD wants to be even.
            emit_buffers.append(b"\x00")
        self._writev(emit_buffers)

        return {
            "ifd_offset": ifd_off,
            "n_segments": n_segments,
            "encoded_bytes": int(total_data_bytes),
            "shape": tuple(arr.shape),
            "dtype": str(arr.dtype),
        }

    def write_pyramid(
        self,
        levels: list[np.ndarray],
        *,
        tile: tuple[int, int] | None = (256, 256),
        compression: str | int = "none",
        compression_level: int | None = None,
        predictor: int = 1,
        photometric: str | int = "auto",
        metadata: str | None = None,
        subifds: bool = False,
    ) -> list[dict]:
        """Write a pyramid.

        ``levels[0]`` is full-resolution; ``levels[1:]`` are progressively
        downsampled. Two storage layouts:

        * ``subifds=False`` (default) — COG convention. Each level is a
          separate top-level IFD. ``NewSubfileType = 1`` flags the
          reduced-resolution pages. Compatible with COG viewers.

        * ``subifds=True`` — bioformats / OME-TIFF convention. Only the
          full-resolution page is a top-level IFD; the reduced-
          resolution levels are referenced via the SubIFDs tag (330).
          Compatible with bioformats, QuPath, NDPI, anything that
          follows the OME-TIFF spec.
        """
        if not subifds:
            infos = []
            for i, arr in enumerate(levels):
                infos.append(self.write_page(
                    arr,
                    tile=tile,
                    compression=compression,
                    compression_level=compression_level,
                    predictor=predictor,
                    photometric=photometric,
                    subfiletype=0 if i == 0 else 1,
                    metadata=metadata if i == 0 else None,
                ))
            return infos

        # SubIFD layout: write the sub-resolution IFDs first (out of
        # the top-level chain) so we know their offsets, then write
        # the main IFD with SubIFDs tag (330) pointing at those
        # offsets.
        sub_infos: list[dict] = []
        for sub in levels[1:]:
            info = self.write_page(
                sub,
                tile=tile,
                compression=compression,
                compression_level=compression_level,
                predictor=predictor,
                photometric=photometric,
                subfiletype=1,
                _in_chain=False,
            )
            sub_infos.append(info)
        sub_ifd_offsets = tuple(int(i["ifd_offset"]) for i in sub_infos)

        # Now the main page goes into the top-level chain with a
        # SubIFDs tag listing each sub-IFD's offset. TIFF type 13
        # (IFD) is the canonical type for SubIFDs but bioformats /
        # libtiff also write type 4 (LONG) — both decode the same.
        main_info = self.write_page(
            levels[0],
            tile=tile,
            compression=compression,
            compression_level=compression_level,
            predictor=predictor,
            photometric=photometric,
            subfiletype=0,
            metadata=metadata,
            extra_tags=[(330, T_LONG, sub_ifd_offsets)],
        )
        return [main_info] + sub_infos

    def write_pyramid_auto(
        self,
        image: np.ndarray,
        *,
        pyramid_levels: int | None = None,
        pyramid_min_size: int = 512,
        pyramid_axes: tuple[int, ...] | str | None = None,
        tile: tuple[int, int] | None = (256, 256),
        compression: str | int = "none",
        compression_level: int | None = None,
        predictor: int = 1,
        photometric: str | int = "auto",
        metadata: str | None = None,
        subifds: bool = False,
    ) -> list[dict]:
        """Write a pyramid built automatically from a single full-res image.

        Opt-in convenience wrapper around :meth:`write_pyramid`. A
        pyramid roughly adds ~33% on-disk size (2D, geometric series)
        on top of the full-res page; the default ``pyramid_min_size=512``
        auto-stops levels at that threshold so a 1024-px-wide input
        produces just 2 levels (no surprise size bloat). Pass
        ``pyramid_levels=N`` to force a specific depth.

        See :func:`opencodecs._pyramid_build.make_pyramid_levels` for
        the downsampling algorithm (2x2 mean pool on the trailing 2
        spatial axes by default).
        """
        from ._pyramid_build import make_pyramid_levels
        levels = make_pyramid_levels(
            image,
            levels=pyramid_levels,
            min_size=pyramid_min_size,
            axes=pyramid_axes,
        )
        return self.write_pyramid(
            levels,
            tile=tile,
            compression=compression,
            compression_level=compression_level,
            predictor=predictor,
            photometric=photometric,
            metadata=metadata,
            subifds=subifds,
        )

    def close(self) -> None:
        """Patch a null next-IFD pointer for the last page and close
        the underlying file handle if we own it."""
        if self._closed:
            return
        self._closed = True
        # Final IFD's next slot stays 0 (default we wrote). Nothing
        # else to patch.
        if self._fd >= 0:
            # Raw-fd writes go to the kernel immediately; no
            # user-space buffer to flush.
            if self._owns_fh:
                os.close(self._fd)
            self._fd = -1
        elif self._fh is not None:
            try:
                self._fh.flush()
            except Exception:
                pass
            if self._owns_fh:
                self._fh.close()

    def __enter__(self) -> "TiffWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    @property
    def n_pages(self) -> int:
        return self._n_pages

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _write_header(self) -> None:
        """Write the TIFF/BigTIFF file header.

        Classic TIFF (8 bytes):
          0..1: byte order ('II' or 'MM')
          2..3: magic = 42
          4..7: first-IFD offset (32-bit) — patched once first page lands

        BigTIFF (16 bytes):
          0..1:   byte order
          2..3:   magic = 43
          4..5:   byte-size-of-offsets = 8
          6..7:   constant = 0
          8..15:  first-IFD offset (64-bit) — patched once first page lands
        """
        bo = self._byte_order
        bo_mark = b"II" if bo == "<" else b"MM"
        magic = struct.pack(bo + "H", self._magic)
        if self._bigtiff:
            const_block = struct.pack(bo + "HH", 8, 0)  # offset-size, const
            first_ifd_off = struct.pack(bo + "Q", 0)
            self._write(bo_mark + magic + const_block + first_ifd_off)
            self._header_first_ifd_slot = 8
        else:
            first_ifd_off = struct.pack(bo + "I", 0)
            self._write(bo_mark + magic + first_ifd_off)
            self._header_first_ifd_slot = 4
        self._next_ifd_offset_slot = self._header_first_ifd_slot
        self._wrote_header = True

    def _coerce_byte_order(self, arr: np.ndarray) -> np.ndarray:
        """Return an array whose byte order matches the writer's
        target. byteswap when needed (returns a copy)."""
        # Multi-byte numeric dtypes only.
        if arr.dtype.itemsize <= 1:
            return arr
        host_le = sys.byteorder == "little"
        target_le = self._byte_order == "<"
        src_bo = arr.dtype.byteorder
        # Normalize "=" / "|" — those mean "host" / "not applicable".
        if src_bo == "=" or src_bo == "|":
            src_is_le = host_le
        elif src_bo == "<":
            src_is_le = True
        else:
            src_is_le = False
        if src_is_le == target_le:
            return arr
        return arr.byteswap()

    def _iter_strip_segments(
        self, arr_le, h, w, rps, spp, n_strips,
    ):
        """Yield (rps, w[, spp]) ndarrays — one per strip."""
        for i in range(n_strips):
            y0 = i * rps
            y1 = min(y0 + rps, h)
            yield arr_le[y0:y1]

    def _iter_tile_segments(
        self, arr_le, h, w, tile_h, tile_w, spp,
        n_tiles_y, n_tiles_x,
    ):
        """Yield (tile_h, tile_w[, spp]) ndarrays — one per tile, padded
        with zeros for boundary tiles. TIFF requires every tile to be
        stored at the full tile dimensions even at the right/bottom
        edges (TIFF 6 §15)."""
        if spp == 1:
            pad_shape = (tile_h, tile_w)
        else:
            pad_shape = (tile_h, tile_w, spp)
        for ty in range(n_tiles_y):
            y0 = ty * tile_h
            y1 = min(y0 + tile_h, h)
            for tx in range(n_tiles_x):
                x0 = tx * tile_w
                x1 = min(x0 + tile_w, w)
                if (y1 - y0) == tile_h and (x1 - x0) == tile_w:
                    # Full tile — return a contiguous slice (no copy
                    # if input is already C-contig).
                    yield np.ascontiguousarray(arr_le[y0:y1, x0:x1])
                else:
                    # Boundary tile: copy into a zero-filled tile.
                    tile = np.zeros(pad_shape, dtype=arr_le.dtype)
                    tile[:y1 - y0, :x1 - x0] = arr_le[y0:y1, x0:x1]
                    yield tile

    def _encode_segment_bytes(
        self, seg: np.ndarray, cmp_code: int, level: int | None,
    ) -> bytes:
        """Compress one segment's bytes via segment_compression."""
        from .core.segment_compression import (
            encode_segment, JPEG, JPEG2000, JXL, WEBP, LERC, LERC_LEGACY,
        )
        if cmp_code in (JPEG, JPEG2000, JXL, WEBP, LERC, LERC_LEGACY):
            # Image-format codecs want the array (they encode shape +
            # bit-depth internally); they reject raw bytes.
            return encode_segment(seg, cmp_code, level=level)
        # Byte-stream codecs: hand them the raw buffer via memoryview.
        buf = memoryview(np.ascontiguousarray(seg)).cast("B")
        return encode_segment(buf, cmp_code, level=level)

    def _ensure_room(self, n: int) -> None:
        if self._pos + n > self._max_offset:
            raise TiffWriterError(
                "classic TIFF: writing this segment would push the file "
                "past 4 GiB; BigTIFF (64-bit offsets) is required but "
                "not yet supported by TiffWriter"
            )

    # ------------------------------------------------------------------
    # Raw-fd / buffered-fh write helpers
    # ------------------------------------------------------------------

    _HAVE_WRITEV = hasattr(os, "writev")
    _HAVE_PWRITE = hasattr(os, "pwrite")
    # IOV_MAX caps how many iovecs we can pass to writev in one call.
    # macOS = 1024, Linux = 1024 typically. Stick to half the platform
    # limit so partial-writev retries that resubmit have headroom.
    try:
        _IOV_MAX = max(64, min(512, os.sysconf("SC_IOV_MAX") // 2))
    except (ValueError, OSError, AttributeError):  # pragma: no cover - non-POSIX
        _IOV_MAX = 512

    def _write(self, data) -> None:
        """Sequential write at the current tracked position.

        ``len(ndarray)`` is the FIRST DIMENSION, not the byte count.
        Normalize ndarrays to a 1-D byte memoryview so cursor tracking
        and short-write retries see byte counts uniformly.
        """
        if isinstance(data, np.ndarray):
            view = memoryview(data).cast("B")
        elif isinstance(data, memoryview):
            view = data if data.format == "B" else data.cast("B")
        else:
            view = data  # bytes / bytearray
        nbytes = len(view)
        if self._fd >= 0:
            n = os.write(self._fd, view)
            if n != nbytes:  # pragma: no cover - short-write retry
                mv = view if isinstance(view, memoryview) else memoryview(view)
                written = n
                while written < nbytes:
                    more = os.write(self._fd, mv[written:])
                    if not more:
                        raise OSError("short write to TIFF file")
                    written += more
            self._pos += nbytes
        else:
            self._fh.write(view)
            self._pos += nbytes

    def _writev(self, buffers) -> None:
        """Scatter-gather write at the current position.

        On raw-fd (POSIX), batches into one ``os.writev`` syscall per
        chunk of up to ``_IOV_MAX`` buffers. For longer lists we issue
        multiple syscalls — still vastly fewer than one-per-buffer,
        and required because the kernel caps writev's iovec count
        (IOV_MAX=1024 on macOS / typical Linux).

        Fallback (file-like dest, or Windows raw-fd without writev):
        serial buffered writes.
        """
        if self._fd >= 0 and self._HAVE_WRITEV:
            # Normalize each buffer to a memoryview that the kernel
            # can use directly. numpy arrays cast to bytes; bytes-like
            # objects pass through.
            vs = [
                memoryview(b).cast("B") if isinstance(b, np.ndarray)
                else (memoryview(b) if not isinstance(
                    b, (bytes, bytearray, memoryview)) else b)
                for b in buffers
            ]
            cap = self._IOV_MAX
            i = 0
            while i < len(vs):
                chunk = vs[i:i + cap]
                chunk_total = sum(len(b) for b in chunk)
                n = os.writev(self._fd, chunk)  # type: ignore[attr-defined]
                if n != chunk_total:  # pragma: no cover - partial writev retry
                    remaining = n
                    for buf in chunk:
                        bl = len(buf)
                        if remaining >= bl:
                            remaining -= bl
                            continue
                        rest = memoryview(buf)[remaining:]
                        written = 0
                        while written < len(rest):
                            m_ = os.write(self._fd, rest[written:])
                            if not m_:
                                raise OSError("short writev tail")
                            written += m_
                        remaining = 0
                self._pos += chunk_total
                i += cap
        else:
            # Buffered-fh fallback (or Windows raw-fd without writev).
            for buf in buffers:
                self._write(buf)

    def _pwrite(self, offset: int, data) -> None:
        """Positional write — for back-patching the IFD chain at close
        without disturbing the sequential cursor."""
        if self._fd >= 0 and self._HAVE_PWRITE:
            os.pwrite(self._fd, data, offset)
        elif self._fd >= 0:  # pragma: no cover - non-POSIX raw-fd fallback
            saved = os.lseek(self._fd, 0, os.SEEK_CUR)
            os.lseek(self._fd, offset, os.SEEK_SET)
            os.write(self._fd, data)
            os.lseek(self._fd, saved, os.SEEK_SET)
        else:
            saved = self._fh.tell()
            self._fh.seek(offset)
            self._fh.write(data)
            self._fh.seek(saved)

    def _write_ifd(self, entries: list[_IFDEntry]) -> int:
        """Write a complete IFD block (entries + out-of-line value
        blobs + next-IFD pointer) at the current file position.

        Returns the absolute offset of the IFD start (which is what
        the previous IFD's "next" pointer slot must be patched to).

        Classic vs BigTIFF differ in:
          * IFD struct = ``entry_count_size + entry_size*N + offset_size``
          * Entry slot width (4 vs 8) controls inline-vs-out-of-line
          * count and slot fields are u32 vs u64
        """
        # Sort entries by tag ascending (TIFF 6 §2 requirement).
        entries.sort(key=lambda e: e.tag)

        bo = self._byte_order
        n_entries = len(entries)
        if not self._bigtiff and n_entries > 0xFFFF:
            raise TiffWriterError(
                f"too many tags ({n_entries}); classic TIFF caps at 65535"
            )

        # Word-align IFD start (TIFF 6 §2).
        if self._pos % 2:
            self._write(b"\x00")
        ifd_off = self._pos
        ifd_struct_size = (
            self._entry_count_size + self._entry_size * n_entries
            + self._offset_size
        )
        ext_data_off = ifd_off + ifd_struct_size

        # Decide for each entry whether its payload fits in the slot
        # (inline, ≤ self._inline_max bytes) or needs an out-of-line offset.
        entry_blocks = bytearray()
        ext_blobs: list[bytes] = []
        ext_cursor = ext_data_off
        slot_pad = self._inline_max

        for ent in entries:
            payload = ent.payload
            if len(payload) <= slot_pad:
                # Inline: right-pad to the slot width.
                slot = payload + b"\x00" * (slot_pad - len(payload))
            else:
                # Out-of-line; entry stores ext_cursor as a 4-or-8 byte uint.
                if ext_cursor > self._max_offset:
                    raise TiffWriterError(
                        f"TIFF: IFD out-of-line value at offset "
                        f"{ext_cursor} would exceed "
                        f"{'16 EiB' if self._bigtiff else '4 GiB'}"
                    )
                slot = struct.pack(bo + self._offset_fmt, ext_cursor)
                ext_blobs.append(payload)
                ext_cursor += len(payload)
                # Word-align each blob.
                if ext_cursor % 2:
                    ext_blobs.append(b"\x00")
                    ext_cursor += 1
            # Entry header: tag(2) + type(2) + count(4 or 8) + slot(4 or 8)
            if self._bigtiff:
                entry_blocks.extend(struct.pack(
                    bo + "HHQ", ent.tag, ent.type_code, ent.count,
                ))
            else:
                entry_blocks.extend(struct.pack(
                    bo + "HHI", ent.tag, ent.type_code, ent.count,
                ))
            entry_blocks.extend(slot)

        # Pack entry count + entries + next-IFD slot (placeholder 0).
        out = bytearray()
        out.extend(struct.pack(bo + self._entry_count_fmt, n_entries))
        out.extend(entry_blocks)
        # Remember slot location for next-IFD patch.
        next_slot_off = (
            ifd_off + self._entry_count_size + self._entry_size * n_entries
        )
        out.extend(struct.pack(bo + self._offset_fmt, 0))  # next-IFD = 0

        # Write IFD struct + external blobs as one scatter-gather
        # syscall (raw-fd path) or a serial sequence (file-like).
        # ``out`` is a bytearray; pass it as bytes for the iovec.
        all_buffers = [bytes(out)]
        all_buffers.extend(ext_blobs)
        self._writev(all_buffers)

        # Stash the location of the next-IFD slot so the *next* page
        # can patch it on write.
        self._next_ifd_offset_slot = next_slot_off
        return ifd_off

    def _patch_next_ifd_offset(self, this_ifd_off: int) -> None:
        """Update the previous IFD's next-pointer (or the header's
        first-IFD slot) to point to ``this_ifd_off``.

        Uses ``os.pwrite`` on the raw-fd path so we don't perturb the
        sequential cursor — one syscall instead of seek+write+seek.
        Slot width is ``self._offset_size`` (4 for classic, 8 for
        BigTIFF)."""
        if self._n_pages == 0:
            # Header's first-IFD slot: bytes 4..7 (classic) or 8..15 (BigTIFF).
            slot = self._header_first_ifd_slot
        else:
            slot = self._prev_next_slot
        self._pwrite(slot, struct.pack(
            self._byte_order + self._offset_fmt, this_ifd_off,
        ))
        # The slot we'll patch NEXT time is the next-IFD slot of the
        # IFD we just wrote. _write_ifd recorded it in self._next_ifd_offset_slot.
        self._prev_next_slot = self._next_ifd_offset_slot


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def imwrite(
    dest: str | Path | io.BufferedWriter,
    arr: np.ndarray,
    **kwargs,
) -> dict:
    """One-shot write: open, write a single page, close.

    Roughly mirrors ``tifffile.imwrite`` for the most common case::

        from opencodecs import imwrite_tiff
        imwrite_tiff("out.tif", arr, compression="zstd", tile=(256, 256))
    """
    with TiffWriter(dest) as w:
        return w.write_page(arr, **kwargs)


_PHOTOMETRIC_NAMES = {
    "miniswhite":   PHOTOMETRIC_MINISWHITE,
    "minisblack":   PHOTOMETRIC_MINISBLACK,
    "rgb":          PHOTOMETRIC_RGB,
    "palette":      PHOTOMETRIC_PALETTE,
    "transparency": PHOTOMETRIC_TRANSPARENCY,
    "cmyk":         PHOTOMETRIC_CMYK,
    "ycbcr":        PHOTOMETRIC_YCBCR,
    "cielab":       PHOTOMETRIC_CIELAB,
}


# Codecs whose decoder returns a flat byte buffer (predictor-eligible).
_BYTE_STREAM_CMP = {
    1,       # NONE
    5,       # LZW
    8,       # DEFLATE
    32773,   # PACKBITS
    32946,   # ADOBE_DEFLATE
    50000,   # ZSTD
}


__all__ = ["TiffWriter", "TiffWriterError", "imwrite"]
