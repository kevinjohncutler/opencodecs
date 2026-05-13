"""TiffCodec — native TIFF reader, no libtiff/tifffile dependency.

Walks the IFD chain via opencodecs.codecs._tiff (Cython), exposes one
TiffPage per IFD, and decodes tiles/strips through opencodecs's existing
native compression codecs.

Initial scope (Tier 5 session 1):
  * Header parse + IFD walk: TIFF 6.0 + BigTIFF
  * Tile / strip layout extraction
  * Decode for compression == NONE (Tier 5 session 2 layers in
    deflate / jpeg / lzw / packbits / zstd / jxl / lerc / jpeg2k)

The Reader API is the same as opencodecs's other multi-frame codecs:
``iter_frames()``, ``[idx]`` random access, ``read()`` materializes
the first IFD.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from .core.codec import Codec, Reader
from .core._optional_backend import import_or_stubs

(
    _tiff_parse_ifd_chain, _tiff_parse_ifd, _tiff_copy_strips,
    _tiff_check_signature,
    _tiff_packbits_decode, _tiff_lzw_decode,
    _tiff_undo_horizontal_u8, _tiff_undo_horizontal_u16,
    _tiff_undo_horizontal_u32, _tiff_undo_floating_point,
    TAG_IMAGE_WIDTH, TAG_IMAGE_LENGTH, TAG_BITS_PER_SAMPLE,
    TAG_COMPRESSION, TAG_PHOTOMETRIC,
    TAG_STRIP_OFFSETS, TAG_SAMPLES_PER_PIXEL, TAG_ROWS_PER_STRIP,
    TAG_STRIP_BYTE_COUNTS, TAG_PLANAR_CONFIG,
    TAG_TILE_WIDTH, TAG_TILE_LENGTH, TAG_TILE_OFFSETS,
    TAG_TILE_BYTE_COUNTS, TAG_SAMPLE_FORMAT,
    TAG_PREDICTOR, TAG_JPEG_TABLES, TAG_SUB_IFDS,
    CMP_NONE, CMP_DEFLATE, CMP_ADOBE_DEFLATE,
    CMP_JPEG, CMP_LZW, CMP_PACKBITS,
    CMP_ZSTD, CMP_WEBP, CMP_JXL, CMP_JPEG2000, CMP_LERC, CMP_LERC_LEGACY,
    CMP_EER_V0, CMP_EER_V1, CMP_EER_V2,
    TAG_EER_SKIPBITS, TAG_EER_HORZBITS, TAG_EER_VERTBITS,
    _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._tiff",
    "parse_ifd_chain", "parse_ifd",
    "copy_strips_from_buffer", "check_signature",
    "packbits_decode", "lzw_decode",
    "undo_horizontal_u8", "undo_horizontal_u16",
    "undo_horizontal_u32", "undo_floating_point",
    "TAG_IMAGE_WIDTH", "TAG_IMAGE_LENGTH", "TAG_BITS_PER_SAMPLE",
    "TAG_COMPRESSION", "TAG_PHOTOMETRIC",
    "TAG_STRIP_OFFSETS", "TAG_SAMPLES_PER_PIXEL", "TAG_ROWS_PER_STRIP",
    "TAG_STRIP_BYTE_COUNTS", "TAG_PLANAR_CONFIG",
    "TAG_TILE_WIDTH", "TAG_TILE_LENGTH", "TAG_TILE_OFFSETS",
    "TAG_TILE_BYTE_COUNTS", "TAG_SAMPLE_FORMAT",
    "TAG_PREDICTOR", "TAG_JPEG_TABLES", "TAG_SUB_IFDS",
    "CMP_NONE", "CMP_DEFLATE", "CMP_ADOBE_DEFLATE",
    "CMP_JPEG", "CMP_LZW", "CMP_PACKBITS",
    "CMP_ZSTD", "CMP_WEBP", "CMP_JXL", "CMP_JPEG2000",
    "CMP_LERC", "CMP_LERC_LEGACY",
    "CMP_EER_V0", "CMP_EER_V1", "CMP_EER_V2",
    "TAG_EER_SKIPBITS", "TAG_EER_HORZBITS", "TAG_EER_VERTBITS",
)


# Lazy imports of opencodecs's existing native codecs — only loaded
# when a TIFF actually uses each compression. Keeps `import opencodecs`
# fast on systems where (e.g.) libjxl isn't built.
def _decode_via(modname: str, attr: str = "decode"):
    """Look up a decode function from another opencodecs codec. Returns
    the callable, or a clear ImportError stub if the backend isn't built."""
    fn, _have = import_or_stubs(modname, attr)
    return fn[0]   # import_or_stubs returns (fn, have) tuple; fn is unwrapped


_DECODERS_CACHE: dict[str, callable] = {}

def _get_decoder(modname: str):
    fn = _DECODERS_CACHE.get(modname)
    if fn is None:
        from importlib import import_module
        try:
            mod = import_module(modname)
            fn = getattr(mod, "decode")
        except ImportError as exc:
            def _missing(*_a, _name=modname, _exc=exc, **_kw):
                raise ImportError(
                    f"TIFF: codec {_name!r} backend not built on this "
                    f"platform — needed for the compressed tile dispatch. "
                    f"({_exc})"
                )
            fn = _missing
        _DECODERS_CACHE[modname] = fn
    return fn


# ---------------------------------------------------------------------------
# Tag → scalar helper
# ---------------------------------------------------------------------------


def _tag(tags: dict, tag_id: int, default=None):
    """Return the resolved value of a tag (None if absent)."""
    e = tags.get(tag_id)
    if e is None:
        return default
    return e[2]


def _tag_seq(tags: dict, tag_id: int) -> tuple:
    """Return tag value as a tuple even if count == 1."""
    v = _tag(tags, tag_id)
    if v is None:
        return ()
    if isinstance(v, (tuple, list)):
        return tuple(v)
    return (v,)


# Sample-format codes per TIFF 6.0 §10:
#   1 = unsigned int (default)
#   2 = signed int
#   3 = IEEE float
#   4 = undefined
def _dtype_for(bits_per_sample: int, sample_format: int) -> np.dtype:
    """Return a numpy dtype for the given (bits, sample_format) pair."""
    if sample_format == 3:  # float
        if bits_per_sample == 16:
            return np.dtype(np.float16)
        if bits_per_sample == 32:
            return np.dtype(np.float32)
        if bits_per_sample == 64:
            return np.dtype(np.float64)
        raise NotImplementedError(
            f"TIFF: float dtype with {bits_per_sample} bps not supported"
        )
    if sample_format == 2:  # signed int
        if bits_per_sample == 8:
            return np.dtype(np.int8)
        if bits_per_sample == 16:
            return np.dtype(np.int16)
        if bits_per_sample == 32:
            return np.dtype(np.int32)
        if bits_per_sample == 64:
            return np.dtype(np.int64)
        raise NotImplementedError(
            f"TIFF: signed int dtype with {bits_per_sample} bps not supported"
        )
    # Default: unsigned int
    if bits_per_sample == 8:
        return np.dtype(np.uint8)
    if bits_per_sample == 16:
        return np.dtype(np.uint16)
    if bits_per_sample == 32:
        return np.dtype(np.uint32)
    if bits_per_sample == 64:
        return np.dtype(np.uint64)
    if bits_per_sample == 1:  # bilevel TIFFs — packed bits, deferred
        raise NotImplementedError(
            "TIFF: 1-bit images need packed-bit unpacking (deferred)"
        )
    raise NotImplementedError(f"TIFF: unsupported bps={bits_per_sample}")


# ---------------------------------------------------------------------------
# TiffPage — one IFD's worth of metadata + decode methods
# ---------------------------------------------------------------------------


class TiffPage:
    """One TIFF page (IFD). Attribute access exposes the common metadata.

    For raw tag access, see ``page.tags`` (dict by tag ID).
    """

    def __init__(self, stream: "TiffStream", index: int, tags: dict):
        self._stream = stream
        self.index = int(index)
        self.tags = tags

        self.width = int(_tag(self.tags, TAG_IMAGE_WIDTH, 0))
        self.height = int(_tag(self.tags, TAG_IMAGE_LENGTH, 0))
        self.compression = int(_tag(self.tags, TAG_COMPRESSION, CMP_NONE))
        self.photometric = int(_tag(self.tags, TAG_PHOTOMETRIC, 0))
        self.samples_per_pixel = int(_tag(self.tags, TAG_SAMPLES_PER_PIXEL, 1))
        self.planar_config = int(_tag(self.tags, TAG_PLANAR_CONFIG, 1))

        bps_raw = _tag(self.tags, TAG_BITS_PER_SAMPLE, 8)
        if isinstance(bps_raw, (tuple, list)):
            # All channels must have the same width for now. (Mixed-width
            # samples are exotic; deferred until we have a use case.)
            if len(set(bps_raw)) > 1:
                raise NotImplementedError(
                    f"TIFF: per-channel bps differs ({bps_raw}); not yet supported"
                )
            self.bits_per_sample = int(bps_raw[0])
        else:
            self.bits_per_sample = int(bps_raw)

        sf_raw = _tag(self.tags, TAG_SAMPLE_FORMAT, 1)
        if isinstance(sf_raw, (tuple, list)):
            if len(set(sf_raw)) > 1:
                raise NotImplementedError(
                    f"TIFF: per-channel SampleFormat differs ({sf_raw}); not yet supported"
                )
            self.sample_format = int(sf_raw[0])
        else:
            self.sample_format = int(sf_raw)

        self.dtype = _dtype_for(self.bits_per_sample, self.sample_format)

        self.predictor = int(_tag(self.tags, TAG_PREDICTOR, 1))
        # JPEGTables (tag 347) — common JPEG quantization/Huffman tables
        # shared across all tiles in a JPEG-compressed TIFF. Stitched
        # into each tile's stream before libjpeg decode.
        jt = _tag(self.tags, TAG_JPEG_TABLES)
        self.jpeg_tables = bytes(jt) if isinstance(jt, (bytes, bytearray, memoryview)) else None

        # Tile vs strip layout. A page is tiled iff TAG_TILE_WIDTH is set;
        # otherwise it's striped (rows-per-strip blocks).
        tw = _tag(self.tags, TAG_TILE_WIDTH)
        tl = _tag(self.tags, TAG_TILE_LENGTH)
        if tw is not None and tl is not None:
            self.is_tiled = True
            self.tile_width = int(tw)
            self.tile_height = int(tl)
            self.offsets = _tag_seq(self.tags, TAG_TILE_OFFSETS)
            self.byte_counts = _tag_seq(self.tags, TAG_TILE_BYTE_COUNTS)
        else:
            self.is_tiled = False
            self.tile_width = self.width
            rps = _tag(self.tags, TAG_ROWS_PER_STRIP, self.height)
            self.tile_height = int(rps)
            self.offsets = _tag_seq(self.tags, TAG_STRIP_OFFSETS)
            self.byte_counts = _tag_seq(self.tags, TAG_STRIP_BYTE_COUNTS)

        # tiles per row, total tiles
        self.tiles_x = (self.width + self.tile_width - 1) // self.tile_width
        self.tiles_y = (self.height + self.tile_height - 1) // self.tile_height

        # SubIFD offsets (TIFF tag 330) — bioformats / pyramid OME-TIFFs
        # use these to attach sub-resolution levels to a top-level IFD.
        # ``_tag`` returns a scalar for count==1 and a tuple otherwise; we
        # always want a tuple here even for a single sub-IFD.
        sub_offsets = self.tags.get(TAG_SUB_IFDS)
        if sub_offsets is None:
            self._subifd_offsets: tuple[int, ...] = ()
        else:
            v = sub_offsets[2]   # (dtype, count, value)
            if isinstance(v, (tuple, list)):
                self._subifd_offsets = tuple(int(o) for o in v)
            else:
                self._subifd_offsets = (int(v),)
        self._subifd_pages: list["TiffPage"] | None = None

    @property
    def subifds(self) -> list["TiffPage"]:
        """Sub-resolution IFDs attached to this page via tag 330.

        bioformats / OME-TIFF's preferred pyramid layout: each
        full-resolution page carries a SubIFDs tag whose values are
        offsets to additional IFDs holding the same image at lower
        resolutions. This walks the chain lazily and returns one
        :class:`TiffPage` per sub-IFD. Returns an empty list when the
        page has no SubIFDs.
        """
        if self._subifd_pages is None:
            self._subifd_pages = []
            for offset in self._subifd_offsets:
                tags, _next = _tiff_parse_ifd(
                    self._stream._read,
                    self._stream._byte_order,
                    self._stream._is_bigtiff,
                    offset,
                )
                # Indexed within parent for diagnostics; share the stream.
                sub = TiffPage(self._stream, -1, tags)
                self._subifd_pages.append(sub)
        return self._subifd_pages

    @property
    def shape(self) -> tuple:
        if self.samples_per_pixel == 1:
            return (self.height, self.width)
        return (self.height, self.width, self.samples_per_pixel)

    def __repr__(self) -> str:
        comp = _COMPRESSION_NAMES.get(self.compression, f"comp={self.compression}")
        layout = "tiled" if self.is_tiled else "striped"
        return (
            f"<TiffPage {self.index} {self.width}x{self.height} "
            f"{self.dtype.name} samples={self.samples_per_pixel} "
            f"{layout} {self.tile_width}x{self.tile_height} {comp}>"
        )

    # ----- decode -----

    def _expected_uncompressed_bytes(self) -> int:
        """Bytes one decompressed tile/strip should produce."""
        # Tiles store the FULL padded tile; strips store actual rows.
        # We compute against the padded tile for tiles, against the
        # nominal strip dimensions for strips. Either way it's the
        # value the byte-stream codecs (deflate, zstd, lzw, packbits)
        # are expected to expand to.
        h = self.tile_height
        w = self.tile_width
        return h * w * self.samples_per_pixel * self.dtype.itemsize

    def _bytes_to_array(self, raw_bytes) -> np.ndarray:
        """Interpret a flat byte buffer as our dtype, honouring file byte
        order. Used by the byte-stream codec paths (none / deflate /
        zstd / lzw / packbits)."""
        file_dtype = self.dtype.newbyteorder(self._stream._byte_order)
        arr = np.frombuffer(raw_bytes, dtype=file_dtype)
        if file_dtype.byteorder not in ("=", "|") and \
                file_dtype.byteorder != np.dtype(self.dtype).byteorder:
            arr = arr.astype(self.dtype, copy=True)
        return arr

    def _decode_segment(self, raw) -> np.ndarray:
        """Decode one tile/strip's raw bytes into a flat or shaped
        ndarray. Image-format codecs (LERC, JXL, JPEG2K, WebP, JPEG)
        return a shaped ndarray; byte-stream codecs return a flat one.
        Caller's `asarray()` knows the difference and handles cropping.
        """
        cmp = self.compression
        if cmp == CMP_NONE:
            return self._bytes_to_array(raw)

        # Byte-stream codecs: decode → bytes → frombuffer → flat array.
        if cmp in (CMP_DEFLATE, CMP_ADOBE_DEFLATE):
            decoded = _get_decoder("opencodecs.codecs._deflate")(bytes(raw))
            return self._bytes_to_array(decoded)
        if cmp == CMP_ZSTD:
            decoded = _get_decoder("opencodecs.codecs._zstd")(bytes(raw))
            return self._bytes_to_array(decoded)
        if cmp == CMP_PACKBITS:
            decoded = _tiff_packbits_decode(
                bytes(raw),
                self._expected_uncompressed_bytes(),
            )
            return self._bytes_to_array(decoded)
        if cmp == CMP_LZW:
            decoded = _tiff_lzw_decode(
                bytes(raw),
                self._expected_uncompressed_bytes(),
            )
            return self._bytes_to_array(decoded)

        # Image-format codecs: decode → already-shaped ndarray.
        if cmp == CMP_LERC or cmp == CMP_LERC_LEGACY:
            return _get_decoder("opencodecs.codecs._lerc")(bytes(raw))
        if cmp == CMP_JXL:
            return _get_decoder("opencodecs.codecs._jxl")(bytes(raw))
        if cmp == CMP_JPEG2000:
            return _get_decoder("opencodecs.codecs._jpeg2k")(bytes(raw))
        if cmp == CMP_WEBP:
            return _get_decoder("opencodecs.codecs._webp")(bytes(raw))
        if cmp == CMP_JPEG:
            return self._decode_jpeg_segment(raw)
        if cmp in (CMP_EER_V0, CMP_EER_V1, CMP_EER_V2):
            return self._decode_eer_segment(raw)

        raise NotImplementedError(
            f"TIFF compression {cmp} ({_COMPRESSION_NAMES.get(cmp, '?')}) "
            f"not implemented in opencodecs's native reader. "
            f"Use tifffile via the existing tiff_reader.py wrapper."
        )

    def _decode_eer_segment(self, raw) -> np.ndarray:
        """Decode one EER (Electron Event Representation) frame.

        Bit-field widths come from the IFD's private tags 65007/8/9
        when present, falling back to per-variant defaults documented
        in the EER spec v3 (M. Leichsenring, 2023):

          - compression 65000: skipbits=8, horzbits=2, vertbits=2
          - compression 65001: skipbits=7, horzbits=2, vertbits=2
          - compression 65002: read from tags (variant per acquisition)

        Output is a ``(H, W)`` uint8 array of event counts (binary
        per-pixel when the source isn't super-resolution).
        """
        from .codecs._eer import decode as _eer_decode
        cmp = self.compression
        if cmp == CMP_EER_V2:
            skipbits = int(_tag(self.tags, TAG_EER_SKIPBITS, 7))
            horzbits = int(_tag(self.tags, TAG_EER_HORZBITS, 2))
            vertbits = int(_tag(self.tags, TAG_EER_VERTBITS, 2))
        elif cmp == CMP_EER_V1:
            skipbits, horzbits, vertbits = 7, 2, 2
        else:  # CMP_EER_V0
            skipbits, horzbits, vertbits = 8, 2, 2
        return _eer_decode(
            bytes(raw),
            (self.height, self.width),
            skipbits, horzbits, vertbits,
            superres=0,
        )

    def _decode_jpeg_segment(self, raw) -> np.ndarray:
        """Decode a JPEG-in-TIFF tile, splicing in JPEGTables if present.

        TIFF 6 Section "JPEG compression": the JPEGTables (tag 347)
        carries a complete tables-only JPEG stream (SOI…DQT/DHT…EOI),
        and each tile carries scan data with its own SOI but no tables.
        Stitch them: take JPEGTables[:-2] (drop EOI) + tile_bytes[2:]
        (drop SOI) so libjpeg sees one self-contained stream.
        """
        tile_bytes = bytes(raw)
        if self.jpeg_tables is not None and len(self.jpeg_tables) >= 4 \
                and len(tile_bytes) >= 2 and tile_bytes[:2] == b"\xff\xd8":
            stitched = self.jpeg_tables[:-2] + tile_bytes[2:]
            return _get_decoder("opencodecs.codecs._jpeg")(stitched)
        return _get_decoder("opencodecs.codecs._jpeg")(tile_bytes)

    def _undo_predictor(self, arr: np.ndarray) -> np.ndarray:
        """Apply the inverse of TAG_PREDICTOR (tag 317) in-place when
        possible. Predictor 1 = identity (no-op)."""
        if self.predictor == 1:
            return arr
        if self.predictor == 2:
            # Horizontal differencing reverse. Cython kernels work in-
            # place on contiguous (rows, cols, samples) views.
            view = arr if arr.ndim == 3 else arr.reshape(arr.shape[0], arr.shape[1], 1)
            if not view.flags["C_CONTIGUOUS"]:
                view = np.ascontiguousarray(view)
                arr = view.reshape(arr.shape) if view is not arr else arr
            if view.dtype == np.uint8:
                _tiff_undo_horizontal_u8(view)
            elif view.dtype == np.uint16:
                _tiff_undo_horizontal_u16(view)
            elif view.dtype == np.uint32:
                _tiff_undo_horizontal_u32(view)
            elif view.dtype == np.int8:
                _tiff_undo_horizontal_u8(view.view(np.uint8))
            elif view.dtype == np.int16:
                _tiff_undo_horizontal_u16(view.view(np.uint16))
            elif view.dtype == np.int32:
                _tiff_undo_horizontal_u32(view.view(np.uint32))
            else:
                raise NotImplementedError(
                    f"TIFF predictor 2 (horizontal) for dtype "
                    f"{view.dtype} not supported"
                )
            return view if arr.ndim == 3 else view.reshape(arr.shape)
        if self.predictor == 3:
            # Floating-point predictor (TIFF Tech Note 3).
            view = arr if arr.ndim == 3 else arr.reshape(arr.shape[0], arr.shape[1], 1)
            view_u8 = np.ascontiguousarray(view).view(np.uint8) \
                .reshape(view.shape[0], view.shape[1], view.shape[2] * view.dtype.itemsize)
            _tiff_undo_floating_point(view_u8, int(view.dtype.itemsize))
            return view if arr.ndim == 3 else view.reshape(arr.shape)
        raise NotImplementedError(
            f"TIFF predictor {self.predictor} not supported"
        )

    def _segment_shape(self, tx: int, ty: int) -> tuple:
        """Pixel dimensions of segment (tx, ty) — last row/col may be cropped."""
        h = min(self.tile_height, self.height - ty * self.tile_height)
        w = min(self.tile_width,  self.width  - tx * self.tile_width)
        if self.samples_per_pixel == 1:
            return (h, w)
        return (h, w, self.samples_per_pixel)

    def _padded_shape(self) -> tuple:
        """Stored (padded) shape of one tile — tile_h × tile_w × samples."""
        if self.samples_per_pixel == 1:
            return (self.tile_height, self.tile_width)
        return (self.tile_height, self.tile_width, self.samples_per_pixel)

    def _read_strips_into(self, out: np.ndarray) -> None:
        """Copy all uncompressed strips directly into ``out``'s buffer.

        Strips in a TIFF are stored row-major: strip 0 is rows 0..rps-1,
        strip 1 is rows rps..2*rps-1, etc. Total bytes ≡ out.nbytes
        (no padding). So we can run a single memcpy per strip into the
        flat byte view of ``out``, skipping the per-strip
        frombuffer + reshape + assignment overhead.

        Caller must already have validated:
          * compression == NONE
          * is_tiled == False
          * byte order matches host (otherwise need byteswap)
        """
        view = out.view(np.uint8).reshape(-1)
        # Hot path: in-memory bytes/memoryview source. The whole TIFF
        # buffer is reachable via the read_at callable's `_buf`
        # attribute; pass it to a Cython memcpy loop and run all strip
        # copies without re-entering Python per strip.
        src_buf = getattr(self._stream._read, "_buf", None)
        if src_buf is not None:
            try:
                _tiff_copy_strips(src_buf, view, self.offsets, self.byte_counts)
                return
            except Exception:
                # Fall back to the generic path below on any unexpected
                # buffer-protocol mismatch (e.g. caller wrapped a read-
                # only memoryview).
                pass

        # Fallback: file-handle / HTTP-range / other read_at sources.
        # If the source supports read_many (HTTPDataSource / FileDataSource),
        # batch the strip fetches into one parallel + coalesced call.
        # Otherwise serial loop matches the old code exactly.
        read_many = getattr(self._stream._read, "read_many", None)
        if read_many is not None and len(self.offsets) > 1:
            ranges = [
                (int(self.offsets[i]), int(self.byte_counts[i]))
                for i in range(len(self.offsets))
            ]
            blobs = read_many(ranges)
            write_off = 0
            for raw, nbytes in zip(blobs, (r[1] for r in ranges)):
                view[write_off:write_off + nbytes] = np.frombuffer(raw, dtype=np.uint8)
                write_off += nbytes
            return

        write_off = 0
        for idx in range(len(self.offsets)):
            offset = int(self.offsets[idx])
            nbytes = int(self.byte_counts[idx])
            raw = self._stream._read(offset, nbytes)
            view[write_off:write_off + nbytes] = np.frombuffer(raw, dtype=np.uint8)
            write_off += nbytes

    def asarray(self) -> np.ndarray:
        """Fully decode this page into a 2D / 3D ndarray."""
        is_byte_stream = self.compression in (
            CMP_NONE, CMP_DEFLATE, CMP_ADOBE_DEFLATE,
            CMP_ZSTD, CMP_PACKBITS, CMP_LZW,
        )
        # Image-format codecs (LERC, JXL, JPEG, JPEG2K, WebP) return
        # already-shaped ndarrays from their own decoders.
        no_predictor = self.predictor == 1

        # Fast path: single segment covers the whole image AND no
        # post-decode steps are needed. Skips np.empty + assignment.
        if (len(self.offsets) == 1
                and self.tile_height >= self.height
                and (not self.is_tiled or self.tile_width >= self.width)
                and is_byte_stream
                and no_predictor):
            offset = int(self.offsets[0])
            nbytes = int(self.byte_counts[0])
            raw = self._stream._read(offset, nbytes)
            decoded = self._decode_segment(raw)
            if self.is_tiled:
                full_shape = self._padded_shape()
                arr = decoded.reshape(full_shape)[:self.height, :self.width]
            else:
                arr = decoded.reshape(self.shape)
            return np.ascontiguousarray(arr)

        out = np.empty(self.shape, dtype=self.dtype)

        # Fast path: uncompressed multi-strip with native byte order
        # and identity predictor — strips are row-contiguous, so we
        # can memcpy straight into out's byte buffer.
        if (not self.is_tiled
                and self.compression == CMP_NONE
                and no_predictor
                and (self.dtype.itemsize == 1 or
                     (self._stream._byte_order in ("<", "=")
                      and self.dtype.byteorder in ("<", "=", "|")))):
            self._read_strips_into(out)
            return out

        # General path: per-tile/strip decode + (optional) predictor +
        # place in out. Handles all compressions and predictors.
        if self.is_tiled:
            full_shape = self._padded_shape()

        # Batched fetch path: when the data source advertises read_many
        # (HTTPDataSource / FileDataSource), pull every segment's bytes
        # in one parallel + coalesced call so we pay 1 round-trip per
        # cluster instead of one per tile. Skip on single-segment images
        # — overhead dominates the benefit.
        prefetched: list[bytes] | None = None
        read_many = getattr(self._stream._read, "read_many", None)
        if read_many is not None and len(self.offsets) > 1:
            ranges = [
                (int(self.offsets[i]), int(self.byte_counts[i]))
                for i in range(len(self.offsets))
            ]
            prefetched = read_many(ranges)

        for ty in range(self.tiles_y):
            for tx in range(self.tiles_x):
                idx = ty * self.tiles_x + tx
                if idx >= len(self.offsets):
                    raise ValueError(
                        f"TIFF: tile index {idx} out of range "
                        f"(have {len(self.offsets)} offsets)"
                    )
                if prefetched is not None:
                    raw = prefetched[idx]
                else:
                    offset = int(self.offsets[idx])
                    nbytes = int(self.byte_counts[idx])
                    raw = self._stream._read(offset, nbytes)
                decoded = self._decode_segment(raw)
                exp_shape = self._segment_shape(tx, ty)

                if is_byte_stream:
                    # decoded is flat; reshape to padded-or-strip shape.
                    if self.is_tiled:
                        tile = decoded.reshape(full_shape)
                        if not no_predictor:
                            tile = self._undo_predictor(tile)
                        tile = tile[:exp_shape[0], :exp_shape[1]]
                    else:
                        if decoded.size != int(np.prod(exp_shape)):
                            raise ValueError(
                                f"TIFF: decoded strip ({decoded.size} elements)"
                                f" does not match expected "
                                f"({int(np.prod(exp_shape))}) for shape {exp_shape}"
                            )
                        tile = decoded.reshape(exp_shape)
                        if not no_predictor:
                            tile = self._undo_predictor(tile)
                else:
                    # Image-format codec — already-shaped ndarray.
                    # TIFF predictors don't apply (these codecs do their
                    # own prediction internally).
                    tile = decoded
                    # Some codecs (LERC) return shape (h, w) for single
                    # samples; align with TIFF's expected shape.
                    if tile.ndim == 2 and self.samples_per_pixel > 1:
                        # Reshape the rare 2D-with-multi-channel case;
                        # most image codecs return (h, w, channels).
                        pass
                    if tile.shape[:2] != exp_shape[:2]:
                        # Tiled images: codec returned padded tile; crop.
                        tile = tile[:exp_shape[0], :exp_shape[1]]

                y0 = ty * self.tile_height
                x0 = tx * self.tile_width
                out[y0:y0 + exp_shape[0], x0:x0 + exp_shape[1]] = tile
        return out


_COMPRESSION_NAMES = {
    CMP_NONE: "none",
    CMP_DEFLATE: "deflate",
    CMP_JPEG: "jpeg",
    CMP_LZW: "lzw",
    CMP_PACKBITS: "packbits",
    CMP_ZSTD: "zstd",
    CMP_WEBP: "webp",
    CMP_JXL: "jxl",
    CMP_JPEG2000: "jpeg2000",
    CMP_LERC: "lerc",
    CMP_EER_V0: "eer-v0",
    CMP_EER_V1: "eer-v1",
    CMP_EER_V2: "eer-v2",
}


# ---------------------------------------------------------------------------
# TiffStream — Reader subclass exposing iter_frames / [idx] random access
# ---------------------------------------------------------------------------


class TiffStream(Reader):
    """Reader for one TIFF file. Supports multi-page (one IFD per page).

    Construct via ``TiffCodec().open(src)`` or directly with a
    ``read_at(offset, n_bytes) -> bytes`` callable for advanced data
    sources (HTTP-range, S3, mmap)."""

    is_chunked = True

    def __init__(self, src: Any, *, read_at: Callable[[int, int], bytes] | None = None):
        self._src = src
        self._owns_fd = False

        # If ``src`` is itself a read_at callable (e.g. HTTPDataSource,
        # FileDataSource, or any user-provided wrapper that takes
        # (offset, n) → bytes), promote it to the read_at slot so the
        # caller doesn't have to spell `read_at=` explicitly.
        if read_at is None and callable(src) and not isinstance(
                src, (str, os.PathLike, bytes, bytearray, memoryview)):
            read_at = src
            self._src = None

        if read_at is not None:
            self._read = read_at
        else:
            self._read = self._open_read_at(src)

        # Lazy: walk ONLY the IFD-chain offsets at open time. Each
        # IFD's tags are resolved on first access via TiffStream.page().
        # This is what makes opening a 1000-page OME-TIFF fast.
        bo, is_bigtiff, ifd_offsets = _tiff_parse_ifd_chain(self._read)
        self._byte_order = bo
        self._is_bigtiff = is_bigtiff
        self._ifd_offsets = list(ifd_offsets)
        self._page_cache: dict[int, TiffPage] = {}
        self.n_frames = len(self._ifd_offsets)

        if not self._ifd_offsets:
            raise ValueError("TIFF: no IFDs found")

        # Populate the single-frame Reader contract from page 0.
        first = self.page(0)
        self.shape = first.shape
        self.dtype = first.dtype

    # ----- I/O ----

    def _open_read_at(self, src: Any) -> Callable[[int, int], bytes]:
        if isinstance(src, (str, os.PathLike)):
            f = open(src, "rb")
            self._owns_fd = True
            self._fd = f

            def _read(offset: int, n: int) -> bytes:
                f.seek(int(offset))
                return f.read(int(n))

            return _read

        if isinstance(src, (bytes, bytearray, memoryview)):
            # Wrap in a memoryview so slicing is zero-copy. Downstream
            # consumers (struct.unpack, np.frombuffer) all accept
            # memoryview, and the eventual decode-into-out copy is a
            # single memcpy instead of two.
            mv = memoryview(src) if not isinstance(src, memoryview) \
                else src
            # uint8 contiguous flatten so the Cython fast-path can take
            # a `const uint8_t[::1]` view without a typecode mismatch.
            if mv.format != "B":
                mv = mv.cast("B")

            def _read(offset: int, n: int):
                end = int(offset) + int(n)
                return mv[int(offset):end]

            # Stash the buffer; parse_ifd_chain looks here for its
            # zero-Python-overhead path.
            _read._buf = mv  # type: ignore[attr-defined]
            return _read

        if hasattr(src, "read") and hasattr(src, "seek"):
            def _read(offset: int, n: int) -> bytes:
                src.seek(int(offset))
                return src.read(int(n))

            return _read

        raise TypeError(
            f"TiffStream: don't know how to read from {type(src).__name__}; "
            "pass a path, bytes, file-like, or a custom read_at=..."
        )

    def close(self) -> None:
        if self._owns_fd:
            try:
                self._fd.close()
            finally:
                self._owns_fd = False

    # ----- Reader contract -----

    def page(self, index: int) -> TiffPage:
        if index < 0:
            index += self.n_frames
        if not (0 <= index < self.n_frames):
            raise IndexError(index)
        page = self._page_cache.get(index)
        if page is None:
            tags, _next = _tiff_parse_ifd(
                self._read, self._byte_order, self._is_bigtiff,
                self._ifd_offsets[index],
            )
            page = TiffPage(self, index, tags)
            self._page_cache[index] = page
        return page

    def iter_frames(self) -> Iterator[np.ndarray]:
        for i in range(self.n_frames):
            yield self.page(i).asarray()

    def __getitem__(self, idx) -> np.ndarray:
        return self.page(int(idx)).asarray()

    def read(self) -> np.ndarray:
        if self.n_frames == 1:
            return self.page(0).asarray()
        return np.stack([self.page(i).asarray() for i in range(self.n_frames)],
                        axis=0)


# ---------------------------------------------------------------------------
# TiffCodec — register under name "tiff"
# ---------------------------------------------------------------------------


class TiffCodec(Codec):
    """Native TIFF reader (no libtiff dependency)."""

    name = "tiff"
    file_extensions = (".tif", ".tiff", ".btf")
    aliases = ("bigtiff",)

    has_native = True
    has_delegate = False
    can_encode = False    # reader-only for now (encode in a future tier)
    can_decode = True
    multi_frame = True
    chunked = True
    streaming_decode = True
    parallel_decode = False  # session 1: single-threaded

    supported_dtypes = (
        np.uint8, np.int8,
        np.uint16, np.int16,
        np.uint32, np.int32,
        np.uint64, np.int64,
        np.float16, np.float32, np.float64,
    )
    supports_color = True

    def signature(self, head: bytes) -> bool:
        return _tiff_check_signature(head)

    def decode(self, src: Any, **opts) -> np.ndarray:
        with self.open(src, **opts) as r:
            return r.read()

    def open(self, src: Any, **opts) -> TiffStream:
        return TiffStream(src, **opts)


__all__ = ["TiffCodec", "TiffStream", "TiffPage"]
