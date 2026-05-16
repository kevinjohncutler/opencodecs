"""DicomRleCodec — DICOM RLE (PS 3.5 Annex G) image compression.

DICOM defines an RLE compression for image pixel data, specified in
DICOM PS 3.5 §G.3. The byte stream is:

* A 64-byte header: ``[u32 num_segments][u32 seg_offset]*15``. The
  number of segments is ``num_segments`` (1..15); subsequent
  ``seg_offset`` fields are byte offsets (from the start of this
  header) into the segment table for each segment. Unused offsets
  must be zero.
* Each segment is a PackBits-encoded byte stream (the same format
  TIFF uses for ``Compression=32773``).

The number of segments depends on the source image:

* 8-bit grayscale → 1 segment.
* 16-bit grayscale → 2 segments (high-byte plane, then low-byte plane).
* 8-bit RGB → 3 segments (R plane, G plane, B plane).
* 16-bit RGB → 6 segments (HR, LR, HG, LG, HB, LB).

Encode + decode are pure Python — the implementation is straightforward
enough that the Cython speedup isn't worth the binding complexity.
PackBits-encoded payloads tend to be small (one strip / tile at a
time in real DICOM workflows), and the I/O cost dominates.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest


# ---------------------------------------------------------------------------
# PackBits (TIFF / DICOM-RLE per-segment)
# ---------------------------------------------------------------------------


def _packbits_encode(buf: bytes) -> bytes:
    """TIFF / DICOM-flavoured PackBits encoder.

    Each emitted "packet" is either a literal run (1..128 unique
    bytes preceded by a count byte 0..127) or a replicate run (2..128
    copies of one byte preceded by a count byte 129..255 = (-(n-1) &
    0xff)). The 0x80 (-128) byte is the standard "no-op" pad marker —
    we don't emit it.
    """
    n = len(buf)
    if n == 0:
        return b""
    out = bytearray()
    i = 0
    while i < n:
        # Try to find a replicate run of length >= 3 (smallest win).
        j = i + 1
        while j < n and j - i < 128 and buf[j] == buf[i]:
            j += 1
        run_len = j - i
        if run_len >= 3:
            out.append((257 - run_len) & 0xFF)  # 257-n encodes -(n-1) in two's-complement byte
            out.append(buf[i])
            i = j
            continue
        # Otherwise gather a literal run until we hit a 3+ replicate
        # or 128 bytes.
        j = i + 1
        while j < n and j - i < 128:
            if j + 2 < n and buf[j] == buf[j + 1] == buf[j + 2]:
                break
            j += 1
        lit_len = j - i
        out.append(lit_len - 1)
        out.extend(buf[i:j])
        i = j
    return bytes(out)


def _packbits_decode(buf: bytes, expected_size: int | None = None) -> bytes:
    out = bytearray()
    i = 0
    n = len(buf)
    while i < n:
        h = buf[i]
        i += 1
        if h <= 127:  # literal run of h+1 bytes
            k = h + 1
            out.extend(buf[i:i + k])
            i += k
        elif h == 128:  # no-op
            continue
        else:  # replicate (257-h) copies
            k = 257 - h
            if i >= n:
                raise ValueError("packbits: replicate run truncated")
            out.extend(bytes([buf[i]]) * k)
            i += 1
    if expected_size is not None and len(out) != expected_size:
        raise ValueError(
            f"packbits: expected {expected_size} bytes, got {len(out)}")
    return bytes(out)


# ---------------------------------------------------------------------------
# DICOM RLE
# ---------------------------------------------------------------------------


def _dicomrle_segments_for(arr: np.ndarray) -> list[np.ndarray]:
    """Split an ndarray into the DICOM-RLE segment list.

    Layout: high-byte planes first per pixel, then low-byte planes;
    channel-major (R high, R low, G high, G low, ...) for multi-byte
    multi-channel inputs. This matches the convention every DICOM
    reader I've checked uses.
    """
    if arr.ndim == 2:
        # Grayscale.
        if arr.dtype.itemsize == 1:
            return [arr.tobytes()]
        # Multi-byte: split into byte planes, MSB first.
        u8 = arr.view(np.uint8).reshape(*arr.shape, arr.dtype.itemsize)
        return [u8[..., arr.dtype.itemsize - 1 - i].tobytes()
                for i in range(arr.dtype.itemsize)]
    elif arr.ndim == 3:
        # Multi-channel.
        h, w, c = arr.shape
        if arr.dtype.itemsize == 1:
            return [arr[..., i].tobytes() for i in range(c)]
        # Multi-byte multi-channel: per-channel MSB-first byte planes.
        u8 = arr.view(np.uint8).reshape(h, w, c, arr.dtype.itemsize)
        segments = []
        for ch in range(c):
            for byte_idx in range(arr.dtype.itemsize):
                segments.append(
                    u8[..., ch, arr.dtype.itemsize - 1 - byte_idx].tobytes()
                )
        return segments
    raise ValueError(f"dicomrle: unsupported ndim {arr.ndim}")


def _assemble_dicomrle_array(
    segments: list[bytes], shape, dtype: np.dtype
) -> np.ndarray:
    """Inverse of :func:`_dicomrle_segments_for`."""
    dt = np.dtype(dtype)
    if len(shape) == 2:
        h, w = shape
        if dt.itemsize == 1:
            if len(segments) != 1:
                raise ValueError(
                    f"dicomrle: 8-bit grayscale needs 1 segment, "
                    f"got {len(segments)}")
            return np.frombuffer(segments[0], dtype=dt).reshape(h, w).copy()
        if len(segments) != dt.itemsize:
            raise ValueError(
                f"dicomrle: {dt.itemsize}-byte grayscale needs "
                f"{dt.itemsize} segments, got {len(segments)}")
        # Reassemble byte-planes (MSB-first) into a host-endian array.
        u8 = np.empty(h * w * dt.itemsize, dtype=np.uint8).reshape(
            h, w, dt.itemsize)
        for i in range(dt.itemsize):
            u8[..., dt.itemsize - 1 - i] = np.frombuffer(
                segments[i], dtype=np.uint8).reshape(h, w)
        return u8.view(dt).reshape(h, w).copy()
    if len(shape) == 3:
        h, w, c = shape
        expected_segs = c * dt.itemsize
        if len(segments) != expected_segs:
            raise ValueError(
                f"dicomrle: {dt.itemsize}-byte × {c}-channel needs "
                f"{expected_segs} segments, got {len(segments)}")
        if dt.itemsize == 1:
            arr = np.empty((h, w, c), dtype=np.uint8)
            for ch in range(c):
                arr[..., ch] = np.frombuffer(
                    segments[ch], dtype=np.uint8).reshape(h, w)
            return arr
        u8 = np.empty((h, w, c, dt.itemsize), dtype=np.uint8)
        idx = 0
        for ch in range(c):
            for byte_idx in range(dt.itemsize):
                u8[..., ch, dt.itemsize - 1 - byte_idx] = np.frombuffer(
                    segments[idx], dtype=np.uint8).reshape(h, w)
                idx += 1
        return u8.view(dt).reshape(h, w, c).copy()
    raise ValueError(f"dicomrle: unsupported shape {shape}")


def _encode_dicomrle(arr: np.ndarray) -> bytes:
    segments = _dicomrle_segments_for(arr)
    if len(segments) > 15:
        raise ValueError(
            f"dicomrle: max 15 segments, got {len(segments)}")
    encoded = [_packbits_encode(s) for s in segments]
    # 64-byte header: num_segments + 15× offset.
    header_words = [len(segments)] + [0] * 15
    offset = 64
    for i, e in enumerate(encoded):
        header_words[1 + i] = offset
        offset += len(e)
    header = struct.pack("<16I", *header_words)
    return header + b"".join(encoded)


def _decode_dicomrle(buf: bytes, shape, dtype: np.dtype) -> np.ndarray:
    if len(buf) < 64:
        raise ValueError("dicomrle: input shorter than 64-byte header")
    fields = struct.unpack("<16I", buf[:64])
    n_segs = fields[0]
    offsets = list(fields[1:1 + n_segs])
    # Each segment ends where the next one starts (or at end-of-buffer
    # for the last segment).
    boundaries = offsets + [len(buf)]
    segments = []
    for i in range(n_segs):
        seg_bytes = buf[boundaries[i]:boundaries[i + 1]]
        decoded = _packbits_decode(seg_bytes)
        segments.append(decoded)
    return _assemble_dicomrle_array(segments, shape, dtype)


class DicomRleCodec(Codec):
    """DICOM RLE Image Compression (PS 3.5 §G)."""

    name = "dicomrle"
    aliases = ("dicom-rle", "dcmrle")
    file_extensions = ()

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.uint8, np.int8, np.uint16, np.int16)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        # No fixed magic; the 64-byte header structure (num_segments
        # in 1..15) is a weak heuristic at best.
        return False

    def encode(self, data: Any, *, dest=None, **opts) -> bytes | None:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        out = _encode_dicomrle(data)
        return _write_dest(out, dest)

    def decode(self, src: Any, *, shape, dtype, out=None,
               **opts) -> np.ndarray:
        buf = _read_src(src)
        arr = _decode_dicomrle(buf, shape, np.dtype(dtype))
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError(
                    f"dicomrle decode: out= must be an ndarray, "
                    f"got {type(out).__name__}")
            if out.shape != arr.shape or out.dtype != arr.dtype:
                raise ValueError(
                    "dicomrle decode: out= shape/dtype mismatch")
            np.copyto(out, arr)
            return out
        return arr


__all__ = ["DicomRleCodec"]
