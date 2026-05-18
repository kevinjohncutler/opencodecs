"""BmpCodec — native BMP encode/decode (no external library).

BMP is a small, well-documented format. The heavy lifting is row-stride
arithmetic and channel reordering, both of which numpy handles at memory
bandwidth — no need for a Cython inner loop. Header parsing is in pure
Python via struct.

Encode parity with imagecodecs:
  - 2D uint8        -> 8-bit paletted with identity grayscale palette
  - (H, W, 3) uint8 -> 24-bit BI_RGB (BGR row order, 4-byte row padding)
  - (H, W, 4) uint8 -> 32-bit BI_BITFIELDS BGRA via BITMAPV4HEADER

Decode supports the formats we actually encounter in the wild:
  - 8-bit paletted  (BI_RGB; grayscale-palette -> 2D, color-palette -> RGB)
  - 24-bit BGR      (BI_RGB)
  - 32-bit BGRA/BGRX (BI_RGB and BI_BITFIELDS with explicit channel masks)
  - 16-bit RGB555/RGB565 (BI_RGB / BI_BITFIELDS)
  - bottom-up (positive height) and top-down (negative height) layouts

Not supported: BI_RLE4/BI_RLE8, BI_JPEG/BI_PNG, OS/2 BA/CI/CP variants,
1- and 4-bit paletted. These are rare and not worth the parser surface.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest


class BmpError(RuntimeError):
    """Raised on malformed or unsupported BMP files."""


_BI_RGB = 0
_BI_BITFIELDS = 3


def _row_stride(width: int, bits_per_pixel: int) -> int:
    """BMP rows are padded to a multiple of 4 bytes."""
    return ((width * bits_per_pixel + 31) // 32) * 4


def _encode(arr: np.ndarray) -> bytes:
    if arr.dtype != np.uint8:
        raise BmpError(f'BMP encode: unsupported dtype {arr.dtype}; need uint8')

    # Pixel rows are written bottom-up; flip vertically.
    if arr.ndim == 2:
        return _encode_paletted8(arr)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return _encode_bgr24(arr)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return _encode_bgra32(arr)
    raise BmpError(
        f'BMP encode: unsupported array shape {arr.shape}; expected 2D or '
        '(H, W, 3|4)')


def _encode_paletted8(arr: np.ndarray) -> bytes:
    h, w = arr.shape
    stride = _row_stride(w, 8)
    pad = stride - w
    rows = arr[::-1]  # bottom-up
    if pad:
        rows = np.concatenate(
            [rows, np.zeros((h, pad), dtype=np.uint8)], axis=1)
    pixels = rows.tobytes()
    palette_size = 256 * 4
    palette = np.zeros((256, 4), dtype=np.uint8)
    palette[:, 0] = palette[:, 1] = palette[:, 2] = np.arange(256)  # BGRX
    palette[:, 3] = 0xFF  # reserved byte; imagecodecs sets 0xFF
    palette_bytes = palette.tobytes()

    info_size = 40
    file_header_size = 14
    pix_offset = file_header_size + info_size + palette_size
    file_size = pix_offset + len(pixels)

    file_hdr = struct.pack('<2sIHHI', b'BM', file_size, 0, 0, pix_offset)
    info_hdr = struct.pack(
        '<IiiHHIIiiII',
        info_size, w, h, 1, 8, _BI_RGB, len(pixels), 3780, 3780, 0, 0)
    return file_hdr + info_hdr + palette_bytes + pixels


def _encode_bgr24(arr: np.ndarray) -> bytes:
    h, w, _ = arr.shape
    stride = _row_stride(w, 24)
    pad = stride - 3 * w
    # Build the BGR-on-bottom-up output directly into a contiguous
    # buffer via per-channel assignment. ~3x faster than the
    # ``np.ascontiguousarray(arr[::-1, :, ::-1])`` it replaced —
    # numpy's contig-from-double-reversed-view path is unexpectedly
    # slow because both Y and channel axes are stride-reversed.
    if pad:
        bgr = np.zeros((h, stride), dtype=np.uint8)
        bgr3 = bgr[:, :3 * w].reshape(h, w, 3)
    else:
        bgr3 = np.empty((h, w, 3), dtype=np.uint8)
    bgr3[:, :, 2] = arr[::-1, :, 0]
    bgr3[:, :, 1] = arr[::-1, :, 1]
    bgr3[:, :, 0] = arr[::-1, :, 2]
    pixels = (bgr if pad else bgr3).tobytes()

    info_size = 40
    file_header_size = 14
    pix_offset = file_header_size + info_size
    file_size = pix_offset + len(pixels)

    file_hdr = struct.pack('<2sIHHI', b'BM', file_size, 0, 0, pix_offset)
    info_hdr = struct.pack(
        '<IiiHHIIiiII',
        info_size, w, h, 1, 24, _BI_RGB, len(pixels), 3780, 3780, 0, 0)
    return file_hdr + info_hdr + pixels


def _encode_bgra32(arr: np.ndarray) -> bytes:
    h, w, _ = arr.shape
    # 32-bit rows are always 4-byte aligned; no padding needed.
    # Build directly into a contig buffer to dodge the slow path
    # numpy hits on doubly-reversed-stride sources (see _encode_bgr24).
    bgra = np.empty((h, w, 4), dtype=np.uint8)
    bgra[:, :, 2] = arr[::-1, :, 0]
    bgra[:, :, 1] = arr[::-1, :, 1]
    bgra[:, :, 0] = arr[::-1, :, 2]
    bgra[:, :, 3] = arr[::-1, :, 3]
    pixels = bgra.tobytes()

    info_size = 108  # BITMAPV4HEADER
    file_header_size = 14
    pix_offset = file_header_size + info_size
    file_size = pix_offset + len(pixels)

    file_hdr = struct.pack('<2sIHHI', b'BM', file_size, 0, 0, pix_offset)
    # BITMAPV4HEADER: 40 base bytes + 4 masks + 4 cs + 36 endpoints + 12 gamma
    info_hdr = struct.pack(
        '<IiiHHIIiiII',
        info_size, w, h, 1, 32, _BI_BITFIELDS, len(pixels), 3780, 3780, 0, 0,
    )
    # Channel masks (BGRA layout in memory; little-endian DWORD pixel reads
    # as 0xAA RR GG BB so masks reflect that).
    masks = struct.pack(
        '<IIII',
        0x00FF0000,  # red
        0x0000FF00,  # green
        0x000000FF,  # blue
        0xFF000000,  # alpha
    )
    cs = struct.pack('<I', 0)  # LCS_CALIBRATED_RGB / unused
    endpoints = b'\x00' * 36
    gamma = struct.pack('<III', 0, 0, 0)
    return file_hdr + info_hdr + masks + cs + endpoints + gamma + pixels


def _decode(data: bytes) -> np.ndarray:
    if len(data) < 14 or data[:2] != b'BM':
        raise BmpError('not a BMP file (missing BM magic)')
    bf_size, _, _, pix_offset = struct.unpack('<IHHI', data[2:14])

    if len(data) < 14 + 4:
        raise BmpError('truncated BMP DIB header')
    info_size = struct.unpack('<I', data[14:18])[0]
    if info_size < 40:
        raise BmpError(f'unsupported DIB header size {info_size} (need >= 40)')

    if len(data) < 14 + info_size:
        raise BmpError('truncated DIB header')
    (
        _info_size, width, height, planes, bpp, compression,
        size_image, _xppm, _yppm, clr_used, _clr_important,
    ) = struct.unpack('<IiiHHIIiiII', data[14:54])
    del _info_size, planes, _xppm, _yppm, _clr_important

    if compression not in (_BI_RGB, _BI_BITFIELDS):
        raise BmpError(f'unsupported BMP compression {compression}')

    top_down = height < 0
    height = abs(height)
    if width <= 0 or height <= 0:
        raise BmpError(f'invalid BMP dimensions {width}x{height}')

    # Pull channel masks (either from BITMAPV4HEADER region or from the
    # 12 bytes that follow a BITMAPINFOHEADER when compression == BI_BITFIELDS).
    masks = None
    alpha_mask = 0
    if compression == _BI_BITFIELDS:
        if info_size >= 108:
            r, g, b, a = struct.unpack('<IIII', data[54:70])
            masks = (r, g, b)
            alpha_mask = a
        else:
            mask_off = 14 + info_size
            r, g, b = struct.unpack('<III', data[mask_off:mask_off + 12])
            masks = (r, g, b)
            # Some 32-bit BI_BITFIELDS files include a 4th alpha mask after.
            if bpp == 32 and len(data) >= mask_off + 16:
                alpha_mask = struct.unpack('<I', data[mask_off + 12:mask_off + 16])[0]

    palette = None
    if bpp <= 8:
        n_colors = clr_used or (1 << bpp)
        pal_off = 14 + info_size
        if compression == _BI_BITFIELDS:  # pragma: no cover - paletted+BI_BITFIELDS rare in wild
            pal_off += 16 if bpp == 32 else 12
        palette = np.frombuffer(
            data, dtype=np.uint8, count=n_colors * 4, offset=pal_off,
        ).reshape(n_colors, 4)

    stride = _row_stride(width, bpp)
    pix_data = data[pix_offset:pix_offset + stride * height]
    if len(pix_data) < stride * height:
        # Some encoders set size_image to 0; verify size from offset is enough.
        if size_image and len(data) - pix_offset >= size_image:  # pragma: no cover - rare encoder bug recovery
            pix_data = data[pix_offset:pix_offset + size_image]
        if len(pix_data) < stride * height:
            raise BmpError('truncated BMP pixel data')

    rows = np.frombuffer(pix_data, dtype=np.uint8).reshape(height, stride)

    if bpp == 8:
        idx = rows[:, :width]
        # Flip vertically unless top-down.
        if not top_down:
            idx = idx[::-1]
        # If palette is identity-grayscale (B==G==R==i), return 2D.
        bgr = palette[:, :3]
        is_gray = (
            np.array_equal(bgr[:, 0], np.arange(len(bgr), dtype=np.uint8))
            and np.array_equal(bgr[:, 1], bgr[:, 0])
            and np.array_equal(bgr[:, 2], bgr[:, 0])
        )
        if is_gray:
            return np.ascontiguousarray(idx)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = palette[idx, 2]  # R from palette[B-channel]
        rgb[..., 1] = palette[idx, 1]
        rgb[..., 2] = palette[idx, 0]
        return rgb

    if bpp == 24:
        bgr = rows[:, :3 * width].reshape(height, width, 3)
        if not top_down:
            bgr = bgr[::-1]
        return np.ascontiguousarray(bgr[..., ::-1])  # BGR -> RGB

    if bpp == 32:
        px = rows[:, :4 * width].reshape(height, width, 4)
        if not top_down:
            px = px[::-1]
        if compression == _BI_BITFIELDS and masks is not None:
            return _unpack_32_bitfields(px, masks, alpha_mask)
        # BI_RGB 32-bit: BGRX in memory; treat as RGB (drop X).
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = px[..., 2]
        rgb[..., 1] = px[..., 1]
        rgb[..., 2] = px[..., 0]
        return rgb

    if bpp == 16:
        # 16-bit pixels stored as little-endian uint16.
        px = np.frombuffer(rows[:, :2 * width].tobytes(), dtype='<u2').reshape(
            height, width)
        if not top_down:
            px = px[::-1]
        if compression == _BI_BITFIELDS and masks is not None:
            return _unpack_16_bitfields(px, masks, alpha_mask)
        # BI_RGB 16-bit is RGB555 by spec.
        return _unpack_16_bitfields(
            px, (0x7C00, 0x03E0, 0x001F), 0)

    raise BmpError(f'unsupported BMP bpp={bpp}')


def _shift_for_mask(mask: int) -> tuple[int, int]:
    """Return (shift, width_in_bits) for a non-zero mask; (0,0) if mask==0."""
    if mask == 0:
        return 0, 0
    shift = 0
    m = mask
    while m & 1 == 0:
        m >>= 1
        shift += 1
    width = 0
    while m & 1:
        m >>= 1
        width += 1
    return shift, width


def _expand_channel(value: np.ndarray, width: int) -> np.ndarray:
    """Expand a `width`-bit channel to 8 bits via top-bit replication."""
    if width >= 8:
        return (value >> (width - 8)).astype(np.uint8)
    # Bit replication: 5-bit -> 8-bit by repeating top bits.
    out = (value << (8 - width)).astype(np.uint8)
    out |= (value >> (2 * width - 8)).astype(np.uint8) if 2 * width >= 8 else 0
    return out


def _unpack_32_bitfields(
    px: np.ndarray, masks: tuple[int, int, int], alpha_mask: int,
) -> np.ndarray:
    # px is (H, W, 4) uint8 in memory order. Reinterpret as little-endian DWORD.
    h, w, _ = px.shape
    dword = np.ascontiguousarray(px).view('<u4').reshape(h, w)
    has_alpha = alpha_mask != 0
    out = np.empty((h, w, 4 if has_alpha else 3), dtype=np.uint8)
    for ch, mask in enumerate(masks):
        shift, width = _shift_for_mask(mask)
        out[..., ch] = _expand_channel((dword & mask) >> shift, width)
    if has_alpha:
        shift, width = _shift_for_mask(alpha_mask)
        out[..., 3] = _expand_channel(
            (dword & alpha_mask) >> shift, width)
    return out


def _unpack_16_bitfields(
    px: np.ndarray, masks: tuple[int, int, int], alpha_mask: int,
) -> np.ndarray:
    h, w = px.shape
    has_alpha = alpha_mask != 0
    out = np.empty((h, w, 4 if has_alpha else 3), dtype=np.uint8)
    for ch, mask in enumerate(masks):
        shift, width = _shift_for_mask(mask)
        out[..., ch] = _expand_channel((px & mask) >> shift, width)
    if has_alpha:
        shift, width = _shift_for_mask(alpha_mask)
        out[..., 3] = _expand_channel((px & alpha_mask) >> shift, width)
    return out


class BmpCodec(Codec):
    """Native BMP codec (no external library)."""

    name = "bmp"
    file_extensions = (".bmp", ".dib")

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.uint8,)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        return len(head) >= 2 and head[:2] == b'BM'

    def encode(self, data: Any, *, dest=None, **opts) -> bytes | None:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        encoded = _encode(data)
        return _write_dest(encoded, dest)

    def decode(self, src: Any, **opts) -> np.ndarray:
        return _decode(_read_src(src))



__all__ = ["BmpCodec", "BmpError"]
