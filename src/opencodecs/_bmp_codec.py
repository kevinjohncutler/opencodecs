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
  - 1-, 2-, 4- and 8-bit paletted (BI_RGB; grayscale-palette -> 2D,
    color-palette -> RGB, or as ``asrgb`` asks)
  - 24-bit BGR      (BI_RGB, and BI_BITFIELDS with explicit channel masks)
  - 32-bit BGRA/BGRX (BI_RGB and BI_BITFIELDS with explicit channel masks)
  - 16-bit RGB555/RGB565 (BI_RGB / BI_BITFIELDS)
  - bottom-up (positive height) and top-down (negative height) layouts

Channel masks (BI_BITFIELDS, BI_ALPHABITFIELDS) are read where the
header defines them: for a 40-byte BITMAPINFOHEADER, exactly three
DWORD masks follow the header (four for BI_ALPHABITFIELDS); the 52- and
56-byte Adobe headers and BITMAPV4HEADER/V5 carry them inside the
header at file offset 54, with the alpha mask from 56 bytes on. A
fourth DWORD after a 40-byte BI_BITFIELDS header is taken as an alpha
mask only when the pixel data starts after it and it is a contiguous
mask disjoint from the color masks, so pixel bytes are never read as a
mask. Masks must be contiguous (Microsoft's BITMAPV4HEADER
documentation); a gapped one raises ``BmpError``, and so do a bitfield
compression on a paletted (1- to 8-bit) image, a pixel data offset that
lies inside the masks after a 40-byte header (the pixels would be read
as masks), and a 24-bit file with a zero color mask (24-bit bitfields
are outside Microsoft's documentation, so only masks that select all
three colors are read).

A file cut short in its headers, masks, color table or uncompressed
pixel rows raises ``BmpError``. An RLE pixel stream is different,
because an encoder may stop it early: cut short anywhere but inside a
delta or an absolute run (which raise ``BmpError``), it decodes, and the
pixels it never reaches keep index 0.

A channel narrower or wider than 8 bits is scaled with the PNG
specification's reference sample-depth equation,
``round(v * 255 / (2**n - 1))``, so every width maps 0 to 0 and its
maximum to 255. imagecodecs truncates instead, so 5- and 6-bit
channels can differ from it by one.

Also decodes BI_RLE8 and BI_RLE4. Those were skipped as "rare" until
the bmpsuite conformance files went into the corpus and turned out to
contain them: they are ordinary output for paletted images, and a
decoder that rejects them is not a BMP decoder.

Not supported: BI_JPEG/BI_PNG and the OS/2 BA/CI/CP variants.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

# Cython encoder. Encode used to be pure-Python+numpy which was ~5x
# slower than imagecodecs because every encode paid two unavoidable
# MB-sized memcpys (ndarray.tobytes() + final concat). The Cython
# path writes directly into a PyBytes_FromStringAndSize buffer with
# a tight RGB->BGR loop that autovectorizes on both NEON and SSE.
_bmp_encode, _bmp_decode_bgr24, _bmp_decode_bgra32, _HAVE_BMP_ENCODE = import_or_stubs(
    "opencodecs.codecs._bmp",
    "encode", "decode_bgr24_to_rgb", "decode_bgra32_to_rgba",
)


def _copy_to_out(result: np.ndarray, out, name: str) -> np.ndarray:
    """Copy ``result`` into a caller's ``out`` as imagecodecs does: the
    dtype must match, ``out`` must be C-contiguous, and its shape must
    equal the result's apart from length-1 axes (the returned array is
    then ``out`` reshaped). Anything else raises ``ValueError`` rather
    than broadcasting or casting into ``out``."""
    if out is None:
        return result
    if not isinstance(out, np.ndarray):
        raise TypeError(f"{name} decode: out must be an ndarray")
    if out.dtype != result.dtype:
        raise ValueError(f"{name} decode: out dtype {out.dtype} is not "
                         f"the image's {result.dtype}")
    if not out.flags.c_contiguous:
        raise ValueError(f"{name} decode: out is not C-contiguous")
    view = out
    if out.shape != result.shape:
        if ([d for d in out.shape if d != 1]
                != [d for d in result.shape if d != 1]):
            raise ValueError(f"{name} decode: out shape {out.shape} does "
                             f"not hold the image's {result.shape}")
        view = out.reshape(result.shape)
    view[...] = result
    return view


class BmpError(RuntimeError):
    """Raised on malformed or unsupported BMP files."""


_BI_RGB = 0
_BI_RLE8 = 1
_BI_RLE4 = 2
_BI_BITFIELDS = 3
_BI_ALPHABITFIELDS = 6


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


# For each encode path we allocate the final output bytearray once
# and write headers + pixels directly into it. That sidesteps the two
# unnecessary memcpys the older "build pixels separately, concat at
# end" pattern paid:
#
#   * ndarray.tobytes()             (one MB-sized copy)
#   * file_hdr + info_hdr + pixels  (another, during concat)
#
# Numpy can write straight into a bytearray via np.frombuffer; the
# only memcpy that remains is the final ``bytes(out)`` immutability
# cast, which is unavoidable in pure Python (PyBytes_FromObject does
# the copy). Net 1 memcpy instead of 3.


def _encode_paletted8(arr: np.ndarray) -> bytes:
    h, w = arr.shape
    stride = _row_stride(w, 8)
    palette_size = 256 * 4
    info_size = 40
    file_header_size = 14
    pix_offset = file_header_size + info_size + palette_size
    pixels_size = h * stride
    total = pix_offset + pixels_size

    out = bytearray(total)
    struct.pack_into('<2sIHHI', out, 0, b'BM', total, 0, 0, pix_offset)
    struct.pack_into(
        '<IiiHHIIiiII', out, file_header_size,
        info_size, w, h, 1, 8, _BI_RGB, pixels_size, 3780, 3780, 0, 0,
    )
    # Grayscale palette (BGRX, with imagecodecs's 0xFF in the reserved byte).
    pal = np.frombuffer(out, dtype=np.uint8, count=palette_size,
                         offset=file_header_size + info_size).reshape(256, 4)
    pal[:, 0] = pal[:, 1] = pal[:, 2] = np.arange(256, dtype=np.uint8)
    pal[:, 3] = 0xFF
    # Pixel rows, bottom-up + row padding to stride.
    view = np.frombuffer(out, dtype=np.uint8, count=pixels_size,
                          offset=pix_offset).reshape(h, stride)
    view[:, :w] = arr[::-1]
    # Anything past column w in each row stays zero (bytearray init).
    return bytes(out)


def _encode_bgr24(arr: np.ndarray) -> bytes:
    h, w, _ = arr.shape
    stride = _row_stride(w, 24)
    info_size = 40
    file_header_size = 14
    pix_offset = file_header_size + info_size
    pixels_size = h * stride
    total = pix_offset + pixels_size

    out = bytearray(total)
    struct.pack_into('<2sIHHI', out, 0, b'BM', total, 0, 0, pix_offset)
    struct.pack_into(
        '<IiiHHIIiiII', out, file_header_size,
        info_size, w, h, 1, 24, _BI_RGB, pixels_size, 3780, 3780, 0, 0,
    )
    view = np.frombuffer(out, dtype=np.uint8, count=pixels_size,
                          offset=pix_offset).reshape(h, stride)
    bgr3 = view[:, :3 * w].reshape(h, w, 3)
    # Per-channel assignment is ~3x faster than
    # np.ascontiguousarray(arr[::-1, :, ::-1]) — numpy's slow path on
    # doubly-reversed-stride views.
    bgr3[:, :, 2] = arr[::-1, :, 0]
    bgr3[:, :, 1] = arr[::-1, :, 1]
    bgr3[:, :, 0] = arr[::-1, :, 2]
    return bytes(out)


def _encode_bgra32(arr: np.ndarray) -> bytes:
    h, w, _ = arr.shape
    # 32-bit rows are always 4-byte aligned; no padding needed.
    info_size = 108  # BITMAPV4HEADER
    file_header_size = 14
    pix_offset = file_header_size + info_size
    pixels_size = h * w * 4
    total = pix_offset + pixels_size

    out = bytearray(total)
    struct.pack_into('<2sIHHI', out, 0, b'BM', total, 0, 0, pix_offset)
    # BITMAPV4HEADER: 40 base bytes + 4 masks + 4 cs + 36 endpoints + 12 gamma
    struct.pack_into(
        '<IiiHHIIiiII', out, file_header_size,
        info_size, w, h, 1, 32, _BI_BITFIELDS, pixels_size, 3780, 3780, 0, 0,
    )
    # Channel masks at offset 14+40 = 54. Little-endian DWORD pixel
    # reads as 0xAA RR GG BB so masks reflect that.
    struct.pack_into('<IIII', out, file_header_size + 40,
                     0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    # cs (LCS_CALIBRATED_RGB / unused) at 70, endpoints (36 zero bytes)
    # at 74, gamma (12 zero bytes) at 110 — all already zero from
    # bytearray init.
    view = np.frombuffer(out, dtype=np.uint8, count=pixels_size,
                          offset=pix_offset).reshape(h, w, 4)
    view[:, :, 2] = arr[::-1, :, 0]
    view[:, :, 1] = arr[::-1, :, 1]
    view[:, :, 0] = arr[::-1, :, 2]
    view[:, :, 3] = arr[::-1, :, 3]
    return bytes(out)


def _apply_palette(idx: np.ndarray, palette: np.ndarray,
                   asrgb: bool | None = None) -> np.ndarray:
    """Map palette indices to RGB, or return 2D when the palette is gray.

    Shared by the uncompressed and RLE paths so both agree on when a
    paletted file is really a grayscale one. ``asrgb`` follows
    imagecodecs: None decides from the palette, True always expands to
    RGB, False returns the palette indices.
    """
    if asrgb is not None and not asrgb:
        return np.ascontiguousarray(idx)
    bgr = palette[:, :3]
    is_gray = (
        np.array_equal(bgr[:, 0], np.arange(len(bgr), dtype=np.uint8))
        and np.array_equal(bgr[:, 1], bgr[:, 0])
        and np.array_equal(bgr[:, 2], bgr[:, 0])
    )
    if is_gray and asrgb is None:
        return np.ascontiguousarray(idx)
    if idx.size and int(idx.max()) >= len(palette):
        # An index past a short color table shows as black rather than
        # failing the whole decode.
        padded = np.zeros((256, 4), dtype=np.uint8)
        padded[:len(palette)] = palette
        palette = padded
    rgb = np.empty(idx.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = palette[idx, 2]           # R lives in the palette's B slot
    rgb[..., 1] = palette[idx, 1]
    rgb[..., 2] = palette[idx, 0]
    return rgb


def _decode_rle(data, start, stop, width, height, four_bit):
    """Expand BI_RLE8 / BI_RLE4 pixel data to a (height, width) index raster.

    Both compressions are the same small state machine. A byte pair is
    either a run (first byte non-zero) or, when the first byte is zero,
    an escape: 0 ends the line, 1 ends the bitmap, 2 introduces a
    two-byte delta, and >= 3 starts an absolute run of that many indices
    padded to a 16-bit boundary.

    Pixels the stream never writes keep index 0. An encoder stopping
    early is legal here rather than a truncation, which is why short
    runs are filled rather than rejected.
    """
    out = np.zeros((height, width), dtype=np.uint8)
    x = y = 0
    i = start
    while i + 1 < stop:
        count = data[i]
        val = data[i + 1]
        i += 2
        if count:
            if y >= height:
                break
            n = min(count, width - x)
            if n > 0:
                if four_bit:
                    hi, lo = val >> 4, val & 0x0F
                    row = out[y]
                    for k in range(n):
                        row[x + k] = hi if k % 2 == 0 else lo
                else:
                    out[y, x:x + n] = val
            x += count
            continue
        if val == 0:                        # end of line
            x, y = 0, y + 1
        elif val == 1:                      # end of bitmap
            break
        elif val == 2:                      # delta
            if i + 1 >= stop:
                raise BmpError('truncated BMP RLE delta')
            x += data[i]
            y += data[i + 1]
            i += 2
        else:                               # absolute run
            n = val
            nbytes = (n + 1) // 2 if four_bit else n
            if i + nbytes > stop:
                raise BmpError('truncated BMP RLE absolute run')
            if y < height:
                row = out[y]
                for k in range(n):
                    if x + k >= width:
                        break
                    if four_bit:
                        b = data[i + (k >> 1)]
                        row[x + k] = (b >> 4) if k % 2 == 0 else (b & 0x0F)
                    else:
                        row[x + k] = data[i + k]
            x += n
            i += nbytes + (nbytes & 1)      # absolute runs pad to a word
    return out[::-1]                        # RLE bitmaps are always bottom-up


def _unpack_indices(rows: np.ndarray, width: int, bpp: int) -> np.ndarray:
    """Split packed 1-, 2- or 4-bit palette indices, most significant
    bits first (the leftmost pixel is in the high bits of each byte)."""
    if bpp == 1:
        return np.unpackbits(rows, axis=1)[:, :width]
    per_byte = 8 // bpp
    mask = (1 << bpp) - 1
    planes = [(rows >> (8 - bpp * (k + 1))) & mask for k in range(per_byte)]
    return np.stack(planes, axis=-1).reshape(rows.shape[0], -1)[:, :width]


def _decode(data: bytes, asrgb: bool | None = None) -> np.ndarray:
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

    if compression not in (_BI_RGB, _BI_RLE8, _BI_RLE4, _BI_BITFIELDS,
                           _BI_ALPHABITFIELDS):
        raise BmpError(f'unsupported BMP compression {compression}')
    if bpp not in (1, 2, 4, 8, 16, 24, 32):
        # Validate before anything derives a stride from it. A bogus
        # depth otherwise reaches the reshape and surfaces as a numpy
        # ValueError about impossible dimensions, which tells the caller
        # nothing about the actual problem.
        raise BmpError(f'invalid BMP bpp={bpp}')
    if compression == _BI_RLE8 and bpp != 8:
        raise BmpError(f'BI_RLE8 requires 8 bpp, got {bpp}')
    if compression == _BI_RLE4 and bpp != 4:
        raise BmpError(f'BI_RLE4 requires 4 bpp, got {bpp}')

    top_down = height < 0
    if top_down and compression in (_BI_RLE8, _BI_RLE4):
        # The spec forbids it: an RLE bitmap is bottom-up by definition,
        # so a negative height here is a malformed file rather than an
        # orientation we could honor.
        raise BmpError('top-down bitmap cannot be RLE compressed')
    height = abs(height)
    if width <= 0 or height <= 0:
        raise BmpError(f'invalid BMP dimensions {width}x{height}')

    # Channel masks. Every header from the 52-byte BITMAPV2INFOHEADER on
    # (V2 and V3 are Adobe's, V4 and V5 Microsoft's) carries R, G, B at
    # file offset 54, and from 56 bytes on also the alpha mask at 66.
    # A 40-byte BITMAPINFOHEADER has none in the header: BI_BITFIELDS
    # appends exactly three DWORD masks (Microsoft's BITMAPINFOHEADER
    # documentation), BI_ALPHABITFIELDS four. A fourth DWORD after a
    # 40-byte BI_BITFIELDS header is normally pixel data, so it is read
    # as alpha only under the narrow leniency below.
    masks = None
    alpha_mask = 0
    n_trailing_masks = 0
    if compression in (_BI_BITFIELDS, _BI_ALPHABITFIELDS):
        # Microsoft documents BI_BITFIELDS for 16 and 32 bpp only; 24 bpp
        # is accepted too, as earlier versions and Pillow read it. A
        # paletted image has no channels for masks to select, so one with
        # a bitfield compression is refused.
        if bpp not in (16, 24, 32):
            raise BmpError(
                f'bitfield compression requires 16, 24 or 32 bpp, got {bpp}')
        if info_size >= 52:
            masks = struct.unpack('<III', data[54:66])
            if info_size >= 56:
                alpha_mask = struct.unpack('<I', data[66:70])[0]
        else:
            n_trailing_masks = 4 if compression == _BI_ALPHABITFIELDS else 3
            mask_end = 14 + info_size + 4 * n_trailing_masks
            if len(data) < mask_end:
                raise BmpError('truncated BMP color masks')
            if pix_offset < mask_end:
                # The masks sit between the header and the pixel data
                # (Microsoft's BITMAPINFOHEADER documentation), so an
                # offset inside them means the file has no masks, and
                # what would be read as masks are pixels.
                raise BmpError(
                    f'BMP pixel data offset {pix_offset} lies inside the '
                    f'color masks, which end at {mask_end}')
            fields = struct.unpack(f'<{n_trailing_masks}I',
                                   data[14 + info_size:mask_end])
            masks = fields[:3]
            if n_trailing_masks == 4:
                alpha_mask = fields[3]
            elif pix_offset >= mask_end + 4 and len(data) >= mask_end + 4:
                # Leniency for writers that put an alpha mask after the
                # three: honor a fourth DWORD only when it lies wholly
                # before the pixel data (and inside the file) and is a
                # contiguous mask selecting bits the color masks do not,
                # so it can never be a pixel misread as a mask.
                extra = struct.unpack('<I', data[mask_end:mask_end + 4])[0]
                if (extra and not extra >> bpp and _contiguous(extra)
                        and not extra & (masks[0] | masks[1] | masks[2])):
                    alpha_mask = extra
        if bpp == 24:
            if not all(masks):
                # No specification covers 24-bit bitfields, imagecodecs
                # refuses them, and Pillow refuses any layout but its
                # known ones, so only masks that select every color are
                # read: a zero mask would turn real pixels black.
                raise BmpError(
                    f'24-bit BMP bitfields need a nonzero mask for each '
                    f'color, got {tuple(hex(m) for m in masks)}')
            if not alpha_mask & 0xFFFFFF:
                # A V4/V5 header's alpha mask selects the high byte of a
                # DWORD, which a 24-bit pixel does not have: no alpha.
                alpha_mask = 0
        for mask in masks + (alpha_mask,):
            if mask >> bpp:
                raise BmpError(
                    f'BMP channel mask {mask:#x} does not fit in {bpp} bits')
            # Microsoft's BITMAPV4HEADER and BITMAPV5HEADER documentation:
            # "The bits in the masks must be contiguous". A gapped mask
            # has no single bit width to scale from.
            if not _contiguous(mask):
                raise BmpError(
                    f'BMP channel mask {mask:#x} is not contiguous')

    palette = None
    if bpp <= 8:
        n_colors = min(clr_used or (1 << bpp), 1 << bpp)
        pal_off = 14 + info_size + 4 * n_trailing_masks
        if pal_off + n_colors * 4 > len(data):
            raise BmpError('truncated BMP color table')
        palette = np.frombuffer(
            data, dtype=np.uint8, count=n_colors * 4, offset=pal_off,
        ).reshape(n_colors, 4)

    if compression in (_BI_RLE8, _BI_RLE4):
        end = pix_offset + size_image if size_image else len(data)
        idx = _decode_rle(memoryview(data), pix_offset, min(end, len(data)),
                          width, height, compression == _BI_RLE4)
        return _apply_palette(idx, palette, asrgb)

    stride = _row_stride(width, bpp)
    # Use a memoryview slice (zero-copy view) instead of a bytes
    # slice (full memcpy). Saves ~1 MB of memcpy on a Kodak-sized
    # BMP decode. ``data[a:b]`` is a fresh bytes object; the
    # memoryview slice is just a stride/length update.
    data_mv = memoryview(data)
    pix_end = pix_offset + stride * height
    if pix_end > len(data):
        # The rows need stride * height bytes whatever biSizeImage says,
        # so a shorter file is truncated. (A recovery branch here used to
        # trust biSizeImage and then failed in numpy's reshape.)
        raise BmpError('truncated BMP pixel data')
    pix_data = data_mv[pix_offset:pix_end]

    rows = np.frombuffer(pix_data, dtype=np.uint8).reshape(height, stride)

    if bpp <= 8:
        if bpp == 8:
            idx = rows[:, :width]
        else:
            idx = _unpack_indices(rows, width, bpp)
        # Flip vertically unless top-down.
        if not top_down:
            idx = idx[::-1]
        return _apply_palette(idx, palette, asrgb)

    if bpp == 24 and masks is not None and (
            masks != (0xFF0000, 0xFF00, 0xFF) or alpha_mask):
        # Masks other than the plain BGR layout: widen each pixel to a
        # DWORD (high byte zero) and unpack it like a 32-bit one.
        px = np.zeros((height, width, 4), dtype=np.uint8)
        px[..., :3] = rows[:, :3 * width].reshape(height, width, 3)
        if not top_down:
            px = px[::-1]
        return _unpack_32_bitfields(px, masks, alpha_mask)

    if bpp == 24:
        # Cython fast path — beats pure-Python+numpy ~14x on a Kodak
        # photo by doing the row-flip + BGR->RGB swap in a tight C
        # loop instead of np.ascontiguousarray on a doubly-reversed-
        # stride view. Falls through to a per-channel numpy assignment
        # when the Cython extension isn't built.
        if _HAVE_BMP_ENCODE:
            return _bmp_decode_bgr24(pix_data, width, height, int(top_down))
        bgr = rows[:, :3 * width].reshape(height, width, 3)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        src = bgr if top_down else bgr[::-1]
        rgb[:, :, 0] = src[:, :, 2]
        rgb[:, :, 1] = src[:, :, 1]
        rgb[:, :, 2] = src[:, :, 0]
        return rgb

    if bpp == 32:
        # 32-bit BI_RGB / BI_BITFIELDS BGRA. Same Cython fast path
        # benefit as bpp==24.
        if masks is not None:
            # Custom channel masks need the slow path — they may not
            # be the canonical RR GG BB AA layout. Rare in practice.
            px = rows[:, :4 * width].reshape(height, width, 4)
            if not top_down:
                px = px[::-1]
            return _unpack_32_bitfields(px, masks, alpha_mask)
        if _HAVE_BMP_ENCODE:
            # 32-bit BI_RGB has a 0xFF reserved byte in alpha slot — we
            # mimic imagecodecs behavior: return (H, W, 3) RGB rather
            # than RGBX. Decode to RGBA via the Cython path, then drop
            # the alpha channel.
            rgba = _bmp_decode_bgra32(pix_data, width, height, int(top_down))
            return rgba[..., :3]
        # Fallback pure-numpy path.
        px = rows[:, :4 * width].reshape(height, width, 4)
        if not top_down:
            px = px[::-1]
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
        if masks is not None:
            return _unpack_16_bitfields(px, masks, alpha_mask)
        # BI_RGB 16-bit is RGB555 by spec.
        return _unpack_16_bitfields(
            px, (0x7C00, 0x03E0, 0x001F), 0)

    raise BmpError(f'unsupported BMP bpp={bpp}')


def _contiguous(mask: int) -> bool:
    """True for zero or a mask whose set bits form one run."""
    return not (mask + (mask & -mask)) & mask


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
    """Scale a ``width``-bit channel to 8 bits as ``round(v*255/max)``.

    This is the PNG specification's reference sample-depth equation
    (PNG section 12.4, "Sample depth scaling"); BMP itself does not say
    how to scale. It maps 0 to 0 and the channel maximum to 255 at every
    width, which the old single bit replication did not do for widths 1
    to 3 (an opaque 1-bit alpha came out as 128).
    """
    if width == 0:
        return np.zeros(value.shape, dtype=np.uint8)
    if width == 8:
        return value.astype(np.uint8)
    maxv = (1 << width) - 1
    if width <= 16:
        levels = np.arange(maxv + 1, dtype=np.uint64)
        lut = ((levels * 510 + maxv) // (2 * maxv)).astype(np.uint8)
        return lut[value]
    v = value.astype(np.uint64)
    return ((v * 510 + maxv) // (2 * maxv)).astype(np.uint8)


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

    def encode(self, data: Any, *, dest=None, ppm=None,
               **opts) -> bytes | None:
        """Encode uint8 2-D (8-bit gray paletted), RGB (24-bit) or RGBA
        (32-bit BITMAPV4HEADER). ``ppm`` sets the horizontal and
        vertical resolution in pixels per meter (an int, or an
        ``(x, y)`` pair); the default is 3780 (96 DPI), and a value
        below 1 is written as 1, as in imagecodecs."""
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        try:
            encoded = _bmp_encode(data) if _HAVE_BMP_ENCODE else _encode(data)
        except Exception as e:
            # Re-raise Cython BmpEncodeError as the wrapper's BmpError so
            # callers can catch a single exception type regardless of
            # which encoder ran. The Cython error type isn't visible to
            # tests that import BmpError from this module.
            if type(e).__name__ == "BmpEncodeError":
                raise BmpError(str(e)) from e
            raise
        if ppm is not None:
            encoded = _set_ppm(encoded, ppm)
        return _write_dest(encoded, dest)

    def decode(self, src: Any, *, asrgb: bool | None = None, out=None,
               **opts) -> np.ndarray:
        """Decode a BMP. ``asrgb`` follows imagecodecs for paletted
        images: None (default) returns 2-D for a grayscale palette and
        RGB otherwise, True always returns RGB, False returns the
        palette indices. Direct-color images ignore it. ``out`` must
        be a C-contiguous uint8 array of the decoded shape (length-1
        axes aside), else ``ValueError``, as in imagecodecs."""
        return _copy_to_out(_decode(_read_src(src), asrgb), out, "bmp")


def _set_ppm(encoded: bytes, ppm) -> bytes:
    """Write the resolution fields (BITMAPINFOHEADER offsets 24 and 28)."""
    if np.ndim(ppm) == 0:
        xppm = yppm = int(ppm)
    else:
        xppm, yppm = (int(v) for v in ppm)
    for v in (xppm, yppm):
        if not -2**31 <= v < 2**31:
            raise BmpError(f'ppm {v} does not fit a signed 32-bit field')
    # BMP gives no meaning to a resolution below one pixel per meter;
    # imagecodecs writes 1 for any such value, and so does this.
    xppm, yppm = max(xppm, 1), max(yppm, 1)
    out = bytearray(encoded)
    struct.pack_into('<ii', out, 14 + 24, xppm, yppm)
    return bytes(out)



__all__ = ["BmpCodec", "BmpError"]
