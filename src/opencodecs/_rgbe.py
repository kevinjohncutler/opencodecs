"""Radiance HDR (RGBE) reader + writer.

Format
------
Greg Ward's RGBE / Radiance .hdr / .pic format. Each pixel is 4 bytes:
``(R, G, B, E)`` where the real RGB color is
``(R, G, B) * 2 ** (E - 128) / 256``. This packs an HDR floating-point
RGB into 32 bits/pixel with a shared exponent per pixel — ~12 stops
of dynamic range at 8 bits of mantissa precision.

The encoder supports two row layouts:

* **Uncompressed**: raw RGBE bytes scan-row by scan-row.
* **New-style RLE** (compression=1): per-row, each of the 4 channels
  is run-length encoded with a simple scheme (lengths >= 128 are
  literal runs; lengths < 128 are byte-fill runs). This is what
  ``radiance``'s own writer emits and what every HDR loader expects.

We implement both on read (auto-detect) and emit new-style RLE on
write (the universal convention).

This module is pure-Python — RGBE is simple enough that a 200-line
implementation is fast enough; we don't need a C extension. The
reader/writer were validated by round-tripping through Bruce Walter's
reference rgbe.c implementation when developing.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


class RgbeError(RuntimeError):
    """Raised on malformed RGBE input."""


_HEADER_TAG_RE = re.compile(rb"^(\w+)\s*=\s*(.+)$")
_RESOLUTION_RE = re.compile(rb"^([+-]Y)\s+(\d+)\s+([+-]X)\s+(\d+)\s*$")


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def _parse_header(buf: bytes) -> tuple[int, int, int, str, bool]:
    """Parse an RGBE header from ``buf``. Returns
    ``(header_size, width, height, orientation, is_rgbe)``."""
    if not buf.startswith(b"#?"):
        raise RgbeError(f"RGBE: missing magic '#?' at start of file")

    # Find blank-line terminator.
    eol = b"\n"
    lines: list[bytes] = []
    cur = 0
    while cur < len(buf):
        nl = buf.find(eol, cur)
        if nl < 0:
            raise RgbeError("RGBE: header has no blank-line terminator")
        line = buf[cur:nl]
        cur = nl + 1
        if line == b"":
            # Blank line terminates the header; next line is the
            # resolution string.
            res_nl = buf.find(eol, cur)
            if res_nl < 0:
                raise RgbeError("RGBE: missing resolution line")
            res = buf[cur:res_nl]
            cur = res_nl + 1
            m = _RESOLUTION_RE.match(res)
            if m is None:
                raise RgbeError(f"RGBE: bad resolution line {res!r}")
            ydir, h, xdir, w = m.groups()
            orientation = (ydir + b" " + xdir).decode("ascii")
            return cur, int(w), int(h), orientation, True
        lines.append(line)
    raise RgbeError("RGBE: ran off the end of the file in header")


def _decode_new_rle_row(buf: bytes, off: int, width: int) -> tuple[np.ndarray, int]:
    """Decode one row in new-style RLE format.

    Layout: a 4-byte row header (0x02 0x02 hi(width) lo(width)) tells
    the decoder this is RLE-format. Then 4 per-channel buffers (R, G,
    B, E) each encoded with: count byte (>128 = literal run len
    count-128; <=128 = fill run length count, next byte is value).
    """
    # First 4 bytes confirm RLE marker.
    if len(buf) < off + 4:
        raise RgbeError("RGBE: short read on RLE row marker")
    a, b, c, d = buf[off], buf[off + 1], buf[off + 2], buf[off + 3]
    if a != 0x02 or b != 0x02 or (c & 0x80) != 0:
        # Not RLE; uncompressed row.
        return None, off
    if ((c << 8) | d) != width:
        raise RgbeError(
            f"RGBE: RLE row width tag {(c << 8) | d} != image width {width}"
        )
    off += 4

    row = np.empty((width, 4), dtype=np.uint8)
    for ch in range(4):
        i = 0
        while i < width:
            if off >= len(buf):
                raise RgbeError("RGBE: short read inside RLE row")
            count = buf[off]
            off += 1
            if count > 128:
                # Fill: next byte repeated (count - 128) times
                count -= 128
                if i + count > width:
                    raise RgbeError("RGBE: RLE fill overflows row width")
                if off >= len(buf):
                    raise RgbeError("RGBE: short read on RLE fill value")
                val = buf[off]; off += 1
                row[i:i + count, ch] = val
                i += count
            else:
                # Literal run: next `count` bytes copied verbatim
                if i + count > width:
                    raise RgbeError("RGBE: RLE literal overflows row width")
                if off + count > len(buf):
                    raise RgbeError("RGBE: short read on RLE literal")
                row[i:i + count, ch] = np.frombuffer(
                    buf, dtype=np.uint8, count=count, offset=off,
                )
                off += count
                i += count
    return row, off


def decode(data) -> np.ndarray:
    """Decode RGBE bytes to an ``(H, W, 3)`` float32 array.

    The output is the linear-RGB color value at each pixel. Channel
    values can exceed 1.0 (that's the entire point of HDR).
    """
    buf = bytes(data)
    hdr, width, height, _, _ = _parse_header(buf)

    out = np.empty((height, width, 4), dtype=np.uint8)
    off = hdr
    for y in range(height):
        row, new_off = _decode_new_rle_row(buf, off, width)
        if row is not None:
            out[y] = row
            off = new_off
        else:
            # Uncompressed scanline (older RGBE writers) or
            # newRLE-incompatible width (<8 or >32767).
            if off + width * 4 > len(buf):
                raise RgbeError(
                    f"RGBE: short read at row {y} (uncompressed path)"
                )
            out[y] = np.frombuffer(
                buf, dtype=np.uint8, count=width * 4, offset=off,
            ).reshape(width, 4)
            off += width * 4

    # RGBE → float RGB. exp 0 means zero pixel.
    R, G, B, E = out[..., 0], out[..., 1], out[..., 2], out[..., 3]
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    nonzero = E > 0
    if nonzero.any():
        scale = (2.0 ** (E[nonzero].astype(np.float32) - 128.0)) / 256.0
        rgb[nonzero, 0] = R[nonzero].astype(np.float32) * scale
        rgb[nonzero, 1] = G[nonzero].astype(np.float32) * scale
        rgb[nonzero, 2] = B[nonzero].astype(np.float32) * scale
    return rgb


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------


def _encode_new_rle_row(row: np.ndarray) -> bytes:
    """Encode one (W, 4) row in new-style RLE format. Each channel is
    independently RLE-encoded into runs of fills (>128) and literals
    (<=128).
    """
    width = row.shape[0]
    out = bytearray()
    # 4-byte row header
    out += bytes([0x02, 0x02, (width >> 8) & 0xFF, width & 0xFF])
    for ch in range(4):
        col = row[:, ch]
        i = 0
        while i < width:
            # Find run length at i (consecutive equal bytes).
            j = i + 1
            while j < width and col[j] == col[i]:
                j += 1
            run = j - i
            if run >= 4:
                # Emit a fill run (capped at 127). 128 is illegal in
                # both fill and literal codepoints.
                while run > 0:
                    n = min(run, 127)
                    out.append(128 + n)
                    out.append(int(col[i]))
                    i += n
                    run -= n
            else:
                # Find length of literal segment ending before next
                # ≥4-long run (or end of row).
                k = j
                while k < width:
                    # Look ahead for 4 consecutive equals → end literal.
                    if k + 3 < width and (
                        col[k] == col[k + 1] == col[k + 2] == col[k + 3]
                    ):
                        break
                    k += 1
                lit_len = k - i
                while lit_len > 0:
                    n = min(lit_len, 128)
                    out.append(n)
                    out.extend(col[i:i + n].tobytes())
                    i += n
                    lit_len -= n
    return bytes(out)


def _float_rgb_to_rgbe(rgb: np.ndarray) -> np.ndarray:
    """Convert an (H, W, 3) float array to (H, W, 4) RGBE bytes.

    Per-pixel: ``v = max(R, G, B)``; if ``v < 1e-32``, output is all
    zeros. Otherwise pick exponent E so that mantissa fits in 0..255:

      E = floor(log2(v)) + 128
      m = floor(v / 2**(E-128) * 256)
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    h, w = rgb.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    vmax = rgb.max(axis=-1)
    nz = vmax > 1e-32
    if nz.any():
        # frexp returns (m, e) with 0.5 <= |m| < 1 and value = m * 2**e
        m, e = np.frexp(vmax[nz].astype(np.float64))
        # Want m * 256 in 0..255 and exponent stored as e + 128
        scale = (m * 256.0) / vmax[nz].astype(np.float64)
        out_r = (rgb[nz, 0].astype(np.float64) * scale).clip(0, 255).astype(np.uint8)
        out_g = (rgb[nz, 1].astype(np.float64) * scale).clip(0, 255).astype(np.uint8)
        out_b = (rgb[nz, 2].astype(np.float64) * scale).clip(0, 255).astype(np.uint8)
        out_e = (e + 128).clip(0, 255).astype(np.uint8)
        out[nz, 0] = out_r
        out[nz, 1] = out_g
        out[nz, 2] = out_b
        out[nz, 3] = out_e
    return out


def encode(rgb: np.ndarray, *, rle: bool = True) -> bytes:
    """Encode an (H, W, 3) float32 array as RGBE / Radiance HDR.

    ``rle=True`` (default) emits new-style RLE per row, matching
    radiance's reference writer.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RgbeError(
            f"rgbe.encode: expected (H, W, 3) array; got shape={rgb.shape}"
        )
    height, width = rgb.shape[:2]
    rgbe = _float_rgb_to_rgbe(rgb)

    out = bytearray()
    out += b"#?RADIANCE\n"
    out += b"FORMAT=32-bit_rle_rgbe\n"
    out += b"GAMMA=1.0\n"
    out += b"\n"
    # Resolution string: Y is the slower axis, descending; X faster.
    out += f"-Y {height} +X {width}\n".encode("ascii")

    if rle and 8 <= width <= 32767:
        for y in range(height):
            out += _encode_new_rle_row(rgbe[y])
    else:
        # Uncompressed fallback for width outside the RLE range.
        out += rgbe.tobytes()
    return bytes(out)


def imread(path: str | Path) -> np.ndarray:
    """Read an .hdr file as a float32 (H, W, 3) array."""
    return decode(Path(path).read_bytes())


def imwrite(path: str | Path, rgb: np.ndarray) -> None:
    """Write a float32 (H, W, 3) array as a Radiance .hdr file."""
    Path(path).write_bytes(encode(rgb))


__all__ = ["encode", "decode", "imread", "imwrite", "RgbeError"]
