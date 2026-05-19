"""Tile decompression for FITS compressed images (BINTABLE + ZIMAGE).

Compressed FITS images live in a ``BINTABLE`` extension HDU with
``ZIMAGE = T`` and one row per tile. The decoder:

1. Parses the BINTABLE header for tile geometry (``ZBITPIX``, ``ZNAXISn``,
   ``ZTILEn``) and compression metadata (``ZCMPTYPE``, ``ZNAMEn``,
   ``ZVALn``).
2. Reads each row's VLA descriptor (8-byte ``P`` or 16-byte ``Q``):
   element-count + heap offset. The heap holds the actual compressed
   tile bytes.
3. Decompresses each tile using the right algorithm and places it into
   the output ndarray at the tile's (row, col) position.

Supported ``ZCMPTYPE`` values:

* ``RICE_1`` — cfitsio Rice coding, int8/int16/int32 tiles.
  Calls ``_rcomp.decode_raw`` (which we shipped earlier without the
  opencodecs framing header so we can decode bare bitstreams).
* ``GZIP_1`` — zlib stream per tile. Standard.
* ``GZIP_2`` — gzip with byte-shuffling: the encoder splits N-byte
  samples into N byte streams (all high bytes, then all next-most
  bytes, ...), gzips, and the decoder reverses. Useful for floats.
* ``NOCOMPRESS`` — tiles are stored raw (no compression).

Deferred: ``HCOMPRESS_1`` (less common, lossy), ``PLIO_1`` (IRAF mask
encoding; rare outside astronomy).
"""

from __future__ import annotations

import struct
import zlib
from typing import Any

import numpy as np

from .codecs._rcomp import decode_raw as _rice_decode_raw, RcompError


_BITPIX_TO_DTYPE_SIGNED = {
    8:   np.dtype("u1"),   # FITS BITPIX=8 is unsigned per spec
    16:  np.dtype(">i2"),
    32:  np.dtype(">i4"),
    64:  np.dtype(">i8"),
    -32: np.dtype(">f4"),
    -64: np.dtype(">f8"),
}


def is_compressed_image(header: dict[str, Any]) -> bool:
    """Return True if the HDU's header describes a compressed image."""
    if header.get("XTENSION", "").strip().upper() != "BINTABLE":
        return False
    return bool(header.get("ZIMAGE"))


def _z_params(header: dict[str, Any]) -> dict[str, int]:
    """Extract ZNAMEn / ZVALn pairs into a flat dict."""
    out: dict[str, int] = {}
    i = 1
    while f"ZNAME{i}" in header:
        name = header[f"ZNAME{i}"]
        if isinstance(name, str):
            out[name] = int(header.get(f"ZVAL{i}", 0))
        i += 1
    return out


# BINTABLE scalar-column TFORM codes (FITS standard table 12): repeat
# count followed by a single-letter type. Map type → byte width.
_TFORM_TYPE_WIDTH = {
    "L": 1,   # logical
    "X": 0,   # bit (sub-byte; handled separately)
    "B": 1,   # unsigned byte
    "I": 2,   # int16
    "J": 4,   # int32
    "K": 8,   # int64
    "A": 1,   # character
    "E": 4,   # float32
    "D": 8,   # float64
    "C": 8,   # complex64
    "M": 16,  # complex128
}


def _scalar_tform_size(tform: str) -> int:
    """Byte width of a scalar (non-VLA) BINTABLE column.

    Parses ``<repeat><type>[(...)]`` where repeat defaults to 1 and
    type is a one-letter code. 'X' (bits) and 'P/Q' (VLA) aren't
    valid here — the caller filters VLA before getting here.
    """
    repeat = 0
    i = 0
    while i < len(tform) and tform[i].isdigit():
        repeat = repeat * 10 + int(tform[i])
        i += 1
    if i == 0:
        repeat = 1
    if i >= len(tform):
        return 0
    type_ch = tform[i]
    if type_ch == "X":
        # Bit columns: repeat is the bit count; round up to byte.
        return (repeat + 7) // 8
    return repeat * _TFORM_TYPE_WIDTH.get(type_ch, 0)


def _read_descriptor(buf: bytes, offset: int, is_q: bool) -> tuple[int, int]:
    """Read a single VLA descriptor (P=8 bytes, Q=16 bytes) and return
    ``(n_elements, heap_byte_offset)``. FITS BINTABLEs store these
    as big-endian int32 (P) or int64 (Q)."""
    if is_q:
        n, off = struct.unpack(">qq", buf[offset:offset + 16])
    else:
        n, off = struct.unpack(">ii", buf[offset:offset + 8])
    return int(n), int(off)


def _gzip_decompress(data: bytes) -> bytes:
    """Decompress a gzip OR zlib stream. cfitsio uses gzip for
    ZCMPTYPE='GZIP_1'; some writers tag plain zlib the same way."""
    # ``wbits=31`` accepts gzip; falling back to zlib (wbits=15) handles
    # the rarer case of zlib-headered tiles labelled GZIP_1.
    try:
        return zlib.decompress(data, wbits=31)
    except zlib.error:
        return zlib.decompress(data, wbits=15)


def _gzip2_byte_unshuffle(unshuffled: bytes, itemsize: int) -> bytes:
    """GZIP_2 stores N-byte samples with bytes split into N streams.
    For an ``itemsize=4`` array of M values, the encoded bytes are
    ``[all bytes-0, all bytes-1, all bytes-2, all bytes-3]``. The
    decoder interleaves them back into ``[v0_b0, v0_b1, v0_b2, v0_b3,
    v1_b0, ...]``. This is identical to ``numpy.transpose`` on a
    ``(itemsize, M)`` view."""
    if itemsize == 1:
        return unshuffled
    nvalues = len(unshuffled) // itemsize
    arr = np.frombuffer(unshuffled, dtype=np.uint8).reshape(itemsize, nvalues)
    return arr.T.tobytes()


def decompress_image(
    parent: "FitsStream",  # noqa: F821 -- typing forward ref
    data_offset: int,
    data_size: int,
    header: dict[str, Any],
) -> np.ndarray:
    """Decode a compressed-image BINTABLE HDU.

    ``parent`` is the FitsStream whose ``_read(offset, n)`` gives bytes
    at any offset; ``data_offset`` and ``data_size`` are the BINTABLE
    HDU's data-block bounds (the rest of this function reads from there).
    """
    zbitpix = int(header["ZBITPIX"])
    znaxis = int(header["ZNAXIS"])
    z_shape = tuple(int(header[f"ZNAXIS{i}"]) for i in range(znaxis, 0, -1))
    z_tile = tuple(
        int(header.get(f"ZTILE{i}", 1 if i > 1 else header[f"ZNAXIS{i}"]))
        for i in range(znaxis, 0, -1)
    )
    ztype = header.get("ZCMPTYPE", "").strip().upper()
    zparams = _z_params(header)

    out_dtype = _BITPIX_TO_DTYPE_SIGNED.get(zbitpix)
    if out_dtype is None:
        raise ValueError(f"compressed FITS: unsupported ZBITPIX={zbitpix}")

    # Lossy float compression: when ZBITPIX is float (-32 / -64) AND
    # ZQUANTIZ is set, tile bytes hold INT32 quantized samples (not
    # float bytes). Per-tile ZSCALE / ZZERO scalar columns hold the
    # affine reverse: ``float = int * ZSCALE + ZZERO``.
    is_quantized_float = (
        zbitpix < 0 and header.get("ZQUANTIZ") is not None
    )

    if znaxis != 2:
        raise NotImplementedError(
            f"compressed FITS: only 2-D images supported, got ZNAXIS={znaxis}"
        )

    # Tile counts per axis. The image is ZNAXIS1 × ZNAXIS2; tiles are
    # ZTILE1 × ZTILE2 (in FITS order). Last tile in each row/col may
    # be partial; tile fits into the image proper (no padding).
    h, w = z_shape           # numpy slowest-first: (rows, cols)
    th, tw = z_tile          # tile (rows, cols)
    n_tile_rows = (h + th - 1) // th
    n_tile_cols = (w + tw - 1) // tw
    expected_tiles = n_tile_rows * n_tile_cols

    # ---------- BINTABLE structure ----------
    # NAXIS1 = bytes per row, NAXIS2 = number of rows.
    # First column is COMPRESSED_DATA; we don't need the column index
    # because that column is always TFORM='1PB(n)' or '1QB(n)' — a
    # variable-length array of bytes — and we can scan TFORMn to find
    # which column it is.
    n_naxis1 = int(header["NAXIS1"])
    n_naxis2 = int(header["NAXIS2"])
    pcount = int(header.get("PCOUNT", 0))
    theap = int(header.get("THEAP", n_naxis1 * n_naxis2))
    tfields = int(header.get("TFIELDS", 0))

    # Walk all BINTABLE columns and record byte offsets of every VLA
    # descriptor we care about. Real compressed-FITS files have up to
    # three relevant variable-length columns:
    #
    #   * COMPRESSED_DATA — the primary ZCMPTYPE-encoded tile.
    #   * GZIP_COMPRESSED_DATA — fallback gzip-encoded tile that's
    #     used when the primary compressor would have made the tile
    #     *bigger* than raw bytes (incompressible-tile escape hatch).
    #   * UNCOMPRESSED_DATA — fallback for tiles stored raw.
    #
    # Plus scalar (non-VLA) columns like ``ZSCALE`` (1D) and ``ZZERO``
    # (1D) for lossy float compression. We don't apply those scale
    # factors yet but we have to walk past them to compute correct
    # offsets for any VLA columns that come after.
    comp_col_offset = -1
    gzip_col_offset = -1
    uncomp_col_offset = -1
    zscale_col_offset = -1
    zzero_col_offset = -1
    is_q = False
    gzip_is_q = False
    uncomp_is_q = False
    cursor = 0
    for col in range(1, tfields + 1):
        ttype = header.get(f"TTYPE{col}", "").strip().upper()
        tform = header.get(f"TFORM{col}", "").strip().upper()
        is_var = "P" in tform or "Q" in tform
        col_is_q = "Q" in tform
        if is_var:
            desc_size = 16 if col_is_q else 8
            if ttype == "COMPRESSED_DATA":
                comp_col_offset = cursor
                is_q = col_is_q
            elif ttype == "GZIP_COMPRESSED_DATA":
                gzip_col_offset = cursor
                gzip_is_q = col_is_q
            elif ttype == "UNCOMPRESSED_DATA":
                uncomp_col_offset = cursor
                uncomp_is_q = col_is_q
            cursor += desc_size
        else:
            # Scalar columns: ZSCALE / ZZERO carry per-tile quantization
            # parameters for lossy float compression. Other scalars are
            # walked past so subsequent VLA offsets stay accurate.
            if ttype == "ZSCALE":
                zscale_col_offset = cursor
            elif ttype == "ZZERO":
                zzero_col_offset = cursor
            cursor += _scalar_tform_size(tform)
    if comp_col_offset < 0:
        raise ValueError(
            "compressed FITS: BINTABLE has no COMPRESSED_DATA column"
        )

    if n_naxis2 != expected_tiles:
        raise ValueError(
            f"compressed FITS: BINTABLE NAXIS2={n_naxis2} != expected "
            f"tile count {expected_tiles}"
        )

    # ---------- read the entire BINTABLE + heap (one block) ----------
    # PCOUNT is the heap size in bytes. THEAP is the offset of the heap
    # start from the beginning of the data area. For the common case
    # THEAP == NAXIS1*NAXIS2, the layout is just [rows][heap].
    total_size = max(theap + pcount, n_naxis1 * n_naxis2)
    if total_size > data_size:
        raise ValueError(
            f"compressed FITS: data area {data_size} bytes is shorter "
            f"than declared THEAP+PCOUNT = {total_size}"
        )
    buf = parent._read(data_offset, total_size)
    heap_start = theap

    # ---------- allocate output, decompress + place each tile ----------
    out = np.empty((h, w), dtype=out_dtype)
    bytes_per_pixel = abs(zbitpix) // 8
    # For quantized floats the encoded samples are int32 even though
    # ZBITPIX says float — switch the bytes-per-pixel / staging dtype
    # accordingly.
    tile_dtype = out_dtype
    if is_quantized_float:
        bytes_per_pixel = 4
        tile_dtype = np.dtype(">i4")

    for row_idx in range(n_naxis2):
        # tile (tr, tc) — row-major: tr = row_idx // n_tile_cols
        tr = row_idx // n_tile_cols
        tc = row_idx % n_tile_cols
        y0 = tr * th
        y1 = min(y0 + th, h)
        x0 = tc * tw
        x1 = min(x0 + tw, w)
        tile_h = y1 - y0
        tile_w = x1 - x0
        tile_nelems = tile_h * tile_w

        row_byte = row_idx * n_naxis1
        # Primary compressed-data descriptor.
        nelems_payload, heap_off = _read_descriptor(
            buf, row_byte + comp_col_offset, is_q,
        )

        # Fallback paths: when COMPRESSED_DATA's primary descriptor has
        # count=0, real-world compressed FITS stores the tile in either
        # GZIP_COMPRESSED_DATA (lossless gzipped *original-dtype* bytes,
        # used when quantization would have hurt the tile too much) or
        # UNCOMPRESSED_DATA (literal original bytes). The fallback tile
        # is NOT quantized — when we hit one in a float-quantized
        # image, the tile decode produces the final float values
        # directly with no ZSCALE / ZZERO rescale applied.
        tile_ztype = ztype
        tile_is_quantized = is_quantized_float
        tile_decode_dtype = tile_dtype
        if nelems_payload == 0 and gzip_col_offset >= 0:
            n2, off2 = _read_descriptor(
                buf, row_byte + gzip_col_offset, gzip_is_q,
            )
            if n2 > 0:
                nelems_payload, heap_off = n2, off2
                tile_ztype = "GZIP_1"
                tile_is_quantized = False
                tile_decode_dtype = out_dtype  # raw float / int bytes
        if nelems_payload == 0 and uncomp_col_offset >= 0:
            n3, off3 = _read_descriptor(
                buf, row_byte + uncomp_col_offset, uncomp_is_q,
            )
            if n3 > 0:
                nelems_payload, heap_off = n3, off3
                tile_ztype = "NOCOMPRESS"
                tile_is_quantized = False
                tile_decode_dtype = out_dtype
        payload = bytes(buf[heap_start + heap_off
                            : heap_start + heap_off + nelems_payload])

        if tile_ztype == "RICE_1" or tile_ztype == "":
            tile_raw_u = _rice_decode_raw(
                payload,
                nelements=tile_nelems,
                blocksize=zparams.get("BLOCKSIZE", 32),
                bytes_per_pixel=zparams.get("BYTEPIX", bytes_per_pixel),
            )
            # rdecomp emits unsigned; view as the signed dtype.
            if tile_decode_dtype.kind == "i":
                tile_arr = tile_raw_u.view(tile_decode_dtype.newbyteorder("=")).copy()
            else:
                tile_arr = tile_raw_u
        elif tile_ztype in ("GZIP_1", "GZIP"):
            raw = _gzip_decompress(payload)
            tile_arr = np.frombuffer(raw, dtype=tile_decode_dtype)
            if tile_arr.size != tile_nelems:
                raise ValueError(
                    f"compressed FITS: tile {row_idx} decoded to "
                    f"{tile_arr.size} elements, expected {tile_nelems}"
                )
        elif tile_ztype == "GZIP_2":
            raw = _gzip2_byte_unshuffle(
                _gzip_decompress(payload), tile_decode_dtype.itemsize,
            )
            tile_arr = np.frombuffer(raw, dtype=tile_decode_dtype)
        elif tile_ztype == "NOCOMPRESS":
            tile_arr = np.frombuffer(payload, dtype=tile_decode_dtype)
        else:
            raise NotImplementedError(
                f"compressed FITS: ZCMPTYPE={tile_ztype!r} is not supported "
                f"yet (only RICE_1 / GZIP_1 / GZIP_2 / NOCOMPRESS)"
            )

        # Apply per-tile ZSCALE / ZZERO for quantized float compression.
        # Skip for fallback tiles (gzip / uncompressed) — those hold
        # the original float bytes already.
        if tile_is_quantized:
            scale = 1.0
            zero = 0.0
            if zscale_col_offset >= 0:
                scale = struct.unpack(
                    ">d",
                    buf[row_byte + zscale_col_offset
                        : row_byte + zscale_col_offset + 8],
                )[0]
            if zzero_col_offset >= 0:
                zero = struct.unpack(
                    ">d",
                    buf[row_byte + zzero_col_offset
                        : row_byte + zzero_col_offset + 8],
                )[0]
            float_tile = (
                tile_arr.astype(np.float64) * scale + zero
            ).astype(out_dtype)
            out[y0:y1, x0:x1] = float_tile.reshape(tile_h, tile_w)
        else:
            out[y0:y1, x0:x1] = tile_arr.reshape(tile_h, tile_w)
    return out
