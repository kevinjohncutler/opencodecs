"""FITS (Flexible Image Transport System) reader — astronomy's
century-old format, still used universally for telescope data.

FITS is fundamentally a sequence of HDUs (Header Data Units), each
consisting of an 80-char-line ASCII header + binary data, both padded
to 2880-byte boundaries. This module implements a minimal but correct
streaming reader for the most common case: primary + N image
extensions, BITPIX values 8 / 16 / 32 / 64 / -32 / -64.

Why a native reader
-------------------
astropy.io.fits is the canonical Python FITS reader, but it pulls in
astropy's entire stack (~80 MB). For pipelines that only need image
HDU pixel data, a 250-line native reader covers everything they need
and works over any ``read_at(offset, n) -> bytes`` callable — so
HTTP-range access to remote FITS files is free (telescope archives
serve them this way).

Scope of v1
-----------
* Primary HDU + N image extensions (HDU class IMAGE_HDU).
* BITPIX 8, 16, 32, 64, -32, -64 — the standard image types.
* BZERO / BSCALE rescaling applied on decode (FITS convention for
  unsigned integers stored in signed BITPIX slots).
* Auto-byteswap from big-endian (FITS is always big-endian) on host
  systems where we're little-endian.

Deferred
--------
* BinTable / ASCII table HDUs (use astropy for tables).
* Tile-compressed (RICE_1 / GZIP / PLIO_1) HDUs — would dispatch
  through the existing opencodecs codec helpers; opt-in follow-up.
* Variable-length array columns.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np


_BLOCK = 2880   # FITS spec: header + data each padded to 2880 bytes


# BITPIX → (numpy dtype, signed?, bytes-per-sample).
_BITPIX_TO_DTYPE = {
    8:   (np.dtype("u1"), False, 1),
    16:  (np.dtype(">i2"), True, 2),
    32:  (np.dtype(">i4"), True, 4),
    64:  (np.dtype(">i8"), True, 8),
    -32: (np.dtype(">f4"), False, 4),
    -64: (np.dtype(">f8"), False, 8),
}


class FitsError(RuntimeError):
    """Raised on malformed FITS input."""


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


def _parse_card(card: bytes) -> tuple[str, str | int | float | bool | None]:
    """Parse one 80-byte FITS card. Returns (keyword, value)."""
    if len(card) < 80:
        card = card + b" " * (80 - len(card))
    keyword = card[:8].decode("ascii", errors="replace").strip()
    # Cards with a value have '= ' at positions 8-9.
    if card[8:10] != b"= " and keyword not in ("END", "COMMENT", "HISTORY", ""):
        return keyword, None
    if keyword in ("END", "COMMENT", "HISTORY", ""):
        return keyword, None

    body = card[10:].decode("ascii", errors="replace")
    # Strip trailing comment (slash outside a string literal).
    in_str = False
    end = len(body)
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "'":
            in_str = not in_str
            if not in_str and i + 1 < len(body) and body[i + 1] == "'":
                # Escaped quote inside string literal.
                i += 1
        elif ch == "/" and not in_str:
            end = i
            break
        i += 1
    value_str = body[:end].strip()
    return keyword, _coerce_value(value_str)


def _coerce_value(s: str):
    s = s.strip()
    if not s:
        return None
    if s.startswith("'") and s.endswith("'"):
        # FITS string literals can have embedded '' (double single
        # quote = escaped single quote).
        return s[1:-1].replace("''", "'").rstrip()
    if s in ("T", "F"):
        return s == "T"
    # Numeric — int or float
    try:
        if "." in s or "e" in s.lower() or "d" in s.lower():
            return float(s.replace("D", "E").replace("d", "e"))
        return int(s)
    except ValueError:
        return s   # leave as raw string


def _parse_header(read_at: Callable[[int, int], bytes], offset: int) -> tuple[dict, int]:
    """Parse a FITS header starting at ``offset``. Returns
    (header_dict, end_offset_after_header) where end_offset is
    aligned to a 2880-byte boundary."""
    header: dict[str, Any] = {}
    cur = offset
    while True:
        block = read_at(cur, _BLOCK)
        if len(block) < _BLOCK:
            raise FitsError(
                f"FITS: short read on header block at offset {cur} "
                f"(got {len(block)}, expected {_BLOCK})"
            )
        cur += _BLOCK
        for i in range(0, _BLOCK, 80):
            kw, val = _parse_card(block[i:i + 80])
            if kw == "END":
                return header, cur
            if kw and kw not in ("COMMENT", "HISTORY", ""):
                # Multi-value keys (NAXIS1, NAXIS2, …) are unique;
                # plain duplicates would be a malformed FITS file.
                header[kw] = val
        # No END in this block → continue.


# ---------------------------------------------------------------------------
# HDU
# ---------------------------------------------------------------------------


class FitsHDU:
    """One Header-Data Unit in a FITS file.

    Attributes
    ----------
    header : dict
        Parsed header keywords. Common ones: NAXIS, NAXIS1, NAXIS2,
        BITPIX, BZERO, BSCALE, EXTNAME.
    shape : tuple
        Pixel shape. FITS stores NAXIS1 as the fastest-varying axis;
        we report shapes with the SLOWEST axis first (numpy/C-order).
    dtype : np.dtype
        Output dtype. uint16 for BITPIX=16 with BZERO=32768 (the
        signed→unsigned offset convention); otherwise the literal
        BITPIX type.
    data_offset : int
        File position of the start of this HDU's pixel data.
    data_size_bytes : int
        Size of the data segment (excluding the trailing 2880-byte
        padding).
    """

    def __init__(
        self,
        stream: "FitsStream",
        header: dict,
        data_offset: int,
    ):
        self._stream = stream
        self.header = header
        self.data_offset = data_offset

        naxis = int(header.get("NAXIS", 0))
        if naxis < 0 or naxis > 999:
            raise FitsError(f"FITS: invalid NAXIS={naxis}")
        bitpix = int(header.get("BITPIX", 8))
        if bitpix not in _BITPIX_TO_DTYPE:
            raise FitsError(
                f"FITS: unsupported BITPIX={bitpix} (expected one of "
                f"{sorted(_BITPIX_TO_DTYPE.keys())})"
            )
        raw_dtype, signed, bps = _BITPIX_TO_DTYPE[bitpix]
        self._raw_dtype = raw_dtype
        self._bps = bps
        self._bitpix = bitpix
        self._bzero = float(header.get("BZERO", 0.0))
        self._bscale = float(header.get("BSCALE", 1.0))

        # NAXIS1 is fastest-varying (x) in FITS. Numpy needs slowest-
        # first, so reverse: shape = (NAXISn, ..., NAXIS2, NAXIS1).
        if naxis == 0:
            self.shape: tuple[int, ...] = ()
        else:
            dims = []
            for i in range(naxis, 0, -1):
                dims.append(int(header.get(f"NAXIS{i}", 0)))
            self.shape = tuple(dims)

        self.data_size_bytes = bps
        for d in self.shape:
            self.data_size_bytes *= d

        # FITS unsigned-integer convention:
        #   BITPIX=16 BZERO=32768 BSCALE=1 → uint16
        #   BITPIX=32 BZERO=2147483648 BSCALE=1 → uint32
        #   BITPIX=64 BZERO=9223372036854775808 BSCALE=1 → uint64
        self._unsigned_int = False
        if bitpix == 16 and self._bzero == 32768.0 and self._bscale == 1.0:
            self.dtype = np.dtype("u2")
            self._unsigned_int = True
        elif bitpix == 32 and self._bzero == 2147483648.0 and self._bscale == 1.0:
            self.dtype = np.dtype("u4")
            self._unsigned_int = True
        elif bitpix == 64 and self._bzero == 9223372036854775808.0 and self._bscale == 1.0:
            self.dtype = np.dtype("u8")
            self._unsigned_int = True
        else:
            # No rescale — output is the raw BITPIX type (native byte
            # order for caller convenience).
            self.dtype = np.dtype(raw_dtype.str.replace(">", "<")
                                  if os.sys.byteorder == "little"
                                  else raw_dtype.str)

    def asarray(self) -> np.ndarray:
        """Decode the full HDU's pixel data."""
        if self.data_size_bytes == 0:
            return np.empty((0,), dtype=self.dtype)
        raw = self._stream._read_at(self.data_offset, self.data_size_bytes)
        # FITS data is always big-endian per the spec.
        arr = np.frombuffer(raw, dtype=self._raw_dtype).reshape(self.shape)
        if self._unsigned_int:
            # signed → unsigned: flip the sign bit. View the bytes as
            # the target unsigned dtype, then add 2^(N-1) modulo 2^N.
            # For uint64, naive int64 arithmetic would overflow on the
            # +2^63 shift; do it in the target unsigned dtype's
            # native modular arithmetic instead.
            target = np.dtype(self.dtype.str)
            shift = target.type(1 << (self._bps * 8 - 1))
            # arr is signed in big-endian; view as unsigned big-endian
            # of the same width, then convert to host order.
            arr_u = arr.view(self.dtype.newbyteorder(">"))
            out = (arr_u + shift).astype(self.dtype)
        elif self._bscale != 1.0 or self._bzero != 0.0:
            out = arr.astype(np.float64) * self._bscale + self._bzero
            # If output should be float32 to save memory, caller can
            # .astype it. Default keeps full precision.
        else:
            out = np.ascontiguousarray(arr).astype(self.dtype, copy=False)
        return out

    def __repr__(self) -> str:
        return (
            f"<FitsHDU shape={self.shape} dtype={self.dtype} "
            f"BITPIX={self._bitpix} BZERO={self._bzero} "
            f"BSCALE={self._bscale}>"
        )


# ---------------------------------------------------------------------------
# FitsStream — top-level reader
# ---------------------------------------------------------------------------


class FitsStream:
    """Streaming FITS reader. Walks the HDU sequence lazily.

    Accepts a path, bytes, file-like, or any ``read_at(offset, n) ->
    bytes`` callable. The callable form pairs with
    :class:`opencodecs._tiff_http.HTTPDataSource` for HTTP-range
    reads of remote FITS files (NASA / ESA / NOAO archives serve them
    this way).

    Examples
    --------
    Open a local file and read the primary HDU::

        with FitsStream("m31.fits") as f:
            arr = f.hdu(0).asarray()

    Stream over HTTPS::

        from opencodecs._tiff_http import HTTPDataSource
        src = HTTPDataSource("https://.../m31.fits")
        with FitsStream(src) as f:
            arr = f.hdu(0).asarray()
    """

    def __init__(self, src: Any, *,
                 read_at: Callable[[int, int], bytes] | None = None):
        self._src = src
        self._owns_fd = False
        # Auto-detect a callable src and promote to read_at.
        if read_at is None and callable(src) and not isinstance(
            src, (str, os.PathLike, bytes, bytearray, memoryview),
        ):
            read_at = src
            self._src = None
        if read_at is not None:
            self._read_at = read_at
        else:
            self._read_at = self._open_read_at(src)
        self._hdu_cache: list[FitsHDU | None] = []
        self._end_of_file = False

    def _open_read_at(self, src) -> Callable[[int, int], bytes]:
        if isinstance(src, (str, os.PathLike)):
            f = open(src, "rb")
            self._owns_fd = True
            self._fd = f
            def _r(o, n):
                f.seek(int(o)); return f.read(int(n))
            return _r
        if isinstance(src, (bytes, bytearray, memoryview)):
            mv = memoryview(src) if not isinstance(src, memoryview) else src
            if mv.format != "B":
                mv = mv.cast("B")
            def _r(o, n):
                return bytes(mv[int(o):int(o) + int(n)])
            return _r
        if hasattr(src, "read") and hasattr(src, "seek"):
            def _r(o, n):
                src.seek(int(o)); return src.read(int(n))
            return _r
        raise TypeError(
            f"FitsStream: don't know how to read from {type(src).__name__}"
        )

    def hdu(self, index: int) -> FitsHDU:
        """Return the ``index``-th HDU (0 = primary)."""
        while len(self._hdu_cache) <= index and not self._end_of_file:
            self._parse_next_hdu()
        if index >= len(self._hdu_cache):
            raise IndexError(f"FITS: no HDU at index {index}")
        return self._hdu_cache[index]

    @property
    def n_hdus(self) -> int:
        """Force a full HDU walk, then return the count."""
        while not self._end_of_file:
            self._parse_next_hdu()
        return len(self._hdu_cache)

    def iter_hdus(self) -> Iterator[FitsHDU]:
        i = 0
        while True:
            try:
                yield self.hdu(i)
            except IndexError:
                return
            i += 1

    def _parse_next_hdu(self) -> None:
        """Parse the next HDU's header + record its data offset."""
        if self._hdu_cache:
            prev = self._hdu_cache[-1]
            assert prev is not None
            cur = prev.data_offset + prev.data_size_bytes
            # Round up to next 2880 boundary.
            pad = (-cur) % _BLOCK
            cur += pad
        else:
            cur = 0

        # Probe: if we can't read 80 bytes, we're at EOF.
        head = self._read_at(cur, 80)
        if len(head) < 80:
            self._end_of_file = True
            return
        # Primary HDU starts with SIMPLE; extensions with XTENSION.
        kw, _ = _parse_card(head)
        if kw not in ("SIMPLE", "XTENSION"):
            self._end_of_file = True
            return
        header, data_off = _parse_header(self._read_at, cur)
        hdu = FitsHDU(self, header, data_off)
        self._hdu_cache.append(hdu)

    def close(self) -> None:
        if self._owns_fd:
            try:
                self._fd.close()
            finally:
                self._owns_fd = False

    def __enter__(self) -> "FitsStream":
        return self

    def __exit__(self, *_) -> bool:
        self.close()
        return False


def imread(path: str | Path) -> np.ndarray:
    """Convenience: read the primary HDU's pixel data as an ndarray."""
    with FitsStream(path) as f:
        return f.hdu(0).asarray()


__all__ = ["FitsStream", "FitsHDU", "FitsError", "imread"]
