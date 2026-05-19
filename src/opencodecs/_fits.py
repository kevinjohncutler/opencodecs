"""FITS (Flexible Image Transport System) reader.

FITS is the canonical archive format for astronomy data (HST, JWST,
Vera Rubin, every major sky survey). A FITS file is a sequence of
Header Data Units (HDUs); each HDU has a 2880-byte-block ASCII
header followed by an optional binary data payload. Headers terminate
with the literal card ``END`` padded to the next 2880-byte boundary;
data payloads are likewise padded to the next 2880-byte boundary
before the next HDU starts.

The reader is HTTP-range friendly by design — at open time we only
read enough to walk the HDU chain (one Range request per HDU header).
Image data is fetched on demand when the caller accesses
``hdu.asarray()``. The same ``read_at`` callable contract that
``TiffStream`` uses works here, so plugging an ``HTTPDataSource`` in
gets you slice-on-demand reads against IRSA / MAST / IDC archives.

Scope of this iteration:

* Primary + extension image HDUs (``BITPIX`` 8 / 16 / 32 / 64 /
  -32 / -64; ``NAXIS`` 1-N).
* ``BSCALE`` / ``BZERO`` linear rescaling on read (the spec
  default of ``output = value * BSCALE + BZERO``).
* Big-endian source bytes (FITS spec mandates big-endian); converted
  to native byte order in the returned ndarray.

Out of scope here (deferred to a follow-up): compressed FITS images
(``XTENSION='BINTABLE'`` with ``ZIMAGE=T`` + ``ZCMPTYPE='RICE_1'`` /
``'GZIP_1'`` / ``'PLIO_1'``) and BINTABLE catalog rows. The Rice
and gzip codecs we already ship would let those work once we add
the tile-stitching loop on top.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterator

import numpy as np

from .core.codec import Reader


_BLOCK = 2880          # FITS header / data block size
_CARD = 80             # one header card width
_CARDS_PER_BLOCK = _BLOCK // _CARD


_BITPIX_TO_DTYPE = {
    8:   np.dtype(">u1"),
    16:  np.dtype(">i2"),
    32:  np.dtype(">i4"),
    64:  np.dtype(">i8"),
    -32: np.dtype(">f4"),
    -64: np.dtype(">f8"),
}


def _parse_value(raw: str) -> Any:
    """Convert a FITS card value field to a Python scalar.

    FITS encodes ints as plain decimals, floats with optional ``D``
    exponent marker, strings in single quotes (doubled to escape),
    and booleans as ``T`` / ``F``.
    """
    v = raw.strip()
    if not v:
        return None
    if v[0] == "'":
        # String value. Trailing single-quote marks the end; FITS
        # escapes embedded quotes by doubling, so 'O''Brien' becomes
        # "O'Brien".
        if v[-1] != "'":
            raise ValueError(f"unterminated FITS string: {raw!r}")
        return v[1:-1].replace("''", "'").rstrip()
    if v == "T":
        return True
    if v == "F":
        return False
    # Numeric. FITS sometimes uses 'D' for exponent ("1.234D5"); numpy
    # / Python only know 'E', so swap.
    v_canon = v.replace("D", "E").replace("d", "e")
    try:
        return int(v_canon)
    except ValueError:
        pass
    try:
        return float(v_canon)
    except ValueError:
        pass
    return v   # unknown — return raw string


def _parse_header(block_bytes: bytes) -> tuple[dict[str, Any], bool]:
    """Decode one 2880-byte block of header cards.

    Returns ``(header_updates, end_seen)``. ``end_seen`` is True when
    the ``END`` card appears in this block.
    """
    updates: dict[str, Any] = {}
    end_seen = False
    for i in range(_CARDS_PER_BLOCK):
        card = block_bytes[i * _CARD : (i + 1) * _CARD].decode("ascii", "replace")
        if card.startswith("END") and card[3:].strip() == "":
            end_seen = True
            break
        # Keyword is bytes 0..7 (left-justified, trailing spaces).
        # Bytes 8..9 are "= " for value cards (HISTORY / COMMENT /
        # blank don't have "="). We only care about value cards.
        key = card[:8].rstrip()
        if not key or card[8:10] != "= ":
            continue
        # Strip inline comment (anything after ``/`` not inside a string).
        body = card[10:]
        in_string = False
        comment_at = len(body)
        for j, ch in enumerate(body):
            if ch == "'":
                in_string = not in_string
            elif ch == "/" and not in_string:
                comment_at = j
                break
        updates[key] = _parse_value(body[:comment_at])
    return updates, end_seen


class FitsHDU:
    """A single Header Data Unit. Header is parsed lazily on first access."""

    def __init__(self, parent: "FitsStream", offset: int, index: int):
        self._parent = parent
        self._offset = offset      # byte offset of this HDU's start
        self.index = index
        self._header: dict[str, Any] | None = None
        self._data_offset: int | None = None
        self._data_size: int | None = None

    # ---------- header ----------

    @property
    def header(self) -> dict[str, Any]:
        if self._header is None:
            self._parse_header()
        return self._header

    def _parse_header(self) -> None:
        header: dict[str, Any] = {}
        end_seen = False
        block_idx = 0
        while not end_seen:
            block = self._parent._read(
                self._offset + block_idx * _BLOCK, _BLOCK
            )
            if len(block) < _BLOCK:
                raise ValueError(
                    f"FITS: short header read at HDU {self.index} "
                    f"(got {len(block)} bytes, expected {_BLOCK})"
                )
            updates, end_seen = _parse_header(block)
            header.update(updates)
            block_idx += 1
        self._header = header
        self._data_offset = self._offset + block_idx * _BLOCK
        self._data_size = self._payload_size_bytes()

    def _payload_size_bytes(self) -> int:
        h = self._header
        naxis = int(h.get("NAXIS", 0))
        if naxis == 0:
            return 0
        bitpix = int(h["BITPIX"])
        size = abs(bitpix) // 8
        for i in range(1, naxis + 1):
            size *= int(h[f"NAXIS{i}"])
        # GROUPS / PCOUNT for random-groups, GCOUNT for tables — most
        # image HDUs have GCOUNT=1, PCOUNT=0 and this collapses to a
        # no-op. Including these makes us safe against extension HDUs.
        gcount = int(h.get("GCOUNT", 1))
        pcount = int(h.get("PCOUNT", 0))
        size = (size + pcount * (abs(bitpix) // 8)) * gcount
        return size

    # ---------- data ----------

    @property
    def shape(self) -> tuple[int, ...]:
        h = self.header
        naxis = int(h.get("NAXIS", 0))
        if naxis == 0:
            return ()
        # FITS NAXIS1 is fastest-varying — reverse for numpy (C order).
        return tuple(int(h[f"NAXIS{i}"]) for i in range(naxis, 0, -1))

    @property
    def dtype(self) -> np.dtype:
        h = self.header
        bitpix = int(h.get("BITPIX", 0))
        if bitpix not in _BITPIX_TO_DTYPE:
            raise ValueError(f"FITS: unsupported BITPIX={bitpix}")
        return _BITPIX_TO_DTYPE[bitpix]

    @property
    def dtype(self) -> np.dtype:
        """Effective output dtype after BZERO/BSCALE interpretation.

        The FITS unsigned-int convention: BITPIX=16 with BSCALE=1 and
        BZERO=2**15 means "this is uint16 stored offset-encoded in a
        signed int16 field". Same for 32 / 64. Floats with BZERO/BSCALE
        applied always come back as float64.
        """
        h = self.header
        raw = self._raw_dtype()
        bscale = float(h.get("BSCALE", 1.0))
        bzero = float(h.get("BZERO", 0.0))
        if bscale == 1.0 and bzero == 0.0:
            return np.dtype(raw.kind + str(raw.itemsize)).newbyteorder("=")
        # Unsigned-int trick: signed ints + integer BZERO at the bias.
        if raw.kind == "i" and bscale == 1.0:
            bias = {1: 1 << 7, 2: 1 << 15, 4: 1 << 31, 8: 1 << 63}[raw.itemsize]
            if int(bzero) == bias and bzero == int(bzero):
                return np.dtype(f"u{raw.itemsize}")
        return np.dtype("f8")

    def _raw_dtype(self) -> np.dtype:
        bitpix = int(self.header.get("BITPIX", 0))
        if bitpix not in _BITPIX_TO_DTYPE:
            raise ValueError(f"FITS: unsupported BITPIX={bitpix}")
        return _BITPIX_TO_DTYPE[bitpix]

    def asarray(self) -> np.ndarray:
        """Read the HDU's data payload into a numpy ndarray.

        Applies the FITS unsigned-int convention when BSCALE=1 and
        BZERO is an integer power-of-two bias (returns uintN). Other
        non-default BSCALE/BZERO combinations return float64 after
        ``output = raw * BSCALE + BZERO``.
        """
        _ = self.header  # ensure parsed
        if self._data_size == 0:
            return np.empty((0,), dtype=np.uint8)
        raw_bytes = self._parent._read(self._data_offset, self._data_size)
        if len(raw_bytes) < self._data_size:
            raise ValueError(
                f"FITS: short data read for HDU {self.index} "
                f"({len(raw_bytes)} < {self._data_size})"
            )
        raw = np.frombuffer(raw_bytes, dtype=self._raw_dtype()).reshape(self.shape)
        h = self.header
        bscale = float(h.get("BSCALE", 1.0))
        bzero = float(h.get("BZERO", 0.0))
        out_dtype = self.dtype
        if bscale == 1.0 and bzero == 0.0:
            # Native byte order, signed-int / float passthrough.
            return raw.astype(out_dtype, copy=True)
        # FITS unsigned-int convention: signed int + integer power-of-
        # two bias = uintN. Modular addition in a wider signed type
        # (one step up from raw.itemsize) gets the bit pattern right
        # without wraparound.
        if out_dtype.kind == "u":
            # FITS unsigned trick is equivalent to flipping the sign
            # bit: signed N-bit value `x` + bias 2**(N-1) = unsigned
            # value (mod 2**N). XORing the high bit on a uintN
            # reinterpretation gives that exactly and works for
            # uint64 where ``int(bzero)`` would overflow C long.
            native = raw.byteswap().view(out_dtype) if raw.dtype.byteorder == ">" else raw.view(out_dtype)
            return (native ^ np.array(1 << (out_dtype.itemsize * 8 - 1),
                                       dtype=out_dtype)).copy()
        return raw.astype(np.float64) * bscale + bzero

    @property
    def total_bytes(self) -> int:
        """Header bytes + data-payload bytes, rounded up to a 2880-byte
        boundary (so the next HDU starts at this HDU's offset +
        total_bytes)."""
        _ = self.header
        data_start = self._data_offset - self._offset
        pad = (-self._data_size) % _BLOCK if self._data_size else 0
        return data_start + self._data_size + pad


class FitsStream(Reader):
    """Reader for one FITS file. One frame per image HDU.

    Construct via ``FitsCodec().open(src)`` or directly with a
    ``read_at(offset, n_bytes) -> bytes`` callable (HTTPDataSource,
    file-like, etc.).
    """

    is_chunked = True

    def __init__(self, src: Any, *, read_at: Callable[[int, int], bytes] | None = None):
        self._src = src

        if read_at is None and callable(src) and not isinstance(
                src, (str, os.PathLike, bytes, bytearray, memoryview)):
            read_at = src
            self._src = None

        if read_at is not None:
            self._read = read_at
            self._owns_fd = False
        else:
            self._read, self._owns_fd = self._open_read_at(src)

        # Probe for FITS magic. If the file doesn't look like FITS we
        # quietly accept it with zero HDUs — matches the contract the
        # existing test suite asserts (``FitsStream(garbage)`` opens
        # cleanly but ``f.n_hdus == 0``).
        try:
            head = self._read(0, _CARD)
        except (OSError, ValueError):
            head = b""
        is_fits = (
            head.startswith(b"SIMPLE  = ") or head.startswith(b"XTENSION= ")
        )

        # Walk HDU chain by reading each header to discover its payload
        # size, then jumping to the next HDU offset. For HTTP-backed
        # reads this is one (small) Range request per HDU header.
        self._hdus: list[FitsHDU] = []
        if is_fits:
            offset = 0
            while True:
                hdu = FitsHDU(self, offset, index=len(self._hdus))
                try:
                    _ = hdu.header
                except (ValueError, KeyError):
                    break
                self._hdus.append(hdu)
                offset += hdu.total_bytes
                try:
                    probe = self._read(offset, 1)
                except (OSError, ValueError):
                    break
                if not probe:
                    break

        self.n_hdus = len(self._hdus)
        self.n_frames = self.n_hdus
        # Optional: a zero-HDU stream (garbage input) is still a valid
        # FitsStream — callers see n_hdus=0 and skip iteration. Skip
        # the primary-HDU shape population in that case.
        if not self._hdus:
            self.shape = ()
            self.dtype = np.dtype("u1")
            return
        primary = self._hdus[0]
        self.shape = primary.shape
        # primary may have NAXIS=0 (header-only); use first data-bearing
        # HDU for the Reader's dtype contract instead.
        first_data = next((h for h in self._hdus if h.header.get("NAXIS", 0)),
                          primary)
        self.dtype = first_data.dtype if first_data.shape else np.dtype("u1")

    # ---------- I/O ----------

    def _open_read_at(self, src: Any) -> tuple[Callable[[int, int], bytes], bool]:
        if isinstance(src, (str, os.PathLike)):
            f = open(src, "rb")

            def read_at(off: int, n: int, _f=f) -> bytes:
                _f.seek(off)
                return _f.read(n)
            return read_at, True
        if isinstance(src, (bytes, bytearray, memoryview)):
            buf = bytes(src)

            def read_at(off: int, n: int, _b=buf) -> bytes:
                return _b[off : off + n]
            return read_at, False
        if hasattr(src, "read") and hasattr(src, "seek"):
            f = src

            def read_at(off: int, n: int, _f=f) -> bytes:
                _f.seek(off)
                return _f.read(n)
            return read_at, False
        raise TypeError(
            f"FITS: unsupported src type {type(src).__name__}; "
            f"pass a path, bytes, file-like, or read_at callable"
        )

    def close(self) -> None:
        if self._owns_fd and self._src is not None:
            # ``self._src`` is a path; the file-handle is captured inside
            # the read_at closure. Close it via the closure's __closure__.
            for cell in (getattr(self._read, "__closure__", None) or ()):
                contents = cell.cell_contents
                if hasattr(contents, "close"):
                    try:
                        contents.close()
                    except OSError:
                        pass
                    break
            self._owns_fd = False

    def __enter__(self) -> "FitsStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---------- Reader contract ----------

    def hdu(self, i: int) -> FitsHDU:
        return self._hdus[i]

    def iter_frames(self) -> Iterator[np.ndarray]:
        for h in self._hdus:
            if h.header.get("NAXIS", 0):
                yield h.asarray()

    def __getitem__(self, idx) -> np.ndarray:
        return self._hdus[idx].asarray()

    def read(self) -> np.ndarray:
        """Read the primary HDU's image data (or the first data-bearing
        HDU when the primary is header-only)."""
        for h in self._hdus:
            if h.header.get("NAXIS", 0):
                return h.asarray()
        raise ValueError("FITS: no image data in this file")


def imread(src: Any) -> np.ndarray:
    """One-shot decode: open ``src``, return the primary HDU's image data
    (or the first data-bearing HDU when the primary is header-only)."""
    with FitsStream(src) as fs:
        return fs.read()


def imwrite(*args, **kwargs):
    """FITS encode is not implemented in opencodecs. Use ``astropy.io.fits``
    or another writer for FITS output; we ship reader-only since the
    Tier 3 streaming-read thesis is what makes FITS interesting for
    opencodecs (HTTP-range archive access)."""
    raise NotImplementedError(
        "FITS encode is not implemented in opencodecs; use astropy.io.fits"
    )


__all__ = ["FitsStream", "FitsHDU", "imread", "imwrite"]
