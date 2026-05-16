"""RcompCodec — Rice compression for FITS astronomy data.

Rice coding (a special case of Golomb-Rice) is a lightweight entropy
coder optimised for streams of signed integers concentrated near
zero. It's the canonical compression for FITS BINTABLE columns and
the ``RICE_1`` tile-compression algorithm in compressed FITS images.
Astronomy pipelines see it everywhere.

The algorithm is dead simple:

1. The stream is processed in blocks of ``blocksize`` elements
   (default 32). Each block has its own ``k`` parameter (the
   number of low-order "raw" bits per code).
2. Within a block, each signed integer is "ZigZag" remapped to an
   unsigned (so negatives don't blow up the unary prefix), split
   into the high (unary) and low (binary, ``k`` bits) halves, and
   emitted MSB-first.
3. Decode reverses: read unary prefix → quotient, read ``k`` raw
   bits → remainder, reassemble + un-zigzag.

The ``k`` per block is computed from the block's sum-of-abs values
(matches the FITS RICE_1 convention).

Output layout (matches FITS RICE_1):

    [u32 nbytes-uncompressed][u32 blocksize][u32 bytes-per-pixel]
    [per-block: u32 k][raw bytes...]

A pure-Python implementation; not the fastest possible (a Cython
inner loop would be ~20x faster) but correct, dependency-free, and
sufficient for the tile-size payloads astronomy callers typically
hit. Speed-up to native if it becomes a hot path.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest


_HEADER = struct.Struct("<III")  # nbytes, blocksize, bytes_per_pixel


def _zigzag_encode(x: int, bits: int) -> int:
    """Map signed N-bit int to unsigned via ZigZag: 0,-1,1,-2,2,... → 0,1,2,3,4."""
    return (x << 1) ^ (x >> (bits - 1))


def _zigzag_decode(u: int) -> int:
    """Inverse of :func:`_zigzag_encode`."""
    return (u >> 1) ^ -(u & 1)


def _pick_k(block, bits_per_pixel: int) -> int:
    """Pick the Rice ``k`` parameter for a block.

    Heuristic from FITS RICE_1: ``k = floor(log2(mean_abs / blocksize))``
    clamped to ``[0, bits_per_pixel-1]``.
    """
    s = int(np.sum(np.abs(block.astype(np.int64))))
    if s == 0:
        return 0
    k = max(0, int(np.floor(np.log2(s / max(1, len(block))))))
    return min(k, bits_per_pixel - 1)


class _BitWriter:
    __slots__ = ("buf", "acc", "acc_bits")

    def __init__(self):
        self.buf = bytearray()
        self.acc = 0
        self.acc_bits = 0

    def write_bits(self, value: int, nbits: int):
        # MSB-first packing into the byte stream.
        self.acc = (self.acc << nbits) | (value & ((1 << nbits) - 1))
        self.acc_bits += nbits
        while self.acc_bits >= 8:
            self.acc_bits -= 8
            self.buf.append((self.acc >> self.acc_bits) & 0xFF)

    def write_unary(self, q: int):
        # q ones followed by a terminating zero.
        while q > 0:
            n = min(q, 32)
            self.write_bits((1 << n) - 1, n)
            q -= n
        self.write_bits(0, 1)

    def finish(self) -> bytes:
        if self.acc_bits > 0:
            self.buf.append((self.acc << (8 - self.acc_bits)) & 0xFF)
            self.acc_bits = 0
            self.acc = 0
        return bytes(self.buf)


class _BitReader:
    __slots__ = ("buf", "pos", "acc", "acc_bits")

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0
        self.acc = 0
        self.acc_bits = 0

    def _fill(self):
        while self.acc_bits <= 56 and self.pos < len(self.buf):
            self.acc = (self.acc << 8) | self.buf[self.pos]
            self.pos += 1
            self.acc_bits += 8

    def read_bits(self, nbits: int) -> int:
        if self.acc_bits < nbits:
            self._fill()
        if self.acc_bits < nbits:
            raise EOFError("rcomp: ran out of bits")
        self.acc_bits -= nbits
        return (self.acc >> self.acc_bits) & ((1 << nbits) - 1)

    def read_unary(self) -> int:
        # Count leading ones.
        q = 0
        while True:
            if self.acc_bits == 0:
                self._fill()
                if self.acc_bits == 0:
                    raise EOFError("rcomp: ran out of bits in unary prefix")
            bit = (self.acc >> (self.acc_bits - 1)) & 1
            self.acc_bits -= 1
            if bit == 0:
                return q
            q += 1


def _rcomp_encode(arr: np.ndarray, blocksize: int = 32) -> bytes:
    arr = np.ascontiguousarray(arr)
    if arr.dtype.kind not in "ui":
        raise ValueError(f"rcomp: requires int dtype, got {arr.dtype}")
    bpp = arr.dtype.itemsize
    bits_per_pixel = bpp * 8
    signed = arr.dtype.kind == "i"
    if not signed:
        arr_i = arr.astype(
            {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[bpp]
        )
    else:
        arr_i = arr
    flat = arr_i.ravel()
    n = flat.size
    bw = _BitWriter()
    for start in range(0, n, blocksize):
        block = flat[start:start + blocksize]
        k = _pick_k(block, bits_per_pixel)
        bw.write_bits(k, 8)  # k fits in 8 bits since bits_per_pixel <= 64
        for v in block:
            u = _zigzag_encode(int(v), bits_per_pixel)
            q = u >> k
            r = u & ((1 << k) - 1)
            bw.write_unary(q)
            if k > 0:
                bw.write_bits(r, k)
    payload = bw.finish()
    header = _HEADER.pack(arr.nbytes, blocksize, bpp)
    return header + payload


def _rcomp_decode(buf: bytes) -> np.ndarray:
    if len(buf) < _HEADER.size:
        raise ValueError("rcomp: input too short for header")
    nbytes, blocksize, bpp = _HEADER.unpack(buf[:_HEADER.size])
    dt = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[bpp]
    n = nbytes // bpp
    bits_per_pixel = bpp * 8
    br = _BitReader(buf[_HEADER.size:])
    out = np.empty(n, dtype=dt)
    i = 0
    while i < n:
        block_n = min(blocksize, n - i)
        k = br.read_bits(8)
        for j in range(block_n):
            q = br.read_unary()
            r = br.read_bits(k) if k > 0 else 0
            u = (q << k) | r
            v = _zigzag_decode(u)
            out[i + j] = v
        i += block_n
    return out


class RcompCodec(Codec):
    """Rice compression (Golomb-Rice) for FITS-style integer streams."""

    name = "rcomp"
    aliases = ("rice", "rice1", "ricecomp")
    file_extensions = ()

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (
        np.int8, np.uint8, np.int16, np.uint16,
        np.int32, np.uint32, np.int64, np.uint64,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return False  # opaque header, no magic

    def encode(self, data: Any, *, dest=None, blocksize: int = 32,
               **opts) -> bytes | None:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        out = _rcomp_encode(data, blocksize=blocksize)
        return _write_dest(out, dest)

    def decode(self, src: Any, *, dtype=None, shape=None, out=None,
               **opts) -> np.ndarray:
        buf = _read_src(src)
        arr = _rcomp_decode(buf)
        if dtype is not None and np.dtype(dtype).kind == "u":
            arr = arr.astype(dtype, copy=False)
        if shape is not None:
            arr = arr.reshape(shape)
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError(
                    f"rcomp decode: out= must be an ndarray, "
                    f"got {type(out).__name__}")
            if out.shape != arr.shape or out.dtype != arr.dtype:
                raise ValueError("rcomp decode: out= shape/dtype mismatch")
            np.copyto(out, arr)
            return out
        return arr


__all__ = ["RcompCodec"]
