"""PackintsCodec — bit-pack arbitrary-width unsigned integers.

Camera raws (10/12/14-bit ADCs), scientific imaging cameras, and
embedded data formats commonly emit integers that don't fit cleanly
into 8/16/32-bit slots — e.g. 12-bit pixels packed 8 to a 12-byte
block. ``packints`` is the filter that converts between the packed
on-wire form and a standard numpy dtype.

Matches imagecodecs's ``packints_encode`` / ``packints_decode``:

* ``encode``: takes an integer ndarray + ``bitspersample=N``;
  emits big-endian-bit-packed bytes (MSB of element 0 in the
  highest bit of byte 0).
* ``decode``: takes packed bytes + ``dtype`` + ``bitspersample=N``;
  unpacks into an ndarray of ``dtype``.

For the common-but-special case of ``bitspersample`` divisible by 8
(8/16/32), the packing is just the natural memory layout — we short-
circuit through ``frombuffer`` for free.

For arbitrary N, the implementation uses NumPy bit-twiddling. Not
the fastest possible (a Cython inner loop would be ~5× faster on
4K+ images) but correct, dependency-free, and matches imagecodecs
output byte-for-byte.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest


def _bitpack(values: np.ndarray, bitspersample: int) -> bytes:
    """MSB-first big-endian bit-packing of unsigned integers."""
    if bitspersample <= 0 or bitspersample > 64:
        raise ValueError(
            f"packints: bitspersample must be in 1..64, got {bitspersample}")
    if bitspersample % 8 == 0:
        # Trivial: just reinterpret in the appropriate dtype, big-endian.
        # imagecodecs's wire layout is big-endian for whole-byte widths.
        target_dt = {
            8: ">u1", 16: ">u2", 24: None, 32: ">u4",
            40: None, 48: None, 56: None, 64: ">u8",
        }.get(bitspersample)
        if target_dt is not None:
            return values.astype(target_dt, copy=False).tobytes()
    # General path: walk the array, shift-and-or into a uint64 accumulator,
    # flush bytes when the buffer has >=8 full bits.
    n = int(values.size)
    nbits = n * bitspersample
    nbytes = (nbits + 7) // 8
    out = bytearray(nbytes)
    acc = 0
    acc_bits = 0
    byte_pos = 0
    flat = values.ravel()
    for v in flat:
        acc = (acc << bitspersample) | (int(v) & ((1 << bitspersample) - 1))
        acc_bits += bitspersample
        while acc_bits >= 8:
            acc_bits -= 8
            out[byte_pos] = (acc >> acc_bits) & 0xFF
            byte_pos += 1
    if acc_bits > 0:
        out[byte_pos] = (acc << (8 - acc_bits)) & 0xFF
    return bytes(out)


def _bitunpack(buf: bytes, dtype: np.dtype, bitspersample: int,
               n_elements: int) -> np.ndarray:
    """Inverse of :func:`_bitpack`."""
    if bitspersample <= 0 or bitspersample > 64:
        raise ValueError(
            f"packints: bitspersample must be in 1..64, got {bitspersample}")
    if bitspersample % 8 == 0:
        target_dt = {
            8: ">u1", 16: ">u2", 32: ">u4", 64: ">u8",
        }.get(bitspersample)
        if target_dt is not None:
            arr = np.frombuffer(buf, dtype=target_dt, count=n_elements)
            return arr.astype(dtype, copy=False)
    # General path: walk bytes into a bit accumulator, slice out groups.
    mask = (1 << bitspersample) - 1
    out = np.empty(n_elements, dtype=dtype)
    acc = 0
    acc_bits = 0
    src = memoryview(buf)
    bp = 0
    for i in range(n_elements):
        while acc_bits < bitspersample:
            if bp >= len(src):
                raise ValueError(
                    f"packints decode: input ran out at element {i}/"
                    f"{n_elements}")
            acc = (acc << 8) | src[bp]
            bp += 1
            acc_bits += 8
        acc_bits -= bitspersample
        out[i] = (acc >> acc_bits) & mask
    return out


class PackintsCodec(Codec):
    """Bit-pack / unpack arbitrary-width unsigned integers."""

    name = "packints"
    aliases = ()
    file_extensions = ()

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (
        np.uint8, np.uint16, np.uint32, np.uint64,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return False  # opaque filter

    def encode(
        self,
        data: Any,
        *,
        dest=None,
        bitspersample: int,
        **opts,
    ) -> bytes | None:
        arr = np.ascontiguousarray(data)
        if arr.dtype.kind != "u":
            raise ValueError(
                f"packints encode: requires unsigned int dtype, got {arr.dtype}")
        out = _bitpack(arr, int(bitspersample))
        return _write_dest(out, dest)

    def decode(
        self,
        src: Any,
        *,
        dtype,
        bitspersample: int,
        n_elements: int | None = None,
        shape=None,
        out=None,
        **opts,
    ) -> np.ndarray:
        if dtype is None:
            raise ValueError("packints decode: dtype= is required")
        buf = _read_src(src)
        if n_elements is None:
            if shape is None:
                # Best effort: derive from input bytes and bitspersample.
                total_bits = len(buf) * 8
                n_elements = total_bits // int(bitspersample)
            else:
                n_elements = int(np.prod(shape))
        result = _bitunpack(buf, np.dtype(dtype), int(bitspersample),
                            int(n_elements))
        if shape is not None:
            result = result.reshape(shape)
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError(
                    f"packints decode: out= must be an ndarray, "
                    f"got {type(out).__name__}")
            if out.shape != result.shape or out.dtype != result.dtype:
                raise ValueError("packints decode: out= shape/dtype mismatch")
            np.copyto(out, result)
            return out
        return result


__all__ = ["PackintsCodec"]
