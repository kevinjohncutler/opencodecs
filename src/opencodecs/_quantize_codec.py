"""QuantizeCodec — IEEE-754 mantissa bit-round / number-of-significant-digits.

Quantization is a lossy *filter*: it rounds float values to a coarser
representation so that downstream byte/bit-level compressors (zstd,
floatpred+zstd, lerc) reach much better ratios. The output is still
the same dtype and shape as the input — quantize doesn't change the
wire format, just reduces the number of distinct values.

Two modes (matching imagecodecs):

* ``mode="bitround"`` + ``bitspersample=N``: keep the top ``N`` bits
  of the mantissa; zero the rest. Cheap (single mask) and bounds the
  relative error to ``2**-N``.

* ``mode="nsd"`` + ``nsd=K``: round to ``K`` significant digits.
  More predictable in absolute error for human eyeballing but more
  expensive (a `log10` per element).

Decode is the identity — quantize is irreversible by design, so the
"decoder" just returns the data unchanged. The asymmetry matches
imagecodecs and keeps the codec chain composable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest


# Mantissa-bits-per-dtype lookup (IEEE 754).
_MANTISSA_BITS = {
    np.dtype("float16"): 10,
    np.dtype("float32"): 23,
    np.dtype("float64"): 52,
}


def _bitround(arr: np.ndarray, keepbits: int) -> np.ndarray:
    """Zero the (mantissa_bits - keepbits) low bits of each float."""
    dt = arr.dtype
    mantissa_bits = _MANTISSA_BITS.get(dt)
    if mantissa_bits is None:
        raise ValueError(
            f"quantize bitround: unsupported dtype {dt}")
    if keepbits < 0 or keepbits > mantissa_bits:
        raise ValueError(
            f"quantize bitround: bitspersample must be in 0..{mantissa_bits}, "
            f"got {keepbits}")
    if keepbits == mantissa_bits:
        return arr.copy()
    # Reinterpret as uint of same size, then mask LSBs.
    uint_view = arr.view(
        np.uint16 if dt.itemsize == 2 else
        np.uint32 if dt.itemsize == 4 else
        np.uint64
    )
    shift = mantissa_bits - keepbits
    mask = ~((np.uint64(1) << np.uint64(shift)) - np.uint64(1))
    # Half-LSB rounding: add 1 << (shift-1) before masking, with overflow
    # treated as banker's rounding via the mask.
    half = np.uint64(1) << np.uint64(shift - 1) if shift > 0 else np.uint64(0)
    out_uint = ((uint_view.astype(np.uint64) + half) & mask).astype(uint_view.dtype)
    return out_uint.view(dt).copy()


def _nsd_round(arr: np.ndarray, nsd: int) -> np.ndarray:
    """Round to ``nsd`` significant decimal digits.

    Implemented as: scale = 10**(nsd - 1 - floor(log10|x|)); round
    x * scale; divide back. Falls back to passing through zeros
    unchanged.
    """
    if nsd < 1:
        raise ValueError(f"quantize nsd: nsd must be >= 1, got {nsd}")
    out = arr.copy()
    nonzero = arr != 0
    if not np.any(nonzero):
        return out
    vals = arr[nonzero]
    # Use log10 of the magnitude to derive the decimal scale.
    mag = np.floor(np.log10(np.abs(vals)))
    scale = np.power(10.0, nsd - 1 - mag)
    rounded = np.round(vals * scale) / scale
    out[nonzero] = rounded.astype(arr.dtype, copy=False)
    return out


class QuantizeCodec(Codec):
    """Lossy mantissa quantization filter."""

    name = "quantize"
    aliases = ()
    file_extensions = ()

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.float16, np.float32, np.float64)
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return False  # transparent filter — no magic

    def encode(
        self,
        data: Any,
        *,
        dest=None,
        mode: str = "bitround",
        bitspersample: int | None = None,
        nsd: int | None = None,
        **opts,
    ) -> bytes | None:
        arr = np.ascontiguousarray(data)
        if arr.dtype.kind != "f":
            raise ValueError(
                f"quantize: requires a floating dtype, got {arr.dtype}")
        if mode == "bitround":
            if bitspersample is None:
                raise ValueError(
                    "quantize bitround: bitspersample= is required")
            out = _bitround(arr, int(bitspersample))
        elif mode == "nsd":
            if nsd is None:
                raise ValueError("quantize nsd: nsd= is required")
            out = _nsd_round(arr, int(nsd))
        else:
            raise ValueError(
                f"quantize: unsupported mode {mode!r}; "
                f"expected 'bitround' or 'nsd'")
        return _write_dest(out.tobytes(), dest)

    def decode(
        self,
        src: Any,
        *,
        dtype,
        shape=None,
        out=None,
        **opts,
    ) -> np.ndarray:
        # quantize is irreversible — decode is a passthrough that just
        # reinterprets the bytes as the original dtype.
        if dtype is None:
            raise ValueError("quantize decode: dtype= is required")
        buf = _read_src(src)
        arr = np.frombuffer(buf, dtype=dtype)
        if shape is not None:
            arr = arr.reshape(shape)
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError(
                    f"quantize decode: out= must be an ndarray, "
                    f"got {type(out).__name__}")
            if out.shape != arr.shape or out.dtype != arr.dtype:
                raise ValueError(
                    "quantize decode: out= shape/dtype mismatch")
            np.copyto(out, arr)
            return out
        # Caller must own the buffer; frombuffer's result is read-only over
        # an immutable bytes input — copy so the result is writable.
        return arr.copy()


__all__ = ["QuantizeCodec"]
