"""DeltaCodec / XorCodec / FloatpredCodec — composable byte-level predictors.

These are *filters*, not compressors: they don't reduce data size,
but they make the bytes more redundant so a downstream compressor
(zstd, lz4, deflate) squeezes harder. Standard preprocessing for
sequential / scientific data.

* **Delta** (TIFF predictor 2): replace each element with its
  difference from the previous element. The result has small
  values clustered near zero for smooth sequences, which entropy
  coders love.

* **XOR**: same shape as delta but XOR rather than subtraction.
  Cheaper than delta on hardware that lacks fast subtraction;
  often equivalent for compression ratio on integer data.

* **Floatpred** (TIFF predictor 3, for IEEE 754): byte-level
  reshuffle that scatters each float into byte-plane streams,
  then delta-encodes the high-byte plane. The standard way to
  compress floating-point TIFFs / scientific arrays.

Mirrors imagecodecs's ``delta_encode`` / ``delta_decode`` /
``xor_encode`` / ``xor_decode`` / ``floatpred_encode`` /
``floatpred_decode``. All use ``axis=-1`` (last axis = innermost
loop in memory) and ``dist=1`` by default.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest


def _resolve_int_dtype_from_itemsize(itemsize: int, signed: bool = False):
    """Map an itemsize (1/2/4/8) to a numpy int dtype."""
    if itemsize == 1:
        return np.int8 if signed else np.uint8
    if itemsize == 2:
        return np.int16 if signed else np.uint16
    if itemsize == 4:
        return np.int32 if signed else np.uint32
    if itemsize == 8:
        return np.int64 if signed else np.uint64
    raise ValueError(f"unsupported itemsize {itemsize}")


def _as_2d_for_axis(arr: np.ndarray, axis: int):
    """View ``arr`` as 2-D ``(outer, inner)`` where ``inner`` is the
    axis we're encoding along. Returns a flat-2D view + the inverse
    reshape callable."""
    axis = axis if axis >= 0 else arr.ndim + axis
    if axis != arr.ndim - 1:
        # Move the predictor axis to the end
        arr_t = np.moveaxis(arr, axis, -1)
    else:
        arr_t = arr
    flat = arr_t.reshape(-1, arr_t.shape[-1]) if arr_t.ndim > 1 else arr_t.reshape(1, -1)
    return arr_t, flat


def _resolve_out_bytes(out, n_bytes: int, label: str):
    """Helper for filter codecs: resolve a writable byte buffer."""
    if out is None:
        return None
    if isinstance(out, int):
        if out < n_bytes:
            raise ValueError(
                f"{label}: out=int({out}) is less than the required "
                f"{n_bytes} bytes")
        return None  # treated as size hint; we still allocate fresh bytes
    if not isinstance(out, (bytearray, memoryview, np.ndarray)):
        raise TypeError(
            f"{label}: out= must be int or writable buffer, "
            f"got {type(out).__name__}")
    return out


# ---------------------------------------------------------------------------
# Delta
# ---------------------------------------------------------------------------


class DeltaCodec(Codec):
    """Delta predictor (TIFF predictor 2)."""

    name = "delta"
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
        np.uint8, np.int8, np.uint16, np.int16,
        np.uint32, np.int32, np.uint64, np.int64,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return False  # filter, no magic

    def encode(
        self,
        data: Any,
        *,
        dest=None,
        axis: int = -1,
        dist: int = 1,
        **opts,
    ) -> bytes | None:
        arr = np.ascontiguousarray(data)
        result = self._apply_along(arr, axis, dist, mode="encode")
        return _write_dest(result.tobytes(), dest)

    def decode(
        self,
        src: Any,
        *,
        dtype=None,
        shape=None,
        axis: int = -1,
        dist: int = 1,
        out=None,
        **opts,
    ) -> np.ndarray:
        buf = _read_src(src)
        if dtype is None:
            dtype = np.uint8
        if shape is None:
            arr = np.frombuffer(buf, dtype=dtype)
        else:
            arr = np.frombuffer(buf, dtype=dtype).reshape(shape)
        result = self._apply_along(arr, axis, dist, mode="decode")
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError(
                    f"delta decode: out= must be an ndarray, "
                    f"got {type(out).__name__}")
            if out.shape != result.shape:
                raise ValueError(
                    f"delta decode: out= shape {out.shape} does not match "
                    f"decoded {result.shape}")
            if out.dtype != result.dtype:
                raise ValueError(
                    f"delta decode: out= dtype {out.dtype} does not match "
                    f"decoded {result.dtype}")
            np.copyto(out, result)
            return out
        return result

    @staticmethod
    def _apply_along(arr: np.ndarray, axis: int, dist: int, mode: str) -> np.ndarray:
        """delta encode = diff; delta decode = cumsum (modular arithmetic
        for unsigned types is exactly numpy's default integer wraparound)."""
        if mode == "encode":
            # out[i] = src[i] - src[i-dist] (mod 2**bits for unsigned)
            shifted = np.roll(arr, dist, axis=axis)
            # Zero out the first ``dist`` slots along axis (np.roll wraps).
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(0, dist)
            shifted[tuple(sl)] = 0
            return (arr - shifted).astype(arr.dtype, copy=False)
        else:
            # cumulative sum modulo dtype range; np.cumsum on unsigned
            # int wraps naturally.
            if dist == 1:
                return np.cumsum(arr, axis=axis, dtype=arr.dtype)
            # General dist>=1: cumsum each lane of stride ``dist``.
            result = arr.copy()
            slc = [slice(None)] * arr.ndim
            for start in range(dist):
                slc[axis] = slice(start, None, dist)
                lane = result[tuple(slc)]
                result[tuple(slc)] = np.cumsum(lane, axis=axis, dtype=arr.dtype)
            return result


# ---------------------------------------------------------------------------
# XOR
# ---------------------------------------------------------------------------


class XorCodec(Codec):
    """XOR predictor — same shape as delta but uses ``^`` instead of ``-``."""

    name = "xor"
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
        np.uint8, np.int8, np.uint16, np.int16,
        np.uint32, np.int32, np.uint64, np.int64,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return False

    def encode(
        self,
        data: Any,
        *,
        dest=None,
        axis: int = -1,
        dist: int = 1,
        **opts,
    ) -> bytes | None:
        arr = np.ascontiguousarray(data)
        result = self._apply_along(arr, axis, dist, mode="encode")
        return _write_dest(result.tobytes(), dest)

    def decode(
        self,
        src: Any,
        *,
        dtype=None,
        shape=None,
        axis: int = -1,
        dist: int = 1,
        out=None,
        **opts,
    ) -> np.ndarray:
        buf = _read_src(src)
        if dtype is None:
            dtype = np.uint8
        arr = (np.frombuffer(buf, dtype=dtype).reshape(shape)
               if shape is not None
               else np.frombuffer(buf, dtype=dtype))
        result = self._apply_along(arr, axis, dist, mode="decode")
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError(
                    f"xor decode: out= must be an ndarray, "
                    f"got {type(out).__name__}")
            if out.shape != result.shape or out.dtype != result.dtype:
                raise ValueError("xor decode: out= shape/dtype mismatch")
            np.copyto(out, result)
            return out
        return result

    @staticmethod
    def _apply_along(arr, axis, dist, mode):
        if mode == "encode":
            shifted = np.roll(arr, dist, axis=axis)
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(0, dist)
            shifted[tuple(sl)] = 0
            return arr ^ shifted
        # decode: running XOR
        result = arr.copy()
        if dist == 1:
            return np.bitwise_xor.accumulate(result, axis=axis)
        slc = [slice(None)] * arr.ndim
        for start in range(dist):
            slc[axis] = slice(start, None, dist)
            result[tuple(slc)] = np.bitwise_xor.accumulate(
                result[tuple(slc)], axis=axis)
        return result


# ---------------------------------------------------------------------------
# Floatpred (TIFF predictor 3)
# ---------------------------------------------------------------------------


class FloatpredCodec(Codec):
    """IEEE-754 byte-plane delta predictor (TIFF predictor 3).

    For an array of floats, splits each element into its constituent
    bytes, concatenates the byte planes (all high bytes first, then
    next, etc.) per row, then delta-encodes. The float bytes within
    one row become two redundant streams (sign+exponent bytes change
    slowly; mantissa low bytes look random); delta on the slow stream
    is what does the compression heavy lifting.
    """

    name = "floatpred"
    aliases = ("float-pred",)
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
        return False

    def encode(
        self,
        data: Any,
        *,
        dest=None,
        axis: int = -1,
        dist: int = 1,
        **opts,
    ) -> bytes | None:
        arr = np.ascontiguousarray(data)
        if arr.dtype.kind != "f":
            raise ValueError(
                f"floatpred encode: requires a floating dtype, got {arr.dtype}")
        bytes_view = self._shuffle_then_delta(arr, axis, dist, encode=True)
        return _write_dest(bytes_view, dest)

    def decode(
        self,
        src: Any,
        *,
        dtype,
        shape=None,
        axis: int = -1,
        dist: int = 1,
        out=None,
        **opts,
    ) -> np.ndarray:
        if dtype is None:
            raise ValueError("floatpred decode: dtype= is required")
        buf = _read_src(src)
        # Restore byte stream → float array.
        result = self._undelta_then_unshuffle(buf, np.dtype(dtype), shape, axis, dist)
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError(
                    f"floatpred decode: out= must be an ndarray, "
                    f"got {type(out).__name__}")
            if out.shape != result.shape or out.dtype != result.dtype:
                raise ValueError("floatpred decode: out= shape/dtype mismatch")
            np.copyto(out, result)
            return out
        return result

    @staticmethod
    def _shuffle_then_delta(arr, axis, dist, encode):
        # arr.shape = (..., n) along ``axis``; itemsize = arr.dtype.itemsize.
        # 1) Reinterpret as uint8 of shape (..., n * itemsize).
        # 2) Reorder columns into byte-plane order (all 1st bytes, then 2nd, ...).
        # 3) Delta-encode along the last axis as uint8.
        # 4) Return the byte stream.
        itemsize = arr.dtype.itemsize
        axis = axis if axis >= 0 else arr.ndim + axis
        if axis != arr.ndim - 1:
            arr = np.moveaxis(arr, axis, -1).copy()
        u8 = arr.view(np.uint8)
        n = arr.shape[-1]
        # u8.shape = (..., n * itemsize); we want (..., itemsize, n) by plane
        u8 = u8.reshape(arr.shape[:-1] + (n, itemsize))
        # Transpose so plane comes first: (..., itemsize, n)
        u8 = np.moveaxis(u8, -1, -2)
        u8 = np.ascontiguousarray(u8)
        # Now apply delta on the trailing axis (per-plane).
        if encode:
            shifted = np.roll(u8, dist, axis=-1)
            sl = [slice(None)] * u8.ndim
            sl[-1] = slice(0, dist)
            shifted[tuple(sl)] = 0
            out = (u8 - shifted).astype(np.uint8, copy=False)
        else:
            out = np.cumsum(u8, axis=-1, dtype=np.uint8)
        return out.tobytes()

    @staticmethod
    def _undelta_then_unshuffle(buf, dtype, shape, axis, dist):
        if shape is None:
            raise ValueError("floatpred decode: shape= is required")
        itemsize = dtype.itemsize
        # Reshape into byte-plane form and reverse the encode pipeline.
        axis_pos = axis if axis >= 0 else len(shape) + axis
        # Move axis to end for processing
        target_shape = tuple(shape)
        if axis_pos != len(target_shape) - 1:
            permuted = list(target_shape)
            inner = permuted.pop(axis_pos)
            permuted.append(inner)
            internal_shape = tuple(permuted)
        else:
            internal_shape = target_shape
        n = internal_shape[-1]
        outer = int(np.prod(internal_shape[:-1])) if len(internal_shape) > 1 else 1
        u8 = np.frombuffer(buf, dtype=np.uint8).reshape(outer, itemsize, n)
        # Undo delta along last axis.
        u8 = np.cumsum(u8, axis=-1, dtype=np.uint8)
        # Move plane axis back: (outer, itemsize, n) -> (outer, n, itemsize)
        u8 = np.moveaxis(u8, -2, -1)
        u8 = np.ascontiguousarray(u8)
        # Reinterpret as the float dtype.
        flat = u8.reshape(outer, n * itemsize).view(dtype)
        # Flat shape: (outer, n); reshape to internal_shape.
        arr = flat.reshape(internal_shape)
        # Move the encoded axis back to its original position.
        if axis_pos != len(target_shape) - 1:
            arr = np.moveaxis(arr, -1, axis_pos)
            arr = arr.reshape(target_shape)
        return np.ascontiguousarray(arr)


__all__ = ["DeltaCodec", "XorCodec", "FloatpredCodec"]
