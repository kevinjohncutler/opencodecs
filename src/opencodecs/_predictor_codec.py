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



_DELTA_KERNEL = "unset"


_XOR_KERNEL = "unset"


def _xor_decode_kernel():
    """The compiled running XOR, or None when the extension is absent."""
    global _XOR_KERNEL
    if _XOR_KERNEL == "unset":
        try:
            from .codecs._bytetools import xor_decode_inplace
            _XOR_KERNEL = xor_decode_inplace
        except ImportError:                              # pragma: no cover
            _XOR_KERNEL = None
    return _XOR_KERNEL


def _delta_decode_kernel():
    """The compiled prefix sum, or None when the extension is absent.

    Resolved once. Everything here still works without it -- the numpy
    path below is the fallback, just slower.
    """
    global _DELTA_KERNEL
    if _DELTA_KERNEL == "unset":
        try:
            from .codecs._bytetools import delta_decode_inplace
            _DELTA_KERNEL = delta_decode_inplace
        except ImportError:                              # pragma: no cover
            _DELTA_KERNEL = None
    return _DELTA_KERNEL


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


def _predictor_into(arr, axis, dist, out, kernel, ufunc):
    """Decode into caller storage without a full result-and-copy temporary."""
    if not isinstance(out, np.ndarray):
        raise TypeError("predictor out must be an ndarray")
    if out.shape != arr.shape or out.dtype != arr.dtype:
        raise ValueError("predictor out shape/dtype mismatch")
    if not out.flags.writeable:
        raise ValueError("predictor out must be writable")
    if dist < 1:
        raise ValueError("predictor distance must be positive")
    # Moving the selected axis to the end can remain a contiguous view.
    target = np.moveaxis(out, axis, -1)
    if (kernel is not None and target.flags.c_contiguous and
            out.dtype.isnative and out.dtype.kind in 'iu' and
            out.dtype.itemsize in (1, 2, 4, 8) and target.shape[-1]):
        np.copyto(out, arr)
        kernel(target.reshape(-1, target.shape[-1]), dist)
        return out
    np.copyto(out, arr)
    selection = [slice(None)] * arr.ndim
    for start in range(min(dist, arr.shape[axis])):
        selection[axis] = slice(start, None, dist)
        lane = out[tuple(selection)]
        # NumPy preserves the destination's byte order and strided layout.
        ufunc.accumulate(lane, axis=axis, dtype=arr.dtype.newbyteorder("="), out=lane)
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
        if out is not None:
            return _predictor_into(arr, axis, dist, out,
                                   _delta_decode_kernel(), np.add)
        return self._apply_along(arr, axis, dist, mode="decode")

    @staticmethod
    def _apply_along(arr: np.ndarray, axis: int, dist: int, mode: str) -> np.ndarray:
        """delta encode = diff; delta decode = cumsum (modular arithmetic
        for unsigned types is exactly numpy's default integer wraparound)."""
        if mode == "encode":
            # out[i] = src[i] - src[i-dist] (mod 2**bits for unsigned).
            #
            # Written as a slice-subtract rather than np.roll: roll
            # allocates a whole shifted copy of the input, and dropping
            # it halves encode time to exactly what imagecodecs takes
            # (0.52 ms against 1.09 ms on 17 MB of uint8).
            out = np.empty_like(arr)
            head = [slice(None)] * arr.ndim
            head[axis] = slice(0, dist)
            tail = [slice(None)] * arr.ndim
            tail[axis] = slice(dist, None)
            prev = [slice(None)] * arr.ndim
            prev[axis] = slice(0, -dist)
            out[tuple(head)] = arr[tuple(head)]
            np.subtract(arr[tuple(tail)], arr[tuple(prev)],
                        out=out[tuple(tail)])
            return out

        # Decode is a prefix sum, and np.cumsum is 5.5x slower than a
        # plain C loop on this shape (34.6 ms against 6.3 ms on 17 MB of
        # uint8) because of its per-element dispatch. A prefix sum is
        # serial either way, so this is not about vectorizing -- it is
        # the same specialization argument as libspng's filter_scanline,
        # applied to a numpy call.
        kern = _delta_decode_kernel()
        # A zero-length axis has nothing to sum, and reshape(-1, 0) cannot
        # resolve its -1, so it takes the NumPy path like _predictor_into.
        if kern is not None and arr.dtype.kind in "iu" and \
                arr.dtype.itemsize in (1, 2, 4, 8) and arr.shape[axis]:
            # A writable, C-contiguous copy with the target axis last.
            # np.frombuffer hands back a read-only array, and the kernel
            # takes a writable memoryview, so this copy is required
            # rather than defensive -- without it the call raises and
            # the slow path silently takes over, which is how this was
            # first written and why it appeared to make no difference.
            moved = np.moveaxis(arr, axis, -1)
            work = np.array(moved, dtype=arr.dtype, order="C", copy=True)
            kern(work.reshape(-1, work.shape[-1]), dist)
            return np.moveaxis(work, -1, axis)

        if dist == 1:
            return np.cumsum(arr, axis=axis, dtype=arr.dtype)
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
        if out is not None:
            return _predictor_into(arr, axis, dist, out,
                                   _xor_decode_kernel(), np.bitwise_xor)
        return self._apply_along(arr, axis, dist, mode="decode")

    @staticmethod
    def _apply_along(arr, axis, dist, mode):
        if mode == "encode":
            # Slice-xor rather than np.roll, which allocates a whole
            # shifted copy of the input just to throw it away. Same
            # reasoning as the delta encoder above.
            out = np.empty_like(arr)
            head = [slice(None)] * arr.ndim
            head[axis] = slice(0, dist)
            tail = [slice(None)] * arr.ndim
            tail[axis] = slice(dist, None)
            prev = [slice(None)] * arr.ndim
            prev[axis] = slice(0, -dist)
            out[tuple(head)] = arr[tuple(head)]
            np.bitwise_xor(arr[tuple(tail)], arr[tuple(prev)],
                           out=out[tuple(tail)])
            return out

        # decode: running XOR. np.bitwise_xor.accumulate carries the
        # same per-element dispatch as np.cumsum did for delta, and the
        # same ~5x gap against a specialized loop.
        kern = _xor_decode_kernel()
        if kern is not None and arr.dtype.kind in "iu" and \
                arr.dtype.itemsize in (1, 2, 4, 8) and arr.shape[axis]:
            moved = np.moveaxis(arr, axis, -1)
            work = np.array(moved, dtype=arr.dtype, order="C", copy=True)
            kern(work.reshape(-1, work.shape[-1]), dist)
            return np.moveaxis(work, -1, axis)

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
    """IEEE-754 floating-point predictor: TIFF predictor 3 (TIFF Technical
    Note 3), the same bytes as imagecodecs' ``floatpred`` and as this
    package's TIFF reader and writer.

    Along ``axis``, a step is the block of all trailing elements (the
    samples of a pixel when ``axis`` is the column axis of a (rows,
    columns, samples) array). Each line's bytes are reordered into byte
    planes, most significant first whatever the machine's byte order, and
    the reordered line is differenced as bytes at ``dist`` steps, across
    plane boundaries. Sign and exponent bytes change slowly, so their
    differences are small, which is what a compressor then exploits.
    imagecodecs takes only the last two axes and ``dist`` of 1 or 2; the
    same rule applies to every axis and distance here.
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
        scratch=None,
        **opts,
    ) -> np.ndarray:
        if dtype is None:
            raise ValueError("floatpred decode: dtype= is required")
        buf = _read_src(src)
        return self._undelta_then_unshuffle(
            buf, np.dtype(dtype), shape, axis, dist, out=out, scratch=scratch)

    @staticmethod
    def _geometry(shape, axis):
        """(lines, steps per line, elements per step) for ``axis``."""
        ndim = len(shape)
        if ndim == 0:
            raise ValueError("floatpred needs at least one dimension")
        ax = axis + ndim if axis < 0 else axis
        if not 0 <= ax < ndim:
            raise ValueError(f"floatpred axis {axis} out of range for {ndim} dimensions")
        lines = int(np.prod(shape[:ax], dtype=np.int64))
        inner = int(np.prod(shape[ax + 1:], dtype=np.int64))
        return lines, shape[ax], inner

    @staticmethod
    def _shuffle_then_delta(arr, axis, dist, encode):
        if dist < 1:
            raise ValueError("predictor distance must be positive")
        native = arr.dtype.newbyteorder("=")
        arr = np.ascontiguousarray(arr, dtype=native)
        lines, n, inner = FloatpredCodec._geometry(arr.shape, axis)
        itemsize = arr.dtype.itemsize
        raw = arr.view(np.uint8).reshape(lines, n * inner, itemsize)
        if _LITTLE_ENDIAN:
            raw = raw[:, :, ::-1]                 # most significant byte first
        row = np.ascontiguousarray(raw.transpose(0, 2, 1)).reshape(lines, -1)
        step = inner * dist
        out = row.copy()
        np.subtract(row[:, step:], row[:, :-step], out=out[:, step:])
        return out.tobytes()

    @staticmethod
    def _undelta_then_unshuffle(buf, dtype, shape, axis, dist, *, out=None, scratch=None):
        if shape is None:
            raise ValueError("floatpred decode: shape= is required")
        if dist < 1:
            raise ValueError("predictor distance must be positive")
        target_shape = tuple(shape)
        native = dtype.newbyteorder("=")
        itemsize = dtype.itemsize
        lines, n, inner = FloatpredCodec._geometry(target_shape, axis)
        if out is not None:
            from .core.buffers import array_output
            array_output(out)
            if out.shape != target_shape or out.dtype != dtype:
                raise ValueError("floatpred out shape/dtype mismatch")
        required = lines * n * inner * itemsize
        source = np.frombuffer(buf, dtype=np.uint8)
        if source.size < required:
            raise ValueError(
                f"floatpred decode: {source.size} bytes, shape {target_shape} needs {required}")
        source = source[:required]
        if scratch is None:
            work = source.copy()
        else:
            from .core.scratch import ScratchBuffer
            from .core.buffers import byte_output
            storage = scratch.bytes(required) if isinstance(scratch, ScratchBuffer) else byte_output(scratch)
            if isinstance(storage, int) or len(storage) < required:
                raise ValueError("floatpred scratch buffer is too small")
            work = np.frombuffer(storage, dtype=np.uint8, count=required)
            if out is not None and np.shares_memory(work, out):
                raise ValueError("floatpred scratch must not overlap output")
            np.copyto(work, source)
        if required:
            kernel = _undo_floating_point_kernel() if dist == 1 and itemsize in (2, 4, 8) else None
            if kernel is not None:
                # The TIFF reader's kernel: undoes the differences at one
                # step (``inner`` bytes) and restores native byte order.
                kernel(work.reshape(lines, n, inner * itemsize), itemsize)
            else:
                rows = work.reshape(lines, -1)
                step = inner * dist
                for start in range(min(step, rows.shape[1])):
                    lane = rows[:, start::step]
                    np.add.accumulate(lane, axis=-1, dtype=np.uint8, out=lane)
                planes = rows.reshape(lines, itemsize, n * inner).transpose(0, 2, 1)
                if _LITTLE_ENDIAN:
                    planes = planes[:, :, ::-1]
                work[...] = np.ascontiguousarray(planes).reshape(-1)
        values = work.view(native).reshape(target_shape)
        if out is None:
            return values.astype(dtype, copy=False) if native != dtype else values
        out[...] = values
        return out


_LITTLE_ENDIAN = np.little_endian


def _undo_floating_point_kernel():
    """The TIFF reader's compiled predictor 3 undo, or None without it."""
    try:
        from .codecs._tiff import undo_floating_point
    except ImportError:
        return None
    return undo_floating_point


__all__ = ["DeltaCodec", "XorCodec", "FloatpredCodec"]
