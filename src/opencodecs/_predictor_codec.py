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

  Both work on each sample's bit pattern as an unsigned integer of
  the same width, in native byte order, modulo 2**bits. That is what
  libtiff does for predictor 2 and what imagecodecs does, and for
  floats it is the only lossless form. The output keeps the input's
  dtype and byte order.

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
from ._filter_args import as_input_array, reject_unknown


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


def _bits_dtype(dtype: np.dtype) -> np.dtype:
    """The native unsigned integer of ``dtype``'s width.

    Delta and XOR run on this view of the samples, whatever their type:
    modular differences of the bit patterns, which is what TIFF predictor
    2 is in libtiff (horizontal differencing chosen by BitsPerSample
    alone, on unsigned words) and what imagecodecs does. For integers the
    result is the same as wrapping arithmetic in the dtype; for floats it
    is the only lossless form, since a difference of float values rounds.
    """
    return np.dtype(f"u{dtype.itemsize}")


def _check_predictor_dtype(dtype: np.dtype, label: str) -> None:
    if dtype.kind not in "iuf" or dtype.itemsize not in (1, 2, 4, 8):
        raise ValueError(
            f"{label}: needs an integer or IEEE floating-point dtype of 1, "
            f"2, 4 or 8 bytes, got {dtype}")


def _check_axis(arr: np.ndarray, axis: int, label: str) -> int:
    """``axis`` as a non-negative index into ``arr``, or a ValueError.

    A 0-d array has no axis to run along; imagecodecs raises "invalid
    axis" for it, and an IndexError from deep inside the slicing said
    nothing about the cause.
    """
    ndim = arr.ndim
    if not -ndim <= axis < ndim:
        raise ValueError(
            f"{label}: invalid axis {axis} for a {ndim}-dimensional array")
    return axis % ndim


def _check_distance(dist: int, label: str) -> None:
    if dist < 1:
        raise ValueError(f"{label}: distance must be positive, got {dist}")


def _accumulate(bits, axis, dist, kernel, ufunc):
    """Undo a predictor in place on ``bits`` (native unsigned) along ``axis``."""
    target = np.moveaxis(bits, axis, -1)
    if not target.size:
        return
    if kernel is not None and target.flags.c_contiguous:
        kernel(target.reshape(-1, target.shape[-1]), dist)
        return
    selection = [slice(None)] * bits.ndim
    for start in range(min(dist, bits.shape[axis])):
        selection[axis] = slice(start, None, dist)
        lane = bits[tuple(selection)]
        ufunc.accumulate(lane, axis=axis, dtype=bits.dtype, out=lane)


def _predict(arr, axis, dist, ufunc, label):
    """Encode: ``out[i] = ufunc(arr[i], arr[i - dist])`` on the bit patterns.

    The result has ``arr``'s dtype, byte order included, as imagecodecs'
    does ("Preserve byteorder"): a big-endian input gives the big-endian
    bytes of the native-order differences.
    """
    _check_predictor_dtype(arr.dtype, label)
    _check_distance(dist, label)
    axis = _check_axis(arr, axis, label)
    native = arr.dtype.newbyteorder("=")
    bits = arr.astype(native, copy=False).view(_bits_dtype(arr.dtype))
    # Written as a slice operation rather than np.roll: roll allocates a
    # whole shifted copy of the input, and dropping it halves encode time
    # to exactly what imagecodecs takes (0.52 ms against 1.09 ms on 17 MB
    # of uint8).
    out = np.empty_like(bits)
    head = [slice(None)] * arr.ndim
    head[axis] = slice(0, dist)
    tail = [slice(None)] * arr.ndim
    tail[axis] = slice(dist, None)
    prev = [slice(None)] * arr.ndim
    prev[axis] = slice(0, -dist)
    out[tuple(head)] = bits[tuple(head)]
    ufunc(bits[tuple(tail)], bits[tuple(prev)], out=out[tuple(tail)])
    return out.view(native).astype(arr.dtype, copy=False)


def _unpredict(arr, axis, dist, kernel, ufunc, label):
    """Decode into a new array of ``arr``'s dtype (byte order preserved)."""
    _check_predictor_dtype(arr.dtype, label)
    _check_distance(dist, label)
    axis = _check_axis(arr, axis, label)
    native = arr.dtype.newbyteorder("=")
    # A writable, C-contiguous, native-order copy with the target axis
    # last. np.frombuffer hands back a read-only array and the kernel
    # takes a writable memoryview of native integers, so this copy is
    # required rather than defensive. A big-endian input used to reach
    # the kernel as is and raise "Big-endian buffer not supported".
    work = np.array(np.moveaxis(arr, axis, -1), dtype=native, order="C")
    _accumulate(work.view(_bits_dtype(native)), -1, dist, kernel, ufunc)
    result = np.moveaxis(work, -1, axis)
    return result if native == arr.dtype else result.astype(arr.dtype)


def _unpredict_into(arr, axis, dist, out, kernel, ufunc, label):
    """Decode into caller storage without a full result-and-copy temporary."""
    if not isinstance(out, np.ndarray):
        raise TypeError("predictor out must be an ndarray")
    if out.shape != arr.shape or out.dtype != arr.dtype:
        raise ValueError("predictor out shape/dtype mismatch")
    if not out.flags.writeable:
        raise ValueError("predictor out must be writable")
    _check_predictor_dtype(arr.dtype, label)
    _check_distance(dist, label)
    axis = _check_axis(arr, axis, label)
    if not out.dtype.isnative:
        np.copyto(out, _unpredict(arr, axis, dist, kernel, ufunc, label))
        return out
    bits = out.view(_bits_dtype(out.dtype))
    np.copyto(bits, arr.view(bits.dtype))
    _accumulate(bits, axis, dist, kernel, ufunc)
    return out


def _decode_array(src, dtype, shape):
    """The stored samples as an array, with imagecodecs' defaults.

    An ndarray carries its own dtype and shape (the form imagecodecs'
    encoders return); anything else is bytes, which mean uint8. Decoding
    an int32 array as flat uint8 used to return wrong values silently.
    """
    if isinstance(src, np.ndarray):
        if dtype is None:
            dtype = src.dtype
        if shape is None and np.dtype(dtype) == src.dtype:
            shape = src.shape
    buf = _read_src(src)
    arr = np.frombuffer(buf, dtype=np.dtype(np.uint8 if dtype is None else dtype))
    return arr if shape is None else arr.reshape(shape)


class _RunningPredictor(Codec):
    """Delta and XOR: one encode ufunc, its inverse running along ``axis``."""

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
        np.float16, np.float32, np.float64,
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
        """Encode ``data`` along ``axis`` at ``dist``.

        An ndarray is taken with its dtype and shape; bytes-like input
        is uint8, as in imagecodecs.
        """
        reject_unknown(f"{self.name} encode", opts)
        arr = as_input_array(data)
        result = _predict(arr, axis, dist, self._encode_ufunc,
                          f"{self.name} encode")
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
        """Decode to an array of ``dtype`` and ``shape``.

        Without them an ndarray ``src`` gives its own dtype and shape, and
        bytes give flat uint8. The result keeps ``dtype``'s byte order.
        """
        reject_unknown(f"{self.name} decode", opts)
        arr = _decode_array(src, dtype, shape)
        if out is not None:
            return _unpredict_into(arr, axis, dist, out, self._kernel(),
                                   self._decode_ufunc, f"{self.name} decode")
        return _unpredict(arr, axis, dist, self._kernel(), self._decode_ufunc,
                          f"{self.name} decode")

    @classmethod
    def _apply_along(cls, arr: np.ndarray, axis: int, dist: int, mode: str) -> np.ndarray:
        """Array in, array of the same dtype out: ``mode`` is encode or decode."""
        if mode == "encode":
            return _predict(arr, axis, dist, cls._encode_ufunc, cls.name)
        return _unpredict(arr, axis, dist, cls._kernel(), cls._decode_ufunc,
                          cls.name)


# ---------------------------------------------------------------------------
# Delta
# ---------------------------------------------------------------------------


class DeltaCodec(_RunningPredictor):
    """Delta predictor (TIFF predictor 2).

    Differences of the samples' bit patterns as unsigned integers of the
    same width, modulo 2**bits: wrapping integer arithmetic for integer
    types, and for floats the lossless form libtiff and imagecodecs use
    (a difference of float values would round). Encoded bytes equal
    imagecodecs' ``delta_encode`` for every dtype, byte order included.
    opencodecs 0.4.0 and earlier stored differences of float values for
    float data; ``decode(..., legacy_float=True)`` reads those streams.
    Decode is a prefix sum, which np.cumsum runs 5.5x slower than a plain
    C loop on this shape (34.6 ms against 6.3 ms on 17 MB of uint8)
    because of its per-element dispatch, so it runs in a compiled kernel.
    """

    name = "delta"
    aliases = ()
    file_extensions = ()
    _encode_ufunc = np.subtract
    _decode_ufunc = np.add

    @staticmethod
    def _kernel():
        return _delta_decode_kernel()

    def decode(
        self,
        src: Any,
        *,
        dtype=None,
        shape=None,
        axis: int = -1,
        dist: int = 1,
        out=None,
        legacy_float: bool = False,
        **opts,
    ) -> np.ndarray:
        """Decode to an array of ``dtype`` and ``shape``.

        Without them an ndarray ``src`` gives its own dtype and shape, and
        bytes give flat uint8. The result keeps ``dtype``'s byte order.

        ``legacy_float=True`` reads a float stream written by opencodecs
        0.4.0 or earlier, whose delta held differences of float values
        rather than of bit patterns. The stream has no header to tell the
        two apart, so the caller says which it is. The differences are
        summed as floats in the stream's dtype, in order along ``axis``,
        which gives the values those versions' decoder returned, bit for
        bit. The result is in ``dtype``'s byte order, as for any decode;
        those versions returned native byte order for a big-endian
        ``dtype``. Their encoder rounded, so the values are not always the
        data first encoded.
        """
        if not legacy_float:
            return super().decode(src, dtype=dtype, shape=shape, axis=axis,
                                  dist=dist, out=out, **opts)
        label = "delta decode"
        reject_unknown(label, opts)
        arr = _decode_array(src, dtype, shape)
        if arr.dtype.kind != "f":
            raise ValueError(
                f"{label}: legacy_float=True reads float streams, got {arr.dtype}")
        _check_predictor_dtype(arr.dtype, label)
        _check_distance(dist, label)
        axis = _check_axis(arr, axis, label)
        native = arr.dtype.newbyteorder("=")
        work = arr.astype(native, copy=True)
        selection = [slice(None)] * arr.ndim
        with np.errstate(over="ignore", invalid="ignore"):   # inf - inf
            for start in range(min(dist, arr.shape[axis])):
                selection[axis] = slice(start, None, dist)
                lane = work[tuple(selection)]
                np.add.accumulate(lane, axis=axis, dtype=native, out=lane)
        result = work if native == arr.dtype else work.astype(arr.dtype)
        if out is None:
            return result
        if not isinstance(out, np.ndarray):
            raise TypeError("predictor out must be an ndarray")
        if out.shape != arr.shape or out.dtype != arr.dtype:
            raise ValueError("predictor out shape/dtype mismatch")
        if not out.flags.writeable:
            raise ValueError("predictor out must be writable")
        np.copyto(out, result)
        return out


# ---------------------------------------------------------------------------
# XOR
# ---------------------------------------------------------------------------


class XorCodec(_RunningPredictor):
    """XOR predictor: same shape as delta but uses ``^`` instead of ``-``.

    Works on the bit patterns, so floats are XORed as IEEE 754 words (the
    Gorilla time-series form) and round-trip exactly, as in imagecodecs'
    ``xor_encode``.
    """

    name = "xor"
    aliases = ()
    file_extensions = ()
    _encode_ufunc = np.bitwise_xor
    _decode_ufunc = np.bitwise_xor

    @staticmethod
    def _kernel():
        return _xor_decode_kernel()


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
        reject_unknown("floatpred encode", opts)
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
        reject_unknown("floatpred decode", opts)
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
