"""Sz3Codec — Codec adapter wrapping the native _sz3 extension.

SZ3 = error-bounded lossy compressor for scientific arrays. Unlike ZFP,
SZ3 uses prediction + Huffman; for time-series and simulation data it
often beats ZFP at the same error budget. Supports float32/64 in 1D-4D
(dimensions of size 1 do not count toward the four).

Modes::

    mode='abs', abs_err=1e-3      # |orig - reconstructed| <= 1e-3 (per pixel)
    mode='rel', rel_err=1e-4      # error <= 1e-4 * (max - min)
    mode='abs_or_rel', ...        # whichever bound is met first
    mode='abs_and_rel', ...       # both bounds

The SZ3 C API (``SZ_compress_args``) implements only these four. It
declares PSNR and L2-norm modes too but exits the process when asked
for them, so ``mode='psnr'`` and ``mode='norm'`` raise ValueError.

``abs`` and ``rel`` (imagecodecs' names) are aliases of ``abs_err`` and
``rel_err``, and ``mode`` also takes imagecodecs' ``SZ3.MODE`` values.
The defaults are imagecodecs': mode ``'abs'`` with both bounds 0.0, so
a call with no arguments writes the same bytes as
``imagecodecs.sz3_encode(data)`` and loses nothing. Pass a bound to
compress lossily.

Wire format
-----------
``encode`` writes SZ3's own stream (magic 0xF342F310, payload, then
the configuration), byte for byte what ``imagecodecs.sz3_encode``
writes. The stream records the dimensions but not reliably the data
type, so ``decode`` needs ``dtype=`` (or an ``out=`` array), as
``imagecodecs.sz3_decode`` does; ``shape=`` defaults to the stored
dimensions, which leave out the ones of size 1. Blobs from releases up
to 0.4.0 start with a private 'SZ3O' preamble that recorded both;
``decode`` still reads them without arguments.

SZ3 reads a payload without bounds checks and throws exceptions its C
API does not catch, so a stream SZ3 cannot read ends the process. Before
SZ3 sees a stream, ``decode`` checks its header, configuration and the
layout of its payload, and raises when the stream is truncated, comes
from another SZ3 data version, or is not laid out the way SZ3 writes its
configuration. The layout also shows the value type wherever the stream holds values SZ3 could not
predict (they are stored as they are), so a ``dtype`` that does not
match raises ValueError. A stream with no such values has the same
layout for float32 and float64, and a wrong ``dtype`` then decodes
without an error. Damage inside the coded data that leaves the layout
intact can still crash SZ3, as it does in imagecodecs.

Example::

    arr = np.random.rand(64, 128, 128).astype(np.float32)
    blob = oc.write(None, arr, format="sz3", mode="abs", abs_err=1e-3)
    back = oc.read(blob, format="sz3", dtype=arr.dtype, shape=arr.shape)
    assert np.abs(arr - back).max() <= 1e-3
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .core.codec import Codec
from .core.buffers import array_output
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _sz3_encode, _sz3_decode_framed, _sz3_decode_raw, _sz3_check_signature,
    _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._sz3",
    "encode", "decode_framed", "decode_raw", "check_signature",
)


def _reject_unknown(opts):
    if opts:
        raise TypeError(f"sz3: unexpected keyword argument(s) {sorted(opts)}")


def _alias(label, ours, theirs):
    if ours is not None and theirs is not None and float(ours) != float(theirs):
        raise ValueError(
            f"sz3: {label} given twice with different values "
            f"({ours} and {theirs})")
    return ours if ours is not None else theirs


class Sz3Codec(Codec):
    """Native SZ3 — modern error-bounded lossy compressor."""

    name = "sz3"
    file_extensions = (".sz3",)

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    # SZ3 v3 C API only dispatches the float types; integer types are
    # declared in sz3c.h but not implemented (decompress raises
    # "dataType N not support"). Use ZFP for integer arrays.
    supported_dtypes = (np.float32, np.float64)
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return _sz3_check_signature(head)

    def encode(self, data: Any, *, dest=None,
               mode="abs",
               abs_err: float | None = None,
               rel_err: float | None = None,
               psnr: float | None = None,
               abs: float | None = None,
               rel: float | None = None,
               **opts) -> bytes | None:
        _reject_unknown(opts)
        if psnr is not None:
            raise ValueError(
                "sz3 encode: PSNR mode is not available; the SZ3 C API does "
                "not implement it (use mode='abs' or 'rel')")
        a = _alias("abs_err", abs_err, abs)
        r = _alias("rel_err", rel_err, rel)
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        if data.dtype.newbyteorder("=") not in (np.dtype(np.float32),
                                                np.dtype(np.float64)):
            raise ValueError(
                f"sz3 encode: only float32/float64 supported "
                f"(got {data.dtype!r}); use 'zfp' or 'aec' for integer arrays"
            )
        if not data.dtype.isnative:
            data = data.astype(data.dtype.newbyteorder("="))
        out = _sz3_encode(
            data, mode="abs" if mode is None else mode,
            abs_err=0.0 if a is None else float(a),
            rel_err=0.0 if r is None else float(r),
        )
        return _write_dest(out, dest)

    def decode(self, src: Any, *, shape=None, dtype=None, out=None,
               **opts) -> np.ndarray:
        _reject_unknown(opts)
        buf = _read_src(src)
        out = array_output(out)
        target_shape = None if shape is None else (
            (int(shape),) if np.isscalar(shape) else tuple(int(s) for s in shape))
        target = None if dtype is None else np.dtype(dtype)
        if bytes(buf[:4]) == b"SZ3O":
            if out is not None and target_shape in (None, out.shape):
                return _sz3_decode_framed(buf, out=out)
            arr = _sz3_decode_framed(buf)
            if target is not None and arr.dtype != target:
                raise ValueError(
                    f"sz3 decode: the blob holds {arr.dtype}, not {target}")
            if target_shape is not None and target_shape != arr.shape:
                if math.prod(target_shape) != arr.size:
                    raise ValueError(
                        f"sz3 decode: shape {target_shape} does not match "
                        f"the {arr.size} decoded values")
                arr = arr.reshape(target_shape)
            if out is not None:
                if out.shape != arr.shape or out.dtype != arr.dtype:
                    raise ValueError("sz3 decode: out= shape/dtype mismatch")
                np.copyto(out, arr)
                return out
            return arr
        if out is not None:
            if target is None:
                target = out.dtype
            if target_shape is None:
                target_shape = out.shape
        if target is None:
            raise ValueError(
                "sz3 decode: an SZ3 stream does not reliably record its data "
                "type; pass dtype= (or out=)")
        return _sz3_decode_raw(buf, dtype=target, shape=target_shape, out=out)


__all__ = ["Sz3Codec"]
