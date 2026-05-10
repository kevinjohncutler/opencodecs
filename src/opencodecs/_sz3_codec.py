"""Sz3Codec — Codec adapter wrapping the native _sz3 extension.

SZ3 = error-bounded lossy compressor for scientific arrays. Unlike ZFP,
SZ3 uses prediction + Huffman; for time-series and simulation data it
often beats ZFP at the same error budget. Supports float32/64 and 8-64
bit signed/unsigned integers, in 1D-5D.

Modes::

    mode='abs', abs_err=1e-3      # |orig - reconstructed| <= 1e-3 (per pixel)
    mode='rel', rel_err=1e-4      # error <= 1e-4 * (max - min)
    mode='abs_or_rel', ...        # whichever bound is met first
    mode='psnr', psnr=80          # target peak SNR in dB
    mode='norm', abs_err=...      # L2-norm bounded

Example::

    arr = np.random.rand(64, 128, 128).astype(np.float32)
    blob = oc.write(None, arr, format="sz3", mode="abs", abs_err=1e-3)
    back = oc.read(blob, format="sz3")
    assert np.abs(arr - back).max() <= 1e-3
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _sz3_encode, _sz3_decode, _sz3_check_signature, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._sz3",
    "encode", "decode", "check_signature",
)


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
               mode: str = "abs",
               abs_err: float = 1e-3,
               rel_err: float = 0.0,
               psnr: float = 0.0,
               **opts) -> bytes | None:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        if data.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(
                f"sz3 encode: only float32/float64 supported "
                f"(got {data.dtype!r}); use 'zfp' or 'aec' for integer arrays"
            )
        out = _sz3_encode(
            data, mode=mode, abs_err=float(abs_err),
            rel_err=float(rel_err), psnr=float(psnr),
        )
        return _write_dest(out, dest)

    def decode(self, src: Any, **opts) -> np.ndarray:
        return _sz3_decode(_read_src(src))


__all__ = ["Sz3Codec"]
