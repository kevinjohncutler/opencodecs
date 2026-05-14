"""SperrCodec — Codec adapter wrapping the native _sperr extension.

SPERR is a wavelet-based error-bounded lossy compressor for scientific
float arrays. It is often the smallest-bitstream option among
ZFP / SZ3 / SPERR at the same PSNR target on smooth fields (climate,
CFD, seismic, lattice QCD).

Modes::

    mode='psnr', psnr=80         # target PSNR in dB (default)
    mode='bpp',  bpp=4.0          # target bits-per-pixel
    mode='pwe',  pwe=1e-3         # point-wise absolute error bound

Example::

    import numpy as np
    import opencodecs as oc

    arr = np.random.rand(64, 128, 128).astype(np.float32)
    blob = oc.write(None, arr, format="sperr", mode="psnr", psnr=80)
    back = oc.read(blob, format="sperr")
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _sperr_encode, _sperr_decode, _sperr_check_signature, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._sperr",
    "encode", "decode", "check_signature",
)


class SperrCodec(Codec):
    """Native SPERR — wavelet-based error-bounded lossy compressor."""

    name = "sperr"
    file_extensions = (".sperr",)

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.float32, np.float64)
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return _sperr_check_signature(head)

    def encode(self, data: Any, *, dest=None,
               mode: str = "psnr",
               psnr: float = 80.0,
               bpp: float = 4.0,
               pwe: float = 1e-3,
               chunk=(256, 256, 256),
               nthreads: int = 0,
               **opts) -> bytes | None:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        if data.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(
                f"sperr encode: only float32/float64 supported "
                f"(got {data.dtype!r}); use 'zfp' or 'sz3' for integer arrays"
            )
        if data.ndim not in (2, 3):
            raise ValueError(
                f"sperr encode: ndim must be 2 or 3 (got {data.ndim})"
            )
        out = _sperr_encode(
            data,
            mode=mode,
            psnr=float(psnr), bpp=float(bpp), pwe=float(pwe),
            chunk=tuple(chunk), nthreads=int(nthreads),
        )
        return _write_dest(out, dest)

    def decode(self, src: Any, **opts) -> np.ndarray:
        return _sperr_decode(_read_src(src))


__all__ = ["SperrCodec"]
