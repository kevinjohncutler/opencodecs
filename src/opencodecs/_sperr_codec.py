"""SperrCodec — Codec adapter wrapping the native _sperr extension.

SPERR is a wavelet-based error-bounded lossy compressor for scientific
float arrays. It is often the smallest-bitstream option among
ZFP / SZ3 / SPERR at the same PSNR target on smooth fields (climate,
CFD, seismic, lattice QCD).

Modes::

    mode='psnr', psnr=80         # target PSNR in dB (default)
    mode='bpp',  bpp=4.0          # target bits-per-pixel
    mode='pwe',  pwe=1e-3         # point-wise absolute error bound

``level`` (imagecodecs' name) sets the target for whichever mode is
chosen, and ``mode`` also takes imagecodecs' ``SPERR.MODE`` values.
``chunks`` and ``numthreads`` are aliases of ``chunk`` and ``nthreads``.

Wire format
-----------
``encode`` writes SPERR's own format, byte for byte what
``imagecodecs.sperr_encode`` and the sperr2d / sperr3d tools write: a
2-D slice carries SPERR's 10-byte header (``header=False`` leaves it
off), a 3-D volume always carries its own. ``decode`` takes shape and
precision from that header; a headerless 2-D stream needs ``shape``,
``dtype`` and ``header=False``. Blobs from releases up to 0.4.0 start
with a private 'SPRR' preamble; ``decode`` still reads them.

SPERR's header has no magic. ``oc.read`` without ``format=`` recognizes
a stream that carries it by its fields in the first 512 bytes (version
byte, flags SPERR sets, dimensions holding 1 to 2**40 values, the chunk
lengths there, and the fixed fields of the first coded stream as far
as those bytes hold them; the whole stream when it is shorter). In a
volume of 117 chunks or more the chunk table can push some or all of
those fields past the 512th byte, and the chunk lengths stand in for
them. A headerless 2-D stream, or a stream of more than 2**40 values,
needs ``format='sperr'``.

In ``psnr`` mode SPERR's quantization step overflows to infinity for
float64 values of very large magnitude, and the stream decodes to NaN.
``encode`` raises ``ValueError`` for such data instead of writing that
stream; ``decode`` reads one from elsewhere as NaN, as imagecodecs and
0.4.0 do.

A big-endian array is coded by its values. ``imagecodecs.sperr_encode``
codes such an array's bytes as if they were native floats, so for it
the bytes differ.

A 3-D volume is compressed as one chunk by default, as imagecodecs
does, so the default bytes are the same. ``chunks=(256, 256, 256)``
gives the chunking of SPERR's own sperr3d tool, which lets the chunks
of a large volume run on separate threads; every chunking reads back
the same way.

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
from .core.buffers import array_output
from .core.pipeline import native_workers
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _sperr_encode, _sperr_decode_framed, _sperr_decode_native,
    _sperr_mode_name, _sperr_check_signature, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._sperr",
    "encode", "decode_framed", "decode_native", "mode_name",
    "check_signature",
)


def _reject_unknown(opts):
    if opts:
        raise TypeError(f"sperr: unexpected keyword argument(s) {sorted(opts)}")


def _alias(label, ours, theirs, default):
    if ours is not None and theirs is not None and ours != theirs:
        raise ValueError(
            f"sperr: {label} given twice with different values "
            f"({ours} and {theirs})")
    if ours is not None:
        return ours
    return default if theirs is None else theirs


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
               mode="psnr",
               psnr: float = 80.0,
               bpp: float = 4.0,
               pwe: float = 1e-3,
               level: float | None = None,
               chunk=None,
               nthreads: int | None = None,
               header: bool = True,
               chunks=None,
               numthreads: int | None = None,
               **opts) -> bytes | None:
        _reject_unknown(opts)
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        if data.dtype.newbyteorder("=") not in (np.dtype(np.float32),
                                                np.dtype(np.float64)):
            raise ValueError(
                f"sperr encode: only float32/float64 supported "
                f"(got {data.dtype!r}); use 'zfp' or 'sz3' for integer arrays"
            )
        if data.ndim not in (2, 3):
            raise ValueError(
                f"sperr encode: ndim must be 2 or 3 (got {data.ndim})"
            )
        if not data.dtype.isnative:
            data = data.astype(data.dtype.newbyteorder("="))
        mode = _sperr_mode_name(mode)
        if level is not None:
            if mode == "psnr":
                psnr = level
            elif mode == "bpp":
                bpp = level
            else:
                pwe = level
        chunk = _alias("chunk", None if chunk is None else tuple(chunk),
                       None if chunks is None else tuple(chunks),
                       None)
        threads = _alias("nthreads", nthreads, numthreads, 0)
        out = _sperr_encode(
            data,
            mode=mode,
            psnr=float(psnr), bpp=float(bpp), pwe=float(pwe),
            chunk=chunk, nthreads=int(native_workers(threads)),
            header=bool(header),
        )
        return _write_dest(out, dest)

    def decode(self, src: Any, *, shape=None, dtype=None, header: bool = True,
               nthreads: int | None = None, numthreads: int | None = None,
               out=None, **opts) -> np.ndarray:
        _reject_unknown(opts)
        buf = _read_src(src)
        out = array_output(out)
        threads = int(native_workers(_alias("nthreads", nthreads, numthreads, 0)))
        if out is not None:
            if dtype is None:
                dtype = out.dtype
            if shape is None:
                shape = out.shape
        if bytes(buf[:4]) == b"SPRR":
            arr = _sperr_decode_framed(buf, nthreads=threads)
            if dtype is not None and np.dtype(dtype) != arr.dtype:
                arr = arr.astype(dtype)
            if shape is not None and tuple(shape) != arr.shape:
                raise ValueError(
                    f"sperr decode: shape {tuple(shape)} does not match the "
                    f"blob's {arr.shape}")
        else:
            arr = _sperr_decode_native(buf, shape=shape, dtype=dtype,
                                       header=bool(header), nthreads=threads)
        if out is not None:
            if out.shape != arr.shape or out.dtype != arr.dtype:
                raise ValueError("sperr decode: out= shape/dtype mismatch")
            np.copyto(out, arr)
            return out
        return arr


__all__ = ["SperrCodec"]
