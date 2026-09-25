"""ZfpCodec — Codec adapter wrapping the native _zfp extension.

ZFP is the standard for *fast* lossy compression of 1D-4D float / int
arrays in HPC. Self-describing blob (full header) carries shape, dtype,
and mode metadata.

Modes (pick one)::

    mode='reversible'                 # lossless
    mode='rate', rate=4               # 4 bits per value (predictable size)
    mode='precision', precision=12    # 12 bits of mantissa (predictable accuracy)
    mode='accuracy', accuracy=1e-3    # absolute error <= 1e-3 (predictable error)

Example::

    arr = np.random.rand(64, 128, 128).astype(np.float32)
    blob = oc.write(None, arr, format="zfp", mode="accuracy", accuracy=1e-4)
    back = oc.read(blob, format="zfp")
    assert np.abs(arr - back).max() <= 1e-4
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs
from .core.parallel import resolve_workers, run_batched

(
    _zfp_encode, _zfp_decode, _zfp_check_signature,
    _zfp_block_grid, _zfp_decode_block, _zfp_decode_block_range,
    _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._zfp",
    "encode", "decode", "check_signature",
    "block_grid", "decode_block", "decode_block_range",
)


class ZfpCodec(Codec):
    """Native ZFP codec — fast lossy compression for 1D-4D arrays."""

    name = "zfp"
    file_extensions = (".zfp",)

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    # In fixed-rate mode every 4x4x4 block occupies the same number of
    # bits, so block N begins at a computed bit offset. That gives both
    # of these at once and neither needs OpenMP, which is what an
    # earlier reading of the zfp docs concluded: decode_block reaches
    # one block without touching the others (0.0010 ms against 0.2094
    # ms for a full decode of the same stream, 216x), and decode()
    # splits the block grid across threads, 114.0 ms to 8.2 ms on 16
    # for a 67 MB volume, bit-identical.
    #
    # The variable-rate modes -- precision, accuracy, reversible --
    # pack blocks at whatever size they need, so there is no offset to
    # compute and both fall back to zfp_decompress.
    chunked = True
    parallel_decode = True

    supported_dtypes = (np.int32, np.int64, np.float32, np.float64)
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return _zfp_check_signature(head)

    def encode(self, data: Any, *, dest=None,
               mode: str = "reversible",
               rate=None, precision=None, accuracy=None,
               **opts) -> bytes | None:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        out = _zfp_encode(
            data, mode=mode,
            rate=rate, precision=precision, accuracy=accuracy,
        )
        return _write_dest(out, dest)

    def decode(self, src: Any, *, out=None, numthreads: int | None = None,
               **opts) -> np.ndarray:
        """Decode a zfp stream.

        A 3-D fixed-rate stream is decoded block-wise across threads;
        anything else goes to zfp_decompress, which is the same answer.
        """
        from .core.pipeline import native_workers
        numthreads = native_workers(numthreads)
        data = _read_src(src)
        from .core.buffers import array_output
        out = array_output(out)
        result = self._decode_parallel(data, numthreads, out=out)
        return _zfp_decode(data, out=out) if result is None else result

    @staticmethod
    def _decode_parallel(data, numthreads, *, out=None):
        """Threaded block decode, or None when it does not apply.

        Returns None rather than raising for every stream this path
        cannot serve, so the caller falls back to zfp_decompress and
        gets the same array either way.
        """
        try:
            grid = _zfp_block_grid(data)
        except Exception:                                # noqa: BLE001
            return None
        if grid["ndim"] != 3 or not grid["fixed_rate"]:
            return None
        nbz, nby, nbx = grid["blocks"]
        n = grid["n_blocks"]
        workers = resolve_workers(numthreads, n,
                                  output_bytes=n * 64 * 4,
                                  min_items=64)
        if workers <= 1:
            return None

        try:
            probe = _zfp_decode_block(data, 0)
        except Exception:
            # The block entry points support floating-point streams;
            # integer fixed-rate streams still use the whole decoder.
            return None
        shape = tuple(grid["shape"])
        if out is None:
            out = np.empty(shape, dtype=probe.dtype)
        elif out.shape != shape or out.dtype != probe.dtype:
            raise ValueError("zfp decode: out shape and dtype must match the stream")

        # Decode complete block rows so one transpose covers a broad
        # output band. Each worker owns bounded contiguous scratch,
        # instead of retaining every padded block plus an assembled copy.
        row_bytes = nbx * 64 * probe.dtype.itemsize
        rows_per_batch = max(1, min(nby, (1 << 20) // row_bytes))
        bands = [(z, y, min(y + rows_per_batch, nby))
                 for z in range(nbz) for y in range(0, nby, rows_per_batch)]
        nz, ny, nx = shape

        def place_band(band):
            z, y, end = band
            start = (z * nby + y) * nbx
            stop = (z * nby + end) * nbx
            blocks = np.empty((stop - start, 4, 4, 4), dtype=probe.dtype)
            _zfp_decode_block_range(data, start, stop, blocks)
            pixels = (blocks.reshape(end - y, nbx, 4, 4, 4)
                            .transpose(2, 0, 3, 1, 4)
                            .reshape(4, (end - y) * 4, nbx * 4))
            z0, y0 = z * 4, y * 4
            z1, y1 = min(z0 + 4, nz), min(end * 4, ny)
            out[z0:z1, y0:y1, :] = pixels[:z1 - z0, :y1 - y0, :nx]

        run_batched(place_band, bands, workers, name="zfp")
        return out

    def block_grid(self, src: Any) -> dict:
        """Block geometry of a stream, without decoding anything."""
        return _zfp_block_grid(_read_src(src))

    def decode_block(self, src: Any, index: int, *, out=None):
        """One 4x4x4 block of a 3-D fixed-rate stream, by index.

        Blocks are numbered x-fastest over the block grid, so block
        ``(kb * nby + jb) * nbx + ib`` is the cube at block coordinates
        (ib, jb, kb). See the extension's docstring for what fixed-rate
        buys and why the other modes cannot do this.
        """
        from .core.buffers import array_output
        return _zfp_decode_block(_read_src(src), index, out=array_output(out))


__all__ = ["ZfpCodec"]
