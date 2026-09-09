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

    def decode(self, src: Any, *, numthreads: int | None = None,
               **opts) -> np.ndarray:
        """Decode a zfp stream.

        A 3-D fixed-rate stream is decoded block-wise across threads;
        anything else goes to zfp_decompress, which is the same answer.
        """
        data = _read_src(src)
        out = self._decode_parallel(data, numthreads)
        return _zfp_decode(data) if out is None else out

    @staticmethod
    def _decode_parallel(data, numthreads):
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

        probe = _zfp_decode_block(data, 0)
        blocks = np.empty((n, 4, 4, 4), dtype=probe.dtype)
        step = (n + workers - 1) // workers
        ranges = [(i, min(i + step, n)) for i in range(0, n, step)]
        run_batched(
            lambda r: _zfp_decode_block_range(data, r[0], r[1],
                                              blocks[r[0]:r[1]]),
            ranges, workers, name="zfp")

        # Blocks are stored x-fastest over the block grid, and each is
        # a 4x4x4 cube, so the assembled volume is the block grid and
        # the intra-block axes interleaved. One transpose, one copy.
        vol = (blocks.reshape(nbz, nby, nbx, 4, 4, 4)
                     .transpose(0, 3, 1, 4, 2, 5)
                     .reshape(nbz * 4, nby * 4, nbx * 4))
        # Edge blocks are padded by the encoder when a dimension is
        # not a multiple of 4. That padding decodes to real values and
        # is not part of the array, so it is trimmed against the
        # extent the header states.
        nz, ny, nx = grid["shape"]
        return np.ascontiguousarray(vol[:nz, :ny, :nx])

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
        return _zfp_decode_block(_read_src(src), index, out=out)


__all__ = ["ZfpCodec"]
