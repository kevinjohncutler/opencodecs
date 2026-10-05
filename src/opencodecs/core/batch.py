"""Decoding many independent compressed chunks in one call.

A chunked array stores its pixels as thousands of small streams (zarr
chunks, TIFF strips and tiles, HDF5 chunks), each of which decodes on its
own. Decoding them one Python call at a time costs some tens of
microseconds of interpreter work per chunk, all of it under the GIL, and
that, not the decoder, is what limits threads: 256 MiB stored as 64 KiB
chunks (4096 of them) decoded more slowly on 32 threads than on 8, and
LZ4 more slowly on either than on one.

:func:`decode_batch` hands the codec module a whole run of chunks per
call instead. The module resolves every source and destination while it
holds the GIL, then decodes the run with one reused decoder context and
the GIL released. A call is split into contiguous runs, one per worker,
on the shared pool, with the worker count from
:func:`~opencodecs.core.parallel.resolve_workers`; so the GIL is taken a
few times per worker rather than once per chunk.
"""

from __future__ import annotations

import importlib
from typing import Any, Sequence

import numpy as np

from .parallel import resolve_workers, run_batched

# name -> (module, run function, error class)
_CODECS = {
    "zstd": ("_zstd", "decode_run", "ZstdError"),
    "deflate": ("_deflate", "decode_run", "ZlibError"),
    "lz4": ("_lz4", "decode_run", "Lz4Error"),
    "lz4block": ("_lz4", "block_decode_run", "Lz4Error"),
}
_ALIASES = {"zlib": "deflate", "lz4f": "lz4", "lz4_block": "lz4block"}


def _resolve(codec: str):
    name = _ALIASES.get(codec, codec)
    if name not in _CODECS:
        raise ValueError(
            f"decode_batch: codec {codec!r} not supported; one of "
            f"{sorted(set(_CODECS) | set(_ALIASES))}")
    module_name, run_name, error_name = _CODECS[name]
    module = importlib.import_module(f"opencodecs.codecs.{module_name}")
    return name, module, getattr(module, run_name), getattr(module, error_name)


def _piece(buffers, offsets, i):
    """Item ``i`` of a sequence of buffers, or of one buffer cut by offsets,
    as a byte memoryview."""
    if offsets is None:
        return memoryview(buffers[i]).cast("B")
    return memoryview(buffers).cast("B")[int(offsets[i]):int(offsets[i + 1])]


def _raise_for(name, module, error, chunks, chunk_offsets, out, offsets,
               index, raw) -> None:
    """Raise what decoding chunk ``index`` alone raises.

    The native run reports only that a chunk failed. Decoding that chunk
    again through the codec's own ``decode`` gives the exception, message
    included, that a loop of ``decode`` calls would have raised.
    """
    target = _piece(out, None if isinstance(out, (list, tuple)) else offsets, index)
    chunk = chunks[index] if chunk_offsets is None else _piece(chunks, chunk_offsets, index)
    if name == "lz4block":
        module.block_decode(chunk, len(target))
    elif name == "deflate":
        module.decode(chunk, out=target, raw=raw)
    else:
        module.decode(chunk, out=target)
    raise error(f"{name}: chunk {index} failed to decode")


def _int64(values, what):
    array = np.ascontiguousarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"decode_batch: {what} must be one-dimensional")
    return array


def decode_batch(codec: str, chunks: Any, out: Any, *, offsets: Any = None,
                 chunk_offsets: Any = None, numthreads: int | None = None,
                 raw: bool = False) -> np.ndarray:
    """Decode every chunk of ``chunks`` into ``out``; return their sizes.

    Parameters
    ----------
    codec : str
        ``"zstd"``; ``"deflate"`` (alias ``"zlib"``) for zlib streams, or
        bare DEFLATE streams with ``raw=True``; ``"lz4"`` for LZ4 frames;
        ``"lz4block"`` for bare LZ4 blocks, each of which must fill its
        destination exactly.
    chunks : sequence of bytes-like, or one bytes-like
        The compressed chunks (bytes, memoryview, numpy arrays, mmap
        slices: anything with a contiguous buffer), or, with
        ``chunk_offsets``, one buffer holding all of them.
    out : writable buffer, or sequence of writable buffers
        One destination per chunk, or one buffer that chunk ``i`` fills
        from ``offsets[i]`` up to ``offsets[i + 1]``. Each chunk decodes
        as the codec's ``decode(chunk, out=destination)`` does.
    offsets : sequence of int, optional
        ``len(chunks) + 1`` byte offsets into a single ``out``. Without
        them a single ``out`` is divided into equal parts, one per chunk.
    chunk_offsets : sequence of int, optional
        ``n + 1`` byte offsets into a single ``chunks`` buffer: chunk ``i``
        is ``chunks[chunk_offsets[i]:chunk_offsets[i + 1]]``. Chunks read
        back to back from a file need no slicing this way.
    numthreads : int, optional
        Threads to decode with. ``None`` sizes the pool from the number of
        chunks and the output size, as the readers do
        (:func:`~opencodecs.core.parallel.resolve_workers`); ``1`` decodes
        serially on the calling thread.
    raw : bool
        For ``"deflate"``: the chunks are bare DEFLATE streams (RFC 1951).

    Returns
    -------
    numpy.ndarray
        int64, the number of bytes each chunk decoded to. A caller that
        expects a full destination checks these.

    Raises
    ------
    The exception the codec's own ``decode`` raises for the first (lowest
    index) chunk that fails, with the same message. Other chunks may have
    been decoded by then; the content of ``out`` is unspecified.
    """
    name, module, run, error = _resolve(codec)
    if raw and name != "deflate":
        raise ValueError(f"decode_batch: raw= applies to deflate, not {name}")
    if chunk_offsets is not None:
        chunk_offsets = _int64(chunk_offsets, "chunk_offsets")
        n = len(chunk_offsets) - 1
        if n < 0:
            raise ValueError("decode_batch: chunk_offsets needs at least one entry")
    else:
        n = len(chunks)
    per_chunk = isinstance(out, (list, tuple))
    if per_chunk:
        output_bytes = sum(memoryview(o).nbytes for o in out)
    else:
        output_bytes = memoryview(out).nbytes
        if offsets is None:
            if n and output_bytes % n:
                raise ValueError(
                    f"decode_batch: {output_bytes} output bytes do not divide "
                    f"into {n} equal parts; pass offsets")
            offsets = np.arange(n + 1, dtype=np.int64) * (output_bytes // n if n else 0)
        else:
            offsets = _int64(offsets, "offsets")
    sizes = np.empty(n, dtype=np.int64)
    if n == 0:
        return sizes
    workers = resolve_workers(numthreads, n, output_bytes=output_bytes)
    keywords = {"raw": True} if (name == "deflate" and raw) else {}
    if workers <= 1:
        failed = run(chunks, out, offsets, 0, n, sizes, chunk_offsets, **keywords)
    else:
        step = -(-n // workers)
        runs = [(i, min(i + step, n)) for i in range(0, n, step)]
        counts = []

        def _run(bounds):
            counts.append(run(chunks, out, offsets, bounds[0], bounds[1], sizes,
                              chunk_offsets, **keywords))

        run_batched(_run, runs, len(runs), name="decode-batch")
        failed = sum(counts)
    if failed:
        index = int(np.flatnonzero(sizes < 0)[0])
        _raise_for(name, module, error, chunks, chunk_offsets, out, offsets,
                   index, raw)
    return sizes


__all__ = ["decode_batch"]
