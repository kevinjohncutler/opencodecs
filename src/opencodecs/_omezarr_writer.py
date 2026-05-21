"""OME-Zarr writer (Zarr v2 + v3) — pixel-equal output to zarr-python.

Companion to :class:`opencodecs._omezarr.OmeZarrArray`. Writes either
a single Zarr array or a full OME-NGFF group (multiple multi-scale
arrays + the multiscales metadata).

Scope of v1
-----------
* Zarr v2 (NGFF v0.4) and Zarr v3 (NGFF v0.5) on the local filesystem.
* Codecs:
    - none / raw
    - zstd via opencodecs native ``_zstd``
    - blosc2 via opencodecs native ``_blosc2`` (v2 + v3)
    - gzip via stdlib (v2 + v3)
* C-order arrays only (zarr-python's default).
* Optional ``shards=`` (Zarr v3 sharding_indexed codec) — outer chunks
  on disk are shards holding many inner sub-chunks plus a trailing
  uint64 index. One file per shard, dramatically lower file-count
  pressure on object stores. See ``write_zarr_array``.

Deferred
--------
* User-supplied filter chains
* Custom dimension separators (we always use "/")
* HTTP write (S3 PUT)
* CRC32C inner / index codec (needs crc32c dep; not blocking sharding
  itself since the reader treats missing CRC as expected)
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _resolve_workers(workers: int | None) -> int:
    """Translate None / negative → cpu_count(); clamp to >=1."""
    if workers is None or int(workers) <= 0:
        n = os.cpu_count() or 1
    else:
        n = int(workers)
    return max(1, n)


# Public name → (v2 numcodecs-id, v3 codec name).
_CODEC_NAME_MAP = {
    "none":   (None, None),
    "raw":    (None, None),
    "zstd":   ("zstd", "zstd"),
    "blosc2": ("blosc2", "blosc2"),     # blosc2 wire format (v2 + v3)
    "gzip":   ("gzip", "gzip"),
    "blosc":  ("blosc", None),          # v2-only (no v3 spec)
}


class OmeZarrWriterError(RuntimeError):
    """Raised on writer state-machine violations."""


# ---------------------------------------------------------------------------
# Codec dispatch (compress only — decode lives in _omezarr)
# ---------------------------------------------------------------------------


def _encode_chunk(raw: bytes, codec: str, level: int | None) -> bytes:
    """Compress one chunk's bytes."""
    if codec in (None, "none", "raw"):
        return raw
    if codec == "zstd":
        from .codecs._zstd import encode as zstd_encode
        return zstd_encode(raw, level=level if level is not None else 3)
    if codec == "blosc2":
        from .codecs._blosc2 import encode as b2_encode
        return b2_encode(raw, level=level if level is not None else 5)
    if codec == "gzip":
        import gzip
        return gzip.compress(raw, compresslevel=level if level is not None else 6)
    # numcodecs fallback (v2 'blosc' etc.)
    import numcodecs
    codec_obj = numcodecs.get_codec({"id": codec})
    return bytes(codec_obj.encode(raw))


# ---------------------------------------------------------------------------
# OmeZarrArrayWriter — single Zarr array
# ---------------------------------------------------------------------------


_DTYPE_TO_V3_NAME = {
    np.dtype("?"):  "bool",
    np.dtype("i1"): "int8",   np.dtype("u1"): "uint8",
    np.dtype("i2"): "int16",  np.dtype("u2"): "uint16",
    np.dtype("i4"): "int32",  np.dtype("u4"): "uint32",
    np.dtype("i8"): "int64",  np.dtype("u8"): "uint64",
    np.dtype("f2"): "float16",
    np.dtype("f4"): "float32",
    np.dtype("f8"): "float64",
    np.dtype("c8"):  "complex64",
    np.dtype("c16"): "complex128",
}


def _v3_dtype_name(dtype: np.dtype) -> str:
    try:
        return _DTYPE_TO_V3_NAME[dtype]
    except KeyError:
        raise OmeZarrWriterError(f"unsupported v3 dtype {dtype}")


def write_zarr_array(
    path: str | Path,
    arr: np.ndarray,
    *,
    chunks: tuple[int, ...] | None = None,
    shards: tuple[int, ...] | None = None,
    compressor: str = "zstd",
    compression_level: int | None = None,
    zarr_format: int = 2,
    fill_value: int | float = 0,
    workers: int | None = None,
) -> None:
    """Write a single Zarr array (v2 or v3) to ``path``.

    Pixel-equal to what zarr-python's ``zarr.create_array`` + ``arr[:] = x``
    would produce — we verify this in tests by reading the same data
    back via zarr-python.

    Parameters
    ----------
    path : path-like
        Directory to write. Will be created (must not exist or must
        be empty).
    arr : ndarray
        Source data.
    chunks : tuple or None
        Chunk shape per axis. ``None`` uses ``arr.shape`` (single chunk),
        which is fine for small arrays but unusual for OME-Zarr.
    shards : tuple or None
        Outer-shard shape (Zarr v3 only). When given, enables the
        ``sharding_indexed`` codec: each shard file on disk holds
        ``prod(shards / chunks)`` inner sub-chunks plus a trailing
        index. Each shard axis must be a multiple of the corresponding
        chunk axis. With ``shards=None`` (default) each chunk is its
        own file. Sharding is the right choice for very large arrays
        on object stores — fewer files, fewer ``PUT`` calls; the
        reader does range-fetches per inner chunk.
    compressor : ``"none"``, ``"zstd"``, ``"blosc2"``, ``"gzip"`` or any
        numcodecs id for ``zarr_format=2``. For ``zarr_format=3`` only
        the named codecs above plus ``"none"`` are supported.
    compression_level : passed through to the codec.
    zarr_format : 2 (NGFF v0.4) or 3 (NGFF v0.5).
    fill_value : Zarr fill value for absent chunks. Defaults to 0.
    workers : int, optional
        Parallel encode workers (ThreadPoolExecutor). ``None`` or ``<=0``
        uses ``os.cpu_count()``; ``1`` forces serial. Chunk encoders
        release the GIL (zstd, blosc2, gzip's zlib path all do),
        producing near-linear speedup on multi-core machines.
    """
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    chunks = tuple(int(c) for c in (chunks or arr.shape))
    if len(chunks) != arr.ndim:
        raise OmeZarrWriterError(
            f"chunks rank {len(chunks)} != array rank {arr.ndim}"
        )

    if shards is not None:
        if zarr_format != 3:
            raise OmeZarrWriterError(
                "sharding (shards=) is a Zarr v3 feature; pass "
                "zarr_format=3 to enable it."
            )
        shards = tuple(int(s) for s in shards)
        if len(shards) != arr.ndim:
            raise OmeZarrWriterError(
                f"shards rank {len(shards)} != array rank {arr.ndim}"
            )
        for s, c, axis in zip(shards, chunks, range(arr.ndim)):
            if s % c != 0:
                raise OmeZarrWriterError(
                    f"shard axis {axis} ({s}) must be a multiple of "
                    f"chunk axis ({c}); each shard holds a whole "
                    f"number of inner chunks."
                )

    n_workers = _resolve_workers(workers)
    if zarr_format == 2:
        _write_v2(root, arr, chunks, compressor, compression_level,
                  fill_value, n_workers)
    elif zarr_format == 3:
        if shards is not None:
            _write_v3_sharded(root, arr, chunks, shards, compressor,
                              compression_level, fill_value, n_workers)
        else:
            _write_v3(root, arr, chunks, compressor, compression_level,
                      fill_value, n_workers)
    else:
        raise OmeZarrWriterError(
            f"zarr_format must be 2 or 3 (got {zarr_format})"
        )


def _chunk_iter(shape, chunks):
    """Yield (chunk_idx, slice_tuple) per chunk in row-major order."""
    n_per_axis = tuple(
        (s + c - 1) // c for s, c in zip(shape, chunks)
    )
    def _walk(axis: int, prefix: tuple[int, ...]):
        if axis == len(shape):
            yield prefix
            return
        for i in range(n_per_axis[axis]):
            yield from _walk(axis + 1, prefix + (i,))
    for idx in _walk(0, ()):
        slc = tuple(
            slice(i * c, min((i + 1) * c, s))
            for i, c, s in zip(idx, chunks, shape)
        )
        yield idx, slc


def _make_chunk_bytes(
    arr: np.ndarray, slc: tuple[slice, ...], chunks: tuple[int, ...],
    fill_value,
) -> bytes:
    """Cut a chunk-sized region out of ``arr`` and serialize to bytes.
    Pads with fill_value when the slice is smaller than chunks (edge
    chunks at the array's right/bottom)."""
    block = arr[slc]
    if block.shape != tuple(chunks):
        padded = np.full(chunks, fill_value, dtype=arr.dtype)
        padded[tuple(slice(0, n) for n in block.shape)] = block
        block = padded
    return np.ascontiguousarray(block).tobytes()


def _encode_one_chunk_v2(args):
    """Worker: cut + compress one chunk → (key_path, encoded_bytes)."""
    arr, slc, chunks, fill_value, compressor, level, key_path = args
    raw = _make_chunk_bytes(arr, slc, chunks, fill_value)
    out = _encode_chunk(raw, compressor, level)
    return key_path, out


def _encode_one_chunk_v3(args):
    """Worker (v3 layout)."""
    arr, slc, chunks, fill_value, compressor, level, key_path = args
    raw = _make_chunk_bytes(arr, slc, chunks, fill_value)
    out = _encode_chunk(raw, compressor, level)
    return key_path, out


def _write_v2(
    root: Path, arr: np.ndarray, chunks: tuple[int, ...],
    compressor: str, level: int | None, fill_value,
    n_workers: int = 1,
) -> None:
    v2_id = _CODEC_NAME_MAP.get(compressor, (compressor, None))[0]
    metadata = {
        "shape": list(arr.shape),
        "chunks": list(chunks),
        "dtype": arr.dtype.str,
        "fill_value": fill_value,
        "order": "C",
        "filters": None,
        "dimension_separator": ".",
        "zarr_format": 2,
    }
    if v2_id in (None,):
        metadata["compressor"] = None
    elif compressor == "zstd":
        metadata["compressor"] = {"id": "zstd",
                                  "level": level if level is not None else 3}
    elif compressor == "blosc2":
        metadata["compressor"] = {"id": "blosc2",
                                  "level": level if level is not None else 5}
    elif compressor == "gzip":
        metadata["compressor"] = {"id": "gzip",
                                  "level": level if level is not None else 6}
    elif compressor == "blosc":
        metadata["compressor"] = {"id": "blosc"}
    else:
        metadata["compressor"] = {"id": v2_id or compressor}
    (root / ".zarray").write_text(json.dumps(metadata))
    (root / ".zattrs").write_text("{}")

    sep = "."
    tasks = [
        (arr, slc, chunks, fill_value, compressor, level,
         root / sep.join(str(i) for i in idx))
        for idx, slc in _chunk_iter(arr.shape, chunks)
    ]

    if n_workers <= 1 or len(tasks) <= 1:
        for t in tasks:
            key_path, out = _encode_one_chunk_v2(t)
            key_path.write_bytes(out)
        return

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_encode_one_chunk_v2, t) for t in tasks]
        for fut in as_completed(futures):
            key_path, out = fut.result()
            key_path.write_bytes(out)


def _write_v3(
    root: Path, arr: np.ndarray, chunks: tuple[int, ...],
    compressor: str, level: int | None, fill_value,
    n_workers: int = 1,
) -> None:
    """Zarr v3 ``zarr.json`` + chunk files at ``c/<i>/<j>/...``."""
    codecs: list[dict] = [
        {"name": "bytes",
         "configuration": {"endian": "little" if arr.dtype.itemsize == 1
                            or arr.dtype.byteorder in ("<", "=", "|")
                            else "big"}},
    ]
    if compressor == "zstd":
        codecs.append({
            "name": "zstd",
            "configuration": {"level": level if level is not None else 3,
                              "checksum": False},
        })
    elif compressor == "blosc2":
        codecs.append({
            "name": "blosc2",
            "configuration": {"clevel": level if level is not None else 5},
        })
    elif compressor == "gzip":
        codecs.append({
            "name": "gzip",
            "configuration": {"level": level if level is not None else 6},
        })
    # else: raw — no extra codec.

    metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(arr.shape),
        "data_type": _v3_dtype_name(arr.dtype),
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(chunks)},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": fill_value,
        "codecs": codecs,
        "attributes": {},
        "storage_transformers": [],
    }
    (root / "zarr.json").write_text(json.dumps(metadata))

    tasks = []
    for idx, slc in _chunk_iter(arr.shape, chunks):
        sub = root / "c"
        for i in idx:
            sub = sub / str(i)
        tasks.append((arr, slc, chunks, fill_value, compressor, level, sub))

    # Pre-create chunk-key parent dirs serially so worker writes are
    # collision-free. With v3's slash-separated chunk keys most chunks
    # share parent dirs; doing this once avoids EEXIST races + mkdir
    # overhead in the hot path.
    seen_parents = set()
    for *_, sub in tasks:
        p = sub.parent
        if p not in seen_parents:
            p.mkdir(parents=True, exist_ok=True)
            seen_parents.add(p)

    if n_workers <= 1 or len(tasks) <= 1:
        for t in tasks:
            key_path, out = _encode_one_chunk_v3(t)
            key_path.write_bytes(out)
        return

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_encode_one_chunk_v3, t) for t in tasks]
        for fut in as_completed(futures):
            key_path, out = fut.result()
            key_path.write_bytes(out)


# ---------------------------------------------------------------------------
# Sharded v3 writer (sharding_indexed codec)
# ---------------------------------------------------------------------------
#
# Layout per shard file (index_location="end", matching zarr-python's
# default and what our :class:`OmeZarrArray` reader expects):
#
#     [chunk_0_bytes][chunk_1_bytes]...[chunk_N-1_bytes][index]
#
# The index is N pairs of little-endian ``uint64`` (offset, nbytes),
# one per inner chunk position in shard-local row-major order. Absent
# chunks (the array doesn't reach this slot — edge-of-array case)
# have both fields set to ``2**64 - 1`` (the Zarr v3 sentinel).
#
# CRC32C on the index footer is supported by the reader but is *not*
# emitted by this writer — it would add a hard runtime dep on the
# ``crc32c`` library and the reader treats absence as a missing
# trailing 4 bytes, not an error.

_EMPTY_SENTINEL = (1 << 64) - 1


def _v3_codecs_for_shard_inner(
    arr: np.ndarray, compressor: str, level: int | None
) -> list[dict]:
    """Build the inner-chunk codec list for ``sharding_indexed``.

    Mirrors the chain produced by :func:`_write_v3` (bytes + optional
    compressor) so the writer is consistent across sharded and
    non-sharded outputs.
    """
    endian = (
        "little" if arr.dtype.itemsize == 1
        or arr.dtype.byteorder in ("<", "=", "|")
        else "big"
    )
    codecs: list[dict] = [
        {"name": "bytes", "configuration": {"endian": endian}},
    ]
    if compressor == "zstd":
        codecs.append({
            "name": "zstd",
            "configuration": {"level": level if level is not None else 3,
                              "checksum": False},
        })
    elif compressor == "blosc2":
        codecs.append({
            "name": "blosc2",
            "configuration": {"clevel": level if level is not None else 5},
        })
    elif compressor == "gzip":
        codecs.append({
            "name": "gzip",
            "configuration": {"level": level if level is not None else 6},
        })
    return codecs


def _encode_one_shard_v3(args):
    """Worker: assemble one shard file.

    Reads its slice of the source array, encodes each inner chunk in
    row-major order, accumulates the concatenated payload + the
    ``(offset, nbytes)`` index, and returns the ready-to-write bytes.
    """
    import struct
    (arr, shard_idx, shard_shape, chunks, fill_value, compressor, level,
     key_path) = args
    ndim = arr.ndim

    # Inner chunk grid within this shard.
    chunks_per_shard = tuple(s // c for s, c in zip(shard_shape, chunks))

    # Walk inner chunks in row-major order, encoding each.
    def _inner_iter(axis: int, prefix: tuple[int, ...]):
        if axis == ndim:
            yield prefix
            return
        for i in range(chunks_per_shard[axis]):
            yield from _inner_iter(axis + 1, prefix + (i,))

    payload = bytearray()
    index_pairs: list[tuple[int, int]] = []
    cur_offset = 0

    for inner in _inner_iter(0, ()):
        # Global chunk index = shard_idx * chunks_per_shard + inner.
        global_idx = tuple(
            si * cps + ii
            for si, cps, ii in zip(shard_idx, chunks_per_shard, inner)
        )
        # Source slice the inner chunk covers in the *array*. Use
        # the same ``min(end, shape)`` clipping the unsharded writer
        # uses (``_chunk_iter`` / ``_make_chunk_bytes``) so edge
        # chunks pad correctly.
        chunk_slc = tuple(
            slice(gi * c, min((gi + 1) * c, dim))
            for gi, c, dim in zip(global_idx, chunks, arr.shape)
        )
        # If the chunk slot is entirely past the array's end on any
        # axis (zero-extent slice), mark it as missing.
        if any(s.stop <= s.start for s in chunk_slc):
            index_pairs.append((_EMPTY_SENTINEL, _EMPTY_SENTINEL))
            continue
        raw = _make_chunk_bytes(arr, chunk_slc, chunks, fill_value)
        encoded = _encode_chunk(raw, compressor, level)
        payload.extend(encoded)
        index_pairs.append((cur_offset, len(encoded)))
        cur_offset += len(encoded)

    # Append the index footer.
    index_bytes = b"".join(
        struct.pack("<QQ", off, n) for off, n in index_pairs
    )
    payload.extend(index_bytes)

    return key_path, bytes(payload)


def _write_v3_sharded(
    root: Path, arr: np.ndarray, chunks: tuple[int, ...],
    shards: tuple[int, ...], compressor: str, level: int | None,
    fill_value, n_workers: int = 1,
) -> None:
    """Zarr v3 ``zarr.json`` + one shard file per outer position."""
    inner_codecs = _v3_codecs_for_shard_inner(arr, compressor, level)
    sharding_cfg = {
        "chunk_shape": list(chunks),
        "codecs": inner_codecs,
        # Bytes-only index (uint64 offset/nbytes pairs). We don't add
        # CRC32C here; see module docstring.
        "index_codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
        ],
        "index_location": "end",
    }
    metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(arr.shape),
        "data_type": _v3_dtype_name(arr.dtype),
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(shards)},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": fill_value,
        "codecs": [
            {"name": "sharding_indexed", "configuration": sharding_cfg},
        ],
        "attributes": {},
        "storage_transformers": [],
    }
    (root / "zarr.json").write_text(json.dumps(metadata))

    # One task per OUTER shard. The shard worker handles all inner
    # chunks for that shard, so chunk-grain parallelism is per-shard.
    # For large arrays with many shards that's still plenty of work
    # to fill ``n_workers`` threads.
    tasks = []
    for shard_idx, _slc in _chunk_iter(arr.shape, shards):
        sub = root / "c"
        for i in shard_idx:
            sub = sub / str(i)
        tasks.append((arr, shard_idx, shards, chunks, fill_value,
                      compressor, level, sub))

    # Pre-create parent dirs serially so worker writes are
    # collision-free (same pattern as the unsharded v3 path).
    seen_parents = set()
    for *_, sub in tasks:
        p = sub.parent
        if p not in seen_parents:
            p.mkdir(parents=True, exist_ok=True)
            seen_parents.add(p)

    if n_workers <= 1 or len(tasks) <= 1:
        for t in tasks:
            key_path, out = _encode_one_shard_v3(t)
            key_path.write_bytes(out)
        return

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_encode_one_shard_v3, t) for t in tasks]
        for fut in as_completed(futures):
            key_path, out = fut.result()
            key_path.write_bytes(out)


# ---------------------------------------------------------------------------
# OmeZarrPyramidWriter — multi-scale group
# ---------------------------------------------------------------------------


def write_omezarr_pyramid(
    path: str | Path,
    levels: list[np.ndarray],
    *,
    chunks: tuple[int, ...] | None = None,
    shards: tuple[int, ...] | None = None,
    compressor: str = "zstd",
    compression_level: int | None = None,
    zarr_format: int = 2,
    axes: list[dict] | None = None,
    fill_value: int | float = 0,
    workers: int | None = None,
) -> None:
    """Write a full OME-NGFF pyramid (group + N arrays + multiscales
    metadata) round-trippable through ``OmeZarrPyramidDataset``.

    Parameters
    ----------
    path
        Group directory (will be created).
    levels
        ``levels[0]`` is full-resolution; subsequent levels are
        downscaled (caller-controlled — we do not downscale).
    chunks
        Per-axis chunk shape. Defaults to ``levels[0].shape``.
    compressor, compression_level
        Per-chunk codec (same options as :func:`write_zarr_array`).
    zarr_format
        2 → NGFF v0.4 (``.zattrs`` at group root holds ``multiscales``).
        3 → NGFF v0.5 (``zarr.json`` ``attributes.ome.multiscales``).
    axes
        NGFF axes spec, e.g. ``[{"name": "y", "type": "space"}, ...]``.
        Defaults to inferring 2-D ``y``/``x`` axes.
    fill_value
        Per-array fill value.
    """
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    if not levels:
        raise OmeZarrWriterError("write_omezarr_pyramid: levels is empty")
    n_dims = levels[0].ndim
    if axes is None:
        # Heuristic: last two are y, x; anything before is channel/etc.
        names = list("tczyx")[-n_dims:]
        type_for = {"t": "time", "c": "channel", "z": "space",
                    "y": "space", "x": "space"}
        axes = [{"name": n, "type": type_for[n]} for n in names]

    # Coordinate transforms: per-level downscale relative to level 0.
    base_shape = levels[0].shape
    datasets = []
    for i, lvl in enumerate(levels):
        scale = [1.0] * n_dims
        # Apply downscale on the trailing 2 spatial axes
        for ax in (-2, -1):
            if lvl.shape[ax] > 0:
                scale[ax] = base_shape[ax] / lvl.shape[ax]
        datasets.append({
            "path": str(i),
            "coordinateTransformations": [
                {"type": "scale", "scale": scale}
            ],
        })

    multiscales = [{
        "version": "0.4" if zarr_format == 2 else "0.5",
        "axes": axes,
        "datasets": datasets,
    }]

    if zarr_format == 2:
        # Group .zgroup + .zattrs at root
        (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        (root / ".zattrs").write_text(
            json.dumps({"multiscales": multiscales})
        )
    else:
        (root / "zarr.json").write_text(json.dumps({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {"ome": {"multiscales": multiscales}},
        }))

    # Adapt shards per level: if the caller pinned ``shards=`` and a
    # later level is smaller than ``shards``, clamp each shard axis
    # down to the level's shape (still keeping it a multiple of
    # ``chunks`` — drop to the smallest multiple of ``chunks`` that
    # fits). For levels with any axis smaller than the chunk, skip
    # sharding on that level entirely.
    for i, lvl in enumerate(levels):
        lvl_chunks = chunks if chunks is not None else lvl.shape
        lvl_shards = shards
        if shards is not None:
            adapted: list[int] = []
            skip = False
            for s, c, dim in zip(shards, lvl_chunks, lvl.shape):
                # smallest multiple of c that's <= dim and <= s
                cap = min(s, (dim // c) * c if dim >= c else 0)
                if cap < c:
                    # Level too small to fit even one chunk on this
                    # axis — fall back to per-chunk layout for it.
                    skip = True
                    break
                adapted.append(cap)
            lvl_shards = None if skip else tuple(adapted)
        write_zarr_array(
            root / str(i),
            lvl,
            chunks=lvl_chunks,
            shards=lvl_shards,
            compressor=compressor,
            compression_level=compression_level,
            zarr_format=zarr_format,
            fill_value=fill_value,
            workers=workers,
        )


def write_omezarr_pyramid_auto(
    path: str | Path,
    image: np.ndarray,
    *,
    pyramid_levels: int | None = None,
    pyramid_min_size: int = 512,
    pyramid_axes: tuple[int, ...] | str | None = None,
    chunks: tuple[int, ...] | None = None,
    shards: tuple[int, ...] | None = None,
    compressor: str = "zstd",
    compression_level: int | None = None,
    zarr_format: int = 2,
    axes: list[dict] | None = None,
    fill_value: int | float = 0,
    workers: int | None = None,
) -> None:
    """Write a multi-scale OME-NGFF pyramid built automatically from a
    single full-res image (opt-in convenience wrapper around
    :func:`write_omezarr_pyramid`).

    A pyramid adds ~33% on-disk size (2D, geometric series) on top of
    the full-res image. The default ``pyramid_min_size=512`` auto-stops
    when an axis would drop below that — so a 1024×1024 input yields
    just 2 levels (no surprise size bloat). Pass ``pyramid_levels=N``
    to override and force a specific depth.

    Parameters
    ----------
    path
        Group directory.
    image
        Single full-resolution array. Downsampled internally via 2x2
        mean pool on the trailing 2 spatial axes (override via
        ``pyramid_axes``).
    pyramid_levels
        Total levels including full-res. ``None`` (default) auto-stops
        at ``pyramid_min_size``.
    pyramid_min_size
        Smallest spatial dimension allowed in the smallest level.
    pyramid_axes
        Override which axes to downsample. See
        :func:`opencodecs._pyramid_build.make_pyramid_levels`.

    All other keyword arguments are forwarded to
    :func:`write_omezarr_pyramid` unchanged.
    """
    from ._pyramid_build import make_pyramid_levels
    levels = make_pyramid_levels(
        image,
        levels=pyramid_levels,
        min_size=pyramid_min_size,
        axes=pyramid_axes,
    )
    write_omezarr_pyramid(
        path, levels,
        chunks=chunks, shards=shards, compressor=compressor,
        compression_level=compression_level,
        zarr_format=zarr_format, axes=axes,
        fill_value=fill_value, workers=workers,
    )


__all__ = [
    "write_zarr_array",
    "write_omezarr_pyramid",
    "write_omezarr_pyramid_auto",
    "OmeZarrWriterError",
]
