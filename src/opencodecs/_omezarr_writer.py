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

Deferred
--------
* Sharded v3 storage (write side — read side shipped previously)
* User-supplied filter chains
* Custom dimension separators (we always use "/")
* HTTP write (S3 PUT)
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

    n_workers = _resolve_workers(workers)
    if zarr_format == 2:
        _write_v2(root, arr, chunks, compressor, compression_level,
                  fill_value, n_workers)
    elif zarr_format == 3:
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
# OmeZarrPyramidWriter — multi-scale group
# ---------------------------------------------------------------------------


def write_omezarr_pyramid(
    path: str | Path,
    levels: list[np.ndarray],
    *,
    chunks: tuple[int, ...] | None = None,
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

    for i, lvl in enumerate(levels):
        write_zarr_array(
            root / str(i),
            lvl,
            chunks=chunks if chunks is not None else lvl.shape,
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
        chunks=chunks, compressor=compressor,
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
