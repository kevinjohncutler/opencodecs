"""OME-Zarr v0.4 (Zarr v2) and v0.5 (Zarr v3) reader.

The cloud-native counterpart to OME-TIFF / COG. An OME-Zarr dataset
is a directory tree of JSON metadata + binary chunk files; pixels at
each pyramid level are stored as a Zarr array, with the group's
``multiscales`` metadata declaring how the levels relate.

This module exposes two readers:

* :class:`OmeZarrArray` — a single Zarr v2 or v3 array. Supports the
  common codecs (raw / zstd / blosc / blosc2 / gzip) via the existing
  opencodecs codec dispatchers, falling back to numcodecs for anything
  else. Chunks intersecting a region are loaded and assembled.

* :class:`OmeZarrPyramidDataset` — an OME-Zarr group containing
  multiple resolution levels. Implements
  :class:`opencodecs.core.pyramid.PyramidReader`, so
  ``read_region(level, y=, x=)`` does the same tile-aware partial
  read the TIFF pyramid reader does, but for Zarr.

Scope of v1
-----------
* Zarr v2 + Zarr v3 array metadata (both formats coexist in the wild).
* Codecs: raw / zstd / blosc(1) / blosc2 / gzip. (blosc1 routes
  through numcodecs because opencodecs's _blosc2 only handles blosc2
  framing.)
* Local filesystem store. HTTP-range support is a follow-up — would
  reuse the existing ``opencodecs._tiff_http.HTTPDataSource`` shape.
* No support yet for sharded v3 storage, transpose codec, or
  user-supplied filter chains (deferred).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .core.pyramid import PyramidLevel, PyramidReader, _normalize_axis


# ---------------------------------------------------------------------------
# Store abstraction — just a key → bytes mapping with bool membership.
# ---------------------------------------------------------------------------


class _FsStore:
    """Local-filesystem store. Each key is a path relative to ``root``."""

    __slots__ = ("root",)

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def __contains__(self, key: str) -> bool:
        return (self.root / key).exists()

    def __getitem__(self, key: str) -> bytes:
        with open(self.root / key, "rb") as f:
            return f.read()


# ---------------------------------------------------------------------------
# Codec dispatcher — zarr codec name → decoder callable.
# ---------------------------------------------------------------------------
#
# Strategy:
#   1. Use opencodecs's native codec for the codecs we have, native
#      paths are usually fastest (we control the bindings).
#   2. Fall back to numcodecs for anything else (filters, exotic codecs,
#      blosc1 with non-trivial sub-codec config).
#
# v2 stores the codec as ``{"id": "<name>", ...config}``.
# v3 stores codecs as an ordered list ``[{"name": "<name>", "configuration": {...}}, ...]``.
# We dispatch on the name; configuration is passed through.

def _decompress_chunk(raw: bytes, codec_spec: dict | list | None) -> bytes:
    """Decompress a single chunk's bytes.

    ``codec_spec`` is the v2 compressor dict (``{"id": ...}``) or the
    v3 codec list. Returns raw (uncompressed) bytes ready to be viewed
    as the chunk's dtype.
    """
    if codec_spec is None:
        # No compression — raw bytes in the file.
        return raw

    # Normalize v2/v3 into a list of (name, config) tuples in pipeline
    # order. v2 has a single compressor and optional filters; we
    # ignore filters here (they're rarely used in OME-Zarr) and would
    # bail if present.
    if isinstance(codec_spec, dict):
        # v2: just the compressor.
        name = codec_spec.get("id") or codec_spec.get("name")
        chain = [(name, codec_spec)]
    else:
        # v3: list of codecs. The "bytes" codec is a typed-array
        # adapter; we handle it implicitly via numpy frombuffer.
        chain = []
        for c in codec_spec:
            n = c.get("name") or c.get("id")
            cfg = c.get("configuration", c)
            chain.append((n, cfg))

    # Reverse: v3 codec lists are encode-order; on decode we run them
    # in reverse. v2 has only one entry so order doesn't matter.
    for name, cfg in reversed(chain):
        if name in (None, "bytes"):
            # "bytes" is an ndarray ↔ bytes adapter — endianness only,
            # no decompression. Handled by frombuffer afterwards.
            continue
        if name == "crc32c":
            # Integrity check; v3 stores it after compression. The
            # trailing 4 bytes are the CRC. Drop them and trust the
            # checksum.
            raw = bytes(raw)[:-4]
            continue
        raw = _run_decoder(name, cfg, raw)
    return raw


_DECODER_CACHE: dict[str, Any] = {}


def _run_decoder(name: str, cfg: dict, data: bytes) -> bytes:
    """Decode one codec step. Prefers opencodecs native; falls back
    to numcodecs."""
    # opencodecs native codecs we have:
    if name == "zstd":
        fn = _DECODER_CACHE.get("zstd")
        if fn is None:
            from .codecs._zstd import decode as _zstd_decode
            fn = _zstd_decode
            _DECODER_CACHE["zstd"] = fn
        return fn(data)
    if name == "blosc2":
        fn = _DECODER_CACHE.get("blosc2")
        if fn is None:
            from .codecs._blosc2 import decode as _b_decode
            fn = _b_decode
            _DECODER_CACHE["blosc2"] = fn
        return fn(data)
    if name == "gzip":
        # Native gzip via Python stdlib. opencodecs's _deflate is raw
        # DEFLATE; gzip adds a header + trailer. stdlib gzip is fine
        # here — it's a thin C wrapper around zlib.
        import gzip
        return gzip.decompress(bytes(data))

    # Fallback: numcodecs. Robust but slower than our native path.
    nc = _DECODER_CACHE.get("__numcodecs__")
    if nc is None:
        try:
            import numcodecs
            nc = numcodecs
            _DECODER_CACHE["__numcodecs__"] = nc
        except ImportError as e:  # pragma: no cover - rare
            raise ImportError(
                f"OmeZarrArray: codec {name!r} not implemented natively "
                f"and numcodecs is unavailable for fallback"
            ) from e
    codec_cls = nc.get_codec({**cfg, "id": name}) if "id" not in cfg \
        else nc.get_codec(cfg)
    return bytes(codec_cls.decode(data))


# ---------------------------------------------------------------------------
# OmeZarrArray — single Zarr v2 / v3 array
# ---------------------------------------------------------------------------


_V3_DTYPE_MAP = {
    "bool": np.dtype("?"),
    "int8": np.dtype("i1"),  "uint8":  np.dtype("u1"),
    "int16": np.dtype("i2"), "uint16": np.dtype("u2"),
    "int32": np.dtype("i4"), "uint32": np.dtype("u4"),
    "int64": np.dtype("i8"), "uint64": np.dtype("u8"),
    "float16": np.dtype("f2"),
    "float32": np.dtype("f4"),
    "float64": np.dtype("f8"),
    "complex64": np.dtype("c8"),
    "complex128": np.dtype("c16"),
}


class OmeZarrArray:
    """Read access to one Zarr v2 or v3 array.

    Created either with a path (local filesystem store) or directly
    given a store object. The metadata is parsed eagerly; chunks are
    loaded on demand by :meth:`read_region`.
    """

    def __init__(self, path: str | Path):
        self._root = Path(path)
        if not self._root.is_dir():
            raise FileNotFoundError(f"OmeZarrArray: not a directory: {self._root}")
        self._store = _FsStore(self._root)

        # Try Zarr v3 first (zarr.json); fall back to v2 (.zarray).
        if "zarr.json" in self._store:
            self._parse_v3()
        elif ".zarray" in self._store:
            self._parse_v2()
        else:
            raise FileNotFoundError(
                f"OmeZarrArray: no zarr.json (v3) or .zarray (v2) in "
                f"{self._root}"
            )

    # ----- metadata -----

    def _parse_v3(self) -> None:
        meta = json.loads(self._store["zarr.json"].decode("utf-8"))
        if meta.get("node_type") != "array":
            raise ValueError(
                f"OmeZarrArray: zarr.json is not an array node "
                f"(node_type={meta.get('node_type')!r})"
            )
        self.zarr_format = 3
        self.shape = tuple(meta["shape"])
        dt_name = meta["data_type"]
        if isinstance(dt_name, dict):
            # v3 extension types (datetime, structured, etc.); not yet
            # supported.
            raise NotImplementedError(
                f"OmeZarrArray: v3 extension dtype not supported: {dt_name}"
            )
        if dt_name not in _V3_DTYPE_MAP:
            raise NotImplementedError(f"OmeZarrArray: unsupported v3 dtype {dt_name!r}")
        self.dtype = _V3_DTYPE_MAP[dt_name]
        grid = meta["chunk_grid"]["configuration"]
        self.chunks = tuple(grid["chunk_shape"])
        self.fill_value = meta.get("fill_value", 0)
        self._codecs = meta.get("codecs", [])
        self._chunk_key_sep = (
            meta.get("chunk_key_encoding", {})
            .get("configuration", {})
            .get("separator", "/")
        )
        # Resolve byte order from the "bytes" codec if present.
        self._byte_order = "<"
        for c in self._codecs:
            if c.get("name") == "bytes":
                self._byte_order = (
                    "<" if c.get("configuration", {}).get("endian", "little")
                          == "little" else ">"
                )

    def _parse_v2(self) -> None:
        meta = json.loads(self._store[".zarray"].decode("utf-8"))
        if int(meta.get("zarr_format", 2)) != 2:
            raise ValueError(
                f"OmeZarrArray: .zarray claims zarr_format != 2 "
                f"({meta.get('zarr_format')})"
            )
        self.zarr_format = 2
        self.shape = tuple(meta["shape"])
        self.chunks = tuple(meta["chunks"])
        self.dtype = np.dtype(meta["dtype"])
        self.fill_value = meta.get("fill_value", 0)
        self._codecs = meta.get("compressor")
        self._chunk_key_sep = meta.get("dimension_separator", ".")
        if meta.get("order", "C") != "C":
            # F-order would mean re-arranging chunk axes on decode;
            # zarr-python writes C by default in OME-Zarr.
            raise NotImplementedError(
                "OmeZarrArray: F-order Zarr v2 arrays not yet supported"
            )
        if meta.get("filters"):
            raise NotImplementedError(
                "OmeZarrArray: v2 filter chains not yet supported"
            )
        self._byte_order = self.dtype.byteorder or "="

    # ----- chunk addressing -----

    def _chunk_key(self, chunk_idx: tuple[int, ...]) -> str:
        if self.zarr_format == 3:
            return "c" + self._chunk_key_sep + self._chunk_key_sep.join(
                str(i) for i in chunk_idx
            )
        # v2
        return self._chunk_key_sep.join(str(i) for i in chunk_idx)

    # ----- chunk decode -----

    def _decode_chunk(self, raw: bytes) -> np.ndarray:
        """Decompress + reshape into a chunk-shaped ndarray."""
        decoded = _decompress_chunk(raw, self._codecs)
        # Interpret as our dtype + byte order.
        dt = self.dtype.newbyteorder(self._byte_order)
        arr = np.frombuffer(decoded, dtype=dt)
        expected = int(np.prod(self.chunks))
        if arr.size != expected:
            raise ValueError(
                f"OmeZarrArray: chunk size mismatch "
                f"(got {arr.size} elements, expected {expected} "
                f"for chunk shape {self.chunks})"
            )
        arr = arr.reshape(self.chunks)
        # Promote to native byte order if file order differs (cheap
        # view + copy on first .astype call; here we copy now so the
        # caller never deals with non-native bytes).
        if dt.byteorder not in ("=", "|") and \
                dt.byteorder != np.dtype(self.dtype).byteorder:
            arr = arr.astype(self.dtype, copy=True)
        return arr

    def _load_chunk(self, chunk_idx: tuple[int, ...]) -> np.ndarray:
        key = self._chunk_key(chunk_idx)
        if key not in self._store:
            # Missing chunk → fill with fill_value (Zarr semantics).
            return np.full(self.chunks, self.fill_value, dtype=self.dtype)
        return self._decode_chunk(self._store[key])

    # ----- public read API -----

    def __getitem__(self, item) -> np.ndarray:
        """``arr[y_slice, x_slice, ...]`` — like a numpy view but
        only the chunks intersecting the slice are loaded."""
        return self.read_region(item)

    def read_region(self, region) -> np.ndarray:
        """Read a region given as a tuple of slices or single slice.

        ``region`` is normalized to one slice per array axis; ``None``
        / missing axes default to the full extent. Negative indices
        are NOT supported in this minimal v1.
        """
        # Normalize region → list of (start, stop) per axis.
        if not isinstance(region, tuple):
            region = (region,)
        # Pad with full slices for trailing axes.
        if len(region) < len(self.shape):
            region = region + (slice(None),) * (len(self.shape) - len(region))
        bounds = []
        for axis, s in enumerate(region):
            full = self.shape[axis]
            if isinstance(s, slice):
                start, stop, step = s.indices(full)
                if step != 1:
                    raise NotImplementedError(
                        f"OmeZarrArray: strided slice on axis {axis} "
                        f"not supported (step={step})"
                    )
            elif isinstance(s, int):
                start, stop = s, s + 1
                if start < 0: start += full
                stop = start + 1
            elif s is None:
                start, stop = 0, full
            else:
                # tuple form (start, stop)
                start, stop = int(s[0]), int(s[1])
            start = max(0, start)
            stop = min(full, stop)
            if stop < start:
                stop = start
            bounds.append((start, stop))

        out_shape = tuple(stop - start for start, stop in bounds)
        out = np.empty(out_shape, dtype=self.dtype)

        # Iterate over the chunks that intersect ``bounds`` and
        # paste their data into ``out``.
        ranges = []
        for axis, (start, stop) in enumerate(bounds):
            c = self.chunks[axis]
            i0 = start // c
            i1 = (stop - 1) // c if stop > start else i0 - 1
            ranges.append(range(i0, i1 + 1))

        # Cartesian product without itertools dependency.
        def _iter_indices(rs, prefix=()):
            if not rs:
                yield prefix
                return
            for i in rs[0]:
                yield from _iter_indices(rs[1:], prefix + (i,))

        for chunk_idx in _iter_indices(ranges):
            chunk = self._load_chunk(chunk_idx)
            # Compute the source slice within the chunk and the
            # destination slice within ``out``.
            src_slices = []
            dst_slices = []
            for axis, ci in enumerate(chunk_idx):
                c = self.chunks[axis]
                chunk_start = ci * c
                chunk_stop = chunk_start + c
                start, stop = bounds[axis]
                s_lo = max(start, chunk_start)
                s_hi = min(stop, chunk_stop)
                src_slices.append(slice(s_lo - chunk_start, s_hi - chunk_start))
                dst_slices.append(slice(s_lo - start, s_hi - start))
            out[tuple(dst_slices)] = chunk[tuple(src_slices)]

        return out

    def read(self) -> np.ndarray:
        """Read the entire array. Convenience for small arrays."""
        return self.read_region(tuple(slice(0, n) for n in self.shape))


# ---------------------------------------------------------------------------
# OmeZarrPyramidDataset — group with multiscales metadata
# ---------------------------------------------------------------------------


def _read_group_attributes(group_root: Path) -> dict:
    """Return the OME attributes block for a Zarr group, regardless
    of zarr_format. v2 stores user attrs in ``.zattrs``; v3 stores them
    inside ``zarr.json`` under ``attributes``."""
    if (group_root / "zarr.json").exists():
        meta = json.loads((group_root / "zarr.json").read_text())
        if meta.get("node_type") != "group":
            raise ValueError(
                f"OmeZarrPyramidDataset: zarr.json at {group_root} is not "
                f"a group (node_type={meta.get('node_type')!r})"
            )
        return meta.get("attributes", {})
    if (group_root / ".zattrs").exists():
        return json.loads((group_root / ".zattrs").read_text())
    return {}


def _extract_multiscales(attrs: dict) -> list[dict]:
    """Locate the OME-NGFF ``multiscales`` block in a group's attrs."""
    # NGFF v0.4 (Zarr v2): top-level "multiscales"
    if "multiscales" in attrs:
        return attrs["multiscales"]
    # NGFF v0.5 (Zarr v3): wrapped under "ome"
    ome = attrs.get("ome")
    if isinstance(ome, dict) and "multiscales" in ome:
        return ome["multiscales"]
    raise ValueError(
        "OmeZarrPyramidDataset: no 'multiscales' block in group "
        "attributes (need OME-NGFF v0.4 or v0.5 layout)"
    )


def _level_downscale(
    arrays: list[OmeZarrArray], y_axis: int, x_axis: int,
) -> list[tuple[int, int]]:
    """Compute (y, x) downscale factors relative to level 0."""
    base_y = arrays[0].shape[y_axis]
    base_x = arrays[0].shape[x_axis]
    factors = []
    for arr in arrays:
        h = arr.shape[y_axis]
        w = arr.shape[x_axis]
        factors.append((
            max(1, round(base_y / h)),
            max(1, round(base_x / w)),
        ))
    return factors


class OmeZarrPyramidDataset(PyramidReader):
    """Pyramid view of an OME-NGFF Zarr group.

    Discovers all resolution levels via the group's ``multiscales``
    metadata, opens an :class:`OmeZarrArray` per level, and exposes
    the standard pyramid API.

    Higher-dim arrays (T, C, Z, Y, X) are supported: ``read_region``
    accepts ``y=``/``x=`` and any other axis as ``**axes_indices``
    (single integer per non-spatial axis, defaulting to 0).

    Examples
    --------
    Read the lowest-resolution overview of channel 0::

        with OmeZarrPyramidDataset("/path/to/group.zarr") as p:
            best = p.best_level_for(max_pixels_y=512)
            overview = p.read_region(best, c=0)

    Crop a region from full resolution::

        crop = p.read_region(level=0, y=(1000, 2000), x=(3000, 4000), c=0)
    """

    def __init__(self, path: str | Path):
        self._root = Path(path)
        if not self._root.is_dir():
            raise FileNotFoundError(
                f"OmeZarrPyramidDataset: not a directory: {self._root}"
            )
        attrs = _read_group_attributes(self._root)
        multiscales = _extract_multiscales(attrs)
        if not multiscales:
            raise ValueError(
                f"OmeZarrPyramidDataset: empty 'multiscales' in {self._root}"
            )
        ms = multiscales[0]  # OME-NGFF allows multiple but ~always 1.
        datasets = ms.get("datasets") or []
        if not datasets:
            raise ValueError(
                f"OmeZarrPyramidDataset: no datasets in multiscales"
            )

        # Identify the y/x axes from the axes metadata (NGFF requires
        # it). Fall back to "last two axes are spatial" if the axes
        # block is absent (older NGFF).
        axes = ms.get("axes")
        if axes:
            # axes is a list of {"name": ..., "type": ...}; spatial
            # types are "space".
            spatial = [
                i for i, a in enumerate(axes)
                if a.get("type") == "space"
            ]
            if len(spatial) >= 2:
                self._y_axis, self._x_axis = spatial[-2], spatial[-1]
            else:
                # Unusual: <2 spatial axes. Fall through.
                self._y_axis = len(axes) - 2
                self._x_axis = len(axes) - 1
        else:
            self._y_axis = -2
            self._x_axis = -1

        self._axes = axes or []
        self._arrays: list[OmeZarrArray] = []
        for ds in datasets:
            rel = ds.get("path")
            if not rel:
                raise ValueError(
                    f"OmeZarrPyramidDataset: dataset entry missing 'path'"
                )
            self._arrays.append(OmeZarrArray(self._root / rel))

        # Normalize axis indices (handle negatives).
        n_dims = len(self._arrays[0].shape)
        if self._y_axis < 0:
            self._y_axis += n_dims
        if self._x_axis < 0:
            self._x_axis += n_dims

        factors = _level_downscale(
            self._arrays, self._y_axis, self._x_axis,
        )
        self._levels = [
            PyramidLevel(
                reader=arr,
                downscale=factors[i],
                shape=arr.shape[-2:] if not axes else
                (arr.shape[self._y_axis], arr.shape[self._x_axis]),
                dtype=arr.dtype,
            )
            for i, arr in enumerate(self._arrays)
        ]

    # ----- ABC contract -----

    @property
    def levels(self) -> list[PyramidLevel]:
        return self._levels

    def close(self) -> None:
        # _FsStore holds no open fds — nothing to close.
        pass

    # ----- region read with non-spatial axis indexing -----

    def read_region(
        self,
        level: int = 0,
        *,
        y: slice | tuple[int, int] | None = None,
        x: slice | tuple[int, int] | None = None,
        **axes_indices,
    ) -> np.ndarray:
        """Read a (y, x) bbox from one pyramid level.

        Non-spatial axes (t, c, z, ...) are selected via keyword
        arguments matching the axes metadata names. Defaults to 0
        for any axis not given. Returns a 2D array (the y/x bbox
        cropped from the selected (t, c, z) hyperplane).
        """
        L = self._arrays[level]
        full_h = L.shape[self._y_axis]
        full_w = L.shape[self._x_axis]
        y0, y1 = _normalize_axis(y, full_h)
        x0, x1 = _normalize_axis(x, full_w)

        # Build per-axis slices for the underlying array.
        region: list = [slice(0, n) for n in L.shape]
        region[self._y_axis] = slice(y0, y1)
        region[self._x_axis] = slice(x0, x1)
        for i, ax in enumerate(self._axes):
            if i in (self._y_axis, self._x_axis):
                continue
            name = ax.get("name")
            idx = int(axes_indices.get(name, 0))
            region[i] = slice(idx, idx + 1)

        out = L.read_region(tuple(region))
        # Squeeze the non-spatial singleton axes we just selected.
        squeeze_axes = tuple(
            i for i in range(len(L.shape))
            if i not in (self._y_axis, self._x_axis)
        )
        if squeeze_axes:
            out = np.squeeze(out, axis=squeeze_axes)
        return out

    def _read_region(self, level, y0, y1, x0, x1):
        """ABC hook — for OmeZarr we override the public read_region
        because non-spatial axis selection is via kwargs. This stub
        is here so the ABC is satisfied; users should call read_region
        directly."""
        raise NotImplementedError(
            "OmeZarrPyramidDataset.read_region needs non-spatial axis "
            "kwargs (t=, c=, z=, …); call read_region(level, y=, x=, "
            "**axes) directly."
        )


__all__ = ["OmeZarrArray", "OmeZarrPyramidDataset"]
