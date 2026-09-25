"""Shared plumbing for the HDF5-backed readers (hdf5, emd, imaris).

Source normalization and checked chunk reconstruction are shared so that
HDF5, EMD, and Imaris readers apply the same selection and filter rules.
"""

from __future__ import annotations

import io
from typing import Any


def h5_source(src: Any) -> Any:
    """What h5py can open, from what a codec is handed.

    An http(s) URL becomes a range-reading file-like, so h5py fetches
    only the chunks a slice touches.

    h5py takes a path or a file-like object but not raw bytes: given
    those it treats them as a filename and raises FileNotFoundError
    with the binary printed as the name, which reads like a missing
    file rather than an unsupported argument. Since it does accept a
    file-like, wrapping is both the clearer error and the working one.

    It really does accept one -- verified against h5py 3.16 with both
    BytesIO and an open file. The HDF5 codec used to write bytes to a
    temporary file instead, on the stated grounds that "h5py needs a
    real file handle for the most common access patterns", which was
    not true and cost a full copy of every in-memory source.
    """
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        # h5py drives its reads through the file-like, so an HDF5 over
        # HTTP fetches the chunks a slice touches and nothing else.
        # _hdf5_http has had this since before EMD and Imaris existed;
        # they just never reached it, because each opened h5py itself.
        from ._hdf5_http import _HTTPFileLike
        from ._tiff_http import HTTPDataSource
        return _HTTPFileLike(HTTPDataSource(src))
    if isinstance(src, (bytes, bytearray, memoryview)):
        return io.BytesIO(bytes(src))
    return src


__all__ = ["h5_source", "read_h5_dataset"]


def read_h5_dataset(dataset, selection=None, *, numthreads=None,
                    keep_integer_axes=False, max_pending_bytes=None,
                    worker_budget=None, pipeline_stats=None):
    """Read an HDF5 selection with a checked parallel filter fast path.

    h5py owns unsupported filters and checksum validation. Supported raw
    chunks are read in the producer thread, then decoded into disjoint
    output slices outside the HDF5 library lock. Integer axes normally
    follow h5py; the legacy HdfReader.read_parallel surface retains them.
    """
    import itertools
    import math
    import numpy as np
    import zlib
    from .core.parallel import resolve_workers, run_batched
    from .core.pipeline import map_bounded, native_workers, range_batches

    idx = (...,) if selection is None else selection
    if not isinstance(idx, tuple):
        idx = (idx,)
    # Normalize simple selections only. h5py remains authoritative for
    # fancy indexing, new axes, strides, and malformed selectors.
    if sum(item is Ellipsis for item in idx) > 1:
        return dataset[idx]
    if any(item is Ellipsis for item in idx):
        pos = next(i for i, item in enumerate(idx) if item is Ellipsis)
        idx = idx[:pos] + (slice(None),) * (dataset.ndim - len(idx) + 1) + idx[pos + 1:]
    idx += (slice(None),) * (dataset.ndim - len(idx))
    if len(idx) != dataset.ndim:
        return dataset[idx]
    bounds, integer_axes = [], []
    normalized = []
    for axis, (item, dim) in enumerate(zip(idx, dataset.shape)):
        if isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_)):
            value = int(item)
            if value < 0:
                value += dim
            if not 0 <= value < dim:
                return dataset[idx]  # preserve h5py's exception
            bounds.append((value, value + 1))
            normalized.append(slice(value, value + 1) if keep_integer_axes else value)
            integer_axes.append(axis)
        elif isinstance(item, slice):
            start, stop, step = item.indices(dim)
            if step != 1:
                return dataset[idx]
            stop = max(start, stop)
            bounds.append((start, stop))
            normalized.append(slice(start, stop))
        else:
            return dataset[idx]
    normalized = tuple(normalized)
    fallback = lambda: dataset[normalized]
    chunks = dataset.chunks
    if chunks is None or not chunks or dataset.dtype.hasobject:
        return fallback()
    shape = tuple(stop - start for start, stop in bounds)
    ranges = [range(start // chunk, (stop + chunk - 1) // chunk)
              if stop > start else range(0)
              for (start, stop), chunk in zip(bounds, chunks)]
    count = math.prod(len(r) for r in ranges)
    # Inspect the actual creation pipeline, not Dataset.compression.
    # A checksum or an unknown filter uses h5py so validation is never
    # silently bypassed by a faster decompressor.
    plist = dataset.id.get_create_plist()
    filters = [plist.get_filter(i) for i in range(plist.get_nfilters())]
    supported = all(f[0] in (1, 2) for f in filters)  # deflate, shuffle
    workers = resolve_workers(native_workers(numthreads), count,
                              has_decode_work=any(f[0] == 1 for f in filters),
                              output_bytes=math.prod(shape) * dataset.dtype.itemsize)
    if not supported or workers <= 1 or count == 0:
        return fallback()

    eager = max_pending_bytes is None and pipeline_stats is None and worker_budget is None
    max_pending_bytes = 64 << 20 if max_pending_bytes is None else max_pending_bytes
    out = np.empty(shape, dtype=dataset.dtype)
    chunk_bytes = math.prod(chunks) * dataset.dtype.itemsize
    chunk_dtype = dataset.dtype
    itemsize = chunk_dtype.itemsize
    fill_value = dataset.fillvalue

    def pieces():
        for ci in itertools.product(*ranges):
            offset = tuple(i * c for i, c in zip(ci, chunks))
            source_slices, target_slices = [], []
            for origin, extent, (start, stop) in zip(offset, chunks, bounds):
                lo, hi = max(origin, start), min(origin + extent, stop)
                source_slices.append(slice(lo - origin, hi - origin))
                target_slices.append(slice(lo - start, hi - start))
            source_slices, target_slices = tuple(source_slices), tuple(target_slices)
            info = dataset.id.get_chunk_info_by_coord(offset)
            if info.size == 0:
                out[target_slices] = fill_value
                continue
            yield offset, int(info.size), source_slices, target_slices

    def fetch(batch):
        fetched = []
        for offset, _, source_slices, target_slices in batch:
            mask, blob = dataset.id.read_direct_chunk(offset)
            fetched.append((mask, blob, source_slices, target_slices))
        return fetched

    def decode_place(piece):
        mask, raw, source_slices, target_slices = piece
        for index in range(len(filters) - 1, -1, -1):
            if mask & (1 << index):
                continue
            filter_id, _, params, _ = filters[index]
            if filter_id == 1:
                raw = zlib.decompress(raw)
            elif filter_id == 2 and itemsize > 1:
                if len(raw) % itemsize:
                    raise ValueError("HDF5 shuffle chunk has an incomplete element")
                raw = np.frombuffer(raw, dtype=np.uint8).reshape(itemsize, -1).T.copy()
        if memoryview(raw).nbytes != chunk_bytes:
            raise ValueError("HDF5 decoded chunk size does not match its declared shape")
        block = np.frombuffer(raw, dtype=chunk_dtype).reshape(chunks)
        out[target_slices] = block[source_slices]

    # Eager input remains the default because interleaved h5py reads and
    # worker handoffs regress fast-source throughput. An explicit byte
    # budget enables bounded overlap. Small selections may still fit the
    # budget eagerly, avoiding handoffs while placing pixels directly.
    if (pipeline_stats is None and worker_budget is None
            and (eager or count * chunk_bytes * 3 <= max_pending_bytes)):
        descriptors = list(pieces())
        if eager or sum(piece[1] for piece in descriptors) + 2 * workers * chunk_bytes <= max_pending_bytes:
            raw_chunks = fetch(descriptors)
            run_batched(decode_place, raw_chunks, workers, name="hdf5")
            return (np.squeeze(out, axis=tuple(integer_axes))
                    if integer_axes and not keep_integer_axes else out)
        # The actual encoded sizes may exceed the decoded size estimate.
        # Reuse the already discovered descriptors without reading twice.
        descriptors = iter(descriptors)
    else:
        descriptors = pieces()

    # Include encoded input and unshuffle/decompression scratch. Metadata
    # discovery and raw chunk reads stay on the producer thread. This also
    # overlaps disk reads with decoding without crossing h5py's lock.
    reservation = lambda piece: piece[1] + 2 * chunk_bytes
    batches = range_batches(descriptors, reservation,
                            max_bytes=max(1, min(2 << 20, max_pending_bytes // (workers + 1))), max_items=16)

    def place_batch(batch):
        for piece in batch:
            decode_place(piece)

    for _ in map_bounded(place_batch, batches, workers,
                         size=lambda batch: sum(reservation(p) for p in batch),
                         prepare=fetch, max_bytes=max_pending_bytes,
                         budget=worker_budget, stats=pipeline_stats, name="hdf5"):
        pass
    if integer_axes and not keep_integer_axes:
        return np.squeeze(out, axis=tuple(integer_axes))
    return out
