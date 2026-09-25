"""HDF5 Reader/Codec wrapping h5py.

HDF5 is a container, not a single image format — files hold a tree of
named datasets that may themselves be 2D / 3D / N-D. The opencodecs
``HdfCodec.open(path)`` exposes the *first* image-shaped dataset in the
file as a Reader; for full tree access, use ``HdfReader.from_path(...)``
which keeps the file open and lets you select a dataset by name.

We deliberately don't try to wrap libhdf5 directly — h5py is the
canonical Python binding and using it lets us match the file format
exactly without a 50k-line reimplementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .core.parallel import resolve_workers

from .core.codec import Codec, Reader

try:
    import h5py
    _HAVE_H5PY = True
except ImportError:  # pragma: no cover - h5py-missing branch
    _HAVE_H5PY = False


class HdfReader(Reader):
    """Reader exposing an HDF5 file's primary dataset (or a named one).

    Random-access via ``[idx]`` reads a single chunk along axis 0; the
    full dataset is materialized via ``read()``. Multi-dataset files
    can be navigated via ``r.dataset_names`` and ``r.select(name)``.
    """

    def __init__(self, path: "str | Path | Any",
                 dataset: str | None = None):
        if not _HAVE_H5PY:  # pragma: no cover - h5py-missing branch
            raise ImportError(
                "h5py is required for HDF5 support: pip install h5py")
        # A file-like goes to h5py untouched. str()-ing it turned a
        # BytesIO into its repr and h5py then looked for a file by that
        # name, which is why the codec above used to write every
        # in-memory source to disk instead.
        if isinstance(path, (str, Path)):
            self._path = str(path)
            src = self._path
        else:
            self._path = getattr(path, "name", repr(path))
            src = path
        self._h5 = h5py.File(src, "r")
        self._dataset_names = _list_image_datasets(self._h5)
        if dataset is None:
            if not self._dataset_names:
                self._h5.close()
                raise ValueError(
                    f"{path}: no image-like datasets found")
            dataset = self._dataset_names[0]
        self._dataset_name = dataset
        self._ds = self._h5[dataset]
        self.shape = tuple(self._ds.shape)
        self.dtype = self._ds.dtype
        self.n_frames = self.shape[0] if self._ds.ndim >= 3 else 1
        self.is_chunked = True

    @property
    def dataset_names(self) -> list[str]:
        return list(self._dataset_names)

    def select(self, name: str) -> "HdfReader":
        """Switch to a different dataset within the same open file."""
        self._dataset_name = name
        self._ds = self._h5[name]
        self.shape = tuple(self._ds.shape)
        self.dtype = self._ds.dtype
        self.n_frames = self.shape[0] if self._ds.ndim >= 3 else 1
        return self

    def iter_frames(self) -> Iterator[np.ndarray]:
        if self._ds.ndim < 3:
            yield self._ds[...]
            return
        for i in range(self.shape[0]):
            yield self._ds[i]

    def read(self, *, numthreads: int | None = None,
             max_pending_bytes: int | None = None, worker_budget=None) -> np.ndarray:
        """The whole dataset, decompressing chunks in parallel.

        read_parallel has been here and unreachable: nothing called it,
        so every read went through h5py one chunk at a time under
        libhdf5's process-wide lock. On 64 gzip chunks of a 33.6 MB
        dataset that was 134.0 ms against 13.1 ms.

        It declines on its own for the cases it cannot serve -- an
        unchunked dataset, an uncompressed one, a filter it has no
        user-space decoder for -- and returns the h5py result, so this
        is a fast path rather than a second implementation.
        """
        workers = resolve_workers(
            numthreads, self._chunk_count(),
            has_decode_work=getattr(self._ds, "compression", None) is not None,
            output_bytes=self._ds.nbytes)
        if workers <= 1:
            return self._ds[...]
        return self.read_parallel(n_workers=workers, max_pending_bytes=max_pending_bytes,
                                  worker_budget=worker_budget)

    def _chunk_count(self) -> int:
        """How many chunks a full read touches, or 1 if unchunked."""
        chunks = self._ds.chunks
        if not chunks:
            return 1
        n = 1
        for dim, chk in zip(self._ds.shape, chunks):
            n *= (dim + chk - 1) // chk
        return n

    def read_parallel(self, idx=None, *, n_workers: int | None = None,
                      max_pending_bytes: int | None = None,
                      worker_budget=None, pipeline_stats=None) -> np.ndarray:
        """Read a selection through the shared checked chunk pipeline.

        Supported gzip/shuffle chunks are decoded outside the HDF5 lock;
        other filters use h5py, including checksum validation. Setting
        max_pending_bytes bounds raw chunks and decode scratch; one
        oversized chunk runs alone. Default eager fetching preserves fast
        local throughput. Integer selectors retain their axis for compatibility with
        this method's original behavior.
        """
        from ._h5_common import read_h5_dataset
        return read_h5_dataset(self._ds, idx, numthreads=n_workers,
                               keep_integer_axes=True, max_pending_bytes=max_pending_bytes,
                               worker_budget=worker_budget, pipeline_stats=pipeline_stats)

    def __getitem__(self, idx) -> np.ndarray:
        return self._ds[idx]

    def close(self) -> None:
        self._h5.close()


def _list_image_datasets(grp: "h5py.Group") -> list[str]:
    """Walk an h5py group and return paths to numeric datasets (>= 1D)."""
    names: list[str] = []

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 1 and (
            obj.dtype.kind in ("u", "i", "f")
        ):
            names.append(name)

    grp.visititems(_visit)
    return names


class HdfCodec(Codec):
    """HDF5 container codec — exposes datasets as Readers."""

    name = "hdf5"
    file_extensions = (".h5", ".hdf5", ".he5")
    aliases = ("hdf",)

    has_native = _HAVE_H5PY
    has_delegate = False
    can_encode = False  # encode would mean creating an h5 file; out of scope here
    can_decode = True
    multi_frame = True
    chunked = True
    streaming_decode = True
    # read() decompresses chunks across threads. The B-tree already
    # gives their offsets and chunks are independent, so this needed no
    # format work -- only for read() to actually call the read_parallel
    # path that was already here and unreachable. Measured on 64 gzip
    # chunks of a 33.6 MB dataset: 134.0 ms to 13.1 ms on 16 workers.
    parallel_decode = True

    supported_dtypes = (np.uint8, np.uint16, np.int16, np.int32, np.float32, np.float64)
    supports_color = False

    def signature(self, head: bytes) -> bool:
        # HDF5 super-block signature is the 8-byte sequence \x89HDF\r\n\x1a\n.
        return len(head) >= 8 and head[:8] == b"\x89HDF\r\n\x1a\n"

    def decode(self, src: Any, *, numthreads: int | None = None,
               max_pending_bytes: int | None = None, worker_budget=None, **opts) -> np.ndarray:
        with self.open(src, **opts) as reader:
            return reader.read(numthreads=numthreads, max_pending_bytes=max_pending_bytes,
                               worker_budget=worker_budget)

    def open(self, src: Any, *, dataset: str | None = None, **opts) -> Reader:
        # URL before path: a str starting with http(s) is not a
        # filename, and checking isinstance(str) first sent it to
        # h5py as one.
        if isinstance(src, str) and src.startswith(("http://", "https://")):
            from ._h5_common import h5_source
            return HdfReader(h5_source(src), dataset=dataset)
        if isinstance(src, (str, Path)):
            return HdfReader(src, dataset=dataset)
        if isinstance(src, (bytes, bytearray, memoryview)) or hasattr(src, "read"):
            # h5py opens a file-like directly. This used to write every
            # non-path source to a temp file, explaining that "h5py
            # needs a real file handle for the most common access
            # patterns" -- which is not so, and which the emd and
            # imaris readers had already been contradicting in their
            # own copies of this helper for as long as they existed.
            from ._h5_common import h5_source
            return HdfReader(h5_source(src), dataset=dataset)
        raise TypeError(f"unsupported HDF5 source: {type(src).__name__}")


__all__ = ["HdfCodec", "HdfReader"]
