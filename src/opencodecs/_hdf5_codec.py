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

    def __init__(self, path: str | Path, dataset: str | None = None):
        if not _HAVE_H5PY:  # pragma: no cover - h5py-missing branch
            raise ImportError(
                "h5py is required for HDF5 support: pip install h5py")
        self._path = str(path)
        self._h5 = h5py.File(self._path, "r")
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

    def read(self) -> np.ndarray:
        return self._ds[...]

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
    parallel_decode = False

    supported_dtypes = (np.uint8, np.uint16, np.int16, np.int32, np.float32, np.float64)
    supports_color = False

    def signature(self, head: bytes) -> bool:
        # HDF5 super-block signature is the 8-byte sequence \x89HDF\r\n\x1a\n.
        return len(head) >= 8 and head[:8] == b"\x89HDF\r\n\x1a\n"

    def decode(self, src: Any, **opts) -> np.ndarray:
        with self.open(src, **opts) as reader:
            return reader.read()

    def open(self, src: Any, *, dataset: str | None = None, **opts) -> Reader:
        if isinstance(src, (str, Path)):
            return HdfReader(src, dataset=dataset)
        # bytes -> dump to a temp file because h5py needs a real file handle
        # for the most common access patterns.
        import tempfile, os
        if isinstance(src, (bytes, bytearray, memoryview)):
            fd, tmp = tempfile.mkstemp(suffix=".h5")
            os.write(fd, bytes(src))
            os.close(fd)
            return HdfReader(tmp, dataset=dataset)
        if hasattr(src, "read"):
            data = src.read()
            fd, tmp = tempfile.mkstemp(suffix=".h5")
            os.write(fd, data)
            os.close(fd)
            return HdfReader(tmp, dataset=dataset)
        raise TypeError(f"unsupported HDF5 source: {type(src).__name__}")


__all__ = ["HdfCodec", "HdfReader"]
