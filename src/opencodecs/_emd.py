"""EMD reader (Electron Microscopy Datasets).

EMD is HDF5 with a convention layered on top, and the trap is that two
incompatible conventions share the extension:

* **Berkeley / NCEM**, the original. A data group is marked with an
  ``emd_group_type`` attribute and holds the array beside one ``dim1``,
  ``dim2``, ... vector per axis giving that axis's coordinates.
* **Velox**, Thermo Fisher's. Arrays live at
  ``/Data/Image/<hash>/Data`` with metadata as JSON in a byte array, and
  the last axis is the frame index rather than the first.

Neither declares itself in the filename, so this detects the schema from
the structure rather than trusting the extension.

One detail no specification prepares you for: the Berkeley schema does
not fix the name of the data array. The canonical name is ``data``, but
real files from the prismatic 4D-STEM simulator call it ``realslice``,
so the array is found as the dataset in the group that is not one of the
``dim`` vectors.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np


class EmdError(Exception):
    """Raised for files that are not EMD, or that use an unknown layout."""


_DIM = re.compile(r"^dim\d+$")


class EmdFile:
    """Reader for one EMD file, in either the Berkeley or Velox schema."""

    def __init__(self, path: Any):
        try:
            import h5py
        except ImportError:                              # pragma: no cover
            raise EmdError(
                "emd: reading .emd needs h5py; install opencodecs[hdf5]"
            ) from None
        self._h5 = h5py.File(path, "r")
        self._h5py = h5py
        self._berkeley = self._find_berkeley()
        self._velox = [] if self._berkeley else self._find_velox()
        if not self._berkeley and not self._velox:
            self._h5.close()
            raise EmdError(
                "emd: no emd_group_type marker and no /Data/Image group; "
                "this is an HDF5 file but not an EMD one")

    # -- discovery ---------------------------------------------------

    def _find_berkeley(self) -> list[str]:
        """Groups carrying a data array, in file order."""
        found: list[str] = []
        h5py = self._h5py

        def visit(name, obj):
            if not isinstance(obj, h5py.Group):
                return
            if "emd_group_type" not in obj.attrs:
                return
            # A marked group is only a *data* group if it actually holds
            # an array; the schema also marks container groups.
            if self._berkeley_array_name(obj) is not None:
                found.append(name)

        self._h5.visititems(visit)
        return found

    def _berkeley_array_name(self, group) -> str | None:
        h5py = self._h5py
        for key, obj in group.items():
            if isinstance(obj, h5py.Dataset) and not _DIM.match(key):
                return key
        return None

    def _find_velox(self) -> list[str]:
        node = self._h5.get("Data")
        if node is None:
            return []
        out = []
        for kind in ("Image", "Spectrum", "SpectrumStream"):
            grp = node.get(kind)
            if grp is None:
                continue
            for key in grp:
                if "Data" in grp[key]:
                    out.append(f"Data/{kind}/{key}")
        return sorted(out)

    @property
    def schema(self) -> str:
        return "berkeley" if self._berkeley else "velox"

    @property
    def n_datasets(self) -> int:
        return len(self._berkeley or self._velox)

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(self._berkeley or self._velox)

    # -- data --------------------------------------------------------

    def _dataset(self, index: int):
        paths = self._berkeley or self._velox
        if not 0 <= index < len(paths):
            raise IndexError(
                f"emd: dataset {index} out of range (0..{len(paths) - 1})")
        group = self._h5[paths[index]]
        if self._berkeley:
            name = self._berkeley_array_name(group)
            return group[name]
        return group["Data"]

    def shape(self, index: int = 0) -> tuple[int, ...]:
        return tuple(self._dataset(index).shape)

    def dtype(self, index: int = 0) -> np.dtype:
        return np.dtype(self._dataset(index).dtype)

    def asarray(self, index: int = 0) -> np.ndarray:
        return self._dataset(index)[...]

    def axes(self, index: int = 0) -> list[np.ndarray]:
        """Per-axis coordinate vectors, where the schema records them.

        Berkeley stores one ``dimN`` dataset per axis, which is how a
        caller recovers real units. Velox keeps the equivalent in its
        JSON metadata instead, so this returns nothing there rather than
        inventing indices.
        """
        if not self._berkeley:
            return []
        group = self._h5[self._berkeley[index]]
        names = sorted((k for k in group if _DIM.match(k)),
                       key=lambda k: int(k[3:]))
        return [group[k][...] for k in names]

    def metadata(self, index: int = 0) -> dict:
        """Velox stores JSON metadata as a uint8 array; decode it."""
        if self._berkeley:
            group = self._h5[self._berkeley[index]]
            return {k: group.attrs[k] for k in group.attrs}
        group = self._h5[self._velox[index]]
        node = group.get("Metadata")
        if node is None:
            return {}
        raw = np.asarray(node[:, 0] if node.ndim == 2 else node[...])
        text = raw.tobytes().rstrip(b"\x00").decode("utf-8", "replace")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def close(self) -> None:
        if getattr(self, "_h5", None) is not None:
            self._h5.close()
            self._h5 = None                              # type: ignore

    def __enter__(self) -> "EmdFile":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"<EmdFile schema={self.schema} datasets={self.n_datasets}>")


__all__ = ["EmdFile", "EmdError"]
