"""Shared plumbing for the HDF5-backed readers (hdf5, emd, imaris).

Small on purpose: it exists so the three modules that open files
through h5py agree about what h5py accepts, which they did not.
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


__all__ = ["h5_source"]
