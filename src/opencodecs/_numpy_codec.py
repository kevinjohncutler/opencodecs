"""NumpyCodec — round-trip an ndarray through the ``.npy`` byte format.

The ``.npy`` format is a thin, self-describing wrapper around raw
array bytes: a magic number, version, and a Python-literal header
naming dtype/shape/fortran-order, followed by the array data. It's
the only ndarray serialization format that is both numpy-native and
specified well enough to share across libraries.

Useful as:

* A trivial fallback "compressor" in pipelines that demand a codec
  interface (e.g. zarr's codec chain) but want raw passthrough.
* A self-describing wire format when you can't ship dtype/shape out
  of band.
* A reference for what "no compression" looks like in a benchmark.

Mirrors imagecodecs's ``numpy_encode`` / ``numpy_decode``: ``encode``
writes the bytes ``numpy.save`` writes (Fortran-ordered input keeps
``fortran_order: True``; datetime64 and timedelta64 are legal), except
that an array holding Python objects raises ``ValueError`` where
``numpy.save`` and imagecodecs pickle it, and
``level`` selects a deflate-compressed ``.npz`` (a zip holding
``arr_0.npy``, the ``numpy.savez_compressed`` layout). ``decode`` reads
either and returns an ndarray, choosing an ``.npz`` member by
``index``.
"""

from __future__ import annotations

import io
import zipfile
from typing import Any

import numpy as np

from .core.codec import Codec, Reader
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest


class NumpyCodec(Codec):
    """``.npy``-format passthrough codec."""

    name = "numpy"
    aliases = ("npy",)
    file_extensions = (".npy",)

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = True
    # A .npy is a short header and then the raw buffer in C order, so
    # a plane along axis 0 is an offset and a length. NpyFile reads it
    # that way, which is what makes opening a 4 GB array over HTTP
    # cost one small range request.
    chunked = True
    streaming_decode = True
    parallel_decode = False

    # Every numpy dtype is supported (it's literal raw bytes plus a
    # dtype header). We list a representative set for the registry's
    # surface; numpy.save itself accepts any np.dtype.
    supported_dtypes = (
        np.uint8, np.int8, np.uint16, np.int16,
        np.uint32, np.int32, np.uint64, np.int64,
        np.float16, np.float32, np.float64,
        np.complex64, np.complex128,
    )
    supports_color = True

    def open(self, src: Any, **opts) -> "Reader":
        """A reader that fetches planes rather than the whole array."""
        from ._numpy_reader import NpyFile
        return NpyFile(src)

    def signature(self, head: bytes) -> bool:
        # ``.npy`` files start with the 6-byte magic ``\x93NUMPY``
        # followed by a 2-byte version (major, minor).
        return len(head) >= 8 and head[:6] == b"\x93NUMPY"

    def encode(self, data: Any, level=None, *, dest=None,
               **opts) -> bytes | None:
        """Encode ``data`` as ``.npy``, or as ``.npz`` when ``level``.

        ``level`` (also by position) follows imagecodecs
        ``numpy_encode(data, level=None, *, out=None)``: falsy writes
        ``.npy``; truthy writes a zip holding ``arr_0.npy`` with deflate,
        at that zlib level when it is 1 to 9, else the zlib default.
        The zip entry carries a fixed timestamp, so equal input gives
        equal bytes. imagecodecs takes no other options, and neither
        does this: any other keyword raises ``TypeError``. An array
        holding Python objects raises ``ValueError``; imagecodecs and
        ``numpy.save`` pickle it.
        """
        from .core._write_helpers import binary_destination
        if opts:
            raise TypeError(
                f"numpy encode: unexpected keyword argument(s) "
                f"{', '.join(sorted(opts))}")
        arr = np.asarray(data)
        if arr.dtype.hasobject:
            raise ValueError("Object arrays cannot be saved when allow_pickle=False")
        stream = io.BytesIO() if dest is None else dest
        with binary_destination(stream) as target:
            if level:
                compresslevel = (int(level) if not isinstance(level, bool)
                                 and 1 <= int(level) <= 9 else None)
                info = zipfile.ZipInfo("arr_0.npy",
                                       date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                try:    # public from Python 3.13, a private slot before
                    info.compress_level = compresslevel
                except AttributeError:
                    info._compresslevel = compresslevel
                with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
                    with zf.open(info, "w", force_zip64=True) as member:
                        _write_npy(member, arr)
            else:
                _write_npy(target, arr)
        return stream.getvalue() if dest is None else None

    def decode(self, src: Any, index=0, *, out=None, **opts) -> np.ndarray:
        """Decode ``.npy``, or member ``index`` of an ``.npz``.

        ``index`` (an int position or a member name, default 0) applies
        to ``.npz`` input only, as in imagecodecs ``numpy_decode``; a
        position or name with no member raises ``KeyError``. Other
        keywords go to ``numpy.load`` (``allow_pickle``,
        ``max_header_size``, ...), as imagecodecs passes them, so one
        that ``numpy.load`` does not take raises ``TypeError``.
        """
        # numpy.load returns a fresh ndarray; if the caller wants
        # the result in their preallocated buffer, copy into it.
        # True zero-alloc isn't possible here because np.load owns
        # the buffer it creates, but out= still saves the second
        # allocation a caller would do.
        loaded = np.load(io.BytesIO(_read_src(src)), **opts)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded:
                files = loaded.files
                if isinstance(index, str):
                    if index not in files:
                        raise KeyError(
                            f"numpy decode: no member {index!r} in .npz "
                            f"(members {files})")
                    name = index
                else:
                    i = int(index)
                    if not -len(files) <= i < len(files):
                        # KeyError, as imagecodecs raises for it
                        raise KeyError(
                            f"numpy decode: index {index} out of range for "
                            f"{len(files)} .npz members")
                    name = files[i]
                arr = loaded[name]
        else:
            arr = loaded
        if out is None:
            return arr
        if not isinstance(out, np.ndarray):
            raise TypeError(
                f"numpy decode: out= must be an ndarray, "
                f"got {type(out).__name__}")
        if out.shape != arr.shape:
            raise ValueError(
                f"numpy decode: out= shape {out.shape} does not match "
                f"decoded {arr.shape}")
        if out.dtype != arr.dtype:
            raise ValueError(
                f"numpy decode: out= dtype {out.dtype} does not match "
                f"decoded {arr.dtype}")
        if not out.flags["C_CONTIGUOUS"]:
            raise ValueError("numpy decode: out= must be C-contiguous")
        np.copyto(out, arr)
        return out


def _dtype_exports_buffer(dtype: np.dtype) -> bool:
    """Whether arrays of ``dtype`` export the buffer protocol (datetime64
    and timedelta64, alone or inside a structure, do not)."""
    try:
        memoryview(np.empty(0, dtype))
    except (ValueError, TypeError, BufferError, NotImplementedError):
        return False
    return True


def _write_npy(target, arr: np.ndarray) -> None:
    """Write ``arr`` as ``.npy``, byte for byte what ``numpy.save`` writes
    (the caller refuses object arrays, which ``numpy.save`` pickles).

    The header is numpy's own (``header_data_from_array_1_0``), so an
    F-contiguous array is stored with ``fortran_order: True`` and its
    data in Fortran order, as NEP 1 allows. The data streams through
    bounded buffers; dtypes the buffer protocol cannot carry go through
    ``numpy.lib.format.write_array``, which also streams.
    """
    from .core._write_helpers import iter_array_buffers, write_all, CompleteWriter
    header = np.lib.format.header_data_from_array_1_0(arr)
    if not _dtype_exports_buffer(arr.dtype):
        np.lib.format.write_array(CompleteWriter(target), arr,
                                  allow_pickle=False)
        return
    header_stream = io.BytesIO()
    try:
        try:
            np.lib.format.write_array_header_1_0(header_stream, header)
        except ValueError:
            np.lib.format.write_array_header_2_0(header_stream, header)
    except UnicodeEncodeError:
        # NumPy's public array writer selects version 3 for Unicode
        # field names and already streams through bounded buffers.
        np.lib.format.write_array(CompleteWriter(target), arr,
                                  allow_pickle=False)
        return
    write_all(target, header_stream.getvalue())
    order = "F" if header["fortran_order"] else "C"
    for block in iter_array_buffers(arr, order=order):
        write_all(target, block)


__all__ = ["NumpyCodec"]
