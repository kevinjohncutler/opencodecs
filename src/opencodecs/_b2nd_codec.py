"""B2ndCodec — Codec adapter wrapping the native _b2nd extension.

Blosc2 NDim ("b2nd") is c-blosc2's multidimensional layer. Each cframe
is a self-contained byte buffer that round-trips an ndarray with full
shape and dtype — no out-of-band metadata required.

This is the natural shape-aware companion to the existing flat blosc2
codec. Use cases:

* Persisting numerical arrays as cframes inside HDF5 / Zarr / a tar
* Network-transfer of multidim arrays (the cframe is a complete record)
* Scientific time-series where each frame is a chunk

Example::

    import numpy as np
    import opencodecs as oc

    arr = np.random.rand(64, 128, 128).astype(np.float32)
    blob = oc.write(None, arr, format="b2nd", compressor="zstd", shuffle="bit")
    back = oc.read(blob, format="b2nd")
    assert back.shape == arr.shape and back.dtype == arr.dtype
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec, Reader
from .core.buffers import array_output
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs
from .core.pipeline import native_workers

(
    _b2nd_encode, _b2nd_decode, _b2nd_inspect, _b2nd_check_signature,
    _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._b2nd",
    "encode", "decode", "inspect", "check_signature",
)


class B2ndCodec(Codec):
    """Native blosc2 NDim — ndarray ↔ self-describing cframe."""

    name = "b2nd"
    aliases = ("blosc2nd", "blosc2-nd")
    file_extensions = (".b2nd",)

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    # A cframe is a grid of independently compressed chunks, so a
    # sub-box decompresses only the chunks it intersects: measured 8x
    # cheaper for a 0.2% box of a 256x256x64 array. This is what b2nd
    # has over flat blosc2, and decode_slice() is where it lives.
    chunked = True
    # Those chunks also decompress across threads, applied to the
    # array's own context rather than the process-global setting.
    parallel_decode = True

    supported_dtypes = (
        np.uint8, np.int8,
        np.uint16, np.int16, np.float16,
        np.uint32, np.int32, np.float32,
        np.uint64, np.int64, np.float64,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return _b2nd_check_signature(head)

    def encode(self, data: Any, *, dest=None,
               level: int = 5,
               compressor: str | None = "zstd",
               shuffle: Any = "bit",
               storage_output: bool = False,
               **opts) -> bytes | None:
        """Encode an array, optionally writing native storage directly to a path.

        ``storage_output=True`` avoids the encoded Python buffer, but may
        reduce throughput. It requires a filesystem destination.
        """
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        if storage_output:
            if dest is None or hasattr(dest, "write"):
                raise ValueError("storage_output requires a filesystem destination")
            import tempfile
            target = Path(dest)
            with tempfile.TemporaryDirectory(prefix=".opencodecs-", dir=target.parent) as folder:
                temporary = Path(folder) / "array.b2nd"
                _b2nd_encode(data, level=int(level), compressor=compressor,
                             shuffle=shuffle, path=temporary)
                os.replace(temporary, target)
            return None
        out = _b2nd_encode(
            data, level=int(level), compressor=compressor, shuffle=shuffle,
        )
        return _write_dest(out, dest)

    def decode(self, src: Any, *, numthreads: int | None = None,
               out=None, **opts) -> np.ndarray:
        if out is None:
            return _b2nd_decode(_read_src(src), numthreads=native_workers(numthreads))
        return _b2nd_decode(_read_src(src), numthreads=native_workers(numthreads),
                            out=array_output(out))

    def decode_slice(self, src: Any, start, stop, *,
                     numthreads: int | None = None,
                     out: np.ndarray | None = None) -> np.ndarray:
        """Decode one n-dimensional sub-box without expanding the rest.

        ``start`` and ``stop`` are per-axis element coordinates of a
        half-open box, each the length of the array's ``ndim``. Only
        the chunks intersecting that box are decompressed.
        """
        if isinstance(src, os.PathLike) or (isinstance(src, str) and "://" not in src):
            with self.open(src, numthreads=numthreads) as reader:
                return reader.read_slice(start, stop, out=out)
        from .codecs._b2nd import decode_slice as _slice
        return _slice(_read_src(src), start, stop,
                      numthreads=native_workers(numthreads),
                      out=out if out is None else array_output(out))

    def inspect(self, src: Any) -> dict:
        """Return {ndim, shape, dtype, itemsize} without decompressing."""
        return _b2nd_inspect(_read_src(src))

    def open(self, src: Any, *, numthreads=None, **opts) -> Reader:
        """Keep local persistent storage indexed for repeated box reads."""
        if isinstance(src, os.PathLike) or (isinstance(src, str) and "://" not in src):
            return _B2ndFileReader(src, numthreads=numthreads)
        return super().open(src, numthreads=numthreads, **opts)


class _B2ndFileReader(Reader):
    n_frames = 1
    is_chunked = True

    def __init__(self, path, *, numthreads=None):
        from .codecs._b2nd import FileReader
        self._native = FileReader(path)
        self.shape = self._native.shape
        self.dtype = self._native.dtype
        self._numthreads = numthreads

    def read_slice(self, start, stop, *, out=None):
        return self._native.read_slice(
            start, stop, out=out,
            numthreads=native_workers(self._numthreads),
        )

    def read(self, *, out=None):
        return self.read_slice((0,) * len(self.shape), self.shape, out=out)

    def iter_frames(self):
        yield self.read()

    def __getitem__(self, index):
        if index not in (0, -1):
            raise IndexError(index)
        return self.read()

    def close(self):
        self._native.close()


__all__ = ["B2ndCodec"]
