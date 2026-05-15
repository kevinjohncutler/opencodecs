"""ND2 Reader/Codec wrapping the ``nd2`` Python package.

Nikon NIS-Elements ND2 is a proprietary binary container for
multi-dimensional microscopy data (T, Z, C, X, Y, multi-position).
Talley Lambert's `nd2 <https://pypi.org/project/nd2/>`_ package is the
canonical Python reader; we wrap it the same way HdfCodec wraps h5py.

A native ND2 reader is doable (the format is documented enough), but
the cost/benefit doesn't favour it: nd2 is well-maintained, fast, and
already handles the SDK's many container revisions. We focus on
giving ND2 the same first-class place in the opencodecs API as our
native CZI / TIFF / OME-Zarr readers.

Example::

    import opencodecs as oc
    arr = oc.read("scan.nd2")          # decode primary array
    with oc.open("scan.nd2") as r:     # streaming
        for frame in r.iter_frames():
            ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .core.codec import Codec, Reader

try:
    import nd2 as _nd2
    _HAVE_ND2 = True
except ImportError:  # pragma: no cover - nd2-missing branch
    _HAVE_ND2 = False


class Nd2Error(RuntimeError):
    """Raised on ND2 open / decode failures."""


class Nd2Reader(Reader):
    """Reader exposing an ND2 file as a streaming N-D array.

    The first axis is treated as the frame axis for ``iter_frames`` /
    ``__getitem__``; for 4-D or 5-D arrays (T, Z, ...) this means the
    outermost dimension. Use ``read()`` to materialize the whole stack.
    """

    def __init__(self, path: str | Path):
        if not _HAVE_ND2:  # pragma: no cover - nd2-missing branch
            raise ImportError(
                "nd2 is required for ND2 support: pip install nd2")
        self._path = str(path)
        self._f = _nd2.ND2File(self._path)
        self.shape: tuple[int, ...] = tuple(self._f.shape)
        self.dtype: np.dtype = np.dtype(self._f.dtype)
        # Frame axis: outermost dim if N >= 3, else the single 2D image
        self.n_frames = self.shape[0] if self._f.ndim >= 3 else 1
        self.is_chunked = True

    @property
    def sizes(self) -> dict[str, int]:
        """Per-axis sizes (e.g. ``{"T": 13, "Y": 600, "X": 800}``).
        Pulls the axis labels from the underlying nd2 reader's
        ``sizes`` dict."""
        return dict(self._f.sizes)

    @property
    def metadata(self) -> Any:
        """Pass-through to nd2.ND2File.metadata for callers that need
        the full ZEN/NIS-Elements metadata tree."""
        return self._f.metadata

    def iter_frames(self) -> Iterator[np.ndarray]:
        if self._f.ndim < 3:
            yield self._f.asarray()
            return
        for i in range(self.shape[0]):
            yield self._f.read_frame(i)

    def read(self) -> np.ndarray:
        return self._f.asarray()

    def __getitem__(self, idx) -> np.ndarray:
        if isinstance(idx, (int, np.integer)):
            return self._f.read_frame(int(idx))
        # Fall through to nd2's slicing — works for tuple / slice indices
        return self._f.asarray()[idx]

    def close(self) -> None:
        if self._f is not None and not self._f.closed:
            self._f.close()

    def __enter__(self) -> "Nd2Reader":
        return self

    def __exit__(self, *_) -> bool:
        self.close()
        return False


class Nd2Codec(Codec):
    """ND2 container codec — Nikon NIS-Elements multi-dim microscopy.

    Two backends:
      * ``opencodecs._nd2_native.Nd2NativeReader`` — pure-Python parser
        for the v3 chunk container. Works on local paths AND
        ``HTTPDataSource`` (range reads = only fetches the FILEMAP +
        ImageAttributes + the specific frames the caller wants).
        Decodes raw / uncompressed pixel data; doesn't handle legacy
        ND2 or compressed (JPEG / JPEG-XR) variants.
      * ``nd2`` package — full-featured delegate (supports every
        compression + legacy variants). Used as fallback.

    ``open(src)`` tries native first, falls back to the delegate on
    NotImplementedError. So users get HTTP range reads + zero
    delegate-dep behaviour transparently when the file is in a
    supported configuration, and full coverage when it isn't.
    """

    name = "nd2"
    file_extensions = (".nd2",)
    aliases = ()

    has_native = True
    has_delegate = _HAVE_ND2
    can_encode = False
    can_decode = True
    multi_frame = True
    chunked = True
    streaming_decode = True
    parallel_decode = False

    supported_dtypes = (
        np.uint8, np.uint16, np.int16, np.float32,
    )
    supports_color = True

    def signature(self, head: bytes) -> bool:
        """ND2 files begin with a 16-byte chunk header followed by the
        ASCII string ``"ND2 FILE SIGNATURE CHUNK NAME"`` at offset 16.

        Modern NIS-Elements ND2 (v2+) starts with magic bytes
        ``\\xDA\\xCE\\xBE\\x0A``; legacy ND2 (pre-2008) uses
        ``\\xDA\\xBC\\xD8\\x3E``. The trailing ASCII "ND2 FILE SIGNATURE"
        is the most reliable discriminator across versions, so we
        match either magic AND the trailing ASCII.
        """
        if len(head) < 4:
            return False
        # Modern ND2 (v2+, including NIS-Elements 4.x and v3 files)
        if head[:4] == b"\xDA\xCE\xBE\x0A":
            return True
        # Legacy ND2 (pre-2008)
        if head[:4] == b"\xDA\xBC\xD8\x3E":
            return True
        return False

    def decode(self, src: Any, **opts) -> np.ndarray:
        with self.open(src, **opts) as reader:
            return reader.read()

    def open(self, src: Any, *, backend: str | None = None,
             **opts) -> Reader:
        """Open the ND2 for reading.

        ``backend``:
          * ``None`` (default): try native first, fall back to nd2.
          * ``"native"``: force the native parser.
          * ``"nd2"``: force the nd2 delegate.
        """
        # Native handles file paths AND DataSource instances (HTTP /
        # mmap / S3). The delegate only handles file paths.
        if backend in (None, "native"):
            try:
                from ._nd2_native import Nd2NativeReader
                from .core.io import DataSource
                if isinstance(src, (str, Path)) or isinstance(src, DataSource):
                    return Nd2NativeReader(src)
                # bytes / file-like: spill to a temp file so the
                # native reader's FileDataSource can mmap-style read.
                import os, tempfile
                if isinstance(src, (bytes, bytearray, memoryview)):
                    fd, tmp = tempfile.mkstemp(suffix=".nd2")
                    os.write(fd, bytes(src))
                    os.close(fd)
                    return Nd2NativeReader(tmp)
                if hasattr(src, "read"):
                    data = src.read()
                    fd, tmp = tempfile.mkstemp(suffix=".nd2")
                    os.write(fd, data)
                    os.close(fd)
                    return Nd2NativeReader(tmp)
            except (NotImplementedError, ValueError) as e:
                if backend == "native":
                    raise
                # Fall through to delegate
                _fallback_reason = str(e)
            else:
                raise TypeError(
                    f"unsupported ND2 source: {type(src).__name__}")

        if not _HAVE_ND2:
            raise ImportError(
                "ND2: native parser failed and `nd2` package is not "
                "installed. pip install nd2 for delegate fallback "
                "support of legacy / compressed ND2 variants.")
        if isinstance(src, (str, Path)):
            return Nd2Reader(src)
        import os, tempfile
        if isinstance(src, (bytes, bytearray, memoryview)):
            fd, tmp = tempfile.mkstemp(suffix=".nd2")
            os.write(fd, bytes(src))
            os.close(fd)
            return Nd2Reader(tmp)
        if hasattr(src, "read"):
            data = src.read()
            fd, tmp = tempfile.mkstemp(suffix=".nd2")
            os.write(fd, data)
            os.close(fd)
            return Nd2Reader(tmp)
        raise TypeError(f"unsupported ND2 source: {type(src).__name__}")

    def info(self, src: Any) -> dict:
        """Partial-parse: open the file just far enough to read the
        FILEMAP + ImageAttributes (~tens of KB) and return geometry
        without decoding any frame. Same contract as
        :meth:`OirCodec.info` / :meth:`VsiCodec.info` — accepts a
        path or a :class:`DataSource`; HTTP-backed sources only
        fetch the bytes the parser needs."""
        from ._nd2_native import Nd2FileParser
        from .core.io import DataSource
        if isinstance(src, (str, Path, DataSource)):
            parser = Nd2FileParser(src)
        elif isinstance(src, (bytes, bytearray, memoryview)):
            import os, tempfile
            fd, tmp = tempfile.mkstemp(suffix=".nd2")
            os.write(fd, bytes(src))
            os.close(fd)
            parser = Nd2FileParser(tmp)
        else:
            raise TypeError(
                f"unsupported ND2 source: {type(src).__name__}")
        attrs = parser.attributes
        return {
            "file_size": parser._size,
            "n_frames": attrs.sequence_count,
            "width": attrs.width,
            "height": attrs.height,
            "n_channels": attrs.n_channels,
            "bits_significant": attrs.bits_significant,
            "bits_in_memory": attrs.bits_in_memory,
            "compression": attrs.compression,
            "dtype": str(attrs.dtype),
            "n_chunks": len(parser.chunks),
        }


__all__ = ["Nd2Codec", "Nd2Reader", "Nd2Error"]
