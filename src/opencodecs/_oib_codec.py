"""OIB / OIF Reader/Codec wrapping the ``oiffile`` Python package.

Olympus OIB (Olympus Image Binary, a single-file OLE-style container)
and OIF (Olympus Image File, a directory + .oif index) are the
proprietary FluoView confocal formats. Christoph Gohlke's
``oiffile`` package is the canonical reader; we wrap it the same way
Nd2Codec / LifCodec wrap their respective libraries.

OIB inside is actually a Microsoft Compound File Binary (CFB / OLE2)
container, with TIFF stacks for the pixel data. Decode returns the
full (T, C, Y, X) stack assembled by oiffile.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .core.codec import Codec, Reader
from .core.io import DataSource

try:
    import oiffile as _oif
    _HAVE_OIFFILE = True
except ImportError:  # pragma: no cover - oiffile-missing branch
    _HAVE_OIFFILE = False


class OibError(RuntimeError):
    """Raised on OIB / OIF open / decode failures."""


class OibReader(Reader):
    """Streaming Reader for Olympus OIB / OIF files."""

    def __init__(self, path: str | Path):
        if not _HAVE_OIFFILE:  # pragma: no cover - oiffile-missing
            raise ImportError(
                "oiffile is required for OIB / OIF support: "
                "pip install oiffile")
        self._path = str(path)
        self._of = _oif.OifFile(self._path)
        # oiffile presents (T, C, Y, X) or (Z, C, Y, X) depending on
        # the original acquisition mode.
        self.shape = tuple(self._of.shape)
        self.dtype = np.dtype(self._of.dtype)
        self.axes = str(self._of.axes)
        self.n_frames = self.shape[0] if len(self.shape) >= 3 else 1
        self.is_chunked = False

    def iter_frames(self) -> Iterator[np.ndarray]:
        full = self.read()
        if full.ndim < 3:
            yield full
            return
        for i in range(full.shape[0]):
            yield full[i]

    def read(self) -> np.ndarray:
        return self._of.asarray()

    def close(self) -> None:
        if self._of is not None:
            try:
                self._of.close()
            except Exception:
                pass
            self._of = None

    def __enter__(self) -> "OibReader":
        return self

    def __exit__(self, *_) -> bool:
        self.close()
        return False


class OibCodec(Codec):
    """Olympus OIB / OIF container codec — FluoView confocal microscopy.

    Two backends:
      * ``opencodecs._oib_native.OibNativeReader`` — pure-Python parser
        for the OLE2 / Compound File Binary container that backs OIB.
        Decodes the per-frame TIFF streams through our native TIFF
        reader. Works on local paths AND ``HTTPDataSource`` (range
        reads → only the OLE2 directory + the requested frame's
        TIFF stream are fetched).
      * ``oiffile`` package — full-featured delegate. Used as a
        fallback. Also the only path for OIF (directory-based)
        variants since the native OLE2 parser only handles OIB.

    ``open(src)`` tries native first, falls back to the delegate.
    """

    name = "oib"
    file_extensions = (".oib", ".oif")
    aliases = ("oif",)

    has_native = True
    has_delegate = _HAVE_OIFFILE
    can_encode = False
    can_decode = True
    multi_frame = True
    chunked = False
    streaming_decode = True
    parallel_decode = False

    supported_dtypes = (np.uint8, np.uint16, np.uint32)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        """OIB is a Microsoft Compound File Binary (CFB / OLE2)
        container; the first 8 bytes are the CFB magic
        ``\\xD0\\xCF\\x11\\xE0\\xA1\\xB1\\x1A\\xE1``. OIF doesn't have
        a magic-byte signature (it's a directory), so we only detect
        OIB here; OIF dispatch is by file extension."""
        return (
            len(head) >= 8
            and head[:8] == b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"
        )

    def decode(self, src: Any, **opts) -> np.ndarray:
        with self.open(src, **opts) as reader:
            return reader.read()

    def open(self, src: Any, *, backend: str | None = None,
             **opts) -> Reader:
        """Open OIB / OIF for reading.

        ``backend``:
          * ``None`` (default): native first, fall back to oiffile.
          * ``"native"``: force the native parser. Won't handle OIF
            (directory variant) — that requires oiffile.
          * ``"oiffile"``: force the oiffile delegate.
        """
        if backend in (None, "native"):
            try:
                from ._oib_native import OibNativeReader
                if isinstance(src, (str, Path)) or isinstance(src, DataSource):
                    return OibNativeReader(src)
                # bytes / file-like: spill to a temp file
                import os, tempfile
                if isinstance(src, (bytes, bytearray, memoryview)):
                    fd, tmp = tempfile.mkstemp(suffix=".oib")
                    os.write(fd, bytes(src))
                    os.close(fd)
                    return OibNativeReader(tmp)
                if hasattr(src, "read"):
                    data = src.read()
                    fd, tmp = tempfile.mkstemp(suffix=".oib")
                    os.write(fd, data)
                    os.close(fd)
                    return OibNativeReader(tmp)
            except (NotImplementedError, ValueError, KeyError) as e:
                if backend == "native":
                    raise
                # Fall through to delegate (e.g. OIF directory variant)
        if not _HAVE_OIFFILE:
            raise ImportError(
                "OIB: native parser couldn't handle this source and "
                "oiffile is not installed. pip install oiffile for "
                "fallback support of OIF (directory) variants.")
        if isinstance(src, (str, Path)):
            return OibReader(src)
        import os, tempfile
        if isinstance(src, (bytes, bytearray, memoryview)):
            fd, tmp = tempfile.mkstemp(suffix=".oib")
            os.write(fd, bytes(src))
            os.close(fd)
            return OibReader(tmp)
        if hasattr(src, "read"):
            data = src.read()
            fd, tmp = tempfile.mkstemp(suffix=".oib")
            os.write(fd, data)
            os.close(fd)
            return OibReader(tmp)
        raise TypeError(f"unsupported OIB source: {type(src).__name__}")


__all__ = ["OibCodec", "OibReader", "OibError"]
