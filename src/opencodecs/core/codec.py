"""Core abstractions for opencodecs.

Three ABCs and a global registry:

  Codec   — encode / decode / open / signature; one instance per format.
  Reader  — uniform iter_frames / read / random access for multi-unit
            formats.
  Writer  — uniform write_frame / close.

Format-specific codecs (JpegXLCodec, PngCodec, …) subclass Codec and
``register_codec(self)`` themselves at import time. Top-level
opencodecs.read / write / open auto-dispatch by extension or magic
signature.

Implementation status flags on each Codec instance let callers check
what's supported (native fast path vs delegated to imagecodecs vs not
yet implemented).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Reader / Writer ABCs (uniform per-format streaming surface)
# ---------------------------------------------------------------------------


class Reader(ABC):
    """Uniform reader interface for multi-frame / multi-chunk formats.

    Single-image formats expose a 1-element iterator (so the same code
    path works for stills and stacks). Random access via ``[idx]`` is
    optional and only available for chunked / frame-indexed formats.
    """

    # Subclasses populate these eagerly during construction (header parse).
    shape: tuple[int, ...]
    dtype: np.dtype
    color: dict | None = None
    icc_profile: bytes | None = None
    n_frames: int | None = None  # None when unknown
    is_chunked: bool = False  # True when [idx] random access is available

    @abstractmethod
    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield one decoded ndarray per frame / chunk."""

    def read(self) -> np.ndarray:
        """Decode the entire stream into a single ndarray.

        Default impl materializes via iter_frames. Subclasses with a
        faster bulk path can override.
        """
        frames = list(self.iter_frames())
        if not frames:
            raise ValueError("empty stream")
        if len(frames) == 1:
            return frames[0]
        return np.stack(frames, axis=0)

    def __getitem__(self, idx) -> np.ndarray:
        """Random access by frame / chunk index (when ``is_chunked``).

        Default impl is O(N) — iterates and picks. Subclasses with a
        frame index or chunk offset table override for O(1).
        """
        if not self.is_chunked:
            raise TypeError(
                f"{type(self).__name__} does not support random access"
            )
        for i, frame in enumerate(self.iter_frames()):
            if i == idx:
                return frame
        raise IndexError(idx)

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.iter_frames()

    def __enter__(self) -> "Reader":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def close(self) -> None:  # pragma: no cover - default no-op for subclasses
        pass  # subclasses override if they hold resources


class Writer(ABC):
    """Uniform writer interface for streaming/multi-frame encode."""

    @abstractmethod
    def write_frame(self, arr: np.ndarray, **opts) -> None:
        """Encode one frame into the stream."""

    def close(self) -> bytes | None:  # pragma: no cover - default no-op for subclasses
        """Finalize. Returns bytes for in-memory mode, else None."""
        return None

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.close()
        return False


# ---------------------------------------------------------------------------
# Codec ABC
# ---------------------------------------------------------------------------


class Codec(ABC):
    """One instance per file format.

    Subclasses set the class attributes at definition time and override
    the methods needed for the format. Most formats only need encode /
    decode / signature; multi-unit formats also override open() to return
    a Reader with iter_frames.
    """

    # Format identity
    name: str = ""                            # "jxl", "png", ...
    file_extensions: tuple[str, ...] = ()     # (".jxl",)
    aliases: tuple[str, ...] = ()             # ("jpegxl", "jpeg-xl")

    # Capability flags — discoverable via list_codecs()
    has_native: bool = False     # opencodecs native impl
    has_delegate: bool = False   # delegated to imagecodecs / tifffile / ...
    can_encode: bool = False
    can_decode: bool = False
    multi_frame: bool = False    # supports stacks / animations
    chunked: bool = False        # supports random-access chunks
    streaming_decode: bool = False  # iter_frames yields without full materialization
    parallel_decode: bool = False   # multi-chunk parallel decode supported

    # Color / dtype support — informational
    supported_dtypes: tuple = ()  # e.g. (np.uint8, np.uint16, np.float16, np.float32)
    supports_color: bool = False  # ColorSpec / ICC profile honored

    # ---- canonical methods ----

    def signature(self, header_bytes: bytes) -> bool:
        """True if these first bytes look like this format's container.

        Default: match by extension only (caller must know the type).
        Overridden by formats with a magic header.
        """
        return False

    def encode(
        self,
        arr: np.ndarray,
        *,
        dest: Any | None = None,
        **opts,
    ) -> bytes | None:
        if not self.can_encode:
            raise NotImplementedError(f"{self.name}: encode not supported")
        raise NotImplementedError  # pragma: no cover - subclass must override

    def decode(self, src: Any, **opts) -> np.ndarray:
        if not self.can_decode:
            raise NotImplementedError(f"{self.name}: decode not supported")
        raise NotImplementedError  # pragma: no cover - subclass must override

    def open(self, src: Any, **opts) -> Reader:
        """Open for streaming / random-access reading.

        Default impl: decode the whole thing eagerly and wrap in a
        single-frame Reader. Codec-specific overrides do real streaming.
        """
        arr = self.decode(src, **opts)
        return _SingleFrameReader(arr)

    def __repr__(self) -> str:
        flags = []
        if self.has_native:
            flags.append("native")
        elif self.has_delegate:
            flags.append("delegate")
        else:
            flags.append("stub")
        if self.can_encode and self.can_decode:
            flags.append("rw")
        elif self.can_decode:
            flags.append("ro")
        elif self.can_encode:
            flags.append("wo")
        if self.multi_frame:
            flags.append("multi")
        if self.chunked:
            flags.append("chunked")
        if self.parallel_decode:
            flags.append("parallel")
        return f"<Codec {self.name} ({', '.join(flags)})>"


class _SingleFrameReader(Reader):
    """Trivial Reader wrapping a single ndarray (for codecs without a
    real streaming impl)."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr
        self.shape = arr.shape
        self.dtype = arr.dtype
        self.n_frames = 1

    def iter_frames(self) -> Iterator[np.ndarray]:
        yield self._arr

    def read(self) -> np.ndarray:
        return self._arr


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, Codec] = {}


def register_codec(codec: Codec) -> None:
    """Add a codec to the global registry under its name + aliases.

    Idempotent: re-registering the same name replaces the previous entry,
    which is what you want during dev (re-import after edit).
    """
    if not codec.name:
        raise ValueError(f"codec {codec!r} has no .name")
    keys = (codec.name, *codec.aliases)
    for key in keys:
        _REGISTRY[key.lower()] = codec


def get_codec(name: str) -> Codec:
    """Look up a codec by name or alias. Raises KeyError if not found."""
    try:
        return _REGISTRY[name.lower()]
    except KeyError:
        raise KeyError(
            f"no codec named {name!r}; "
            f"registered: {sorted({c.name for c in _REGISTRY.values()})}"
        ) from None


def list_codecs() -> list[dict]:
    """Return one descriptor dict per unique registered codec.

    >>> for c in list_codecs(): print(c['name'], c['flags'])
    """
    seen: dict[int, Codec] = {}
    for c in _REGISTRY.values():
        seen[id(c)] = c
    out = []
    for c in seen.values():
        out.append({
            "name": c.name,
            "aliases": list(c.aliases),
            "extensions": list(c.file_extensions),
            "native": c.has_native,
            "delegate": c.has_delegate,
            "encode": c.can_encode,
            "decode": c.can_decode,
            "multi_frame": c.multi_frame,
            "chunked": c.chunked,
            "parallel_decode": c.parallel_decode,
            "dtypes": [np.dtype(d).name for d in c.supported_dtypes],
        })
    out.sort(key=lambda d: d["name"])
    return out


def has_codec(name: str, *, op: str | None = None) -> bool:
    """True if a codec exists, optionally with the requested operation."""
    try:
        c = get_codec(name)
    except KeyError:
        return False
    if op == "encode":
        return c.can_encode
    if op == "decode":
        return c.can_decode
    return True


def codec_for_path(path: str | os.PathLike) -> Codec:
    """Resolve a codec by file extension. Raises KeyError if unknown."""
    ext = Path(path).suffix.lower()
    if not ext:
        raise KeyError(f"no extension on {path!r}")
    for c in {id(v): v for v in _REGISTRY.values()}.values():
        if ext in c.file_extensions:
            return c
    raise KeyError(f"no codec registered for extension {ext!r}")


def codec_for_bytes(data: bytes | memoryview, n: int = 32) -> Codec:
    """Resolve a codec by sniffing the magic bytes of the data."""
    head = bytes(data[:n])
    for c in {id(v): v for v in _REGISTRY.values()}.values():
        try:
            if c.signature(head):
                return c
        except Exception:  # pragma: no cover - signature() defensive catch
            continue
    raise KeyError("no codec recognizes this data signature")


def _resolve_codec(src: Any, *, format: str | None = None) -> Codec:
    """Best-effort dispatch used by top-level read / open."""
    if format is not None:
        return get_codec(format)
    if isinstance(src, (str, os.PathLike)):
        try:
            return codec_for_path(src)
        except KeyError:
            # Path didn't have a known extension. Fall back to peeking
            # the file's magic bytes.
            try:
                with open(src, "rb") as f:
                    head = f.read(64)
                return codec_for_bytes(head)
            except Exception:
                pass
            raise KeyError(f"can't determine codec for {src!r}") from None
    if isinstance(src, (bytes, bytearray, memoryview)):
        return codec_for_bytes(src)
    if hasattr(src, "read"):
        # File-like — peek + reset
        if hasattr(src, "seek") and hasattr(src, "tell"):
            pos = src.tell()
            head = src.read(64)
            src.seek(pos)
            return codec_for_bytes(head)
    raise KeyError(f"can't determine codec for {type(src).__name__}")


__all__ = [
    "Codec",
    "Reader",
    "Writer",
    "register_codec",
    "get_codec",
    "list_codecs",
    "has_codec",
    "codec_for_path",
    "codec_for_bytes",
]
