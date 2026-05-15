"""Native OIB (Olympus FluoView) parser — no `oiffile` package dep.

OIB is a Microsoft Compound File Binary (CFB / OLE2) container with:

  * ``OibInfo.txt``: index file mapping Stream IDs to friendly path
    names (``s_C{c:03d}Z{z:03d}.tif``, ``s_C{c}Z{z}.pty``, etc.)
  * ``Stream<NNNNN>``: the main ``.oif`` INI file describing the
    experiment + per-frame TIFFs

We layer:

    OibNativeReader  →  OleReader (binary container)
                     →  parse OibInfo.txt + main .oif
                     →  per-frame _tiff.TiffStream (existing native TIFF)

Built on the same DataSource abstraction as TIFF / ND2 — so the OIB
reader transparently handles local files and HTTP range reads. The
OLE2 walk only needs the directory + FAT + a handful of metadata
streams (typically < 100 KB total) to fully open an OIB; per-frame
TIFFs are loaded on demand.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .core.codec import Reader
from .core.io import DataSource
from ._ole2 import OleReader


# Per the FluoView OIF spec, axis codes are letters:
#   X / Y = spatial
#   C     = channel
#   Z     = z-slice
#   T     = time
_AXIS_ORDER = ("T", "Z", "C", "Y", "X")


@dataclass
class OibFrameRef:
    """Pointer to one (C, Z, T) TIFF stream inside the OIB."""
    c: int
    z: int
    t: int
    stream: str   # OLE2 stream name


@dataclass
class OibLayout:
    """Decoded geometry of an OIB experiment from the main .oif file."""
    width: int
    height: int
    n_channels: int
    n_z: int
    n_t: int
    dtype: np.dtype
    axis_order: str       # e.g. "XYCZ"
    frames: dict[tuple[int, int, int], str] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        """ndarray shape with singletons squeezed except (H, W).

        Axis order matches FluoView convention (and oiffile's output):
        ``(T?, C?, Z?, H, W)`` — channels OUTSIDE z-slice axis.
        Singletons collapse: a single-channel 6-slice stack returns
        ``(6, H, W)``.
        """
        base: list[int] = []
        if self.n_t > 1:
            base.append(self.n_t)
        if self.n_channels > 1:
            base.append(self.n_channels)
        if self.n_z > 1:
            base.append(self.n_z)
        base.extend([self.height, self.width])
        return tuple(base)


class OibFileParser:
    """OIB container reader. Reads OibInfo.txt + the main .oif INI
    eagerly; per-frame TIFFs are loaded on demand."""

    def __init__(self, src: Any):
        self._ole = OleReader(src)
        self._streams_by_name = self._read_stream_map()
        self.layout = self._parse_main_oif()

    # ----- container metadata -----

    def _read_stream_map(self) -> dict[str, str]:
        """Read OibInfo.txt → dict mapping friendly path → Stream ID.
        Example: ``"Storage00001/s_C001Z001.tif" -> "Stream00001"``."""
        if "OibInfo.txt" not in self._ole.list_streams():
            raise ValueError(
                "OIB: missing OibInfo.txt — not a valid Olympus OIB")
        info = self._ole.read_stream("OibInfo.txt").decode("utf-16-le")
        # Strip BOM
        if info and info[0] == "﻿":
            info = info[1:]
        result: dict[str, str] = {}
        for line in info.splitlines():
            if "=" in line and line.startswith("Stream"):
                key, val = line.split("=", 1)
                result[val.strip()] = key.strip()
            elif "=" in line and line.startswith("Storage"):
                key, val = line.split("=", 1)
                result[val.strip()] = key.strip()
        return result

    def _parse_main_oif(self) -> OibLayout:
        """Read the main .oif INI to get axis sizes + dtype +
        per-frame stream pointers."""
        main_path = self._main_oif_path()
        main_stream = self._streams_by_name.get(main_path)
        if main_stream is None:
            raise ValueError(
                f"OIB: main .oif file {main_path!r} not in stream map")
        text = self._ole.read_stream(main_stream).decode(
            "utf-16-le", errors="replace")
        if text and text[0] == "﻿":
            text = text[1:]
        ini = _parse_ini(text)
        # Pull axis sizes from "Axis N Parameters Common" sections.
        # Each Axis section has AxisCode (X/Y/C/Z/T/...) and MaxSize.
        axis_sizes: dict[str, int] = {}
        for section, kvs in ini.items():
            m = re.match(r"Axis \d+ Parameters Common", section)
            if not m:
                continue
            code = kvs.get("AxisCode", "").strip()
            max_size = int(kvs.get("MaxSize", "0"))
            if code and max_size > 0:
                axis_sizes[code] = max_size
        axis_order = ini.get(
            "Axis Parameter Common", {}).get("AxisOrder", "XYCZ")

        width = axis_sizes.get("X", 0)
        height = axis_sizes.get("Y", 0)
        n_channels = axis_sizes.get("C", 1)
        n_z = axis_sizes.get("Z", 1)
        n_t = axis_sizes.get("T", 1)

        # Build the (C, Z, T) → friendly file path → stream map.
        # OIB names TIFF frames as s_C{c:03d}Z{z:03d}T{t:03d}.tif (T
        # is omitted when there's only one timepoint).
        # The storage path prefix comes from OibInfo's StorageNNNNN
        # entry value (e.g. "...oif.files").
        storage_prefix = self._storage_prefix()
        frames: dict[tuple[int, int, int], str] = {}
        for c in range(1, n_channels + 1):
            for z in range(1, n_z + 1):
                for t in range(1, n_t + 1):
                    name = self._frame_friendly_name(
                        c, z, t, n_t, storage_prefix)
                    stream = self._streams_by_name.get(name)
                    if stream is not None:
                        frames[(c - 1, z - 1, t - 1)] = stream

        # Dtype: derive from the first TIFF frame's IFD. Only peek at
        # the first 2 KB of the TIFF stream — that's enough for the
        # header + first IFD's BitsPerSample / SampleFormat tags, and
        # avoids fetching the full 2 MB stream over HTTP just to learn
        # one byte of metadata.
        dtype = np.dtype("u2")  # OIB is typically uint16
        if frames:
            stream = next(iter(frames.values()))
            tiff_head = self._ole.read_stream(stream, length=2048)
            dtype = _peek_tiff_dtype(tiff_head) or dtype

        return OibLayout(
            width=width,
            height=height,
            n_channels=n_channels,
            n_z=n_z,
            n_t=n_t,
            dtype=dtype,
            axis_order=axis_order,
            frames=frames,
        )

    def _main_oif_path(self) -> str:
        info = self._ole.read_stream("OibInfo.txt").decode("utf-16-le")
        m = re.search(r"^MainFileName=(\S+)", info, re.MULTILINE)
        if not m:
            raise ValueError("OIB: OibInfo.txt has no MainFileName entry")
        main_stream = m.group(1).strip()
        for friendly, sid in self._streams_by_name.items():
            if sid == main_stream:
                return friendly
        raise ValueError(
            f"OIB: main stream {main_stream!r} not in stream map")

    def _storage_prefix(self) -> str:
        """OibInfo.txt records frame paths using the STORAGE ID
        ("Storage00001/..."), not the friendly storage filename
        ("....oif.files/..."). Find the storage ID."""
        for friendly, sid in self._streams_by_name.items():
            if sid.startswith("Storage"):
                return sid
        return ""

    def _frame_friendly_name(
        self, c: int, z: int, t: int, n_t: int, prefix: str,
    ) -> str:
        # OibInfo.txt entries look like "Storage00001/s_C001Z001.tif"
        # — note the STORAGE ID prefix, not the friendly storage
        # filename. The frame file pattern is s_C{c:03d}Z{z:03d}.tif
        # for spatial-only experiments; with time-lapse a T component
        # is added per the FluoView spec, but that's untested here.
        if n_t > 1:
            base = f"s_C{c:03d}Z{z:03d}T{t:03d}.tif"
        else:
            base = f"s_C{c:03d}Z{z:03d}.tif"
        return f"{prefix}/{base}" if prefix else base

    # ----- public read API -----

    @property
    def shape(self) -> tuple[int, ...]:
        return self.layout.shape

    @property
    def dtype(self) -> np.dtype:
        return self.layout.dtype

    def read_frame(self, c: int, z: int, t: int = 0) -> np.ndarray:
        """Decode one (C, Z, T) frame to a 2D ndarray (H, W)."""
        stream = self.layout.frames.get((c, z, t))
        if stream is None:
            raise KeyError(
                f"OIB: no frame at (c={c}, z={z}, t={t}); have "
                f"{len(self.layout.frames)} frames")
        tiff_bytes = self._ole.read_stream(stream)
        from .codecs._tiff import check_signature as _tiff_sig
        from ._tiff_codec import TiffCodec
        if not _tiff_sig(tiff_bytes):
            raise ValueError(
                f"OIB: stream {stream!r} is not a TIFF")
        arr = TiffCodec().decode(tiff_bytes)
        return np.squeeze(arr)

    def read_all(self) -> np.ndarray:
        """Decode every frame, assemble the full (T?, C?, Z?, H, W)
        ndarray (FluoView axis order)."""
        L = self.layout
        out = np.empty(L.shape, dtype=L.dtype)
        for (c, z, t), stream in L.frames.items():
            frame = self.read_frame(c, z, t)
            idx = []
            if L.n_t > 1: idx.append(t)
            if L.n_channels > 1: idx.append(c)
            if L.n_z > 1: idx.append(z)
            out[tuple(idx)] = frame
        return out

    def close(self) -> None:
        self._ole.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_ini(text: str) -> dict[str, dict[str, str]]:
    """Lightweight INI parser. .oif files use UTF-16-LE-with-BOM,
    Windows line endings, and the values are unquoted but may
    contain ``=``. The standard ``configparser`` would work but
    pulls in a lot for one trivial format."""
    result: dict[str, dict[str, str]] = {}
    section: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            result.setdefault(section, {})
            continue
        if "=" not in line or section is None:
            continue
        key, val = line.split("=", 1)
        result[section][key.strip()] = val.strip().strip('"')
    return result


def _peek_tiff_dtype(tiff_bytes: bytes) -> np.dtype | None:
    """Read the BitsPerSample tag from a TIFF without full decode.
    Returns the matching np.dtype, or None if we can't tell."""
    if len(tiff_bytes) < 16:
        return None
    if tiff_bytes[:2] not in (b"II", b"MM"):
        return None
    le = tiff_bytes[:2] == b"II"
    fmt = "<H" if le else ">H"
    if struct.unpack_from(fmt, tiff_bytes, 2)[0] != 42:
        return None
    long_fmt = "<I" if le else ">I"
    ifd_off = struct.unpack_from(long_fmt, tiff_bytes, 4)[0]
    if ifd_off + 2 > len(tiff_bytes):
        return None
    n_entries = struct.unpack_from(fmt, tiff_bytes, ifd_off)[0]
    bps = 0
    sample_format = 1   # 1=uint, 2=int, 3=float
    for i in range(n_entries):
        e = ifd_off + 2 + i * 12
        tag = struct.unpack_from(fmt, tiff_bytes, e)[0]
        if tag == 258:   # BitsPerSample
            bps = struct.unpack_from(fmt, tiff_bytes, e + 8)[0]
        elif tag == 339:   # SampleFormat
            sample_format = struct.unpack_from(fmt, tiff_bytes, e + 8)[0]
    if bps == 8:
        return np.dtype("u1" if sample_format == 1 else "i1")
    if bps == 16:
        return np.dtype("<u2" if sample_format == 1 else "<i2")
    if bps == 32:
        if sample_format == 3:
            return np.dtype("<f4")
        return np.dtype("<u4" if sample_format == 1 else "<i4")
    return None


# ---------------------------------------------------------------------------
# Reader wrapper
# ---------------------------------------------------------------------------


class OibNativeReader(Reader):
    """Native Olympus OIB / OIF reader — no oiffile dependency."""

    def __init__(self, src: Any):
        self._parser = OibFileParser(src)
        self.shape = self._parser.shape
        self.dtype = self._parser.dtype
        L = self._parser.layout
        self.axes = L.axis_order
        # Use the outermost present axis as the frame axis.
        if L.n_t > 1:
            self.n_frames = L.n_t
        elif L.n_z > 1:
            self.n_frames = L.n_z
        elif L.n_channels > 1:
            self.n_frames = L.n_channels
        else:
            self.n_frames = 1
        self.is_chunked = False

    def iter_frames(self) -> Iterator[np.ndarray]:
        full = self._parser.read_all()
        if full.ndim < 3:
            yield full
            return
        for i in range(self.n_frames):
            yield full[i]

    def read(self) -> np.ndarray:
        return self._parser.read_all()

    def close(self) -> None:
        self._parser.close()

    def __enter__(self) -> "OibNativeReader":
        return self

    def __exit__(self, *_) -> bool:
        self.close()
        return False


__all__ = ["OibNativeReader", "OibFileParser", "OibLayout"]
