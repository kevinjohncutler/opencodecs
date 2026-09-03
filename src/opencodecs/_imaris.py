"""Imaris (.ims) reader.

Imaris files are HDF5 with a fixed layout that Bitplane's software
writes and that a lot of light-microscopy work ends up stored in::

    /DataSet/ResolutionLevel N/TimePoint T/Channel C/Data   (Z, Y, X)
    /DataSetInfo/Image        attributes: X, Y, Z, ExtMin*, ExtMax*
    /DataSetInfo/Channel N    attributes: Name, Color, ...

Two conventions make it more than "an HDF5 file with arrays in it", and
both are only discoverable from a file Imaris actually wrote.

**The stored array is padded.** ``Data`` is allocated on chunk-friendly
bounds, so a 1949-pixel image is stored 2048 wide with the remainder
zeroed. The real extent lives in ``ImageSizeX/Y/Z`` on the *channel*
group, per resolution level, and a reader that returns ``Data`` as-is
hands back a border of fabricated zeros.

**Attributes are arrays of single characters**, not HDF5 strings. Read
``Image.attrs["X"]`` and you get ``[b'1', b'9', b'4', b'9']``, so
anything expecting a string gets nothing usable.

Because it is a genuine resolution pyramid, this implements the shared
:class:`opencodecs.core.pyramid.PyramidReader` rather than inventing a
second level API.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.pyramid import PyramidLevel, PyramidReader


class ImarisError(Exception):
    """Raised for files that are not Imaris, or that we cannot interpret."""


def _attr_str(group: Any, key: str) -> str | None:
    """Decode Imaris's character-array attribute convention."""
    if key not in group.attrs:
        return None
    v = group.attrs[key]
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, str):
        return v
    try:
        return b"".join(bytes(c) for c in v.tolist()).decode("utf-8", "replace")
    except Exception:                                    # noqa: BLE001
        return str(v)


def _attr_int(group: Any, key: str) -> int | None:
    s = _attr_str(group, key)
    if s is None:
        return None
    try:
        return int(float(s.strip()))
    except ValueError:
        return None


class _ImarisLevelAccessor:
    """Pixel access for one resolution level, cropped to the real extent."""

    def __init__(self, reader: "ImarisReader", level: int):
        self._reader = reader
        self._level = level

    @property
    def shape(self) -> tuple[int, ...]:
        """Full ``(Z, Y, X)`` for this level.

        Deliberately different from the enclosing PyramidLevel.shape,
        which is the ``(Y, X)`` plane the region API measures against.
        """
        return self._reader.level_shape(self._level)

    @property
    def dtype(self) -> np.dtype:
        return self._reader.dtype

    def asarray(self, *, timepoint: int = 0, channel: int = 0) -> np.ndarray:
        return self._reader.read(level=self._level, timepoint=timepoint,
                                 channel=channel)

    def __repr__(self) -> str:
        return f"<ImarisLevel {self._level} shape={self.shape}>"


class ImarisReader(PyramidReader):
    """Reader for one Imaris .ims file."""

    def __init__(self, path: Any, *, timepoint: int = 0, channel: int = 0):
        try:
            import h5py
        except ImportError:                              # pragma: no cover
            raise ImarisError(
                "imaris: reading .ims needs h5py; install opencodecs[hdf5]"
            ) from None
        self._h5 = h5py.File(path, "r")
        if "DataSet" not in self._h5:
            self._h5.close()
            raise ImarisError(
                "imaris: no /DataSet group; this is an HDF5 file but not an "
                "Imaris one")

        ds = self._h5["DataSet"]
        # "ResolutionLevel 10" must sort after "ResolutionLevel 9".
        self._level_names = sorted(
            (k for k in ds.keys() if k.startswith("ResolutionLevel")),
            key=lambda k: int(k.rsplit(" ", 1)[-1]))
        if not self._level_names:
            self._h5.close()
            raise ImarisError("imaris: /DataSet has no ResolutionLevel groups")

        first = ds[self._level_names[0]]
        self._timepoint_names = sorted(
            (k for k in first.keys() if k.startswith("TimePoint")),
            key=lambda k: int(k.rsplit(" ", 1)[-1]))
        self._channel_names = sorted(
            (k for k in first[self._timepoint_names[0]].keys()
             if k.startswith("Channel")),
            key=lambda k: int(k.rsplit(" ", 1)[-1]))

        self.timepoint = timepoint
        self.channel = channel
        self._levels_cache: list[PyramidLevel] | None = None

    # -- structure ---------------------------------------------------

    @property
    def n_timepoints(self) -> int:
        return len(self._timepoint_names)

    @property
    def n_channels(self) -> int:
        return len(self._channel_names)

    def _channel_group(self, level: int, timepoint: int, channel: int):
        try:
            return (self._h5["DataSet"][self._level_names[level]]
                    [self._timepoint_names[timepoint]]
                    [self._channel_names[channel]])
        except IndexError:
            raise IndexError(
                f"imaris: level={level} timepoint={timepoint} "
                f"channel={channel} out of range "
                f"({self.n_levels}, {self.n_timepoints}, {self.n_channels})"
            ) from None

    def level_shape(self, level: int) -> tuple[int, int, int]:
        """True ``(Z, Y, X)`` for a level, ignoring the stored padding.

        Each level carries its own ImageSize attributes; the global
        DataSetInfo/Image extent only ever describes level 0, so using it
        everywhere would over-crop every coarser level.
        """
        g = self._channel_group(level, self.timepoint, self.channel)
        stored = g["Data"].shape
        sizes = tuple(_attr_int(g, f"ImageSize{a}") for a in ("Z", "Y", "X"))
        return tuple(
            s if (s is not None and 0 < s <= st) else st
            for s, st in zip(sizes, stored))              # type: ignore

    @property
    def dtype(self) -> np.dtype:
        g = self._channel_group(0, self.timepoint, self.channel)
        return np.dtype(g["Data"].dtype)

    @property
    def levels(self) -> list[PyramidLevel]:
        if self._levels_cache is None:
            base = self.level_shape(0)
            out = []
            for i in range(len(self._level_names)):
                shape = self.level_shape(i)
                # Pyramid levels are described by their (y, x) downscale.
                dy = max(1, round(base[1] / shape[1])) if shape[1] else 1
                dx = max(1, round(base[2] / shape[2])) if shape[2] else 1
                # PyramidLevel.shape is the (y, x) plane, which is what
                # the shared region API measures bounding boxes against.
                # Passing (z, y, x) here makes read_region treat the z
                # extent as the image height and best_level_for choose on
                # it, so a single-plane stack reports height 1 and every
                # level looks small enough.
                out.append(PyramidLevel(
                    reader=_ImarisLevelAccessor(self, i),
                    downscale=(dy, dx),
                    shape=shape[1:],
                    dtype=self.dtype,
                ))
            self._levels_cache = out
        return self._levels_cache

    # -- pixels ------------------------------------------------------

    def read(self, *, level: int = 0, timepoint: int | None = None,
             channel: int | None = None) -> np.ndarray:
        """Read a whole level as ``(Z, Y, X)``, cropped to the real extent."""
        tp = self.timepoint if timepoint is None else timepoint
        ch = self.channel if channel is None else channel
        g = self._channel_group(level, tp, ch)
        saved_tp, saved_ch = self.timepoint, self.channel
        self.timepoint, self.channel = tp, ch
        try:
            z, y, x = self.level_shape(level)
        finally:
            self.timepoint, self.channel = saved_tp, saved_ch
        return g["Data"][:z, :y, :x]

    def _read_region(self, level: PyramidLevel, y0: int, y1: int,
                     x0: int, x1: int) -> np.ndarray:
        """Backend hook for PyramidReader.read_region.

        Slices the HDF5 dataset directly, so h5py fetches only the chunks
        overlapping the box rather than materializing the level. The box
        is a (y, x) rectangle taken across the whole z extent, so a
        single-plane acquisition returns ``(y, x)`` and a stack returns
        ``(z, y, x)``.
        """
        # Take the index off the accessor rather than searching the list:
        # PyramidLevel is a dataclass, so list.index would compare fields
        # and depend on the accessor's identity semantics.
        index = level.reader._level
        g = self._channel_group(index, self.timepoint, self.channel)
        z, _, _ = self.level_shape(index)
        region = g["Data"][:z, y0:y1, x0:x1]
        # A 2-D acquisition is the common case; drop the singleton z so
        # read_region returns (y, x) like the other pyramid backends.
        return region[0] if region.shape[0] == 1 else region

    # -- metadata ----------------------------------------------------

    @property
    def info(self) -> dict[str, str]:
        """Decoded /DataSetInfo/Image attributes."""
        if "DataSetInfo" not in self._h5 or "Image" not in self._h5["DataSetInfo"]:
            return {}
        img = self._h5["DataSetInfo"]["Image"]
        return {k: _attr_str(img, k) or "" for k in img.attrs}

    def channel_info(self, channel: int = 0) -> dict[str, str]:
        info = self._h5.get("DataSetInfo")
        key = f"Channel {channel}"
        if info is None or key not in info:
            return {}
        return {k: _attr_str(info[key], k) or "" for k in info[key].attrs}

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        """``(z, y, x)`` physical size per voxel, from the extent bounds."""
        img = self._h5.get("DataSetInfo", {}).get("Image") \
            if "DataSetInfo" in self._h5 else None
        if img is None:
            return (0.0, 0.0, 0.0)
        shape = self.level_shape(0)
        out = []
        for axis, n in zip((2, 1, 0), shape):            # z, y, x
            lo = _attr_str(img, f"ExtMin{axis}")
            hi = _attr_str(img, f"ExtMax{axis}")
            try:
                out.append((float(hi) - float(lo)) / n if n else 0.0)
            except (TypeError, ValueError):
                out.append(0.0)
        return tuple(out)                                 # type: ignore

    def close(self) -> None:
        if getattr(self, "_h5", None) is not None:
            self._h5.close()
            self._h5 = None                               # type: ignore

    def __repr__(self) -> str:
        return (f"<ImarisReader levels={self.n_levels} "
                f"timepoints={self.n_timepoints} channels={self.n_channels} "
                f"shape={self.level_shape(0)} dtype={self.dtype.str}>")


__all__ = ["ImarisReader", "ImarisError"]
