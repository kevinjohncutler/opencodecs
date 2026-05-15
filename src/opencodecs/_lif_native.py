"""Native LIF (Leica Image File) parser — no readlif dependency.

**STATUS: experimental, not the default registered LIF codec.** The
production path is still ``LifCodec`` via the ``readlif`` package
(see ``_lif_codec.py``). This module establishes the binary-parsing
+ XML-metadata pattern that future native vendor readers (ND2, OIB)
can follow.

The LIF binary container is straightforward: a UTF-16-LE XML header
describing the experiment structure, followed by N "memory blocks"
each holding a contiguous pixel-data buffer for one image. The XML
maps image elements to their memory blocks via a MemoryBlockID
reference; each image's <Dimensions> sub-tree declares the axis order
and per-axis stride (``BytesInc``).

Decoding a LIF image is therefore:

    1. Find the memory block whose name matches the image's
       MemoryBlockID.
    2. Read that block as bytes.
    3. ``np.frombuffer`` + reshape using the per-axis NumberOfElements
       (sorted by BytesInc ascending). Channels are an extra axis
       declared in <Channels><ChannelDescription> entries.

We support the common case of a contiguous, row-major memory layout
where each dimension's BytesInc is the product of the inner dims —
i.e. a simple ndarray with one stride per axis.

**Known limitation: frame-order metadata.** Some Leica LIFs (notably
those produced by LAS-X tile-scan workflows) embed FrameOrder /
ScanOrder XML attributes that override the natural memory-stride
ordering for mosaic / Z / T axes. ``readlif`` honors these; we
don't. For files without frame-order overrides (the common case)
this module produces byte-identical output to readlif. For files
with overrides (the "frameOrderCombinedScanTypes" corpus example),
only the m=0 mosaic slice matches; m>=1 slices are returned in
memory-stride order rather than the frame-order-corrected order.

LIF binary structure (per Leica LAS-X SDK + observed bytes):

    [magic 0x70 LE u32]
    [block payload size u32]
    [test 0x2A]
    [xml char count u32]   # UTF-16-LE chars, NOT bytes
    [XML metadata (UTF-16-LE)]
    Then for each memory block:
      [magic 0x70 LE u32]
      [header size u32]
      [test 0x2A]
      [memory size u64]
      [test 0x2A]
      [desc char count u32]   # UTF-16-LE chars
      [desc string (UTF-16-LE)]
      [data ... memory size bytes]
"""

from __future__ import annotations

import os
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .core.codec import Reader


_MAGIC = b"\x70\x00\x00\x00"
_TEST = 0x2A


# DimID values from the LIF spec.
_DIM_NAMES = {
    1:  "X",
    2:  "Y",
    3:  "Z",
    4:  "T",
    5:  "C",   # confocal lambda (color in some software)
    6:  "S",
    10: "M",   # mosaic / tile position
}


@dataclass
class LifMemoryBlock:
    """One memory block in the LIF file. `name` is the LIF-internal
    identifier (e.g. "MemBlock_86") that the XML references."""
    name: str
    offset: int
    size: int


@dataclass
class LifImage:
    """One LIF image element. Maps the XML Element to a memory block,
    plus the axis ordering / stride / channel info needed to reshape
    the block's bytes into a labelled ndarray."""
    name: str
    memblock_id: str
    dtype: np.dtype
    # axis_order is the dimensions in MEMORY order (fastest-changing
    # axis first), each entry is (axis_label, n_elements, bytes_inc).
    axis_order: list[tuple[str, int, int]] = field(default_factory=list)
    n_channels: int = 1
    channel_bytes_inc: int = 0
    bytes_per_sample: int = 1

    def shape(self) -> tuple[int, ...]:
        """Reshape shape (slowest-changing axis first), so the data
        reads as ``arr[T, Z, C, Y, X]`` etc."""
        # axis_order is fastest-first; reshape wants slowest-first.
        ordered = list(reversed(self.axis_order))
        out: list[int] = []
        # Channels are a separate axis declared in <Channels> with
        # their own bytes_inc. Insert it at the right position: just
        # before the first axis whose bytes_inc is greater than the
        # channel bytes_inc.
        chan_inserted = False
        for name, n, _ in ordered:
            if not chan_inserted and self.n_channels > 1:
                # Find a position. Compare channel_bytes_inc against
                # the current axis's bytes_inc to decide ordering.
                pass
            out.append(n)
        # Simple convention: channels go just before the innermost
        # 2 spatial dims (Y, X). For most LIFs this is correct.
        if self.n_channels > 1:
            # axis_order has X (innermost) and Y next. Insert C before Y.
            # Reversed order: [...slowest..., Z, Y, X]; insert C before Y.
            shape = list(out)
            if len(shape) >= 2:
                # Insert C at index -2 (before Y, since Y is at -2 in
                # reverse order? wait need to think).
                # ordered = reversed(axis_order) gives slowest-first.
                # axis_order fast-first: [X, Y, Z, M, T] → reversed: [T, M, Z, Y, X]
                # Channel axis goes between Z and Y per BytesInc.
                shape.insert(-2, self.n_channels)
            else:
                shape.insert(0, self.n_channels)
            return tuple(shape)
        return tuple(out)


class LifFileParser:
    """Lazy parser over a LIF file. Reads the XML header eagerly,
    enumerates memory blocks lazily, decodes individual images on
    request."""

    def __init__(self, path: str | Path):
        self._path = str(path)
        with open(self._path, "rb") as f:
            self._data = memoryview(f.read())
        self.xml: str = self._parse_xml_header()
        self.blocks: dict[str, LifMemoryBlock] = self._scan_blocks()
        self.images: list[LifImage] = self._parse_images_from_xml()

    # ----- header / block walk -----

    def _parse_xml_header(self) -> str:
        d = self._data
        if d[0:4] != _MAGIC:
            raise ValueError(
                f"LIF: bad magic at offset 0: {bytes(d[0:4]).hex()}"
            )
        # bytes 4-7: block payload size (we don't need it)
        if d[8] != _TEST:
            raise ValueError(
                f"LIF: bad test byte at offset 8: {d[8]:#x}"
            )
        xml_chars = struct.unpack_from("<I", d, 9)[0]
        xml_bytes = xml_chars * 2
        return bytes(d[13:13 + xml_bytes]).decode("utf-16-le")

    def _scan_blocks(self) -> dict[str, LifMemoryBlock]:
        d = self._data
        xml_chars = struct.unpack_from("<I", d, 9)[0]
        off = 13 + xml_chars * 2
        blocks: dict[str, LifMemoryBlock] = {}
        while off < len(d):
            if d[off:off + 4] != _MAGIC:
                raise ValueError(
                    f"LIF: bad memblock magic at offset {off}: "
                    f"{bytes(d[off:off+4]).hex()}"
                )
            if d[off + 8] != _TEST:
                raise ValueError(
                    f"LIF: bad memblock test1 byte at {off + 8}"
                )
            mem_size = struct.unpack_from("<Q", d, off + 9)[0]
            if d[off + 17] != _TEST:
                raise ValueError(
                    f"LIF: bad memblock test2 byte at {off + 17}"
                )
            desc_chars = struct.unpack_from("<I", d, off + 18)[0]
            desc = bytes(
                d[off + 22:off + 22 + desc_chars * 2]
            ).decode("utf-16-le")
            data_off = off + 22 + desc_chars * 2
            blocks[desc] = LifMemoryBlock(
                name=desc, offset=data_off, size=mem_size,
            )
            off = data_off + mem_size
        return blocks

    # ----- XML element walk -----

    def _parse_images_from_xml(self) -> list[LifImage]:
        try:
            root = ET.fromstring(self.xml)
        except ET.ParseError as e:
            raise ValueError(f"LIF: malformed XML header: {e}")
        images: list[LifImage] = []
        # Walk every Element that contains a Data/Image sub-tree.
        # Element nesting depth varies, so use iter() over all descendants.
        for elem in root.iter("Element"):
            image_data = elem.find("./Data/Image")
            if image_data is None:
                continue
            # An Element can have multiple <Memory> tags (a real LIF
            # often pairs a placeholder Memory[Size=0] with the real
            # data Memory[Size>0]). Pick the non-empty one.
            mb_id = ""
            for mem in elem.iter("Memory"):
                size_str = mem.get("Size", "0")
                try:
                    if int(size_str) > 0:
                        mb_id = mem.get("MemoryBlockID", "")
                        break
                except ValueError:
                    continue
            if not mb_id:
                continue
            image = self._parse_image_element(
                elem, image_data, mb_id,
            )
            if image is not None:
                images.append(image)
        return images

    def _parse_image_element(
        self, elem: ET.Element, image_data: ET.Element, mb_id: str,
    ) -> LifImage | None:
        name = elem.get("Name", "<unnamed>")
        desc = image_data.find("ImageDescription")
        if desc is None:
            return None
        dims = desc.find("Dimensions")
        chans = desc.find("Channels")
        if dims is None or chans is None:
            return None

        # Channels
        chan_entries = list(chans.findall("ChannelDescription"))
        n_channels = len(chan_entries) if chan_entries else 1
        chan_resolution = (
            int(chan_entries[0].get("Resolution", "8"))
            if chan_entries else 8
        )
        # BytesInc for the channel axis (the offset between channels)
        channel_bytes_inc = (
            int(chan_entries[1].get("BytesInc", "0"))
            if len(chan_entries) > 1 else 0
        )
        bytes_per_sample = max(1, (chan_resolution + 7) // 8)
        dtype = np.dtype(f"u{bytes_per_sample}")

        # Dimensions — sort by BytesInc ascending (fastest-first)
        dim_entries: list[tuple[str, int, int]] = []
        for d in dims.findall("DimensionDescription"):
            dim_id = int(d.get("DimID", "0"))
            n = int(d.get("NumberOfElements", "1"))
            bi = int(d.get("BytesInc", "0"))
            label = _DIM_NAMES.get(dim_id, f"D{dim_id}")
            dim_entries.append((label, n, bi))
        dim_entries.sort(key=lambda t: t[2])

        return LifImage(
            name=name,
            memblock_id=mb_id,
            dtype=dtype,
            axis_order=dim_entries,
            n_channels=n_channels,
            channel_bytes_inc=channel_bytes_inc,
            bytes_per_sample=bytes_per_sample,
        )

    # ----- decode -----

    def array_for_image(self, image: LifImage) -> np.ndarray:
        """Materialize one image as an ndarray with axes in
        slowest-first order. Layout matches what readlif's
        ``LifImage.as_array()`` produces for non-mosaic LIFs and is
        a (mosaic, ...) stack for mosaic LIFs."""
        block = self.blocks.get(image.memblock_id)
        if block is None:
            raise KeyError(
                f"LIF: image {image.name!r} references missing "
                f"memory block {image.memblock_id!r}"
            )
        buf = bytes(self._data[block.offset:block.offset + block.size])
        flat = np.frombuffer(buf, dtype=image.dtype).copy()

        # Compute shape using the SORTED axis_order + channels.
        # Strides in memory: each axis stride = BytesInc / bytes_per_sample.
        # We can recover the per-axis element count from NumberOfElements.
        # The full memory layout (fastest→slowest) is:
        #   X, Y, C, Z, M, T  (typical), but order depends on BytesInc.
        # The channel axis is interleaved at channel_bytes_inc.

        # Build a list of (label, n, stride_elems) including channels.
        axes_with_channel: list[tuple[str, int, int]] = []
        for label, n, bi in image.axis_order:
            axes_with_channel.append(
                (label, n, bi // image.bytes_per_sample)
            )
        if image.n_channels > 1:
            chan_stride = (
                image.channel_bytes_inc // image.bytes_per_sample
            )
            axes_with_channel.append(("C", image.n_channels, chan_stride))
            axes_with_channel.sort(key=lambda t: t[2])

        # Reshape: slowest-first means reversed order.
        shape_slow_first = tuple(
            n for _, n, _ in reversed(axes_with_channel)
        )
        # frombuffer + reshape via the contiguous layout (assumes the
        # natural row-major layout matches the stride product). This
        # holds for every LIF I've seen.
        return flat.reshape(shape_slow_first)


# ---------------------------------------------------------------------------
# Reader / Codec wrapper
# ---------------------------------------------------------------------------


class LifNativeReader(Reader):
    """Native-Python LIF reader (no readlif dependency)."""

    def __init__(self, path: str | Path, image: int | str | None = None):
        self._parser = LifFileParser(path)
        if not self._parser.images:
            raise ValueError(f"LIF: no image elements found in {path}")
        self._image_idx = self._resolve_image(
            image if image is not None else 0
        )
        self._image = self._parser.images[self._image_idx]
        self.shape = tuple(self._compute_shape())
        self.dtype = self._image.dtype
        self.n_images = len(self._parser.images)
        self.n_frames = self.n_images
        self.is_chunked = False

    def _compute_shape(self) -> tuple[int, ...]:
        # Mirror the layout of LifFileParser.array_for_image.
        axes = list(self._image.axis_order)
        if self._image.n_channels > 1:
            chan_stride = (
                self._image.channel_bytes_inc // self._image.bytes_per_sample
            )
            axes.append(("C", self._image.n_channels, chan_stride))
            axes.sort(key=lambda t: t[2])
        return tuple(n for _, n, _ in reversed(axes))

    def _resolve_image(self, key: int | str) -> int:
        if isinstance(key, (int, np.integer)):
            return int(key)
        for i, im in enumerate(self._parser.images):
            if im.name == key:
                return i
        raise KeyError(f"LIF: no image named {key!r}")

    @property
    def image_names(self) -> list[str]:
        return [im.name for im in self._parser.images]

    def image(self, key: int | str) -> "LifNativeReader":
        self._image_idx = self._resolve_image(key)
        self._image = self._parser.images[self._image_idx]
        self.shape = self._compute_shape()
        self.dtype = self._image.dtype
        return self

    def iter_frames(self) -> Iterator[np.ndarray]:
        for i in range(self.n_images):
            self.image(i)
            yield self.read()

    def read(self) -> np.ndarray:
        return self._parser.array_for_image(self._image)

    def close(self) -> None:
        self._parser._data = b""

    def __enter__(self) -> "LifNativeReader":
        return self

    def __exit__(self, *_) -> bool:
        self.close()
        return False


__all__ = ["LifFileParser", "LifImage", "LifMemoryBlock", "LifNativeReader"]
