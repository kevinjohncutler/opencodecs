"""Native ND2 (Nikon NIS-Elements) parser — no `nd2` package dependency.

ND2 v3 is a chunk-based container similar to RIFF:

    [chunk-0]
    [chunk-1]
    ...
    [chunk-N]   <- the LAST chunk is the FILEMAP: a TOC listing every
                   other chunk's (name, offset, length).

Each chunk has a 16-byte header:

    bytes 0-3:  magic  0xDA 0xCE 0xBE 0x0A
    bytes 4-7:  name-field length (u32; name + null padding, may be
                large enough to align the data to a page boundary)
    bytes 8-15: data length (u64)

followed by ``name_field_length`` bytes of (name + null padding) and
``data_length`` bytes of data. Chunks aren't tightly packed — the
encoder pads the name field so each chunk's data starts at a 4096-byte
boundary. The FILEMAP at the end lists every chunk so a reader doesn't
need to scan linearly.

Three chunk classes matter:

  * ``ImageAttributes!`` (~XML, UTF-8): width/height/dtype/frame-count
  * ``ImageDataSeq|N!``: raw / compressed pixel bytes for frame N
  * ``ImageMetadataSeq|0!`` (~XML): per-axis labels and per-frame
    coordinates (T, Z, P, ...)

For HTTP / remote ND2s, the read pattern is:

  1. Read the last ~16 KB of the file to find the FILEMAP chunk
  2. Parse FILEMAP → get every chunk's offset
  3. Range-request ImageAttributes! + ImageDataSeq|N! for the frames
     the caller wants

The reader uses :class:`~opencodecs.core.io.DataSource` so the same
code path drives local files, mmap, and ``HTTPDataSource``.

**STATUS: experimental, not yet the default registered ND2 codec.**
The production path is still ``Nd2Codec`` via the ``nd2`` package
(see ``_nd2_codec.py``). This module is the second proof-of-concept
for the native-vendor-reader pattern (LIF was first).
"""

from __future__ import annotations

import os
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .core.codec import Reader
from .core.io import DataSource


_MAGIC = b"\xDA\xCE\xBE\x0A"
_MAGIC_LEGACY = b"\xDA\xBC\xD8\x3E"   # pre-2008 ND2; not supported here

_CHUNKMAP_SIG = b"ND2 CHUNK MAP SIGNATURE 0000001!"   # 32 bytes, in trailer
_FILEMAP_SIG  = b"ND2 FILEMAP SIGNATURE NAME 0001!"   # 32 bytes, sentinel


@dataclass
class Nd2Chunk:
    """Pointer to one chunk inside an ND2 file. ``offset`` is the
    chunk's HEADER start; ``data_offset`` is where the payload begins
    (after magic + name_field)."""
    name: str
    offset: int
    data_offset: int
    data_length: int


@dataclass
class Nd2Attributes:
    """Image geometry extracted from the ImageAttributes! XML chunk."""
    width: int
    height: int
    n_channels: int
    bits_in_memory: int
    bits_significant: int
    sequence_count: int      # = total frames (T * Z * P, depending)
    width_bytes: int         # row stride (= width * bits_in_memory/8 * channels)
    compression: int         # eCompression: 0/2 raw, 1 JPEG, 3 JPEG2k

    @property
    def dtype(self) -> np.dtype:
        if self.bits_in_memory <= 8:
            return np.dtype("u1")
        if self.bits_in_memory <= 16:
            return np.dtype("<u2")
        if self.bits_in_memory == 32:
            return np.dtype("<f4")
        raise ValueError(
            f"ND2: unsupported bits_in_memory={self.bits_in_memory}")

    @property
    def frame_size_bytes(self) -> int:
        return self.height * self.width_bytes


class Nd2FileParser:
    """Parses an ND2 v3 container backed by a DataSource.

    The constructor only reads the FILEMAP + ImageAttributes — about
    a dozen KB. Individual frames are loaded on demand via
    :meth:`read_frame`. For HTTP-backed sources this means opening an
    ND2 costs O(1) requests regardless of frame count.
    """

    def __init__(self, src: Any):
        if isinstance(src, DataSource):
            self._src = src
            self._owns_src = False
        elif isinstance(src, (str, os.PathLike)):
            # Lazy import — avoids pulling _tiff_http into modules that
            # don't need HTTP at all.
            from ._tiff_http import FileDataSource
            self._src = FileDataSource(str(src))
            self._owns_src = True
        else:
            raise TypeError(
                f"ND2: unsupported source {type(src).__name__}; pass "
                f"a path or a DataSource")
        # File size (we need it to locate the FILEMAP near EOF).
        # FileDataSource has a .size attribute; HTTPDataSource has a
        # .total_size property (lazily populated on first range read).
        size = getattr(self._src, "size", None)
        if size is None:
            size = getattr(self._src, "total_size", None)
        if size is None:
            # Force a tiny read; HTTPDataSource's first range request
            # populates total_size via the Content-Range response.
            self._src.read_at(0, 4)
            size = getattr(self._src, "total_size", None)
        if size is None:
            raise RuntimeError(
                "ND2: DataSource didn't expose its total size; can't "
                "locate the FILEMAP at EOF")
        self._size = int(size)

        # Verify the file starts with ND2 magic.
        head = self._src.read_at(0, 4)
        if head == _MAGIC_LEGACY:
            raise NotImplementedError(
                "ND2: pre-2008 legacy format (magic 0xDABCD83E) is not "
                "supported by the native parser; use the `nd2` package "
                "via Nd2Codec instead")
        if head != _MAGIC:
            raise ValueError(
                f"ND2: not a v3 ND2 (magic={head.hex()}, expected "
                f"{_MAGIC.hex()})")

        self.chunks: dict[str, Nd2Chunk] = self._read_filemap()
        self.attributes: Nd2Attributes = self._parse_image_attributes()

    # ----- FILEMAP -----

    def _read_filemap(self) -> dict[str, Nd2Chunk]:
        """Locate the FILEMAP via the trailing (signature, location)
        record, then parse its entries.

        The last 40 bytes of every ND2 v3 file are:
            32 bytes: signature ``ND2 FILEMAP SIGNATURE NAME 0001!``
            8 bytes:  u64 location of the FILEMAP chunk's header
        """
        trailer = self._src.read_at(self._size - 40, 40)
        # NIS-Elements wrote two different EOF signatures historically:
        #   * "ND2 CHUNK MAP SIGNATURE 0000001!" (modern)
        #   * "ND2 FILEMAP SIGNATURE NAME 0001!" (older)
        # Both point to a chunk whose internal sentinel name is the
        # FILEMAP one; the trailer signature can be either.
        if trailer[:32] not in (_CHUNKMAP_SIG, _FILEMAP_SIG):
            raise ValueError(
                f"ND2: missing chunkmap/filemap signature at EOF "
                f"(got {trailer[:32]!r}); not a valid v3 ND2")
        location = struct.unpack_from("<Q", trailer, 32)[0]
        return self._parse_filemap_chunk(location)

    def _parse_filemap_chunk(self, offset: int) -> dict[str, Nd2Chunk]:
        header = self._src.read_at(offset, 16)
        if header[:4] != _MAGIC:
            raise ValueError(
                f"ND2: FILEMAP magic mismatch at offset {offset}: "
                f"{header[:4].hex()}")
        name_len = struct.unpack_from("<I", header, 4)[0]
        data_len = struct.unpack_from("<Q", header, 8)[0]
        body = self._src.read_at(offset + 16 + name_len, data_len)
        # FILEMAP body: repeating <chunk_name>!<u64 offset><u64 size>.
        # Chunk names end with '!' (it's part of the name); the parser
        # finds each name by scanning for '!'. The last entry is the
        # FILEMAP signature itself, repeated as a sentinel — break on it.
        result: dict[str, Nd2Chunk] = {}
        pos = 0
        while pos < len(body):
            bang = body.find(b"!", pos)
            if bang < 0:
                break
            name = body[pos:bang + 1].decode("latin-1", errors="replace")
            after = bang + 1
            # The FILEMAP body ends with the CHUNKMAP signature
            # repeated as a sentinel (32 bytes ending in '!').
            if name == _CHUNKMAP_SIG.decode("latin-1"):
                break
            if after + 16 > len(body):
                break
            entry_off = struct.unpack_from("<Q", body, after)[0]
            entry_size = struct.unpack_from("<Q", body, after + 8)[0]
            # FILEMAP records the chunk's HEADER offset (= magic byte).
            # data_offset (where the payload starts) is resolved
            # on first read by reading the per-chunk name_field length.
            result[name] = Nd2Chunk(
                name=name,
                offset=entry_off,
                data_offset=-1,
                data_length=entry_size,
            )
            pos = after + 16
        return result

    def _resolve_chunk_data_offset(self, chunk: Nd2Chunk) -> Nd2Chunk:
        """Lazily resolve a FILEMAP entry's name_field_length so we
        know where the payload starts (= header_offset + 16 + name_len)."""
        if chunk.data_offset >= 0:
            return chunk
        header = self._src.read_at(chunk.offset, 16)
        if header[:4] != _MAGIC:
            raise ValueError(
                f"ND2: chunk {chunk.name!r} magic mismatch at offset "
                f"{chunk.offset}: {header[:4].hex()}")
        name_len = struct.unpack_from("<I", header, 4)[0]
        data_len = struct.unpack_from("<Q", header, 8)[0]
        # FILEMAP's recorded data_length is the WHOLE chunk size
        # (header + name + data); recompute from the actual header.
        chunk.data_offset = chunk.offset + 16 + name_len
        chunk.data_length = data_len
        return chunk

    # ----- ImageAttributes -----

    def _parse_image_attributes(self) -> Nd2Attributes:
        chunk = self.chunks.get("ImageAttributes!")
        if chunk is None:
            raise ValueError("ND2: missing ImageAttributes! chunk")
        chunk = self._resolve_chunk_data_offset(chunk)
        raw = self._src.read_at(chunk.data_offset, chunk.data_length)
        xml_text = raw.decode("utf-8")
        return _parse_attributes_xml(xml_text)

    # ----- public read API -----

    @property
    def n_frames(self) -> int:
        return self.attributes.sequence_count

    def read_frame(self, index: int) -> np.ndarray:
        """Decode frame ``index`` (range 0..n_frames-1) into an
        ndarray of shape (H, W, C) or (H, W) (channels squeezed)."""
        if not 0 <= index < self.n_frames:
            raise IndexError(
                f"ND2: frame {index} out of range [0, {self.n_frames})")
        chunk_name = f"ImageDataSeq|{index}!"
        chunk = self.chunks.get(chunk_name)
        if chunk is None:
            raise KeyError(
                f"ND2: missing {chunk_name!r}; corrupt FILEMAP?")
        chunk = self._resolve_chunk_data_offset(chunk)
        # Each ImageDataSeq chunk is: [8 bytes frame metadata] + pixels.
        # The 8-byte prefix is a uint32 + uint32 (timestamps; not
        # required for pixel decode).
        attrs = self.attributes
        if attrs.compression in (0, 2):
            # Raw / "compression=2 but actually raw" — both observed
            # in real files. Read frame_size_bytes after the 8-byte
            # metadata prefix.
            payload = self._src.read_at(
                chunk.data_offset + 8, attrs.frame_size_bytes)
            arr = np.frombuffer(payload, dtype=attrs.dtype).copy()
        else:
            raise NotImplementedError(
                f"ND2: compression {attrs.compression} not implemented "
                f"in the native parser yet; use Nd2Codec (delegate) "
                f"for this file")
        if attrs.n_channels > 1:
            arr = arr.reshape(attrs.height, attrs.width, attrs.n_channels)
        else:
            arr = arr.reshape(attrs.height, attrs.width)
        return arr

    def close(self) -> None:
        if self._owns_src and hasattr(self._src, "close"):
            self._src.close()


def _parse_attributes_xml(xml_text: str) -> Nd2Attributes:
    root = ET.fromstring(xml_text)
    fields: dict[str, str] = {}
    # The structure is <variant><no_name><uiWidth value="800"/>...
    for elem in root.iter():
        v = elem.attrib.get("value")
        if v is not None:
            fields[elem.tag] = v
    def _int(key: str, default: int = 0) -> int:
        return int(fields.get(key, default))
    return Nd2Attributes(
        width=_int("uiWidth"),
        height=_int("uiHeight"),
        n_channels=_int("uiComp", 1),
        bits_in_memory=_int("uiBpcInMemory", 16),
        bits_significant=_int("uiBpcSignificant", 16),
        sequence_count=_int("uiSequenceCount", 1),
        width_bytes=_int("uiWidthBytes"),
        compression=_int("eCompression", 0),
    )


# ---------------------------------------------------------------------------
# Reader interface
# ---------------------------------------------------------------------------


class Nd2NativeReader(Reader):
    """Native-Python ND2 Reader. Streaming-friendly: opening an ND2
    only reads the FILEMAP + ImageAttributes (~16 KB); per-frame data
    is loaded on demand."""

    def __init__(self, src: Any):
        self._parser = Nd2FileParser(src)
        attrs = self._parser.attributes
        self.shape = self._compute_shape(attrs)
        self.dtype = attrs.dtype
        self.n_frames = self._parser.n_frames
        self.is_chunked = True

    @staticmethod
    def _compute_shape(attrs: Nd2Attributes) -> tuple[int, ...]:
        if attrs.sequence_count > 1:
            base = (attrs.sequence_count, attrs.height, attrs.width)
        else:
            base = (attrs.height, attrs.width)
        if attrs.n_channels > 1:
            return base + (attrs.n_channels,)
        return base

    def iter_frames(self) -> Iterator[np.ndarray]:
        for i in range(self.n_frames):
            yield self._parser.read_frame(i)

    def __getitem__(self, idx) -> np.ndarray:
        if isinstance(idx, (int, np.integer)):
            return self._parser.read_frame(int(idx))
        raise TypeError(
            f"Nd2NativeReader: only int frame indexing is supported "
            f"(got {type(idx).__name__}); use iter_frames() or read() "
            f"for the full stack")

    def read(self) -> np.ndarray:
        if self.n_frames == 1:
            return self._parser.read_frame(0)
        frames = [self._parser.read_frame(i) for i in range(self.n_frames)]
        return np.stack(frames)

    def close(self) -> None:
        self._parser.close()

    def __enter__(self) -> "Nd2NativeReader":
        return self

    def __exit__(self, *_) -> bool:
        self.close()
        return False


__all__ = [
    "Nd2FileParser",
    "Nd2NativeReader",
    "Nd2Attributes",
    "Nd2Chunk",
]
