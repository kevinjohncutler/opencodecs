"""OIR (Olympus FluoView Newer) codec.

OIR is the successor to OIB in Olympus FluoView / Evident software.
We have a clean-room partial understanding of the format from
inspecting real files (no proprietary docs / no bioformats code
read for this implementation):

* File starts with 16-byte ASCII signature ``OLYMPUSRAWFORMAT``
* Header (96 bytes) contains:
    - @0x10: u64 = 12 (header length?)
    - @0x20: u64 = file_size
    - @0x28: u64 = offset of footer
    - @0x30: u64 = record count N
* Body: N records. Each record is one of:
    - **Frame record** (paired in the footer 67 bytes apart):
      record header + small intermediate header + uint16 pixel data
    - **XML metadata record** (``<?xml ...<lsmframe:frameProperties``):
      describes per-frame width/height/bitCounts/channel info
* Footer (at offset @0x28): N × 8-byte entries. Each entry's
  low 4 bytes are zero, high 4 bytes are a u32 file offset. The
  list interleaves frame-data-start, frame-payload-start, and
  metadata-record-start offsets.

The structure is clear enough to enumerate records + extract
metadata; per-record pixel-data dimensions are tricky — the
242 KB per "frame record" doesn't trivially correspond to a full
512×512×u16 frame, suggesting per-channel splitting or chunked
encoding that we'd need a frame-of-known-content to verify.

Until full pixel decoding is verified, this codec exposes:
  * signature() — works
  * info(path) — returns {n_records, footer entries, XML metadata}
  * decode() / open() — raise NotImplementedError with a clear
    pointer at the limitation. Bioformats is the only complete
    public OIR reader.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec, Reader


_OIR_MAGIC = b"OLYMPUSRAWFORMAT"


def _read_oir_info(path: str | Path) -> dict:
    """Clean-room partial parse of an OIR file's header + footer +
    first XML metadata record. Returns what we know without
    decoding pixels."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:16] != _OIR_MAGIC:
        raise ValueError(
            f"OIR: bad magic at offset 0: {data[:16]!r}")
    file_size = len(data)
    rec_file_size = struct.unpack_from("<Q", data, 0x20)[0]
    footer_off = struct.unpack_from("<Q", data, 0x28)[0]
    n_records = struct.unpack_from("<Q", data, 0x30)[0]
    if rec_file_size != file_size:
        raise ValueError(
            f"OIR: header file-size mismatch: header says "
            f"{rec_file_size} but file is {file_size}")
    # Read footer u64 offsets: low-u32 is zero, high-u32 is offset.
    record_offsets: list[int] = []
    for i in range(n_records):
        e = footer_off + i * 8
        if e + 8 > file_size:
            break
        record_offsets.append(struct.unpack_from("<I", data, e + 4)[0])
    # First XML metadata record describes the geometry. Search for
    # the canonical opening tag.
    xml_meta: dict[str, str] = {}
    xml_start = data.find(b"<?xml")
    if xml_start >= 0:
        xml_end = data.find(b"</lsmframe:frameProperties>", xml_start)
        if xml_end >= 0:
            xml_end += len(b"</lsmframe:frameProperties>")
            xml = data[xml_start:xml_end].decode("utf-8", errors="replace")
            for tag in ("width", "height", "depth", "bitCounts",
                        "colorType"):
                m = re.search(rf"<base:{tag}>([^<]+)</base:{tag}>", xml)
                if m:
                    xml_meta[tag] = m.group(1).strip()
    return {
        "file_size": file_size,
        "footer_offset": footer_off,
        "n_records": n_records,
        "record_offsets": record_offsets,
        "frame_metadata": xml_meta,
    }


class OirCodec(Codec):
    """Olympus OIR — partial native parse (header + metadata).

    Files start with the 16-byte ASCII signature ``OLYMPUSRAWFORMAT``.
    Native parsing extracts file structure (footer record table) +
    per-frame XML metadata (width / height / depth / bitCounts). Full
    pixel decoding awaits a frame-of-known-content verification pass —
    until then, ``decode()`` and ``open()`` raise NotImplementedError.
    ``oc.get_codec("oir").info(path)`` returns the partial-parse dict.
    """

    name = "oir"
    file_extensions = (".oir",)
    aliases = ()

    has_native = False
    has_delegate = False
    can_encode = False
    can_decode = False
    multi_frame = True
    chunked = True
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.uint8, np.uint16)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        return len(head) >= 16 and head[:16] == _OIR_MAGIC

    def info(self, src: Any) -> dict:
        """Partial-parse: returns ``{file_size, footer_offset,
        n_records, record_offsets, frame_metadata}``. Useful for
        inspecting an OIR without full pixel decoding."""
        if not isinstance(src, (str, Path)):
            raise TypeError("OirCodec.info: pass a file path")
        return _read_oir_info(src)

    def decode(self, src: Any, **opts) -> np.ndarray:
        raise NotImplementedError(
            "OIR: Olympus FluoView Newer format. The format is "
            "OLYMPUSRAWFORMAT-prefixed but the binary container is "
            "undocumented. No native parser yet — use bioformats "
            "(via python-bioformats / scyjava) for the time being. "
            "Tracking issue: opencodecs#future-oir-native.")

    def open(self, src: Any, **opts) -> Reader:
        raise NotImplementedError(
            "OIR: Olympus FluoView Newer — see OirCodec.decode().")


__all__ = ["OirCodec"]
