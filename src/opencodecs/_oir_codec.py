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
    decoding pixels.

    Records that come in 67-byte-apart pairs in the footer are
    "frame records": the first offset is the record header (67 bytes
    of name + size), the second is the payload. Each payload starts
    with an 8-byte intermediate header (data_size u32 + flags u32)
    then the raw uint16 pixel buffer.

    For the OME etienne corpus sample (amy_slice_z_stack.oir):
      - 99 frame-record pairs in the footer
      - Most have data_size=0x3b400 (242,688 bytes = 512×237×u16)
      - Three smaller-size records (38,912 bytes) at every 3rd
        position — possibly thumbnails or supplementary data
      - Frame names alternate "REF_LSM0_*_N" (3 reference records)
        and "t001_*_*_*_N" (per-timepoint, per-channel, per-z)

    The per-record byte count (512×237) doesn't match the XML's
    advertised geometry (512×512). Until cross-checked against a
    ground-truth decoder we don't ship full-decode — see the
    ``frame_records`` list returned here for raw record metadata.
    """
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
    record_offsets: list[int] = []
    for i in range(n_records):
        e = footer_off + i * 8
        if e + 8 > file_size:
            break
        record_offsets.append(struct.unpack_from("<I", data, e + 4)[0])
    # Detect 67-byte-apart pairs = frame records, read intermediate
    # header (8 bytes: payload_size + flags) to get accurate byte
    # count per frame.
    frame_records: list[dict] = []
    i = 0
    while i < len(record_offsets) - 1:
        a, b = record_offsets[i], record_offsets[i+1]
        if b - a == 67 and a + 67 + 8 <= file_size:
            payload_size = struct.unpack_from("<I", data, b)[0]
            # Record name: 0x3b-byte name follows the 12-byte header
            # prefix (4 bytes record_type + 4 bytes subtype + 4 bytes
            # zero + 4 bytes data_size + 4 bytes name_len).
            name_len = struct.unpack_from("<I", data, a + 16)[0]
            name = data[a + 20:a + 20 + name_len].decode(
                "latin-1", errors="replace")
            frame_records.append({
                "record_offset": a,
                "payload_offset": b + 8,
                "payload_size": payload_size,
                "name": name,
            })
            i += 2
        else:
            i += 1
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
        "frame_records": frame_records,
    }


def _read_oir_raw_records(path: str | Path) -> list[np.ndarray]:
    """**Experimental.** Returns each OIR frame record as a raw
    uint16 ndarray. The reshape is best-effort based on the
    observed payload size — for the corpus sample's 242,688-byte
    records, we reshape as 512 × 237 (matches the byte count).
    Doesn't match the XML's stated 512 × 512 geometry; further
    interpretation requires a ground-truth reference."""
    info = _read_oir_info(path)
    with open(path, "rb") as f:
        data = f.read()
    out: list[np.ndarray] = []
    for rec in info["frame_records"]:
        off = rec["payload_offset"]
        size = rec["payload_size"]
        flat = np.frombuffer(data[off:off + size], dtype="<u2").copy()
        # Best-effort reshape: prefer N×237 when count = N×237
        n_pix = len(flat)
        h = 0
        for cand_h in (237, 474, 256, 473, 512, 1024):
            if n_pix % cand_h == 0:
                h = cand_h
                break
        if h == 0:
            out.append(flat)
        else:
            out.append(flat.reshape(h, n_pix // h))
    return out


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
        n_records, record_offsets, frame_metadata, frame_records}``.
        Useful for inspecting an OIR without full pixel decoding."""
        if not isinstance(src, (str, Path)):
            raise TypeError("OirCodec.info: pass a file path")
        return _read_oir_info(src)

    def raw_records(self, src: Any) -> list[np.ndarray]:
        """**Experimental** — returns each frame record as a raw
        uint16 ndarray, reshaped using a best-effort guess at the
        per-record dimensions (the per-record byte count doesn't
        match the XML's stated geometry, so the reshape is a
        heuristic). Useful for inspecting WHAT the records contain
        before we have a ground-truth decoder. Don't rely on these
        arrays being correctly oriented — that's the open question
        this method exists to help answer."""
        if not isinstance(src, (str, Path)):
            raise TypeError("OirCodec.raw_records: pass a file path")
        return _read_oir_raw_records(src)

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
