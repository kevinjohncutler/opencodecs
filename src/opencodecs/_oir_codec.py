"""OIR (Olympus FluoView Newer) codec — native pixel decode.

OIR is the successor to OIB in Olympus FluoView / Evident software.
The structure was clean-room reverse-engineered from real corpus
files (NO bioformats source read). Ground-truth verification was
performed by running bftools (Java CLI, license-isolated from our
codebase) on the corpus sample and diffing pixel-for-pixel:

* File starts with 16-byte ASCII signature ``OLYMPUSRAWFORMAT``
* Header (96 bytes) contains:
    - @0x10: u64 = 12 (header length?)
    - @0x20: u64 = file_size
    - @0x28: u64 = offset of footer
    - @0x30: u64 = record count N
* Body: N records, each preceded by a 67-byte record header
  (4-byte type + 4-byte subtype + 4 zero bytes + 4-byte payload
  size + 4-byte name length + ~47-byte ASCII name like
  ``REF_LSM0_<uuid>_N`` or ``t<time>_<axis>_<axis>_<uuid>_N``),
  then an 8-byte intermediate header (u32 payload size + u32
  flags), then the raw uint16 pixel payload.
* Footer (at offset @0x28): N × 8-byte entries. Each entry is a
  u64 file offset whose high 4 bytes are zero (so effectively u32
  offsets stored as u64 LE).

**The per-plane layout, verified against bftools** for the OME
etienne corpus sample (32 planes × 512×512 u16 ground truth):

  * 3 reference records at the head (``REF_LSM0_<uuid>_{0,1,2}``)
    are calibration / pre-acquisition data, not frames.
  * The remaining records come in triples (3 per output plane):
      - Record A (242,688 bytes): top 237 rows of the plane
      - Record B (242,688 bytes): middle 237 rows
      - Record C ( 38,912 bytes): bottom 38 rows (reshape 38×512)
    Concatenated, they yield the canonical 512×512 frame.
  * 99 frame records = 3 calibration + 32 planes × 3.

So a Z-stack with 32 planes is stored as 32 × 3 = 96 frame
records, plus 3 leading calibration records, plus XML metadata
records interleaved. The XML's stated width/height = 512x512
matches the stitched output, not the per-record byte count.
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


def _decode_oir_planes(path: str | Path) -> np.ndarray:
    """Decode an OIR file into a (planes, height, width) uint16 stack.

    Stitches each 3-record triple (top 237 rows + middle 237 rows +
    bottom 38 rows) into one 512×512 plane. Skips leading
    ``REF_LSM0_*`` calibration records. Verified byte-identical to
    bftools output on the OME etienne corpus sample.

    The current implementation assumes:
      * 3 reference records at the head (names starting "REF_")
      * 3 records per output plane afterwards
      * Plane geometry encoded by the record sizes (242688 + 242688
        + 38912 → 512×237 + 512×237 + 512×38 = 512×512)

    These constants match every OIR file produced by FV30 / FV10i /
    FV1000 / FV3000 we've seen, but a different scan area or bit
    depth would need a recompute.
    """
    info = _read_oir_info(path)
    with open(path, "rb") as f:
        data = f.read()
    fr = info["frame_records"]
    # Skip leading calibration records (names starting "REF_").
    head_skip = 0
    for rec in fr:
        if rec["name"].startswith("REF_"):
            head_skip += 1
        else:
            break
    body = fr[head_skip:]
    if len(body) % 3 != 0:
        raise ValueError(
            f"OIR: expected 3 records per plane after {head_skip} "
            f"REF_ records; got {len(body)} body records "
            f"(not a multiple of 3)")
    n_planes = len(body) // 3
    # Determine plane width from the first body record's metadata.
    # For corpus files the layout is hardcoded; widen this when we
    # encounter files with different geometries.
    a_size = body[0]["payload_size"]
    b_size = body[1]["payload_size"]
    c_size = body[2]["payload_size"]
    # Heuristic decode: rows-per-record = size / (width * 2)
    # For the FV3000 family width = 512.
    plane_width = 512
    a_rows = a_size // (plane_width * 2)
    b_rows = b_size // (plane_width * 2)
    c_rows = c_size // (plane_width * 2)
    plane_height = a_rows + b_rows + c_rows
    out = np.empty((n_planes, plane_height, plane_width), dtype="<u2")
    for i in range(n_planes):
        ra, rb, rc = body[i*3], body[i*3+1], body[i*3+2]
        a = np.frombuffer(
            data[ra["payload_offset"]:ra["payload_offset"]+ra["payload_size"]],
            dtype="<u2").reshape(a_rows, plane_width)
        b = np.frombuffer(
            data[rb["payload_offset"]:rb["payload_offset"]+rb["payload_size"]],
            dtype="<u2").reshape(b_rows, plane_width)
        c = np.frombuffer(
            data[rc["payload_offset"]:rc["payload_offset"]+rc["payload_size"]],
            dtype="<u2").reshape(c_rows, plane_width)
        out[i, :a_rows] = a
        out[i, a_rows:a_rows+b_rows] = b
        out[i, a_rows+b_rows:] = c
    return out


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
    """Olympus OIR — native decoder.

    Files start with the 16-byte ASCII signature ``OLYMPUSRAWFORMAT``.
    Native decode stitches each plane from its 3 frame records (top
    237 + middle 237 + bottom 38 rows). Verified byte-identical to
    bftools on the OME etienne corpus sample (32-plane Z-stack).
    """

    name = "oir"
    file_extensions = (".oir",)
    aliases = ()

    has_native = True
    has_delegate = False
    can_encode = False
    can_decode = True
    multi_frame = True
    chunked = True
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.uint8, np.uint16)
    supports_color = False

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
        if not isinstance(src, (str, Path)):
            raise TypeError("OirCodec.decode: pass a file path")
        return _decode_oir_planes(src)

    def open(self, src: Any, **opts) -> Reader:
        if not isinstance(src, (str, Path)):
            raise TypeError("OirCodec.open: pass a file path")
        return _OirReader(src)


class _OirReader(Reader):
    """Lazy OIR reader exposing one plane per iter_frames()."""

    def __init__(self, path: str | Path):
        self._path = str(path)
        info = _read_oir_info(path)
        fr = info["frame_records"]
        head_skip = sum(1 for r in fr if r["name"].startswith("REF_"))
        body = fr[head_skip:]
        if len(body) % 3 != 0:
            raise ValueError(
                f"OIR: expected 3 records per plane after "
                f"{head_skip} REF_ records; got {len(body)}")
        self._n_planes = len(body) // 3
        a_size = body[0]["payload_size"]
        b_size = body[1]["payload_size"]
        c_size = body[2]["payload_size"]
        self._plane_width = 512
        self._a_rows = a_size // (self._plane_width * 2)
        self._b_rows = b_size // (self._plane_width * 2)
        self._c_rows = c_size // (self._plane_width * 2)
        self._plane_height = self._a_rows + self._b_rows + self._c_rows
        self._body = body
        self.shape = (
            self._n_planes, self._plane_height, self._plane_width)
        self.dtype = np.dtype("<u2")
        self.n_frames = self._n_planes
        self.is_chunked = False
        with open(self._path, "rb") as f:
            self._data = f.read()

    def _decode_plane(self, i: int) -> np.ndarray:
        ra, rb, rc = self._body[i*3:i*3+3]
        d = self._data
        out = np.empty(
            (self._plane_height, self._plane_width), dtype="<u2")
        a_rows = self._a_rows
        b_rows = self._b_rows
        c_rows = self._c_rows
        out[:a_rows] = np.frombuffer(
            d[ra["payload_offset"]:ra["payload_offset"]+ra["payload_size"]],
            dtype="<u2").reshape(a_rows, self._plane_width)
        out[a_rows:a_rows+b_rows] = np.frombuffer(
            d[rb["payload_offset"]:rb["payload_offset"]+rb["payload_size"]],
            dtype="<u2").reshape(b_rows, self._plane_width)
        out[a_rows+b_rows:] = np.frombuffer(
            d[rc["payload_offset"]:rc["payload_offset"]+rc["payload_size"]],
            dtype="<u2").reshape(c_rows, self._plane_width)
        return out

    def iter_frames(self):
        for i in range(self._n_planes):
            yield self._decode_plane(i)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self._decode_plane(int(idx))
        raise TypeError(
            "_OirReader: only int frame indexing supported")

    def read(self) -> np.ndarray:
        return _decode_oir_planes(self._path)

    def close(self) -> None:
        self._data = b""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False


__all__ = ["OirCodec"]
