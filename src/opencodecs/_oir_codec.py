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

import os
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec, Reader
from .core.io import DataSource


_OIR_MAGIC = b"OLYMPUSRAWFORMAT"


def _src_from(src: Any) -> tuple[DataSource, bool, int]:
    """Coerce a path or DataSource to (DataSource, owns_src, size)."""
    if isinstance(src, DataSource):
        size = getattr(src, "size", None)
        if size is None:
            size = getattr(src, "total_size", None)
        if size is None:
            src.read_at(0, 4)
            size = getattr(src, "total_size", None)
        if size is None:
            raise RuntimeError(
                "OIR: DataSource didn't expose total_size")
        return src, False, int(size)
    if isinstance(src, (str, os.PathLike)):
        from ._tiff_http import FileDataSource
        ds = FileDataSource(str(src))
        return ds, True, int(ds.size)
    raise TypeError(f"OIR: unsupported source {type(src).__name__}")


def _read_oir_info(src: Any) -> dict:
    """Clean-room partial parse of an OIR file's header + footer.

    Accepts a path or any :class:`~opencodecs.core.io.DataSource`.
    HTTP-backed sources only fetch the header (64 bytes) + footer
    (n_records × 8) + per-frame-record headers (67+8 bytes each) —
    typically a few KB for a multi-megabyte OIR.

    Records that come in 67-byte-apart pairs in the footer are
    "frame records": the first offset is the record header (67 bytes
    of name + size), the second is the payload. Each payload starts
    with an 8-byte intermediate header (data_size u32 + flags u32)
    then the raw uint16 pixel buffer.

    For the OME etienne corpus sample (amy_slice_z_stack.oir):
      - 99 frame records in the footer
      - Each Z-plane = 3 records: top 237 rows + middle 237 + bottom 38
      - First 3 records are calibration ("REF_LSM0_*_{0,1,2}")
      - The remaining 96 records = 32 Z-planes × 3 records
    """
    ds, owns, file_size = _src_from(src)
    try:
        hdr = ds.read_at(0, 64)
        if hdr[:16] != _OIR_MAGIC:
            raise ValueError(
                f"OIR: bad magic at offset 0: {hdr[:16]!r}")
        rec_file_size = struct.unpack_from("<Q", hdr, 0x20)[0]
        footer_off = struct.unpack_from("<Q", hdr, 0x28)[0]
        n_records = struct.unpack_from("<Q", hdr, 0x30)[0]
        if rec_file_size != file_size:
            raise ValueError(
                f"OIR: header file-size mismatch: header says "
                f"{rec_file_size} but DataSource is {file_size}")
        # Footer: n_records × 8 bytes
        footer_bytes = ds.read_at(footer_off, n_records * 8)
        record_offsets: list[int] = []
        for i in range(n_records):
            e = i * 8
            if e + 8 > len(footer_bytes):
                break
            record_offsets.append(
                struct.unpack_from("<I", footer_bytes, e + 4)[0])
        # For each 67-byte-apart pair, read the 67-byte record header
        # + 8-byte intermediate header (75 bytes total per record).
        frame_records: list[dict] = []
        i = 0
        while i < len(record_offsets) - 1:
            a, b = record_offsets[i], record_offsets[i + 1]
            if b - a == 67 and b + 8 <= file_size:
                rec_hdr = ds.read_at(a, 67 + 8)
                # name_len at offset 16 within the record header
                name_len = struct.unpack_from("<I", rec_hdr, 16)[0]
                name = rec_hdr[20:20 + name_len].decode(
                    "latin-1", errors="replace")
                payload_size = struct.unpack_from(
                    "<I", rec_hdr, 67)[0]
                frame_records.append({
                    "record_offset": a,
                    "payload_offset": b + 8,
                    "payload_size": payload_size,
                    "name": name,
                })
                i += 2
            else:
                i += 1
        # Find XML metadata records (offsets that aren't part of any
        # 67-byte pair). For info() we just need the first XML chunk
        # to learn geometry.
        xml_meta: dict[str, str] = {}
        # Walk the non-frame record offsets and probe each for XML.
        non_frame = set(record_offsets)
        for r in frame_records:
            non_frame.discard(r["record_offset"])
            non_frame.discard(r["payload_offset"] - 8)
        for cand in sorted(non_frame):
            head = ds.read_at(cand, 200)
            if b"<?xml" in head:
                xs = head.index(b"<?xml")
                # Read up to ~8 KB starting from the candidate, scan
                # for the closing tag.
                blob = ds.read_at(cand + xs, 8192)
                end = blob.find(b"</lsmframe:frameProperties>")
                if end > 0:
                    xml = blob[:end].decode(
                        "utf-8", errors="replace")
                    for tag in ("width", "height", "depth",
                                "bitCounts", "colorType"):
                        m = re.search(
                            rf"<base:{tag}>([^<]+)</base:{tag}>", xml)
                        if m:
                            xml_meta[tag] = m.group(1).strip()
                    break
        return {
            "file_size": file_size,
            "footer_offset": footer_off,
            "n_records": n_records,
            "record_offsets": record_offsets,
            "frame_metadata": xml_meta,
            "frame_records": frame_records,
        }
    finally:
        if owns:
            ds.close()


def _decode_oir_planes(src: Any) -> np.ndarray:
    """Decode an OIR source into a (planes, height, width) uint16 stack.

    Accepts a path or any :class:`~opencodecs.core.io.DataSource`.
    Each plane is stitched from 3 consecutive frame records (top
    237 rows + middle 237 rows + bottom 38 rows = 512×512 for the
    FV3000 family). Leading ``REF_LSM0_*`` calibration records
    are skipped.

    For an HTTP-backed DataSource: opens the file with ~4 KB of
    range reads (header + footer + per-frame headers) then fetches
    each plane's 3 payloads as 3 contiguous range reads. Total
    wire bytes = sum of payload sizes (= raw plane bytes) +
    open overhead — same shape as ND2/OIB's HTTP behavior.
    """
    ds, owns, _ = _src_from(src)
    try:
        info = _read_oir_info(ds)
        fr = info["frame_records"]
        # Skip leading calibration records.
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
        plane_width = 512
        a_size = body[0]["payload_size"]
        b_size = body[1]["payload_size"]
        c_size = body[2]["payload_size"]
        a_rows = a_size // (plane_width * 2)
        b_rows = b_size // (plane_width * 2)
        c_rows = c_size // (plane_width * 2)
        plane_height = a_rows + b_rows + c_rows
        out = np.empty(
            (n_planes, plane_height, plane_width), dtype="<u2")
        for i in range(n_planes):
            ra, rb, rc = body[i*3:i*3+3]
            # Three range reads per plane — coalescing isn't a win
            # because the three records aren't contiguous on disk.
            out[i, :a_rows] = np.frombuffer(
                ds.read_at(ra["payload_offset"], ra["payload_size"]),
                dtype="<u2").reshape(a_rows, plane_width)
            out[i, a_rows:a_rows+b_rows] = np.frombuffer(
                ds.read_at(rb["payload_offset"], rb["payload_size"]),
                dtype="<u2").reshape(b_rows, plane_width)
            out[i, a_rows+b_rows:] = np.frombuffer(
                ds.read_at(rc["payload_offset"], rc["payload_size"]),
                dtype="<u2").reshape(c_rows, plane_width)
        return out
    finally:
        if owns:
            ds.close()


def _read_oir_raw_records(src: Any) -> list[np.ndarray]:
    """**Experimental.** Returns each OIR frame record as a raw
    uint16 ndarray. The reshape is best-effort based on the
    observed payload size — for the corpus sample's 242,688-byte
    records, we reshape as 512 × 237 (matches the byte count).
    Doesn't match the XML's stated 512 × 512 geometry; further
    interpretation requires a ground-truth reference."""
    ds, owns, _ = _src_from(src)
    try:
        info = _read_oir_info(ds)
        out: list[np.ndarray] = []
        for rec in info["frame_records"]:
            payload = ds.read_at(rec["payload_offset"], rec["payload_size"])
            flat = np.frombuffer(payload, dtype="<u2").copy()
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
    finally:
        if owns:
            ds.close()


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
        Accepts a path or a DataSource (HTTP-backed sources only
        fetch the few KB needed for the parse)."""
        return _read_oir_info(src)

    def raw_records(self, src: Any) -> list[np.ndarray]:
        """Returns each frame record as a uint16 ndarray reshaped to
        the per-record geometry. For the FV3000 family three records
        per plane combine to a 512×512 plane (top 237 + middle 237
        + bottom 38). The full :meth:`decode` stitches them."""
        return _read_oir_raw_records(src)

    def decode(self, src: Any, **opts) -> np.ndarray:
        return _decode_oir_planes(src)

    def open(self, src: Any, **opts) -> Reader:
        return OirNativeReader(src)


class OirNativeReader(Reader):
    """Lazy OIR reader. Each :meth:`__getitem__` fetches only that
    plane's 3 record payloads via the underlying DataSource —
    HTTP-backed sources thus pay per-plane I/O cost (1/n_planes of
    the file plus open overhead) rather than reading everything
    upfront."""

    def __init__(self, src: Any):
        self._ds, self._owns_src, _ = _src_from(src)
        info = _read_oir_info(self._ds)
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

    def _decode_plane(self, i: int) -> np.ndarray:
        ra, rb, rc = self._body[i*3:i*3+3]
        out = np.empty(
            (self._plane_height, self._plane_width), dtype="<u2")
        a_rows = self._a_rows
        b_rows = self._b_rows
        c_rows = self._c_rows
        out[:a_rows] = np.frombuffer(
            self._ds.read_at(ra["payload_offset"], ra["payload_size"]),
            dtype="<u2").reshape(a_rows, self._plane_width)
        out[a_rows:a_rows+b_rows] = np.frombuffer(
            self._ds.read_at(rb["payload_offset"], rb["payload_size"]),
            dtype="<u2").reshape(b_rows, self._plane_width)
        out[a_rows+b_rows:] = np.frombuffer(
            self._ds.read_at(rc["payload_offset"], rc["payload_size"]),
            dtype="<u2").reshape(c_rows, self._plane_width)
        return out

    def iter_frames(self):
        for i in range(self._n_planes):
            yield self._decode_plane(i)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self._decode_plane(int(idx))
        raise TypeError(
            "OirNativeReader: only int frame indexing supported")

    def read(self) -> np.ndarray:
        return _decode_oir_planes(self._ds)

    def close(self) -> None:
        if self._owns_src:
            self._ds.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False


__all__ = ["OirCodec", "OirNativeReader"]
