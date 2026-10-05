"""HEIF and AVIF containers (ISOBMFF items), read and written in Python.

Shared by the hardware backends, which hand the coded pictures to a
hardware codec and so need the container's metadata without libheif or
libavif: which item is primary, its coded data (``iloc``), its
properties (``hvcC``/``av1C``, ``ispe``, ``colr``, transforms) and, for a
grid, how its tiles are laid out. Writing covers what a hardware
encoder's output needs: one coded picture as the primary item, with
``ispe``, ``pixi``, ``colr`` and its codec configuration.

Only metadata is parsed here, never pixels, so this costs microseconds.
"""

from __future__ import annotations

import struct

# Auxiliary-image types that mean "this is the alpha plane" (HEVC's and
# MPEG-B's spellings); a depth map or other auxiliary image does not
# change how the primary image decodes.
ALPHA_URNS = (b"urn:mpeg:hevc:2015:auxid:1",
              b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha")


# ------------------------------------------------------------------ read

def boxes(data, start, end):
    """(type, payload start, box end) for each ISOBMFF box in a range."""
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, pos)
        head = 8
        if size == 1:
            if pos + 16 > end:
                return
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            head = 16
        elif size == 0:
            size = end - pos
        if size < head or pos + size > end:
            return
        yield kind, pos + head, pos + size
        pos += size


def _uint(data, at, size):
    return (int.from_bytes(data[at:at + size], "big") if size else 0), at + size


class Items:
    """The item structure of a HEIF/AVIF ``meta`` box."""

    def __init__(self, data):
        self.data = data
        self.primary = None
        self.types = {}       # item id -> 4-byte type
        self.refs = []        # (type, from id, [to ids])
        self.props = []       # (type, payload start, end), 1-based in ipma
        self.assoc = {}       # item id -> [property index]
        self.iloc = {}        # item id -> (construction method, [(offset, length)])
        self.idat = None      # (start, end) of the idat payload

    def properties(self, item) -> dict:
        """{property type: (payload start, end)}, the first of each type."""
        out = {}
        for i in self.assoc.get(item, ()):
            if 1 <= i <= len(self.props):
                kind, s, e = self.props[i - 1]
                out.setdefault(kind, (s, e))
        return out

    def references(self, kind, item) -> list:
        """Item ids that ``item`` references with ``kind`` (e.g. dimg)."""
        for k, src, dst in self.refs:
            if k == kind and src == item:
                return list(dst)
        return []

    def item_data(self, item) -> bytes:
        """The item's bytes; ValueError for a layout not handled here."""
        if item not in self.iloc:
            raise ValueError(f"item {item} has no location")
        method, extents = self.iloc[item]
        data = self.data
        if method == 0:
            base, limit = 0, len(data)
        elif method == 1 and self.idat is not None:
            base, limit = self.idat
        else:
            raise ValueError(f"item {item} uses construction method {method}")
        parts = []
        for offset, length in extents:
            start = base + offset
            stop = limit if length == 0 else start + length
            if stop > limit or start > stop:
                raise ValueError(f"item {item} extends past its data")
            parts.append(bytes(data[start:stop]))
        return b"".join(parts)

    def size(self, item):
        """(width, height) from the item's ispe, or None."""
        ispe = self.properties(item).get(b"ispe")
        if ispe is None or ispe[1] - ispe[0] < 12:
            return None
        return struct.unpack_from(">II", self.data, ispe[0] + 4)

    def nclx(self, item):
        """(primaries, transfer, matrix, full_range) from a colr nclx box
        of the item, or None."""
        for i in self.assoc.get(item, ()):
            if not 1 <= i <= len(self.props):
                continue
            kind, s, e = self.props[i - 1]
            if kind == b"colr" and e - s >= 11 and self.data[s:s + 4] == b"nclx":
                cp, tc, mc = struct.unpack_from(">HHH", self.data, s + 4)
                return cp, tc, mc, bool(self.data[s + 10] & 0x80)
        return None


def parse(data):
    """:class:`Items` for ``data``, or None if it has no readable meta box."""
    meta = next(((s, e) for k, s, e in boxes(data, 0, len(data))
                 if k == b"meta"), None)
    if meta is None:
        return None
    out = Items(data)
    for kind, s, e in boxes(data, meta[0] + 4, meta[1]):
        version = data[s]
        if kind == b"pitm":
            out.primary = (struct.unpack_from(">H", data, s + 4)[0]
                           if version == 0 else
                           struct.unpack_from(">I", data, s + 4)[0])
        elif kind == b"iinf":
            first = s + (6 if version == 0 else 8)
            for k2, s2, e2 in boxes(data, first, e):
                if k2 != b"infe" or data[s2] < 2:
                    continue
                if data[s2] == 2:
                    item = struct.unpack_from(">H", data, s2 + 4)[0]
                    at = s2 + 8
                else:
                    item = struct.unpack_from(">I", data, s2 + 4)[0]
                    at = s2 + 10
                out.types[item] = bytes(data[at:at + 4])
        elif kind == b"iref":
            wide = version != 0
            for k2, s2, e2 in boxes(data, s + 4, e):
                fmt, step = (">I", 4) if wide else (">H", 2)
                src = struct.unpack_from(fmt, data, s2)[0]
                count = struct.unpack_from(">H", data, s2 + step)[0]
                at = s2 + step + 2
                dst = [struct.unpack_from(fmt, data, at + i * step)[0]
                       for i in range(count)]
                out.refs.append((bytes(k2), src, dst))
        elif kind == b"iloc":
            _parse_iloc(data, s, version, out.iloc)
        elif kind == b"idat":
            out.idat = (s, e)
        elif kind == b"iprp":
            for k2, s2, e2 in boxes(data, s, e):
                if k2 == b"ipco":
                    out.props = [(bytes(k3), s3, e3)
                                 for k3, s3, e3 in boxes(data, s2, e2)]
                elif k2 == b"ipma":
                    _parse_ipma(data, s2, out.assoc)
    return out


def _parse_iloc(data, s, version, iloc):
    at = s + 4
    offset_size, length_size = data[at] >> 4, data[at] & 15
    base_size = data[at + 1] >> 4
    index_size = data[at + 1] & 15 if version in (1, 2) else 0
    at += 2
    if version < 2:
        count = struct.unpack_from(">H", data, at)[0]
        at += 2
    else:
        count = struct.unpack_from(">I", data, at)[0]
        at += 4
    for _ in range(count):
        if version < 2:
            item = struct.unpack_from(">H", data, at)[0]
            at += 2
        else:
            item = struct.unpack_from(">I", data, at)[0]
            at += 4
        method = 0
        if version in (1, 2):
            method = struct.unpack_from(">H", data, at)[0] & 15
            at += 2
        at += 2                                 # data_reference_index
        base, at = _uint(data, at, base_size)
        extents = struct.unpack_from(">H", data, at)[0]
        at += 2
        spans = []
        for _ in range(extents):
            _, at = _uint(data, at, index_size)
            offset, at = _uint(data, at, offset_size)
            length, at = _uint(data, at, length_size)
            spans.append((base + offset, length))
        iloc[item] = (method, spans)


def _parse_ipma(data, s, assoc):
    version, flags = data[s], data[s + 3]
    count = struct.unpack_from(">I", data, s + 4)[0]
    at = s + 8
    for _ in range(count):
        if version < 1:
            item = struct.unpack_from(">H", data, at)[0]
            at += 2
        else:
            item = struct.unpack_from(">I", data, at)[0]
            at += 4
        n = data[at]
        at += 1
        idx = []
        for _ in range(n):
            if flags & 1:
                idx.append(struct.unpack_from(">H", data, at)[0] & 0x7FFF)
                at += 2
            else:
                idx.append(data[at] & 0x7F)
                at += 1
        assoc.setdefault(item, []).extend(idx)


def clap_is_identity(data, props) -> bool:
    """True if the clean-aperture box keeps the whole coded image."""
    ispe = props.get(b"ispe")
    if ispe is None:
        return False
    width, height = struct.unpack_from(">II", data, ispe[0] + 4)
    wn, wd, hn, hd, xn, xd, yn, yd = struct.unpack_from(
        ">IIIIiIiI", data, props[b"clap"][0])
    if not (wd and hd and xd and yd):
        return False
    return wn == width * wd and hn == height * hd and xn == 0 and yn == 0


def transform_reason(items, item) -> str:
    """Why the item's own transform properties keep it off a hardware
    path that returns the coded pixels as they are, or ""."""
    data = items.data
    mine = items.properties(item)
    if b"imir" in mine:
        return "the image is mirrored (imir)"
    if b"irot" in mine and data[mine[b"irot"][0]] & 3:
        return "the image is rotated (irot)"
    if b"clap" in mine and not clap_is_identity(data, mine):
        return "the image is cropped (clap)"
    return ""


def has_alpha(items, item) -> bool:
    """True if an auxiliary image of ``item`` is its alpha plane."""
    data = items.data
    for kind, src, dst in items.refs:
        if kind == b"auxl" and item in dst:
            auxc = items.properties(src).get(b"auxC")
            # auxC is a full box; its aux_type is a NUL-terminated URN.
            urn = (b"" if auxc is None else
                   bytes(data[auxc[0] + 4:auxc[1]]).split(b"\0")[0])
            if auxc is None or urn in ALPHA_URNS:
                return True
    return False


def grid_layout(items, item):
    """(rows, columns, output width, output height) of a grid item."""
    desc = items.item_data(item)
    if len(desc) < 8 or desc[0] != 0:
        raise ValueError("unsupported grid descriptor")
    rows, cols = desc[2] + 1, desc[3] + 1
    if desc[1] & 1:
        width, height = struct.unpack_from(">II", desc, 4)
    else:
        width, height = struct.unpack_from(">HH", desc, 4)
    return rows, cols, width, height


def hvcc(data, start, end) -> dict:
    """The fields of an hvcC box needed to decode its stream."""
    if end - start < 23:
        raise ValueError("hvcC too short")
    out = dict(chroma=data[start + 16] & 3,
               luma_bits=8 + (data[start + 17] & 7),
               chroma_bits=8 + (data[start + 18] & 7),
               length_size=(data[start + 21] & 3) + 1, nals=[])
    count, at = data[start + 22], start + 23
    for _ in range(count):
        n = struct.unpack_from(">H", data, at + 1)[0]
        at += 3
        for _ in range(n):
            size = struct.unpack_from(">H", data, at)[0]
            out["nals"].append(bytes(data[at + 2:at + 2 + size]))
            at += 2 + size
        if at > end:
            raise ValueError("hvcC arrays run past the box")
    return out


def length_prefixed_to_annexb(payload, length_size) -> bytes:
    """Length-prefixed NAL units (as stored in HEIF) to an Annex-B stream."""
    out, at, end = [], 0, len(payload)
    while at + length_size <= end:
        size = int.from_bytes(payload[at:at + length_size], "big")
        at += length_size
        if at + size > end:
            raise ValueError("NAL unit runs past the item")
        out.append(b"\0\0\0\1")
        out.append(payload[at:at + size])
        at += size
    return b"".join(out)


# ----------------------------------------------------------------- write

def _box(kind: bytes, *payload: bytes) -> bytes:
    body = b"".join(payload)
    return struct.pack(">I4s", 8 + len(body), kind) + body


def _full(kind: bytes, version: int, flags: int, *payload: bytes) -> bytes:
    return _box(kind, struct.pack(">I", (version << 24) | flags), *payload)


def colr_nclx(primaries, transfer, matrix, full_range) -> bytes:
    return _box(b"colr", b"nclx",
                struct.pack(">HHHB", primaries, transfer, matrix,
                            0x80 if full_range else 0))


def write_single_item(*, brands, item_type: bytes, config: bytes,
                      width: int, height: int, bits: int, channels: int,
                      payload: bytes, nclx, icc: bytes | None = None,
                      clap: bytes | None = None) -> bytes:
    """A HEIF/AVIF file holding one coded picture as its primary item.

    ``brands`` is (major, [compatible]); ``config`` is the whole hvcC or
    av1C box, marked essential as both formats require; ``nclx`` is
    (primaries, transfer, matrix, full_range) for a ``colr`` box, which
    an ``icc`` profile joins as a second ``colr`` box. A ``clap`` box
    (whole) crops the coded picture; as a transformative property it is
    listed last and marked essential.
    """
    major, compatible = brands
    ftyp = _box(b"ftyp", major, struct.pack(">I", 0), *compatible)
    hdlr = _full(b"hdlr", 0, 0, struct.pack(">I", 0), b"pict",
                 b"\0" * 12, b"\0")
    pitm = _full(b"pitm", 0, 0, struct.pack(">H", 1))
    infe = _full(b"infe", 2, 0, struct.pack(">HH", 1, 0), item_type, b"\0")
    iinf = _full(b"iinf", 0, 0, struct.pack(">H", 1), infe)
    props = [config,
             _full(b"ispe", 0, 0, struct.pack(">II", width, height)),
             _full(b"pixi", 0, 0, bytes([channels] + [bits] * channels)),
             colr_nclx(*nclx)]
    if icc:
        props.append(_box(b"colr", b"prof", bytes(icc)))
    essential = [True] + [False] * (len(props) - 1)
    if clap:
        props.append(clap)
        essential.append(True)
    ipco = _box(b"ipco", *props)
    ipma = _full(b"ipma", 0, 0, struct.pack(">IHB", 1, 1, len(props)),
                 bytes((0x80 if e else 0) | (i + 1)
                       for i, e in enumerate(essential)))
    iprp = _box(b"iprp", ipco, ipma)

    def iloc(offset):
        # version 0, 4-byte offsets and lengths, no base offset.
        return _full(b"iloc", 0, 0, bytes([0x44, 0x00]),
                     struct.pack(">HHHHII", 1, 1, 0, 1, offset, len(payload)))

    def meta(offset):
        return _full(b"meta", 0, 0, hdlr, pitm, iloc(offset), iinf, iprp)

    head = len(ftyp) + len(meta(0)) + 8      # the mdat header
    if len(payload) + head > 0xFFFFFFFF:
        raise ValueError("coded picture too large for a 32-bit iloc")
    return ftyp + meta(head) + _box(b"mdat", payload)
