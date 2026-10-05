"""HEVC and AV1 headers, as far as the hardware backends need them.

Decoding a HEIF on a hardware decoder needs the color signaling libheif
would use when the container has none: the HEVC sequence parameter
set's VUI (matrix, full-range flag). Wrapping a hardware encoder's
output in a container needs the codec configuration record (hvcC from
the SPS, av1C from the AV1 sequence header), and for AV1 the sequence
header's color description set to what the pixels were converted with.
"""

from __future__ import annotations

import struct


class _Bits:
    """MSB-first bit reader over a bytes-like object."""

    def __init__(self, data, pos=0):
        self.data = data
        self.pos = pos            # in bits

    def u(self, n):
        value = 0
        for _ in range(n):
            byte = self.data[self.pos >> 3]
            value = (value << 1) | ((byte >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return value

    def ue(self):
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
            if zeros > 31:
                raise ValueError("bad exp-Golomb code")
        return (1 << zeros) - 1 + self.u(zeros)

    def se(self):
        k = self.ue()
        return (k + 1) // 2 if k & 1 else -(k // 2)


class _BitWriter:
    def __init__(self):
        self.bits = []

    def u(self, n, value):
        self.bits.extend((value >> (n - 1 - i)) & 1 for i in range(n))

    def tobytes(self):
        out = bytearray()
        for i in range(0, len(self.bits), 8):
            chunk = self.bits[i:i + 8]
            chunk += [0] * (8 - len(chunk))
            out.append(int("".join(map(str, chunk)), 2))
        return bytes(out)


# ------------------------------------------------------------------ HEVC

def rbsp(nal: bytes) -> bytes:
    """A NAL unit's payload without its 2-byte header and emulation
    prevention bytes."""
    out = bytearray()
    zeros = 0
    for b in nal[2:]:
        if zeros >= 2 and b == 3:
            zeros = 0
            continue
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


def nal_type(nal: bytes) -> int:
    return (nal[0] >> 1) & 0x3F


def split_annexb(stream: bytes) -> list:
    """The NAL units of an Annex-B byte stream, start codes removed."""
    out, i, n = [], 0, len(stream)
    starts = []
    while True:
        j = stream.find(b"\0\0\1", i)
        if j < 0:
            break
        starts.append(j + 3)
        i = j + 3
    for k, s in enumerate(starts):
        e = starts[k + 1] - 3 if k + 1 < len(starts) else n
        nal = stream[s:e]
        # a 4-byte start code leaves one zero byte on the previous unit
        while k + 1 < len(starts) and nal.endswith(b"\0"):
            nal = nal[:-1]
        if nal:
            out.append(nal)
    return out


def _profile_tier_level(r: _Bits, max_sub_layers_minus1: int):
    r.u(2 + 1 + 5 + 32 + 48 + 8)      # general profile, flags, level
    present = [(r.u(1), r.u(1)) for _ in range(max_sub_layers_minus1)]
    if max_sub_layers_minus1 > 0:
        r.u(2 * (8 - max_sub_layers_minus1))
    for profile, level in present:
        if profile:
            r.u(88)
        if level:
            r.u(8)


def _scaling_list_data(r: _Bits):
    for size_id in range(4):
        for _ in range(0, 6, 3 if size_id == 3 else 1):
            if not r.u(1):
                r.ue()
            else:
                count = min(64, 1 << (4 + (size_id << 1)))
                if size_id > 1:
                    r.se()
                for _ in range(count):
                    r.se()


def _short_term_ref_pic_sets(r: _Bits, count: int):
    """Skip the SPS's st_ref_pic_set()s, deriving each set's delta POCs
    (H.265 7.4.8), which a predicted set's syntax depends on."""
    sets = []                   # (negative deltas S0, positive deltas S1)
    for idx in range(count):
        if idx and r.u(1):      # inter_ref_pic_set_prediction_flag
            sign = r.u(1)
            delta_rps = (1 - 2 * sign) * (r.ue() + 1)
            s0, s1 = sets[idx - 1]
            n = len(s0) + len(s1)
            use = []
            for _ in range(n + 1):
                used = r.u(1)
                use.append(1 if used else r.u(1))
            neg, pos = [], []
            for j in range(len(s1) - 1, -1, -1):
                d = s1[j] + delta_rps
                if d < 0 and use[len(s0) + j]:
                    neg.append(d)
            if delta_rps < 0 and use[n]:
                neg.append(delta_rps)
            for j in range(len(s0)):
                d = s0[j] + delta_rps
                if d < 0 and use[j]:
                    neg.append(d)
            for j in range(len(s0) - 1, -1, -1):
                d = s0[j] + delta_rps
                if d > 0 and use[j]:
                    pos.append(d)
            if delta_rps > 0 and use[n]:
                pos.append(delta_rps)
            for j in range(len(s1)):
                d = s1[j] + delta_rps
                if d > 0 and use[len(s0) + j]:
                    pos.append(d)
            sets.append((neg, pos))
        else:
            n_neg, n_pos = r.ue(), r.ue()
            if n_neg > 16 or n_pos > 16:
                raise ValueError("bad short-term reference set")
            neg, pos, poc = [], [], 0
            for _ in range(n_neg):
                poc -= r.ue() + 1
                r.u(1)
                neg.append(poc)
            poc = 0
            for _ in range(n_pos):
                poc += r.ue() + 1
                r.u(1)
                pos.append(poc)
            sets.append((neg, pos))


def hevc_sps(nal: bytes) -> dict:
    """Size, chroma format, bit depths and VUI color fields of an SPS.

    ``vui`` is (primaries, transfer, matrix, full_range) as libde265
    reports them, its defaults (2, 2, 2, False) where the stream has no
    VUI or no color description.
    """
    data = rbsp(nal)
    r = _Bits(data)
    r.u(4)
    max_sub_layers_minus1 = r.u(3)
    r.u(1)
    _profile_tier_level(r, max_sub_layers_minus1)
    r.ue()
    chroma = r.ue()
    if chroma == 3:
        r.u(1)
    width, height = r.ue(), r.ue()
    if r.u(1):                                  # conformance window
        for _ in range(4):
            r.ue()
    luma_bits, chroma_bits = 8 + r.ue(), 8 + r.ue()
    out = dict(chroma=chroma, width=width, height=height,
               luma_bits=luma_bits, chroma_bits=chroma_bits,
               vui=(2, 2, 2, False), dpb=[], at={})
    log2_max_poc_lsb = r.ue() + 4
    ordering = r.u(1)
    out["at"]["dpb"] = r.pos
    for _ in range(0 if ordering else max_sub_layers_minus1,
                   max_sub_layers_minus1 + 1):
        out["dpb"].append((r.ue(), r.ue(), r.ue()))
    out["at"]["dpb_end"] = r.pos
    for _ in range(6):
        r.ue()
    if r.u(1) and r.u(1):                       # scaling lists, in the SPS
        _scaling_list_data(r)
    r.u(1)                                      # amp
    r.u(1)                                      # sao
    if r.u(1):                                  # pcm
        r.u(4)
        r.u(4)
        r.ue()
        r.ue()
        r.u(1)
    out["at"]["rps"] = r.pos
    _short_term_ref_pic_sets(r, r.ue())
    out["at"]["rps_end"] = r.pos
    if r.u(1):                                  # long-term reference pictures
        for _ in range(r.ue()):
            r.u(log2_max_poc_lsb)
            r.u(1)
    r.u(1)                                      # temporal mvp
    r.u(1)                                      # strong intra smoothing
    out["at"]["vui"] = r.pos
    if r.u(1):                                  # VUI
        if r.u(1):                              # aspect ratio
            if r.u(8) == 255:
                r.u(32)
        if r.u(1):                              # overscan
            r.u(1)
        out["at"]["signal"] = r.pos
        if r.u(1):                              # video signal type
            r.u(3)
            full = bool(r.u(1))
            cp = tc = mc = 2
            if r.u(1):
                cp, tc, mc = r.u(8), r.u(8), r.u(8)
            out["vui"] = (cp, tc, mc, full)
        out["at"]["signal_end"] = r.pos
        if r.u(1):                              # chroma location
            r.ue()
            r.ue()
        r.u(3)                                  # neutral chroma, field info
        if r.u(1):                              # default display window
            for _ in range(4):
                r.ue()
        if r.u(1):                              # timing info
            r.u(64)
            if r.u(1):
                r.ue()
            out["at"]["hrd"] = r.pos
            if r.u(1):
                _hrd_parameters(r, max_sub_layers_minus1)
            out["at"]["hrd_end"] = r.pos
    return out


def _hrd_parameters(r: _Bits, max_sub_layers_minus1: int):
    """Skip hrd_parameters(1, max_sub_layers_minus1) (H.265 E.2.2)."""
    nal, vcl = r.u(1), r.u(1)
    sub_pic = 0
    if nal or vcl:
        sub_pic = r.u(1)
        if sub_pic:
            r.u(8 + 5 + 1 + 5)
        r.u(8)                                  # bit rate and CPB size scales
        if sub_pic:
            r.u(4)
        r.u(15)
    for _ in range(max_sub_layers_minus1 + 1):
        fixed = r.u(1)
        if not fixed:
            fixed = r.u(1)
        low_delay = 0
        if fixed:
            r.ue()
        else:
            low_delay = r.u(1)
        count = r.ue() + 1 if not low_delay else 1
        for _ in range(nal + vcl):
            for _ in range(count):
                r.ue()
                r.ue()
                if sub_pic:
                    r.ue()
                    r.ue()
                r.u(1)


def _ue_bits(value):
    n = (value + 1).bit_length()
    return [0] * (n - 1) + [((value + 1) >> (n - 1 - i)) & 1 for i in range(n)]


def _ebsp(payload: bytes) -> bytes:
    """Emulation prevention: a 3 after any two zero bytes followed by a
    byte of 3 or less."""
    out = bytearray()
    zeros = 0
    for b in payload:
        if zeros >= 2 and b <= 3:
            out.append(3)
            zeros = 0
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


def hevc_sps_rewrite(nal: bytes, *, dpb=None, signal=None,
                     drop_rps=False, drop_hrd=False) -> bytes:
    """The SPS with its DPB sizes or VUI video signal type replaced.

    ``dpb`` is (max_dec_pic_buffering_minus1, max_num_reorder_pics,
    max_latency_increase_plus1) for every sub-layer; ``drop_rps`` removes
    the short-term reference picture sets (only non-IDR slices can refer
    to them, so a stream of IDR pictures never does); ``drop_hrd``
    removes the VUI's HRD parameters (buffering for streaming, meaningless
    for a still image; libde265 rejects the bit-rate values NVENC writes
    there for pictures above 8.9 megapixels); ``signal`` is
    (primaries, transfer, matrix, full_range), written as the VUI's
    video signal type (format 5, unspecified). Every other bit is copied;
    the SPS must already have a VUI for ``signal``.
    """
    info = hevc_sps(nal)
    edits = []
    if dpb is not None:
        new = []
        for _ in info["dpb"]:
            for v in dpb:
                new += _ue_bits(v)
        edits.append((info["at"]["dpb"], info["at"]["dpb_end"], new))
    if signal is not None:
        if "signal" not in info["at"]:
            raise ValueError("SPS without a VUI")
        cp, tc, mc, full = signal
        new = [1] + [1, 0, 1] + [1 if full else 0] + [1]
        for v in (cp, tc, mc):
            new += [(v >> (7 - i)) & 1 for i in range(8)]
        edits.append((info["at"]["signal"], info["at"]["signal_end"], new))
    if drop_rps:
        edits.append((info["at"]["rps"], info["at"]["rps_end"], [1]))
    if drop_hrd and "hrd" in info["at"]:
        edits.append((info["at"]["hrd"], info["at"]["hrd_end"], [0]))
    return _rewrite_bits(nal, edits)


def _rewrite_bits(nal: bytes, edits) -> bytes:
    """Replace bit ranges of a NAL unit's RBSP: ``edits`` is a list of
    (start, end, new bits); the stop bit and alignment are redone and
    emulation prevention applied again."""
    data = rbsp(nal)
    bits = [(byte >> (7 - i)) & 1 for byte in data for i in range(8)]
    while bits and bits[-1] == 0:
        bits.pop()
    bits.pop()                                  # rbsp_stop_one_bit
    for start, end, new in sorted(edits, reverse=True):
        bits[start:end] = new
    bits.append(1)
    bits += [0] * (-len(bits) % 8)
    payload = bytes(int("".join(map(str, bits[i:i + 8])), 2)
                    for i in range(0, len(bits), 8))
    return bytes(nal[:2]) + _ebsp(payload)


def hevc_vps_rewrite(nal: bytes, dpb) -> bytes:
    """The VPS with every sub-layer's DPB sizes set to ``dpb``."""
    r = _Bits(rbsp(nal))
    r.u(4 + 1 + 1 + 6)
    max_sub_layers_minus1 = r.u(3)
    r.u(1 + 16)
    _profile_tier_level(r, max_sub_layers_minus1)
    ordering = r.u(1)
    start = r.pos
    new = []
    for _ in range(0 if ordering else max_sub_layers_minus1,
                   max_sub_layers_minus1 + 1):
        r.ue()
        r.ue()
        r.ue()
        for v in dpb:
            new += _ue_bits(v)
    return _rewrite_bits(nal, [(start, r.pos, new)])


def hvcc_record(vps: bytes, sps: bytes, pps: bytes) -> bytes:
    """An hvcC box for one VPS, SPS and PPS (length size 4)."""
    data = rbsp(sps)
    info = hevc_sps(sps)
    max_sub_layers_minus1 = (data[0] >> 1) & 7
    nested = data[0] & 1
    # The general profile_tier_level (profile space, tier, profile,
    # compatibility and constraint flags, level) is the SPS's bytes 1 to
    # 12, laid out exactly as hvcC stores them.
    body = (bytes([1]) + data[1:13]
            + struct.pack(">H", 0xF000)              # min spatial segmentation
            + bytes([0xFC,                            # parallelism type 0
                     0xFC | info["chroma"],
                     0xF8 | (info["luma_bits"] - 8),
                     0xF8 | (info["chroma_bits"] - 8)])
            + struct.pack(">H", 0)                    # average frame rate
            + bytes([((max_sub_layers_minus1 + 1) << 3) | (nested << 2) | 3,
                     3]))
    for kind, nal in ((32, vps), (33, sps), (34, pps)):
        body += bytes([0x80 | kind]) + struct.pack(">HH", 1, len(nal)) + nal
    return struct.pack(">I4s", 8 + len(body), b"hvcC") + body


# ------------------------------------------------------------------- AV1

OBU_SEQUENCE_HEADER = 1
OBU_TEMPORAL_DELIMITER = 2
OBU_PADDING = 15


def _leb128(data, at):
    value = shift = 0
    for i in range(8):
        byte = data[at + i]
        value |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return value, at + i + 1
    raise ValueError("bad leb128")


def leb128(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def split_obus(stream: bytes) -> list:
    """(type, whole OBU bytes, payload start within it) for each OBU of a
    low-overhead (size-field) AV1 stream."""
    out, at = [], 0
    while at < len(stream):
        header = stream[at]
        kind = (header >> 3) & 15
        ext = (header >> 2) & 1
        if not (header >> 1) & 1:
            raise ValueError("AV1 OBU without a size field")
        size, payload = _leb128(stream, at + 1 + ext)
        end = payload + size
        if end > len(stream):
            raise ValueError("AV1 OBU runs past the stream")
        out.append((kind, bytes(stream[at:end]), payload - at))
        at = end
    return out


def av1_sequence_header(payload: bytes) -> dict:
    """The fields of an AV1 sequence header OBU payload needed for av1C,
    and the bit position where color_config starts."""
    r = _Bits(payload)
    profile = r.u(3)
    still = r.u(1)
    reduced = r.u(1)
    out = dict(profile=profile, still=still, reduced=reduced, tier=0)
    if reduced:
        out["level"] = r.u(5)
    else:
        decoder_model = initial_delay = 0
        buffer_delay_bits = 0
        if r.u(1):                                    # timing info
            r.u(32)
            r.u(32)
            if r.u(1):
                zeros = 0
                while r.u(1) == 0:
                    zeros += 1
                r.u(zeros)
            decoder_model = r.u(1)
            if decoder_model:
                buffer_delay_bits = r.u(5) + 1
                r.u(32)
                r.u(5)
                r.u(5)
        initial_delay = r.u(1)
        for i in range(r.u(5) + 1):
            r.u(12)
            level = r.u(5)
            tier = r.u(1) if level > 7 else 0
            if i == 0:
                out["level"], out["tier"] = level, tier
            if decoder_model and r.u(1):
                r.u(2 * buffer_delay_bits + 1)
            if initial_delay and r.u(1):
                r.u(4)
    wbits, hbits = r.u(4) + 1, r.u(4) + 1
    out["width"], out["height"] = r.u(wbits) + 1, r.u(hbits) + 1
    if not reduced and r.u(1):                        # frame id numbers
        r.u(4)
        r.u(3)
    r.u(3)                                            # 128x128, filter intra, intra edge
    order_hint = 0
    if not reduced:
        r.u(4)                                        # interintra .. dual filter
        order_hint = r.u(1)
        if order_hint:
            r.u(2)
        force_sct = 2 if r.u(1) else r.u(1)
        if force_sct > 0 and not r.u(1):
            r.u(1)
        if order_hint:
            r.u(3)
    r.u(3)                                            # superres, cdef, restoration
    out["color_config_at"] = r.pos
    high = r.u(1)
    bits = 8
    if profile == 2 and high:
        bits = 12 if r.u(1) else 10
    elif high:
        bits = 10
    mono = r.u(1) if profile != 1 else 0
    cp = tc = mc = 2
    if r.u(1):
        cp, tc, mc = r.u(8), r.u(8), r.u(8)
    sx = sy = 1
    position = 0
    if mono:
        full = r.u(1)
    elif cp == 1 and tc == 13 and mc == 0:
        full, sx, sy = 1, 0, 0
    else:
        full = r.u(1)
        if profile == 0:
            sx = sy = 1
        elif profile == 1:
            sx = sy = 0
        elif bits == 12:
            sx = r.u(1)
            sy = r.u(1) if sx else 0
        else:
            sx, sy = 1, 0
        if sx and sy:
            position = r.u(2)
    separate_uv = r.u(1) if not mono else 0
    out.update(bits=bits, mono=mono, color=(cp, tc, mc, bool(full)),
               subsampling=(sx, sy), sample_position=position,
               separate_uv_delta_q=separate_uv, film_grain=r.u(1))
    return out


def av1_set_color(obu: bytes, payload_at: int, color) -> bytes:
    """The sequence header OBU with its color description replaced by
    ``color`` = (primaries, transfer, matrix, full_range).

    Everything before color_config is copied bit for bit; color_config
    is written again with the same depth, subsampling and sample
    position; the film-grain flag and trailing bits follow.
    """
    payload = obu[payload_at:]
    info = av1_sequence_header(payload)
    cp, tc, mc, full = color
    profile, bits, (sx, sy) = info["profile"], info["bits"], info["subsampling"]
    if info["mono"]:
        raise ValueError("monochrome AV1 is not rewritten")
    if cp == 1 and tc == 13 and mc == 0:
        raise ValueError("identity (sRGB 4:4:4) color is not rewritten")
    w = _BitWriter()
    r = _Bits(payload)
    w.u(info["color_config_at"], r.u(info["color_config_at"]))
    w.u(1, 1 if bits > 8 else 0)
    if profile == 2 and bits > 8:
        w.u(1, 1 if bits == 12 else 0)
    if profile != 1:
        w.u(1, 0)                                     # mono_chrome
    w.u(1, 1)                                         # color description present
    w.u(8, cp)
    w.u(8, tc)
    w.u(8, mc)
    w.u(1, 1 if full else 0)
    if profile == 2 and bits == 12:
        w.u(1, sx)
        if sx:
            w.u(1, sy)
    if sx and sy:
        w.u(2, info["sample_position"])
    w.u(1, info["separate_uv_delta_q"])
    w.u(1, info["film_grain"])
    w.u(1, 1)                                         # trailing one bit
    new_payload = w.tobytes()
    header = obu[0]
    ext = obu[1:2] if (header >> 2) & 1 else b""
    return bytes([header]) + ext + leb128(len(new_payload)) + new_payload


def av1c_record(info: dict, sequence_header_obu: bytes) -> bytes:
    """An av1C box for a parsed sequence header, carrying the header."""
    sx, sy = info["subsampling"]
    body = bytes([
        0x81,
        (info["profile"] << 5) | info["level"],
        (info["tier"] << 7) | ((1 if info["bits"] > 8 else 0) << 6)
        | ((1 if info["bits"] == 12 else 0) << 5) | (info["mono"] << 4)
        | (sx << 3) | (sy << 2) | info["sample_position"],
        0,
    ]) + sequence_header_obu
    return struct.pack(">I4s", 8 + len(body), b"av1C") + body
