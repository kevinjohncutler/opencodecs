"""Argument handling shared by the ``_jpeg`` and ``_mozjpeg`` extensions.

Both extensions take the parameters ``imagecodecs.jpeg8_encode`` and
``imagecodecs.jpeg8_decode`` define, with the same spellings and
meanings. The parsing lives here once so the two cannot drift apart:
colorspace names, subsampling spellings, the JPEG tables and header
splices, and a marker scan that reads the frame header without
decoding anything.
"""

from __future__ import annotations

from typing import NamedTuple


# libjpeg's J_COLOR_SPACE values. imagecodecs accepts these integers as
# well as the names below, so a caller moving between the two libraries
# can pass the same argument.
_JCS_NAMES = {
    0: "unknown", 1: "gray", 2: "rgb", 3: "ycbcr", 4: "cmyk", 5: "ycck",
    6: "rgb", 7: "rgbx", 8: "bgr", 9: "bgrx", 10: "xbgr", 11: "xrgb",
    12: "rgba", 13: "bgra", 14: "abgr", 15: "argb", 16: "rgb565",
}

# The string spellings imagecodecs accepts (case insensitive), plus the
# explicit pixel orders libjpeg-turbo names as JCS_EXT_*.
_CS_NAMES = {
    "unknown": "unknown",
    "gray": "gray", "grayscale": "gray",
    "minisblack": "gray", "miniswhite": "gray",
    "rgb": "rgb", "ycbcr": "ycbcr",
    "cmyk": "cmyk", "separated": "cmyk", "ycck": "ycck",
    "rgba": "rgba", "rgbx": "rgbx", "bgr": "bgr", "bgrx": "bgrx",
    "xbgr": "xbgr", "xrgb": "xrgb", "bgra": "bgra", "abgr": "abgr",
    "argb": "argb", "rgb565": "rgb565",
}

# Samples per pixel of each packed input or output layout.
CS_SAMPLES = {
    "gray": 1, "rgb": 3, "bgr": 3, "ycbcr": 3, "rgb565": 3,
    "cmyk": 4, "ycck": 4, "rgba": 4, "rgbx": 4, "bgrx": 4, "xbgr": 4,
    "xrgb": 4, "bgra": 4, "abgr": 4, "argb": 4,
}


# Packed input orders whose fourth sample is alpha. A JPEG has no alpha
# channel, so encoding one would drop data; the X orders name padding,
# which is meant to be dropped.
_ALPHA_ORDERS = frozenset(("rgba", "bgra", "argb", "abgr"))


def reject_alpha(in_cs: str, colorspace, what: str) -> None:
    """Raise for an input order that carries alpha.

    imagecodecs raises for ``"rgba"`` too; for ``"bgra"``, ``"argb"``
    and ``"abgr"`` it stores the fourth sample as a fourth component.
    TurboJPEG reads these orders as RGB and would drop the alpha
    samples without a word, so they raise here instead.
    """
    if in_cs in _ALPHA_ORDERS:
        pad = in_cs.replace("a", "x")
        raise ValueError(
            f"{what}: colorspace={colorspace!r}: JPEG has no alpha "
            f"channel, so the alpha samples would be lost; pass the color "
            f"samples only, or colorspace={pad!r} to drop the fourth "
            f"sample as padding")


# TIFF PhotometricInterpretation names (TIFF 6.0, TIFF/EP, DNG) that
# name no libjpeg colorspace. tifffile's jpeg_decode_colorspace passes
# PHOTOMETRIC(photometric).name as the decode outcolorspace for every
# photometric above 3 it does not map itself (CFA and LinearRaw in DNG,
# CIELab, ...), and imagecodecs reads each as JCS_UNKNOWN: decode with
# the library's default for the stream. These names keep that meaning
# here, so a DNG-style lossless JPEG tile reads through tifffile_patch
# as it does through imagecodecs.
_PHOTOMETRIC_DEFAULT = frozenset((
    "palette", "mask", "cielab", "icclab", "itulab", "cfa", "logl",
    "logluv", "linear_raw", "depth_map", "semantic_mask",
))


def colorspace_name(value, what: str, *, decode: bool = False) -> str | None:
    """Return the canonical name of a colorspace argument, or ``None``.

    ``None`` and ``"unknown"`` both mean "not specified", as they do in
    imagecodecs. With ``decode=True`` the TIFF photometric names that
    are not JPEG colorspaces (``"CFA"``, ``"LINEAR_RAW"``, ``"CIELAB"``,
    ...) mean "not specified" too, as imagecodecs reads them. Any other
    unrecognized value raises instead of falling back to a default:
    imagecodecs maps every unknown string to JCS_UNKNOWN silently, and a
    misspelled colorspace that quietly does nothing is exactly the kind
    of dropped argument this module exists to prevent.
    """
    if value is None:
        return None
    if isinstance(value, str):
        key = value.strip().lower()
        if decode and key in _PHOTOMETRIC_DEFAULT:
            return None
        name = _CS_NAMES.get(key)
    elif isinstance(value, int) and not isinstance(value, bool):
        name = _JCS_NAMES.get(int(value))
    else:
        name = None
    if name is None:
        raise ValueError(f"jpeg: unknown {what} {value!r}")
    return None if name == "unknown" else name


# Luma (horizontal, vertical) sampling factors per subsampling name, the
# tuple spelling imagecodecs accepts alongside the strings.
_SUBSAMPLING = {
    "444": "444", "422": "422", "420": "420", "440": "440", "411": "411",
    "gray": "gray", "grayscale": "gray",
    (1, 1): "444", (2, 1): "422", (2, 2): "420", (1, 2): "440",
    (4, 1): "411",
}


def subsampling_name(value) -> str | None:
    """Return ``"444"``, ``"422"``, ``"420"``, ``"440"``, ``"411"``,
    ``"gray"`` or ``None`` for a subsampling argument."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        key = tuple(int(v) for v in value)
    else:
        key = str(value).strip().lower()
    name = _SUBSAMPLING.get(key)
    if name is None:
        raise ValueError(
            f"jpeg: unknown subsampling {value!r}; expected one of "
            f"'444', '422', '420', '440', '411', 'gray' or a (h, v) "
            f"luma sampling tuple")
    return name


def splice_stream(data, tables=None, header=None) -> bytes:
    """Return one self-contained JPEG stream from abbreviated pieces.

    ``header`` is prepended and an EOI marker appended, as in
    ``imagecodecs.jpeg_decode``: it carries the leading markers a
    container stores apart from the entropy-coded data.

    ``tables`` is an abbreviated table-specification stream (ITU-T T.81
    B.5: SOI, DQT and DHT segments, EOI), such as TIFF's JPEGTables tag.
    Dropping its EOI and the image's SOI and joining the two gives the
    stream libjpeg would assemble by reading the tables first: the
    tables are defined before the frame that uses them.
    """
    if header is None and tables is None:
        return data if isinstance(data, (bytes, bytearray)) else bytes(data)
    data = bytes(data)
    if header is not None:
        data = bytes(header) + data + b"\xff\xd9"
    if tables is not None:
        tables = bytes(tables)
        if len(tables) < 4 or tables[:2] != b"\xff\xd8" \
                or tables[-2:] != b"\xff\xd9":
            raise ValueError(
                "jpeg: tables must be a table-specification stream that "
                "starts with SOI and ends with EOI")
        if data[:2] != b"\xff\xd8":
            raise ValueError("jpeg: data must start with an SOI marker "
                             "to be combined with tables")
        data = tables[:-2] + data[2:]
    return data


class FrameHeader(NamedTuple):
    """The fields of a JPEG frame header (ITU-T T.81 B.2.2)."""

    sof: int          # marker code, 0xC0 to 0xCF
    precision: int    # P, bits per sample
    height: int       # Y
    width: int        # X
    components: int   # Nf

    @property
    def lossless(self) -> bool:
        """SOF3, SOF7, SOF11 and SOF15 are the lossless processes."""
        return self.sof in (0xC3, 0xC7, 0xCB, 0xCF)


def _segments(data):
    """Yield (marker, start, end) for the marker segments before SOS,
    then (0xDA, start, len(data)) for the scan and everything after."""
    mv = memoryview(data)
    n = len(mv)
    i = 2
    while i + 1 < n:
        if mv[i] != 0xFF:
            raise ValueError("jpeg: corrupt marker segment")
        marker = mv[i + 1]
        if marker == 0xFF:          # fill byte before a marker
            yield 0xFF, i, i + 1
            i += 1
            continue
        if marker == 0xDA or marker == 0xD9:
            yield marker, i, n
            return
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            yield marker, i, i + 2
            i += 2
            continue
        if i + 3 >= n:
            raise ValueError("jpeg: truncated marker segment")
        end = i + 2 + ((mv[i + 2] << 8) | mv[i + 3])
        yield marker, i, end
        i = end


# Adobe APP14 transform flag (Adobe Technical Note 5116): 0 means the
# components are stored as is (RGB or CMYK), 1 YCbCr, 2 YCCK.
_ADOBE_TRANSFORM = {"rgb": 0, "cmyk": 0, "ycbcr": 1, "ycck": 2}
_STORABLE = {1: ("gray",), 3: ("rgb", "ycbcr"), 4: ("cmyk", "ycck")}


def override_colorspace(data, target: str) -> bytes:
    """Return the stream with markers that declare it ``target``.

    A decoder learns a JPEG's colorspace only from markers (T.81 leaves
    it to the application): a JFIF APP0 marker means YCbCr, and an
    Adobe APP14 marker's transform flag means RGB/CMYK (0), YCbCr (1) or
    YCCK (2). A container that knows better, such as a TIFF whose
    PhotometricInterpretation is RGB around a JPEG without those
    markers, needs the stream read as it says. This drops any JFIF and
    Adobe markers and puts one Adobe marker carrying ``target`` after
    SOI; the frame and scan data are untouched.
    """
    data = bytes(data)
    fh = frame_header(data)
    if fh is None:
        raise ValueError("jpeg: no frame header found")
    if target not in _STORABLE.get(fh.components, ()):
        raise NotImplementedError(
            f"jpeg: a {fh.components}-component stream cannot be decoded "
            f"as {target}")
    if target == "gray":
        return data
    adobe = (b"\xff\xee\x00\x0eAdobe\x00\x64\x00\x00\x00\x00"
             + bytes([_ADOBE_TRANSFORM[target]]))
    parts = [b"\xff\xd8", adobe]
    for marker, start, end in _segments(data):
        if marker == 0xEE and data[start + 4:start + 9] == b"Adobe":
            continue
        if marker == 0xE0 and data[start + 4:start + 9] == b"JFIF\x00" \
                and target != "ycbcr":
            continue
        parts.append(data[start:end])
    return b"".join(parts)


# The JFIF APP0 segment libjpeg writes by default (ITU-T T.871): version
# 1.01, no density units, aspect ratio 1:1, no thumbnail.
_JFIF_APP0 = (b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01"
              b"\x00\x00")
_PROGRESSIVE_SOF = (0xC2, 0xC6, 0xCA, 0xCE)


def label_ycbcr(data) -> bytes:
    """Return a three-component stream labeled the way JFIF labels YCbCr.

    TurboJPEG writes YCbCr samples only through RGB storage, which
    names the components R, G and B (82, 71, 66) and carries an Adobe
    APP14 marker with transform 0. Decoders infer the colorspace in
    different orders: libjpeg-turbo trusts the JFIF and Adobe markers
    first, while IJG libjpeg 9 trusts the component ids first and reads
    R, G, B as RGB whatever the markers say. This rewrites the frame
    header's component ids and every scan header's component selectors
    to 1, 2 and 3 and replaces the JFIF and Adobe markers by a JFIF APP0
    marker after SOI: the ids and the marker libjpeg itself writes for a
    YCbCr JPEG (T.871), which IJG libjpeg 9 and libjpeg-turbo read as YCbCr. Tables,
    sampling factors and entropy-coded data are untouched.
    """
    data = bytes(data)
    n = len(data)
    if n < 4 or data[:2] != b"\xff\xd8":
        raise ValueError("jpeg: not a JPEG stream")
    parts = [b"\xff\xd8", _JFIF_APP0]
    ids = None
    progressive = False
    i = 2
    while i < n:
        if i + 1 >= n or data[i] != 0xFF:
            raise ValueError("jpeg: corrupt marker segment")
        marker = data[i + 1]
        if marker == 0xFF:              # fill byte before a marker
            i += 1
            continue
        if marker == 0xD9:              # EOI and anything after it
            parts.append(data[i:])
            break
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            parts.append(data[i:i + 2])
            i += 2
            continue
        if i + 3 >= n:
            raise ValueError("jpeg: truncated marker segment")
        end = i + 2 + ((data[i + 2] << 8) | data[i + 3])
        if end > n:
            raise ValueError("jpeg: truncated marker segment")
        seg = bytearray(data[i:end])
        if (marker == 0xE0 and seg[4:9] == b"JFIF\x00") \
                or (marker == 0xEE and seg[4:9] == b"Adobe"):
            i = end
            continue
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if len(seg) < 19 or seg[9] != 3:
                raise ValueError(
                    "jpeg: only a three-component frame is labeled YCbCr")
            ids = {seg[10 + 3 * k]: k + 1 for k in range(3)}
            if len(ids) != 3:
                raise ValueError("jpeg: repeated component ids")
            for k in range(3):
                seg[10 + 3 * k] = k + 1
            progressive = marker in _PROGRESSIVE_SOF
            parts.append(bytes(seg))
            i = end
            continue
        if marker != 0xDA:
            parts.append(bytes(seg))
            i = end
            continue
        # Scan header (T.81 B.2.3): Ns, then a selector and table byte
        # per component.
        if ids is None or len(seg) < 5:
            raise ValueError("jpeg: scan before frame header")
        ns = seg[4]
        try:
            for k in range(ns):
                seg[5 + 2 * k] = ids[seg[5 + 2 * k]]
        except (KeyError, IndexError):
            raise ValueError("jpeg: scan names an unknown component")
        parts.append(bytes(seg))
        if not progressive and ns == 3:
            # A sequential (or lossless) frame codes each component in
            # exactly one scan, so an interleaved scan of all three is
            # the frame's only one: the rest is entropy-coded data,
            # perhaps a DNL marker, and EOI.
            parts.append(data[end:])
            break
        # Skip the entropy-coded data to the next marker: 0xFF is
        # followed by 0x00 (a stuffed byte) or RSTn inside it (B.1.1.5).
        j = end
        while True:
            k = data.find(b"\xff", j)
            if k < 0 or k + 1 >= n:
                k = n
                break
            nxt = data[k + 1]
            if nxt == 0x00 or 0xD0 <= nxt <= 0xD7 or nxt == 0xFF:
                j = k + 1 if nxt == 0xFF else k + 2
                continue
            break
        parts.append(data[end:k])
        i = k
    if ids is None:
        raise ValueError("jpeg: no frame header found")
    return b"".join(parts)


# The output a stream decodes to when neither colorspace nor
# outcolorspace is given: what the stream declares, converted to RGB or
# CMYK. The decoders read this directly on that common path.
DEFAULT_OUTPUT = {"gray": "gray", "rgb": "rgb", "ycbcr": "rgb",
                  "cmyk": "cmyk", "ycck": "cmyk"}


def decode_plan(stream_cs, colorspace, outcolorspace, packed):
    """Return (declared, pixels): the colorspace the stream must declare
    and the packed pixel layout to decode it to.

    ``colorspace`` and ``outcolorspace`` are canonical names (or None)
    with libjpeg's meanings, as ``imagecodecs.jpeg8_decode`` passes them
    through: ``colorspace`` is how the stored components are to be
    read, ``outcolorspace`` the output, and an output left unset with a
    colorspace given is that same colorspace, unconverted. TurboJPEG
    outputs only packed RGB, grayscale and CMYK, so unconverted YCbCr or
    YCCK is produced by declaring the stream RGB or CMYK: libjpeg then
    copies the stored components through, which is what an unconverted
    YCbCr output is.
    """
    stored = colorspace if colorspace is not None else stream_cs
    if stored is None:
        raise NotImplementedError(
            "jpeg decode: unsupported component layout (TurboJPEG decodes "
            "1-, 3- and 4-component JPEG)")
    if outcolorspace is None:
        outcolorspace = (colorspace if colorspace is not None
                         else DEFAULT_OUTPUT[stored])
    if outcolorspace == stored and stored in ("ycbcr", "ycck"):
        raw = "rgb" if stored == "ycbcr" else "cmyk"
        return raw, raw
    if outcolorspace not in packed:
        raise NotImplementedError(
            f"jpeg decode: TurboJPEG cannot convert {stored} to "
            f"{outcolorspace}")
    return stored, outcolorspace


def frame_header(data) -> FrameHeader | None:
    """Read the first frame header of a JPEG stream, or ``None``.

    Walks the marker segments from SOI (T.81 B.1.1.4: every marker
    other than SOI, EOI, RSTn and TEM carries a two-byte length) and
    stops at the first SOFn or at SOS. Never touches entropy-coded data.
    """
    mv = memoryview(data)
    n = len(mv)
    if n < 4 or mv[0] != 0xFF or mv[1] != 0xD8:
        return None
    i = 2
    while i + 3 < n:
        if mv[i] != 0xFF:
            return None
        marker = mv[i + 1]
        if marker == 0xFF:          # fill byte before a marker
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker in (0xD9, 0xDA):  # EOI or SOS before any frame
            return None
        length = (mv[i + 2] << 8) | mv[i + 3]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 9 >= n:
                return None
            return FrameHeader(
                sof=marker, precision=mv[i + 4],
                height=(mv[i + 5] << 8) | mv[i + 6],
                width=(mv[i + 7] << 8) | mv[i + 8],
                components=mv[i + 9])
        i += 2 + length
    return None
