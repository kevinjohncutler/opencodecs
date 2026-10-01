"""BMP header variants, channel scaling and low-depth palettes.

Every file is built byte by byte from the layouts in Microsoft's
BITMAPINFOHEADER / BITMAPV4HEADER documentation and the Adobe 52- and
56-byte headers, so the expected pixels come from the format rather
than from our own encoder. Pillow and imagecodecs are used as second
references where they decode the case correctly.

* A 40-byte header with BI_BITFIELDS is followed by exactly three DWORD
  masks; nothing after them is an alpha mask.
* The 52-byte (V2) header carries the R, G, B masks inside the header,
  the 56-byte (V3) header also the alpha mask, at file offset 54.
* Channels of any width scale as round(v * 255 / (2**n - 1)), the PNG
  specification's reference sample-depth equation.
"""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from opencodecs import get_codec
from opencodecs._bmp_codec import BmpError

W, H = 5, 3
RNG = np.random.default_rng(1)
RGB = RNG.integers(1, 255, (H, W, 3), dtype=np.uint8)
ALPHA = RNG.integers(1, 255, (H, W), dtype=np.uint8)
MASKS = (0xFF0000, 0xFF00, 0xFF)


def _file(header: bytes, extra: bytes, pixels: bytes) -> bytes:
    off = 14 + len(header) + len(extra)
    return (b"BM" + struct.pack("<IHHI", off + len(pixels), 0, 0, off)
            + header + extra + pixels)


def _base(info_size, bpp, compression, nbytes, w=W, h=H, clr_used=0):
    return struct.pack("<IiiHHIIiiII", info_size, w, h, 1, bpp,
                       compression, nbytes, 2835, 2835, clr_used, 0)


def _pixels32(with_alpha):
    """Bottom-up 32-bit words A<<24 | R<<16 | G<<8 | B. Without alpha the
    top byte holds junk (0x5A), which a decoder must not show."""
    rows = []
    for y in range(H - 1, -1, -1):
        for x in range(W):
            a = int(ALPHA[y, x]) if with_alpha else 0x5A
            rows.append(struct.pack(
                "<I", (a << 24) | (int(RGB[y, x, 0]) << 16)
                | (int(RGB[y, x, 1]) << 8) | int(RGB[y, x, 2])))
    return b"".join(rows)


def _bitfields32(info_size, alpha_mask=0, with_alpha=False, compression=3):
    pix = _pixels32(with_alpha)
    masks = struct.pack("<III", *MASKS)
    base = _base(info_size, 32, compression, len(pix))
    if info_size == 40:
        extra = masks
        if compression == 6:
            extra += struct.pack("<I", alpha_mask)
        return _file(base, extra, pix)
    hdr = base + masks
    if info_size >= 56:
        hdr += struct.pack("<I", alpha_mask)
    if info_size == 108:
        hdr += b"BGRs" + b"\0" * 48
    return _file(hdr, b"", pix)


def _decode(blob, **kw):
    return get_codec("bmp").decode(blob, **kw)


def _pillow(blob):
    try:
        from PIL import Image
    except ImportError:
        return None
    return np.asarray(Image.open(io.BytesIO(blob)))


def _imagecodecs():
    try:
        import imagecodecs
    except ImportError:
        return None
    return imagecodecs


def test_40_byte_header_reads_three_masks_and_no_alpha():
    blob = _bitfields32(40)
    got = _decode(blob)
    assert got.shape == (H, W, 3)
    np.testing.assert_array_equal(got, RGB)
    pil = _pillow(blob)
    if pil is not None:
        np.testing.assert_array_equal(got, pil[..., :3])


def test_40_byte_header_fourth_mask_only_before_the_pixels():
    """A fourth DWORD that sits in a gap before the pixel data and is
    disjoint from the color masks is honored as alpha (as imagecodecs
    does); when the pixels start right after three masks it is not."""
    pix = _pixels32(True)
    base = _base(40, 32, 3, len(pix))
    four = struct.pack("<IIII", *MASKS, 0xFF000000)
    got = _decode(_file(base, four, pix))
    np.testing.assert_array_equal(got[..., :3], RGB)
    np.testing.assert_array_equal(got[..., 3], ALPHA)
    # Same bytes, but the pixel data begins after three masks: the
    # "fourth mask" is now the first pixel and must not become alpha.
    three = struct.pack("<III", *MASKS)
    blob = _file(base, three, struct.pack("<I", 0xFF000000) + pix[4:])
    assert _decode(blob).shape == (H, W, 3)


def test_40_byte_header_alpha_bitfields_reads_four_masks():
    blob = _bitfields32(40, alpha_mask=0xFF000000, with_alpha=True,
                        compression=6)
    got = _decode(blob)
    assert got.shape == (H, W, 4)
    np.testing.assert_array_equal(got[..., :3], RGB)
    np.testing.assert_array_equal(got[..., 3], ALPHA)


@pytest.mark.parametrize("info_size,alpha_mask,with_alpha", [
    (52, 0, False),
    (56, 0xFF000000, True),
    (56, 0, False),
    (108, 0xFF000000, True),
    (108, 0, False),
])
def test_masks_inside_v2_v3_v4_headers(info_size, alpha_mask, with_alpha):
    blob = _bitfields32(info_size, alpha_mask, with_alpha)
    got = _decode(blob)
    assert got.shape == (H, W, 4 if with_alpha else 3)
    np.testing.assert_array_equal(got[..., :3], RGB)
    if with_alpha:
        np.testing.assert_array_equal(got[..., 3], ALPHA)
    ic = _imagecodecs()
    if ic is not None:      # imagecodecs reads these headers correctly
        np.testing.assert_array_equal(got, ic.bmp_decode(blob))
    pil = _pillow(blob)
    if pil is not None:
        np.testing.assert_array_equal(got, pil)


def _pixels16(values):
    h, w = values.shape
    stride = ((w * 16 + 31) // 32) * 4
    rows = [values[y].astype("<u2").tobytes().ljust(stride, b"\0")
            for y in range(h - 1, -1, -1)]
    return b"".join(rows)


def _scale(v, n):
    return (np.asarray(v, np.int64) * 255 * 2 + (2**n - 1)) // (2 * (2**n - 1))


def test_rgb565_in_v2_header():
    """This header made the old decoder read pixels as masks and raise
    OverflowError."""
    v = RNG.integers(0, 65536, (H, W)).astype(np.uint16)
    pix = _pixels16(v)
    blob = _file(_base(52, 16, 3, len(pix)) + struct.pack(
        "<III", 0xF800, 0x07E0, 0x001F), b"", pix)
    got = _decode(blob)
    want = np.stack([_scale(v >> 11, 5), _scale((v >> 5) & 63, 6),
                     _scale(v & 31, 5)], -1)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("n,mask", [(5, 0x001F), (6, 0x07E0)])
def test_every_5_and_6_bit_level_scales_to_the_rounded_value(n, mask):
    levels = np.arange(2**n)
    shift = (mask & -mask).bit_length() - 1
    v = (levels << shift).astype(np.uint16).reshape(1, -1)
    pix = _pixels16(v)
    blob = _file(_base(40, 16, 3, len(pix), w=v.shape[1], h=1),
                 struct.pack("<III", 0xF800, 0x07E0, 0x001F), pix)
    got = _decode(blob)
    ch = 2 if mask == 0x001F else 1
    want = np.round(levels * 255 / (2**n - 1)).astype(np.uint8)
    np.testing.assert_array_equal(got[0, :, ch], want)


def test_one_bit_alpha_is_opaque_at_255():
    """ARGB1555 in a V4 header: alpha 1 must be 255, not 128."""
    v = np.array([[0x8000 | 0x7C00, 0x001F]], np.uint16)
    pix = _pixels16(v)
    hdr = (_base(108, 16, 3, len(pix), w=2, h=1)
           + struct.pack("<IIII", 0x7C00, 0x03E0, 0x001F, 0x8000)
           + b"BGRs" + b"\0" * 48)
    got = _decode(_file(hdr, b"", pix))
    np.testing.assert_array_equal(got[0, 0], [255, 0, 0, 255])
    np.testing.assert_array_equal(got[0, 1], [0, 0, 255, 0])


def test_a2r10g10b10_scales_both_ways():
    a, r, g, b = 3, 1023, 512, 1
    word = (a << 30) | (r << 20) | (g << 10) | b
    pix = struct.pack("<I", word)
    hdr = (_base(108, 32, 3, 4, w=1, h=1)
           + struct.pack("<IIII", 0x3FF00000, 0x000FFC00, 0x000003FF,
                         0xC0000000)
           + b"BGRs" + b"\0" * 48)
    got = _decode(_file(hdr, b"", pix))
    np.testing.assert_array_equal(
        got[0, 0], [255, _scale(512, 10), _scale(1, 10), 255])


def test_mask_wider_than_the_pixel_is_an_error():
    pix = _pixels16(np.zeros((H, W), np.uint16))
    blob = _file(_base(40, 16, 3, len(pix)),
                 struct.pack("<III", 0xF80000, 0x07E0, 0x001F), pix)
    with pytest.raises(BmpError, match="does not fit"):
        _decode(blob)


def _paletted(bpp, idx, palette):
    h, w = idx.shape
    stride = ((w * bpp + 31) // 32) * 4
    rows = []
    for y in range(h - 1, -1, -1):
        bits = "".join(format(int(i), f"0{bpp}b") for i in idx[y])
        bits = bits.ljust(stride * 8, "0")
        rows.append(int(bits, 2).to_bytes(stride, "big"))
    pix = b"".join(rows)
    pal = palette[:1 << bpp].tobytes()
    return _file(_base(40, bpp, 0, len(pix), w=w, h=h), pal, pix)


def _palette(gray):
    p = np.zeros((256, 4), np.uint8)
    if gray:
        p[:, 0] = p[:, 1] = p[:, 2] = np.arange(256)
    else:
        p[:, :3] = np.random.default_rng(3).integers(0, 256, (256, 3))
    return p


@pytest.mark.parametrize("bpp", [1, 2, 4, 8])
@pytest.mark.parametrize("gray", [True, False])
def test_uncompressed_low_depth_palettes(bpp, gray):
    idx = np.random.default_rng(bpp).integers(0, 1 << bpp, (3, 13))
    pal = _palette(gray)
    blob = _paletted(bpp, idx, pal)
    rgb = pal[idx][..., 2::-1][..., :3]
    np.testing.assert_array_equal(_decode(blob, asrgb=False), idx)
    np.testing.assert_array_equal(_decode(blob, asrgb=True), rgb)
    np.testing.assert_array_equal(_decode(blob), idx if gray else rgb)
    ic = _imagecodecs()
    if ic is not None and bpp != 2:   # imagecodecs has no 2-bit
        np.testing.assert_array_equal(_decode(blob), ic.bmp_decode(blob))
        np.testing.assert_array_equal(_decode(blob, asrgb=True),
                                      ic.bmp_decode(blob, asrgb=True))
    # Pillow has no 2-bit BMP and fails on a 16-entry gray-ramp palette
    # at 4 bits ("codec configuration error"), so it checks the rest.
    if bpp != 2 and not (gray and bpp == 4):
        try:
            from PIL import Image
        except ImportError:
            return
        want = np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
        np.testing.assert_array_equal(_decode(blob, asrgb=True), want)


def test_ppm_matches_imagecodecs():
    img = np.random.default_rng(4).integers(0, 256, (3, 5, 3), np.uint8)
    blob = get_codec("bmp").encode(img, ppm=1000)
    assert struct.unpack("<ii", blob[38:46]) == (1000, 1000)
    assert struct.unpack("<ii", get_codec("bmp").encode(img)[38:46]) == (
        3780, 3780)
    ic = _imagecodecs()
    if ic is not None:
        for arr in (img, img[..., 0], np.dstack([img, img[..., :1]])):
            assert get_codec("bmp").encode(arr, ppm=1000) == ic.bmp_encode(
                arr, ppm=1000)


@pytest.mark.parametrize("ppm, want", [
    (0, 1), (-1, 1), (-5, 1), (-2**31, 1), (0.6, 1), (1, 1), (1.5, 1),
    (2, 2), (2**31 - 1, 2**31 - 1)])
def test_ppm_below_one_writes_one(ppm, want):
    """A resolution below one pixel per meter is written as 1, the value
    imagecodecs writes for it (fractions truncate first)."""
    img = np.random.default_rng(5).integers(0, 256, (2, 3, 3), np.uint8)
    blob = get_codec("bmp").encode(img, ppm=ppm)
    assert struct.unpack("<ii", blob[38:46]) == (want, want)
    ic = _imagecodecs()
    if ic is not None:
        assert blob == bytes(ic.bmp_encode(img, ppm=ppm))


@pytest.mark.parametrize("info_size", [40, 56, 108])
@pytest.mark.parametrize("masks", [
    (0xF0F000, 0xFF00, 0xFF, 0),           # a gap inside red
    (0xFF0000, 0xFF00, 0xFF, 0x81000000),  # a gap inside alpha
])
def test_non_contiguous_mask_is_an_error(info_size, masks):
    """Microsoft's BITMAPV4HEADER documentation: "The bits in the masks
    must be contiguous". A gapped mask used to reach the scaling table
    with values past its end and raise numpy's IndexError."""
    pix = _pixels32(True)
    base = _base(info_size, 32, 3 if not masks[3] or info_size > 40 else 6,
                 len(pix))
    packed = struct.pack("<IIII", *masks)
    if info_size == 40:
        blob = _file(base, packed if masks[3] else packed[:12], pix)
    else:
        blob = _file(base + packed + b"\0" * (info_size - 56), b"", pix)
    with pytest.raises(BmpError, match="contiguous"):
        _decode(blob)


def test_gapped_fourth_dword_is_not_taken_for_alpha():
    """The fourth DWORD after a 40-byte BI_BITFIELDS header is honored
    as alpha only when it is a valid mask; a gapped one is ignored, not
    an error, because it is normally pixel data."""
    pix = _pixels32(True)
    four = struct.pack("<IIII", *MASKS, 0x81000000)
    got = _decode(_file(_base(40, 32, 3, len(pix)), four, pix))
    np.testing.assert_array_equal(got, RGB)


def test_every_truncation_is_refused_with_bmperror():
    """The bmpsuite rule: malformed input raises BmpError. A file cut
    inside the fourth DWORD after the masks used to raise struct.error,
    and a file shorter than its rows (with biSizeImage set) raised
    numpy's reshape ValueError."""
    pix = _pixels32(True)
    four = struct.pack("<IIII", *MASKS, 0xFF000000)
    blobs = [_file(_base(40, 32, 3, len(pix)), four, pix),
             _bitfields32(108, 0xFF000000, True),
             _paletted(4, np.arange(H * W).reshape(H, W) % 16,
                       _palette(False))]
    for blob in blobs:
        for n in range(len(blob)):
            try:
                _decode(blob[:n])
            except BmpError:
                pass


def _pixels24(rgb, order):
    """Bottom-up 24-bit pixels with each channel at the byte ``order``
    gives (R, G, B byte positions within the 3-byte pixel)."""
    h, w, _ = rgb.shape
    rows = []
    for y in range(h - 1, -1, -1):
        row = bytearray(3 * w)
        for x in range(w):
            for ch in range(3):
                row[3 * x + order[ch]] = int(rgb[y, x, ch])
        rows.append(bytes(row) + b"\0" * (-len(row) % 4))
    return b"".join(rows)


@pytest.mark.parametrize("info_size", [40, 108])
@pytest.mark.parametrize("masks, order", [
    ((0xFF0000, 0xFF00, 0xFF), (2, 1, 0)),     # the plain BGR layout
    ((0xFF, 0xFF00, 0xFF0000), (0, 1, 2)),     # red in the low byte
])
def test_24_bit_bitfields_decode_through_their_masks(info_size, masks,
                                                     order):
    """Microsoft documents BI_BITFIELDS for 16 and 32 bits, and
    imagecodecs refuses 24, but opencodecs 0.4.0 read 24-bit files with
    the plain BGR masks (it ignored the masks), so they still decode;
    other masks are now honored. The expected pixels are the ones each
    mask selects from the bytes built here. A V4 header's alpha mask
    selects a fourth byte a 24-bit pixel lacks, so it gives no alpha."""
    pix = _pixels24(RGB, order)
    base = _base(info_size, 24, 3, len(pix))
    packed = struct.pack("<III", *masks)
    if info_size == 40:
        blob = _file(base, packed, pix)
    else:
        blob = _file(base + packed + struct.pack("<I", 0xFF000000)
                     + b"BGRs" + b"\0" * 48, b"", pix)
    np.testing.assert_array_equal(_decode(blob), RGB)
    ic = _imagecodecs()
    if ic is not None:
        with pytest.raises(Exception, match="16 or 32"):
            ic.bmp_decode(blob)


def _pillow_refuses(blob):
    """Pillow's BMP reader refuses a bitfield layout it does not know."""
    try:
        from PIL import Image
    except ImportError:
        return
    with pytest.raises(OSError, match="bitfields layout"):
        Image.open(io.BytesIO(blob)).load()


@pytest.mark.parametrize("info_size", [40, 108])
@pytest.mark.parametrize("masks", [
    (0, 0, 0),
    (0xFF0000, 0, 0xFF),
    (0, 0xFF00, 0xFF),
])
def test_24_bit_bitfields_with_a_zero_color_mask_is_an_error(info_size,
                                                             masks):
    """24-bit bitfields are outside Microsoft's documentation, so they
    are read only when every color has a mask. A zero mask would turn
    real pixels black without an error; Pillow refuses these files
    ("Unsupported BMP bitfields layout") and imagecodecs refuses every
    24-bit bitfield file."""
    pix = _pixels24(RGB, (2, 1, 0))
    base = _base(info_size, 24, 3, len(pix))
    packed = struct.pack("<III", *masks)
    if info_size == 40:
        blob = _file(base, packed, pix)
    else:
        blob = _file(base + packed + struct.pack("<I", 0)
                     + b"BGRs" + b"\0" * 48, b"", pix)
    with pytest.raises(BmpError, match="nonzero mask"):
        _decode(blob)
    _pillow_refuses(blob)
    ic = _imagecodecs()
    if ic is not None:
        with pytest.raises(Exception, match="16 or 32"):
            ic.bmp_decode(blob)


@pytest.mark.parametrize("bpp", [16, 24, 32])
@pytest.mark.parametrize("compression", [3, 6])
def test_pixel_offset_inside_the_masks_is_an_error(bpp, compression):
    """Microsoft's BITMAPINFOHEADER documentation puts the three
    BI_BITFIELDS masks (four for BI_ALPHABITFIELDS) between a 40-byte
    header and the pixels, and bfOffBits gives where the pixels start.
    A file whose bfOffBits is 54 has no room for masks: what would be
    read as masks are its first pixels. With black first pixels those
    are zero masks, which would decode the image to black without an
    error."""
    w = 6
    stride = (w * bpp + 31) // 32 * 4
    rows = [b"\0" * stride, bytes(range(1, stride + 1))]  # black bottom row
    pix = b"".join(rows)
    base = _base(40, bpp, compression, len(pix), w=w, h=2)
    blob = b"BM" + struct.pack("<IHHI", 54 + len(pix), 0, 0, 54) + base + pix
    with pytest.raises(BmpError, match="inside the color masks"):
        _decode(blob)
    if bpp == 24 and compression == 3:
        _pillow_refuses(blob)


def test_pixel_offset_just_past_the_masks_decodes():
    """The boundary of the check above: pixels that start right after
    the three masks are read. imagecodecs takes the first pixel word as a
    fourth, alpha mask here, so only its color channels agree."""
    blob = _bitfields32(40)
    off = struct.unpack("<I", blob[10:14])[0]
    assert off == 14 + 40 + 12
    np.testing.assert_array_equal(_decode(blob), RGB)


def test_paletted_bitfields_is_an_error():
    """A bitfield compression has no meaning for palette indices;
    imagecodecs refuses it the same way."""
    blob = bytearray(_paletted(8, np.arange(H * W).reshape(H, W),
                               _palette(False)))
    struct.pack_into("<I", blob, 30, 3)       # biCompression = BI_BITFIELDS
    with pytest.raises(BmpError, match="16, 24 or 32"):
        _decode(bytes(blob))
    ic = _imagecodecs()
    if ic is not None:
        with pytest.raises(Exception, match="16 or 32"):
            ic.bmp_decode(bytes(blob))


def test_rle_cut_between_codes_still_decodes():
    """The documented exception to the truncation rule: an RLE8 stream
    may stop early, and the pixels it never reaches keep index 0."""
    idx = np.zeros((2, 4), np.uint8)
    idx[1] = 7                                # bottom row, stored first
    rle = bytes([4, 7, 0, 0, 4, 9, 0, 1])     # run, EOL, run, end
    pal = _palette(False)[:256].tobytes()
    full = _file(_base(40, 8, 1, len(rle), w=4, h=2), pal, rle)
    top = _decode(full, asrgb=False)
    want = np.array([[9] * 4, [7] * 4], np.uint8)
    np.testing.assert_array_equal(top, want)
    cut = full[:len(full) - 4]                # drop the second row's run
    want[0] = 0
    np.testing.assert_array_equal(_decode(cut, asrgb=False), want)


@pytest.mark.parametrize("make", ["24", "gray8"])
def test_decode_out_must_match_like_imagecodecs(make):
    """``out`` follows imagecodecs: same dtype, C-contiguous, and the
    decoded shape apart from length-1 axes; anything else raises
    ValueError instead of broadcasting or casting into it."""
    codec = get_codec("bmp")
    img = RGB if make == "24" else RGB[..., 0]
    blob = codec.encode(img)
    ok = np.empty(img.shape, np.uint8)
    assert codec.decode(blob, out=ok) is ok
    np.testing.assert_array_equal(ok, img)
    got = codec.decode(blob, out=np.empty((1,) + img.shape, np.uint8))
    np.testing.assert_array_equal(got, img)
    bad = [np.empty((2,) + img.shape, np.uint8),          # would broadcast
           np.empty(img.shape, np.float32),               # would cast
           np.empty((img.size,), np.uint8),               # same size
           np.empty((2 * H,) + img.shape[1:], np.uint8)[::2]]
    ic = _imagecodecs()
    for out in bad:
        with pytest.raises(ValueError):
            codec.decode(blob, out=out)
        if ic is not None:
            with pytest.raises(ValueError):
                ic.bmp_decode(blob, out=out)
