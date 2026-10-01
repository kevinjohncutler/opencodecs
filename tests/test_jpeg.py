"""JPEG codec (libjpeg-turbo, TurboJPEG v3) against independent references.

Each test pins behavior to something other than our own round trip:
imagecodecs 2026.8.16 (``jpeg8_encode`` / ``jpeg8_decode`` link
libjpeg-turbo directly), the frame header as ITU-T T.81 B.2.2 defines it
(parsed here by hand), or streams assembled by hand from the spec.

Covers the 0.4.1 fixes:

* ``JpegCodec.encode`` forwards ``subsampling``, ``lossless`` and every
  other ``jpeg8_encode`` parameter, and unknown options raise.
* 12-bit DCT, 2- to 16-bit lossless, and CMYK/YCCK streams decode.
* (H, W, 1) is grayscale.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc

J = pytest.importorskip("opencodecs.codecs._jpeg")
ic = pytest.importorskip("imagecodecs")
if not getattr(getattr(ic, "JPEG8", None), "available", False):
    pytest.skip("imagecodecs has no jpeg8", allow_module_level=True)

ALL_PRECISIONS = bool(getattr(ic.JPEG8, "all_precisions", False))


def _we_have_all_precisions() -> bool:
    """libjpeg-turbo 3.1 added lossless precisions other than 8/12/16."""
    try:
        J.encode(np.zeros((8, 8), np.uint8), lossless=True, bitspersample=5)
    except J.JpegError:
        return False
    return True


def _sof(stream: bytes):
    """(marker, precision, [(id, h, v), ...]) from the first SOFn."""
    i = 2
    while i < len(stream):
        assert stream[i] == 0xFF
        m = stream[i + 1]
        length = stream[i + 2] << 8 | stream[i + 3]
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            seg = stream[i + 4:i + 2 + length]
            nf = seg[5]
            comps = [(seg[6 + 3 * k], seg[7 + 3 * k] >> 4, seg[7 + 3 * k] & 15)
                     for k in range(nf)]
            return m, seg[0], comps
        i += 2 + length
    raise AssertionError("no SOF marker")


def _rng(seed=0):
    return np.random.default_rng(seed)


def _rgb(shape=(37, 53, 3), seed=0):
    return _rng(seed).integers(0, 256, shape, dtype=np.uint8)


# ---------------------------------------------------------------------------
# JpegCodec.encode forwards every parameter
# ---------------------------------------------------------------------------

# Luma sampling factors (H, V) per T.81 B.2.2 for each subsampling.
_LUMA = {"444": (1, 1), "422": (2, 1), "420": (2, 2), "440": (1, 2),
         "411": (4, 1)}


@pytest.mark.parametrize("ss", ["444", "422", "420", "440", "411"])
def test_codec_encode_honors_subsampling(ss):
    rgb = _rgb()
    codec = oc.get_codec("jpeg")
    enc = codec.encode(rgb, level=90, subsampling=ss)
    marker, precision, comps = _sof(enc)
    assert (marker, precision) == (0xC0, 8)
    assert comps[0][1:] == _LUMA[ss]
    assert [c[1:] for c in comps[1:]] == [(1, 1), (1, 1)]
    assert enc == ic.jpeg8_encode(rgb, level=90, subsampling=ss)
    # The tuple spelling imagecodecs accepts.
    assert codec.encode(rgb, level=90, subsampling=_LUMA[ss]) == enc
    # The top-level path goes through the same adapter.
    assert oc.write(None, rgb, format="jpeg", level=90, subsampling=ss) == enc


def test_codec_encode_honors_lossless():
    rgb = _rgb()
    enc = oc.get_codec("jpeg").encode(rgb, lossless=True)
    assert _sof(enc)[0] == 0xC3          # SOF3: lossless, Huffman
    np.testing.assert_array_equal(ic.jpeg8_decode(enc), rgb)
    assert enc == ic.jpeg8_encode(rgb, lossless=True)


def test_codec_encode_honors_colorspace_options():
    rgb = _rgb()
    codec = oc.get_codec("jpeg")
    assert codec.encode(rgb, outcolorspace="rgb") == \
        ic.jpeg8_encode(rgb, outcolorspace="rgb")
    assert codec.encode(rgb, outcolorspace="gray") == \
        ic.jpeg8_encode(rgb, outcolorspace="gray")
    assert codec.encode(rgb, optimize=True) == \
        ic.jpeg8_encode(rgb, optimize=True)


@pytest.mark.parametrize("name", ["jpeg", "mozjpeg"])
def test_codec_rejects_unknown_options(name):
    if not oc.has_codec(name):
        pytest.skip(f"{name} not built")
    codec = oc.get_codec(name)
    rgb = _rgb()
    with pytest.raises(TypeError):
        codec.encode(rgb, bogus_kw=1)
    with pytest.raises(TypeError):
        codec.decode(ic.jpeg8_encode(rgb), bogus_kw=1)


# ---------------------------------------------------------------------------
# Encode parameters that cannot be honored raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs, exc", [
    (dict(smoothing=10), NotImplementedError),
    (dict(predictor=2), ValueError),                    # needs lossless
    (dict(lossless=True, subsampling="420"), ValueError),
    (dict(lossless=True, outcolorspace="ycbcr"), ValueError),
    (dict(lossless=True, optimize=False), ValueError),  # always optimal
    (dict(lossless=True, predictor=8), ValueError),
    (dict(outcolorspace="rgb", subsampling="gray"), ValueError),
    (dict(outcolorspace="cmyk"), ValueError),
    (dict(colorspace="ycck"), NotImplementedError),
    (dict(colorspace="ycbcr", outcolorspace="rgb"), NotImplementedError),
    (dict(colorspace="ycbcr", subsampling="gray"), ValueError),
    (dict(colorspace="ycbcr", lossless=True, subsampling="420"), ValueError),
    (dict(colorspace="cmyk"), ValueError),              # 4 samples needed
    (dict(colorspace="nonsense"), ValueError),
    (dict(subsampling="433"), ValueError),
    (dict(bitspersample=12), ValueError),               # uint8 holds 2-8
])
def test_encode_raises_instead_of_dropping(kwargs, exc):
    with pytest.raises(exc):
        J.encode(_rgb(), **kwargs)


def test_encode_rejects_values_wider_than_precision():
    u16 = _rng().integers(0, 4096, (9, 11), dtype=np.uint16)
    u16[3, 4] = 60000
    # 12-bit lossy cannot hold 60000; imagecodecs writes a stream that
    # decodes to something else (0 here), so silence would lose data.
    with pytest.raises(ValueError, match="does not fit"):
        J.encode(u16)
    with pytest.raises(ValueError, match="does not fit"):
        J.encode(u16, lossless=True, bitspersample=12)
    u8 = _rng().integers(0, 256, (9, 11), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not fit"):
        J.encode(u8, lossless=True, bitspersample=5)
    with pytest.raises(ValueError):
        J.encode(_rng().integers(0, 4096, (9, 11), dtype=np.uint16),
                 optimize=False)  # 12-bit always uses optimized tables


def test_encode_rejects_non_image_dtype_and_shape():
    with pytest.raises(J.JpegError):
        J.encode(np.zeros((8, 8), np.float32))
    with pytest.raises(J.JpegError):
        J.encode(np.zeros((8, 8, 2), np.uint8))
    # A list of Python ints is a signed integer array: rejected, as
    # imagecodecs rejects it, not cast to uint8 as 0.4.0 did.
    with pytest.raises(Exception):
        ic.jpeg8_encode([[1, 2], [3, 4]])
    with pytest.raises(J.JpegError, match="dtype"):
        J.encode([[1, 2], [3, 4]])
    with pytest.raises(J.JpegError, match="dtype"):
        J.encode([[1, 2], [3, 300]])


@pytest.mark.parametrize("kwargs", [
    dict(subsampling="420"),
    dict(subsampling=(2, 1)),
    dict(outcolorspace="ycbcr"),
    dict(outcolorspace="gray"),
])
def test_lossless_limits_are_turbojpegs(kwargs):
    # T.81 Annex H lets a lossless frame carry any sampling factors and
    # any stored colorspace; TurboJPEG's lossless mode writes neither,
    # and imagecodecs, the reference, ignores both: it writes the same
    # unsubsampled RGB frame as with no options. The error names the
    # TurboJPEG limit rather than the format.
    rgb = _rgb()
    ref = ic.jpeg8_encode(rgb, lossless=True, **kwargs)
    assert ref == ic.jpeg8_encode(rgb, lossless=True)
    assert [c[1:] for c in _sof(ref)[2]] == [(1, 1)] * 3
    with pytest.raises(ValueError, match="TurboJPEG's lossless mode"):
        J.encode(rgb, lossless=True, **kwargs)
    assert J.encode(rgb, lossless=True) == ref


def test_lossless_optimize_true_is_what_turbojpeg_writes():
    # TurboJPEG's lossless mode computes the Huffman tables from the
    # data on every encode, so optimize=True is already honored; the
    # reference is imagecodecs, which accepts it and writes the same
    # bytes as without it. Two images with different statistics get
    # different DHT code-length counts, which shows the tables are
    # computed rather than fixed.
    def dht_bits(stream):
        i = stream.index(b"\xff\xc4")
        return stream[i + 5:i + 21]

    rgb = _rgb()
    flat = np.full((37, 53, 3), 7, np.uint8)
    for px in (rgb, flat):
        ours = J.encode(px, lossless=True, optimize=True)
        assert ours == ic.jpeg8_encode(px, lossless=True, optimize=True)
        assert ours == J.encode(px, lossless=True)
    assert dht_bits(J.encode(rgb, lossless=True)) != \
        dht_bits(J.encode(flat, lossless=True))
    u16 = _rng(2).integers(0, 65536, (9, 11), dtype=np.uint16)
    enc = J.encode(u16, lossless=True, optimize=True)
    np.testing.assert_array_equal(ic.jpeg8_decode(enc), u16)


@pytest.mark.parametrize("ss", [None, "444", "422", "420"])
def test_ycbcr_input_is_stored_unconverted(ss):
    # YCbCr samples are stored as they are, the way imagecodecs stores
    # them with colorspace="ycbcr". At quality 100 every quantization
    # step is 1, so the reference, imagecodecs' stream, decodes to the
    # same pixels as ours through imagecodecs, both converted to RGB
    # and as the stored YCbCr samples.
    ycc = _rgb((48, 64, 3), seed=11)
    kw = {} if ss is None else {"subsampling": ss}
    ours = J.encode(ycc, level=100, colorspace="ycbcr", **kw)
    theirs = ic.jpeg8_encode(ycc, level=100, colorspace="ycbcr",
                             outcolorspace="ycbcr", **kw)
    luma = _LUMA[ss or "420"]
    assert [c[1:] for c in _sof(ours)[2]] == [luma, (1, 1), (1, 1)]
    assert [c[1:] for c in _sof(ours)[2]] == \
        [c[1:] for c in _sof(theirs)[2]]
    np.testing.assert_array_equal(ic.jpeg8_decode(ours),
                                  ic.jpeg8_decode(theirs))
    np.testing.assert_array_equal(
        ic.jpeg8_decode(ours, outcolorspace="ycbcr"),
        ic.jpeg8_decode(theirs, outcolorspace="ycbcr"))
    np.testing.assert_array_equal(J.decode(ours), ic.jpeg8_decode(ours))
    if ss == "444":
        stored = ic.jpeg8_decode(ours, outcolorspace="ycbcr")
        assert np.abs(stored.astype(int) - ycc).max() <= 1
    # Lossless: the stored samples are the input, exactly.
    ll = J.encode(ycc, colorspace="ycbcr", lossless=True)
    np.testing.assert_array_equal(
        ic.jpeg8_decode(ll, outcolorspace="ycbcr"), ycc)


def _markers(stream: bytes):
    """[(marker, segment)] for the segments before the first SOS."""
    out, i = [], 2
    while stream[i + 1] != 0xDA:
        length = stream[i + 2] << 8 | stream[i + 3]
        out.append((stream[i + 1], stream[i:i + 2 + length]))
        i += 2 + length
    length = stream[i + 2] << 8 | stream[i + 3]
    out.append((0xDA, stream[i:i + 2 + length]))
    return out


def _inferred_colorspace(stream: bytes, decoder: str) -> str:
    """The colorspace a decoder infers for a three-component JPEG.

    T.81 leaves the colorspace to the application. IJG libjpeg 9
    (jdapimin.c, default_decompress_parms) checks the frame's component
    ids first: 1, 2, 3 is YCbCr and "R", "G", "B" is RGB whatever the
    markers say; only other ids fall back to the JFIF marker (YCbCr)
    and the Adobe APP14 transform (0 RGB, 1 YCbCr). libjpeg-turbo
    (jdapimin.c) checks the JFIF marker and the Adobe transform first
    and the ids "R", "G", "B" only after them.
    """
    segs = _markers(stream)
    ids = tuple(c[0] for c in _sof(stream)[2])
    jfif = any(m == 0xE0 and seg[4:9] == b"JFIF\x00" for m, seg in segs)
    adobe = [seg[15] for m, seg in segs
             if m == 0xEE and seg[4:9] == b"Adobe"]
    by_ids = {(1, 2, 3): "ycbcr", (0x52, 0x47, 0x42): "rgb"}.get(ids)
    by_markers = ("ycbcr" if jfif else
                  {0: "rgb", 1: "ycbcr"}.get(adobe[0]) if adobe else None)
    if decoder == "ijg":
        return by_ids or by_markers or "ycbcr"
    return by_markers or ("rgb" if ids == (0x52, 0x47, 0x42) else "ycbcr")


@pytest.mark.parametrize("kind", ["lossy", "12bit", "lossless"])
def test_ycbcr_input_is_labeled_ycbcr_for_every_decoder(kind):
    # TurboJPEG stores YCbCr samples through RGB storage, which names
    # the components R, G, B. IJG libjpeg reads those ids as RGB before
    # it looks at any marker, so the stream must carry what libjpeg and
    # imagecodecs write for YCbCr: component ids 1, 2, 3 in the frame
    # and in the scan, and the JFIF APP0 marker (T.871).
    if kind == "lossy":
        ycc, kw = _rgb((48, 64, 3), seed=14), {"level": 100}
    elif kind == "12bit":
        ycc = _rng(15).integers(0, 4096, (48, 64, 3), dtype=np.uint16)
        kw = {"level": 100, "subsampling": "444"}
    else:
        ycc, kw = _rgb((48, 64, 3), seed=16), {"lossless": True}
    ours = J.encode(ycc, colorspace="ycbcr", **kw)
    theirs = ic.jpeg8_encode(ycc, colorspace="ycbcr", **kw)
    assert [c[0] for c in _sof(ours)[2]] == [1, 2, 3]
    assert [c[0] for c in _sof(ours)[2]] == [c[0] for c in _sof(theirs)[2]]
    ours_m, theirs_m = _markers(ours), _markers(theirs)
    assert ours_m[0] == theirs_m[0]              # the same JFIF APP0
    assert ours_m[0][0] == 0xE0
    assert all(m != 0xEE for m, _ in ours_m)     # no Adobe marker
    sos = ours_m[-1][1]
    assert [sos[5 + 2 * k] for k in range(sos[4])] == [1, 2, 3]
    for decoder in ("ijg", "turbo"):
        assert _inferred_colorspace(ours, decoder) == "ycbcr"
        assert _inferred_colorspace(theirs, decoder) == "ycbcr"
    # The check above tells the two decoders apart: an RGB JPEG from
    # imagecodecs reads as RGB in both.
    rgb_stream = ic.jpeg8_encode(_rgb(), outcolorspace="rgb")
    assert _inferred_colorspace(rgb_stream, "ijg") == "rgb"
    assert _inferred_colorspace(rgb_stream, "turbo") == "rgb"
    if kind == "lossless":
        np.testing.assert_array_equal(
            ic.jpeg8_decode(ours, outcolorspace="ycbcr"), ycc)
    else:
        np.testing.assert_array_equal(ic.jpeg8_decode(ours),
                                      ic.jpeg8_decode(theirs))
        np.testing.assert_array_equal(J.decode(ours), ic.jpeg8_decode(ours))


def test_label_ycbcr_walks_every_scan():
    # A progressive stream has many scans, with tables between them; the
    # labeling must step over each scan's entropy-coded data to reach
    # the next scan header. A MozJPEG stream already has ids 1, 2, 3
    # and a JFIF marker, so labeling it changes nothing, and the result
    # decodes as before.
    mz = pytest.importorskip("opencodecs.codecs._mozjpeg")
    from opencodecs.codecs._jpeg_common import label_ycbcr
    stream = mz.encode(_rgb((40, 56, 3), seed=17), level=90)
    assert _sof(stream)[0] == 0xC2
    assert stream.count(b"\xff\xda") > 3
    labeled = label_ycbcr(stream)
    if _markers(stream)[0][1] == _markers(labeled)[0][1]:
        assert labeled == stream
    np.testing.assert_array_equal(ic.jpeg8_decode(labeled),
                                  ic.jpeg8_decode(stream))
    with pytest.raises(ValueError, match="three-component"):
        label_ycbcr(J.encode(_rgb()[..., 0]))


# The packed orders and their libjpeg J_COLOR_SPACE integers (JCS_EXT_*).
_PACKED = {"rgbx": 7, "bgr": 8, "bgrx": 9, "xbgr": 10, "xrgb": 11,
           "bgra": 13, "abgr": 14, "argb": 15}
# The orders whose fourth sample is alpha; encode raises for them.
_ALPHA = {"rgba": 12, "bgra": 13, "abgr": 14, "argb": 15}
# Where the red, green and blue samples sit in each order.
_RGB_AT = {"rgbx": [0, 1, 2], "bgr": [2, 1, 0], "bgrx": [2, 1, 0],
           "xbgr": [3, 2, 1], "xrgb": [1, 2, 3], "bgra": [2, 1, 0],
           "abgr": [3, 2, 1], "argb": [1, 2, 3]}


@pytest.mark.parametrize("name", sorted(set(_PACKED) - set(_ALPHA)))
def test_encode_packed_orders_read_color_in_order_named(name):
    # jpeg reads a packed order as libjpeg-turbo's JCS_EXT_* layout: the
    # color samples in the order named, the fourth sample not stored.
    # The reference is imagecodecs encoding the same color samples
    # given as RGB: byte-identical.
    n = 3 if name == "bgr" else 4
    px = _rgb((37, 53, n), seed=5)
    rgb = np.ascontiguousarray(px[..., _RGB_AT[name]])
    enc = J.encode(px, colorspace=name)
    assert enc == ic.jpeg8_encode(rgb)
    assert len(_sof(enc)[2]) == 3
    # imagecodecs does not name these orders. Given the J_COLOR_SPACE
    # integer it raises; given the name it reads an unknown colorspace
    # and stores every sample unconverted as its own component, so it
    # keeps a fourth sample that jpeg does not store.
    with pytest.raises(Exception):
        ic.jpeg8_encode(px, colorspace=_PACKED[name])
    theirs = ic.jpeg8_encode(px, colorspace=name, level=100,
                             subsampling="444")
    assert len(_sof(theirs)[2]) == n
    if n == 4:
        assert np.abs(ic.jpeg8_decode(theirs).astype(int) - px).max() <= 1


@pytest.mark.parametrize("name", sorted(_ALPHA))
def test_encode_alpha_orders_raise(name):
    # A JPEG has no alpha channel. imagecodecs raises for "rgba" and for
    # every J_COLOR_SPACE integer with alpha, and for the other names
    # keeps the fourth sample as a fourth component, which decodes back.
    # TurboJPEG would read these orders as RGB and drop the alpha, so
    # jpeg, mozjpeg and write() raise rather than lose it.
    px = _rgb((37, 53, 4), seed=5)
    with pytest.raises(Exception):
        ic.jpeg8_encode(px, colorspace=_ALPHA[name])
    if name == "rgba":
        with pytest.raises(Exception):
            ic.jpeg8_encode(px, colorspace=name)
    else:
        theirs = ic.jpeg8_encode(px, colorspace=name, level=100,
                                 subsampling="444")
        assert len(_sof(theirs)[2]) == 4
    for cs in (name, _ALPHA[name]):
        with pytest.raises(ValueError, match="no alpha"):
            J.encode(px, colorspace=cs)
        with pytest.raises(ValueError, match="no alpha"):
            oc.write(None, px, format="jpeg", colorspace=cs)
        if oc.has_codec("mozjpeg"):
            with pytest.raises(ValueError, match="no alpha"):
                oc.get_codec("mozjpeg").encode(px, colorspace=cs)
    # Naming the fourth sample padding drops it, as asked.
    pad = name.replace("a", "x")
    assert J.decode(J.encode(px, colorspace=pad)).shape == (37, 53, 3)


@pytest.mark.parametrize("name", sorted(_PACKED))
def test_decode_packed_orders_follow_libjpeg(name):
    # outcolorspace="bgr" and the other packed orders return libjpeg-
    # turbo's JCS_EXT_* layouts, which is what imagecodecs returns for
    # the J_COLOR_SPACE integers. imagecodecs reads the names themselves
    # as unknown and returns RGB, so there the two libraries differ.
    ref = ic.jpeg8_encode(_rgb((37, 53, 3), seed=6))
    theirs = ic.jpeg8_decode(ref, outcolorspace=_PACKED[name])
    ours = J.decode(ref, outcolorspace=name)
    np.testing.assert_array_equal(ours, theirs)
    np.testing.assert_array_equal(ours[..., _RGB_AT[name]],
                                  ic.jpeg8_decode(ref))
    np.testing.assert_array_equal(ic.jpeg8_decode(ref, outcolorspace=name),
                                  ic.jpeg8_decode(ref))


# ---------------------------------------------------------------------------
# High precision: encode bytes equal imagecodecs, decode of its streams
# ---------------------------------------------------------------------------


def test_12bit_dct_matches_imagecodecs():
    u12 = _rng(1).integers(0, 4096, (9, 11), dtype=np.uint16)
    enc = J.encode(u12)
    assert _sof(enc)[:2] == (0xC1, 12)    # extended DCT, P=12
    assert enc == ic.jpeg8_encode(u12)
    ref = ic.jpeg8_encode(u12, bitspersample=12)
    out = J.decode(ref)
    assert out.dtype == np.uint16 and out.shape == (9, 11)
    np.testing.assert_array_equal(out, ic.jpeg8_decode(ref))

    rgb12 = _rng(2).integers(0, 4096, (10, 13, 3), dtype=np.uint16)
    ref = ic.jpeg8_encode(rgb12)
    assert J.encode(rgb12) == ref
    np.testing.assert_array_equal(J.decode(ref), ic.jpeg8_decode(ref))
    # DCT-domain scaling works at 12 bits too.
    assert J.decode(ref, scale=2).shape == (5, 7, 3)


@pytest.mark.parametrize("bps", [2, 5, 7, 8, 9, 12, 13, 16])
def test_lossless_precisions_match_imagecodecs(bps):
    if bps not in (8, 12, 16) and not (ALL_PRECISIONS
                                       and _we_have_all_precisions()):
        pytest.skip("libjpeg-turbo before 3.1 lacks 2-16 bit lossless")
    dtype = np.uint8 if bps <= 8 else np.uint16
    a = _rng(bps).integers(0, 1 << bps, (9, 11), dtype=dtype)
    ref = ic.jpeg8_encode(a, lossless=True, bitspersample=bps)
    assert _sof(ref)[:2] == (0xC3, bps)
    out = J.decode(ref)
    assert out.dtype == dtype
    np.testing.assert_array_equal(out, a)
    enc = J.encode(a, lossless=True, bitspersample=bps)
    assert enc == ref


@pytest.mark.parametrize("predictor", range(1, 8))
def test_lossless_predictors_match_imagecodecs(predictor):
    rgb = _rgb()
    enc = J.encode(rgb, lossless=True, predictor=predictor)
    assert enc == ic.jpeg8_encode(rgb, lossless=True, predictor=predictor)
    np.testing.assert_array_equal(ic.jpeg8_decode(enc), rgb)


def test_lossless_uint16_default_keeps_data():
    # Data that fits 12 bits gets imagecodecs' default precision, 12.
    small = _rng(3).integers(0, 4096, (9, 11), dtype=np.uint16)
    enc = J.encode(small, lossless=True)
    assert _sof(enc)[:2] == (0xC3, 12)
    assert enc == ic.jpeg8_encode(small, lossless=True)
    # Wider data is stored at 16 bits rather than declared 12-bit.
    wide = _rng(4).integers(0, 65536, (9, 11), dtype=np.uint16)
    enc = J.encode(wide, lossless=True)
    assert _sof(enc)[:2] == (0xC3, 16)
    np.testing.assert_array_equal(ic.jpeg8_decode(enc), wide)
    np.testing.assert_array_equal(J.decode(enc), wide)


def test_lossless_decode_cannot_be_scaled():
    enc = J.encode(_rgb((64, 96, 3)), lossless=True)
    with pytest.raises(ValueError, match="lossless"):
        J.decode(enc, scale=2)


# ---------------------------------------------------------------------------
# Four components: CMYK and YCCK (Adobe APP14)
# ---------------------------------------------------------------------------


def test_cmyk_and_ycck_decode_match_imagecodecs():
    c4 = _rng(5).integers(0, 256, (37, 53, 4), dtype=np.uint8)
    for ref in (ic.jpeg8_encode(c4, colorspace="cmyk", outcolorspace="cmyk"),
                ic.jpeg8_encode(c4, colorspace="cmyk", outcolorspace="ycck"),
                ic.jpeg8_encode(c4)):  # 4 components, no Adobe marker
        out = J.decode(ref)
        assert out.shape == (37, 53, 4) and out.dtype == np.uint8
        np.testing.assert_array_equal(out, ic.jpeg8_decode(ref))
    c12 = _rng(6).integers(0, 4096, (8, 8, 4), dtype=np.uint16)
    ref = ic.jpeg8_encode(c12)
    np.testing.assert_array_equal(J.decode(ref), ic.jpeg8_decode(ref))


def test_cmyk_encode_matches_imagecodecs():
    c4 = _rng(7).integers(0, 256, (37, 53, 4), dtype=np.uint8)
    cmyk = J.encode(c4)
    # Same bytes as imagecodecs with both colorspaces named: Adobe APP14
    # transform 0, components C, M, Y, K at 1x1. imagecodecs needs both;
    # colorspace="cmyk" alone raises there.
    assert cmyk == ic.jpeg8_encode(c4, colorspace="cmyk",
                                   outcolorspace="cmyk")
    with pytest.raises(Exception):
        ic.jpeg8_encode(c4, colorspace="cmyk")
    assert b"Adobe" not in ic.jpeg8_encode(c4)
    assert [c[0] for c in _sof(ic.jpeg8_encode(c4))[2]] == [0, 1, 2, 3]
    assert [c[0] for c in _sof(cmyk)[2]] == [ord(x) for x in "CMYK"]
    # And the same pixels as imagecodecs' default (raw components).
    np.testing.assert_array_equal(ic.jpeg8_decode(cmyk),
                                  ic.jpeg8_decode(ic.jpeg8_encode(c4)))
    ycck = J.encode(c4, outcolorspace="ycck")
    assert ycck == ic.jpeg8_encode(c4, colorspace="cmyk",
                                   outcolorspace="ycck")
    lossless = J.encode(c4, lossless=True)
    np.testing.assert_array_equal(ic.jpeg8_decode(lossless), c4)


@pytest.mark.parametrize("ss", ["444", "422", "420", "440", "411"])
def test_rgb_and_cmyk_subsampling_is_honored(ss):
    # T.81 B.2.2 gives each component its own sampling factors, so an
    # RGB or CMYK JPEG can be subsampled. TurboJPEG gives the first (and
    # fourth) component the factors the subsampling names and samples
    # the others once per MCU. imagecodecs ignores subsampling here, so
    # the references are the hand-parsed frame header and imagecodecs'
    # decode of the stream.
    h, v = _LUMA[ss]
    rgb = _rgb()
    enc = J.encode(rgb, outcolorspace="rgb", subsampling=ss)
    assert [c[1:] for c in _sof(enc)[2]] == [(h, v), (1, 1), (1, 1)]
    np.testing.assert_array_equal(J.decode(enc), ic.jpeg8_decode(enc))
    c4 = _rng(7).integers(0, 256, (37, 53, 4), dtype=np.uint8)
    enc = J.encode(c4, subsampling=ss)
    assert [c[1:] for c in _sof(enc)[2]] == \
        [(h, v), (1, 1), (1, 1), (h, v)]
    np.testing.assert_array_equal(J.decode(enc), ic.jpeg8_decode(enc))


def test_rgb_and_cmyk_default_is_not_subsampled():
    rgb = _rgb()
    enc = J.encode(rgb, outcolorspace="rgb")
    assert [c[1:] for c in _sof(enc)[2]] == [(1, 1)] * 3
    # Unsubsampled, the bytes are imagecodecs' (which never subsamples
    # an RGB JPEG), with or without subsampling="444".
    assert enc == ic.jpeg8_encode(rgb, outcolorspace="rgb")
    assert J.encode(rgb, outcolorspace="rgb", subsampling="444") == \
        ic.jpeg8_encode(rgb, outcolorspace="rgb", subsampling="444")


def test_validate_is_accepted_like_imagecodecs():
    # jpeg8_encode takes validate= for compatibility and ignores it.
    rgb = _rgb()
    for value in (True, False):
        assert J.encode(rgb, validate=value) == J.encode(rgb)
        assert oc.get_codec("jpeg").encode(rgb, validate=value) == \
            ic.jpeg8_encode(rgb, validate=value)


# ---------------------------------------------------------------------------
# (H, W, 1) is grayscale
# ---------------------------------------------------------------------------


def test_trailing_singleton_channel_is_grayscale():
    g = _rgb((32, 48, 1))
    enc = J.encode(g)
    assert enc == J.encode(g[:, :, 0])
    assert enc == ic.jpeg8_encode(g)
    assert oc.write(None, g, format="jpeg") == enc
    assert J.decode(enc).shape == (32, 48)


# ---------------------------------------------------------------------------
# Decode options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcs", ["gray", "rgb", "rgba", "ycbcr", 2, 1])
def test_decode_outcolorspace_matches_imagecodecs(outcs):
    ref = ic.jpeg8_encode(_rgb())
    np.testing.assert_array_equal(J.decode(ref, outcolorspace=outcs),
                                  ic.jpeg8_decode(ref, outcolorspace=outcs))


def test_decode_outcolorspace_unavailable_raises():
    ref = ic.jpeg8_encode(_rgb())
    with pytest.raises(NotImplementedError):
        J.decode(ref, outcolorspace="ycck")
    with pytest.raises(NotImplementedError):
        J.decode(ref, outcolorspace="rgb565")
    with pytest.raises(ValueError):
        J.decode(ref, outcolorspace="nonsense")


@pytest.mark.parametrize("colorspace, outcolorspace", [
    ("rgb", None),        # YCbCr stream read as RGB, unconverted
    (2, 2),               # what tifffile passes for RGB without JFIF
    ("ycbcr", None),      # unconverted YCbCr components
    ("ycbcr", "rgb"),     # the stream's own reading, converted
    (None, "ycbcr"),
])
def test_decode_colorspace_on_jfif_stream_matches_imagecodecs(
        colorspace, outcolorspace):
    ref = ic.jpeg8_encode(_rgb())
    np.testing.assert_array_equal(
        J.decode(ref, colorspace=colorspace, outcolorspace=outcolorspace),
        ic.jpeg8_decode(ref, colorspace=colorspace,
                        outcolorspace=outcolorspace))


@pytest.mark.parametrize("colorspace, outcolorspace", [
    ("ycbcr", None), ("ycbcr", "rgb"), ("rgb", None), (None, "gray"),
])
def test_decode_colorspace_on_adobe_rgb_stream_matches_imagecodecs(
        colorspace, outcolorspace):
    ref = ic.jpeg8_encode(_rgb(), outcolorspace="rgb")   # Adobe transform 0
    np.testing.assert_array_equal(
        J.decode(ref, colorspace=colorspace, outcolorspace=outcolorspace),
        ic.jpeg8_decode(ref, colorspace=colorspace,
                        outcolorspace=outcolorspace))


@pytest.mark.parametrize("colorspace, outcolorspace", [
    ("cmyk", None), ("ycck", None), ("ycck", "cmyk"), (None, "ycck"),
])
def test_decode_colorspace_on_ycck_stream_matches_imagecodecs(
        colorspace, outcolorspace):
    c4 = _rng(9).integers(0, 256, (16, 24, 4), dtype=np.uint8)
    ref = ic.jpeg8_encode(c4, colorspace="cmyk", outcolorspace="ycck")
    np.testing.assert_array_equal(
        J.decode(ref, colorspace=colorspace, outcolorspace=outcolorspace),
        ic.jpeg8_decode(ref, colorspace=colorspace,
                        outcolorspace=outcolorspace))


# TIFF photometric names that are no JPEG colorspace. tifffile passes
# them as outcolorspace for photometric > 3, and imagecodecs reads them
# as JCS_UNKNOWN, the library default.
_PHOTOMETRIC_NAMES = ["CFA", "LINEAR_RAW", "CIELAB", "ICCLAB", "ITULAB",
                      "MASK", "PALETTE", "LOGL", "LOGLUV", "DEPTH_MAP",
                      "SEMANTIC_MASK", "linear_raw"]


@pytest.mark.parametrize("name", _PHOTOMETRIC_NAMES)
def test_decode_photometric_names_match_imagecodecs(name):
    M = pytest.importorskip("opencodecs.codecs._mozjpeg")
    gray = _rgb((37, 53))
    streams = [ic.jpeg8_encode(_rgb()),
               ic.jpeg8_encode(gray),
               ic.jpeg8_encode(gray.astype(np.uint16) * 257,
                               lossless=True, bitspersample=16)]
    for stream in streams:
        for kw in ("outcolorspace", "colorspace"):
            ref = ic.jpeg8_decode(stream, **{kw: name})
            np.testing.assert_array_equal(J.decode(stream, **{kw: name}), ref)
            np.testing.assert_array_equal(M.decode(stream, **{kw: name}), ref)


def test_unknown_colorspace_names_still_raise():
    stream = ic.jpeg8_encode(_rgb())
    for bad in ("rbg", "nonsense", 99):
        with pytest.raises(ValueError, match="unknown"):
            J.decode(stream, outcolorspace=bad)
    # Photometric names mean "library default" only when decoding: an
    # encoder asked to store CFA has no such colorspace to write.
    with pytest.raises(ValueError, match="unknown"):
        J.encode(_rgb(), outcolorspace="CFA")


def test_decode_colorspace_impossible_raises():
    ref = ic.jpeg8_encode(_rgb())
    with pytest.raises(NotImplementedError):
        J.decode(ref, colorspace="cmyk")       # 3 components
    with pytest.raises(NotImplementedError):
        J.decode(ref, colorspace="gray")


def test_decode_fast_upsampling_replicates_chroma():
    """fancyupsampling=False replicates each chroma sample over its 2x2
    block (T.81 A.1.1 sampling; libjpeg's merged upsampler). Converting
    the output back with the JFIF YCbCr equations shows the chroma
    constant within a block, up to rounding, which interpolation is not.

    imagecodecs 2026.8.16 sets this flag before jpeg_read_header resets
    it, so its output does not change and cannot be the reference here.
    """
    # Mid-range values keep the RGB outputs off the 0/255 clamp, which
    # would break the inverse conversion.
    rgb = _rng(10).integers(80, 176, (64, 64, 3), dtype=np.uint8)
    ref = ic.jpeg8_encode(rgb, level=95, subsampling="420")

    def chroma_spread(img):
        f = img.astype(np.float64)
        cb = -0.168736 * f[..., 0] - 0.331264 * f[..., 1] + 0.5 * f[..., 2]
        blocks = cb.reshape(32, 2, 32, 2).transpose(0, 2, 1, 3)
        blocks = blocks.reshape(32, 32, 4)
        return float((blocks.max(-1) - blocks.min(-1)).max())

    fast = J.decode(ref, fancyupsampling=False)
    smooth = J.decode(ref)
    np.testing.assert_array_equal(smooth, J.decode(ref, fancyupsampling=True))
    np.testing.assert_array_equal(smooth, ic.jpeg8_decode(ref))
    assert chroma_spread(fast) <= 1.5
    assert chroma_spread(smooth) > 5.0


def _split_tables(full: bytes):
    """Split a full stream into a T.81 B.5 table-specification stream
    and the abbreviated image stream that needs it."""
    tables, image = [], []
    i = 2
    while full[i + 1] != 0xDA:
        length = full[i + 2] << 8 | full[i + 3]
        seg = full[i:i + 2 + length]
        (tables if full[i + 1] in (0xDB, 0xC4) else image).append(seg)
        i += 2 + length
    return (b"\xff\xd8" + b"".join(tables) + b"\xff\xd9",
            b"\xff\xd8" + b"".join(image) + full[i:], i)


def test_decode_tables_and_header():
    full = ic.jpeg8_encode(_rgb())
    tables, abbreviated, sos = _split_tables(full)
    with pytest.raises(J.JpegError):
        J.decode(abbreviated)            # no quantization tables
    expect = ic.jpeg8_decode(full)
    np.testing.assert_array_equal(J.decode(abbreviated, tables=tables), expect)
    np.testing.assert_array_equal(
        ic.jpeg8_decode(abbreviated, tables=tables), expect)
    # header: imagecodecs.jpeg_decode decodes header + data + EOI.
    header, body = full[:sos], full[sos:-2]
    np.testing.assert_array_equal(J.decode(body, header=header), expect)
    np.testing.assert_array_equal(ic.jpeg_decode(body, header=header), expect)
    codec = oc.get_codec("jpeg")
    np.testing.assert_array_equal(
        codec.decode(abbreviated, tables=tables), expect)


def test_decode_shape_and_bitspersample():
    ref = ic.jpeg8_encode(_rgb())
    # Below 65500 pixels the frame header decides, as in imagecodecs.
    assert J.decode(ref, shape=(37, 53)).shape == (37, 53, 3)
    with pytest.raises(NotImplementedError):
        J.decode(ref, shape=(70000, 53))
    assert J.decode(ref, bitspersample=8).dtype == np.uint8
    with pytest.raises(ValueError):
        J.decode(ref, bitspersample=12)
    u12 = ic.jpeg8_encode(_rng().integers(0, 4096, (9, 11), dtype=np.uint16))
    assert J.decode(u12, bitspersample=12).dtype == np.uint16
    assert J.decode(u12, bitspersample=16).dtype == np.uint16


def test_decode_out_buffer_uint16():
    ref = ic.jpeg8_encode(_rng().integers(0, 4096, (9, 11), dtype=np.uint16))
    out = np.empty((9, 11), np.uint16)
    res = J.decode(ref, out=out)
    assert res is out
    np.testing.assert_array_equal(out, ic.jpeg8_decode(ref))
    with pytest.raises(ValueError):
        J.decode(ref, out=np.empty((9, 11), np.uint8))


def test_decoder_context_resets_options():
    ref = ic.jpeg8_encode(_rgb(), subsampling="420")
    with oc.get_codec("jpeg").decoder() as dec:
        fast = dec.decode(ref, fancyupsampling=False)
        np.testing.assert_array_equal(fast, J.decode(ref, fancyupsampling=False))
        # The next call on the same handle is back to the default.
        np.testing.assert_array_equal(dec.decode(ref), ic.jpeg8_decode(ref))
        u12 = ic.jpeg8_encode(_rng().integers(0, 4096, (9, 11), dtype=np.uint16))
        np.testing.assert_array_equal(dec.decode(u12), ic.jpeg8_decode(u12))


# ---------------------------------------------------------------------------
# tifffile through the patch: options reach the encoder and decoder
# ---------------------------------------------------------------------------


def test_tifffile_patch_forwards_jpeg_options(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    from opencodecs import tifffile_patch as patch

    rgb = _rgb((64, 64, 3))
    path = tmp_path / "rgb444.tif"
    with patch.patched():
        tifffile.imwrite(path, rgb, photometric="rgb", compression="jpeg",
                         compressionargs={"level": 90}, subsampling=(1, 1),
                         tile=(32, 32))
        ours = tifffile.imread(path)
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        assert page.tags["YCbCrSubSampling"].value == (1, 1)
        tile = page.parent.filehandle
        tile.seek(page.dataoffsets[0])
        stream = tile.read(page.databytecounts[0])
        theirs = page.asarray()
    # The stream holds what the tag says: Y sampled 1x1 (4:4:4).
    assert _sof(stream)[2][0][1:] == (1, 1)
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize("photometric, shape, dtype, args", [
    (34892, (32, 48, 3), np.uint8, {}),                  # LINEAR_RAW
    (32803, (32, 48), np.uint8, {}),                     # CFA
    ("cielab", (32, 48, 3), np.uint8, {}),
    (32803, (32, 48), np.uint16, {"lossless": True}),
    (34892, (32, 48, 3), np.uint16, {"lossless": True}),
    (32803, (32, 48), np.uint16, {"lossless": True, "bitspersample": 12}),
])
def test_tifffile_patch_reads_photometric_jpeg(tmp_path, photometric, shape,
                                               dtype, args):
    # tifffile decodes these with outcolorspace set to the photometric
    # name ("LINEAR_RAW", "CFA", "CIELAB"); the file is written and read
    # back by tifffile with imagecodecs, the reference.
    tifffile = pytest.importorskip("tifffile")
    from opencodecs import tifffile_patch as patch

    top = 4096 if args.get("bitspersample") == 12 else np.iinfo(dtype).max + 1
    data = _rng(3).integers(0, top, shape, dtype=dtype)
    path = tmp_path / "photometric.tif"
    kw = dict(photometric=photometric, compression="jpeg",
              compressionargs=args)
    if "bitspersample" in args:
        kw["bitspersample"] = args["bitspersample"]
    tifffile.imwrite(path, data, **kw)
    theirs = tifffile.imread(path)
    with patch.patched():
        ours = tifffile.imread(path)
    np.testing.assert_array_equal(ours, theirs)
    if args.get("lossless"):
        np.testing.assert_array_equal(ours, data)


def test_tifffile_patch_writes_rgb_jpeg_like_imagecodecs(tmp_path):
    # tifffile passes subsampling=(2, 2) with every RGB JPEG it writes,
    # also when compressionargs keep the JPEG RGB. The TIFF then says
    # PhotometricInterpretation RGB, where YCbCrSubSampling means
    # nothing, and imagecodecs writes the JPEG unsubsampled; with the
    # patch the file is the same, byte for byte.
    tifffile = pytest.importorskip("tifffile")
    from opencodecs import tifffile_patch as patch

    rgb = _rgb((64, 64, 3))
    kw = dict(photometric="rgb", compression="jpeg", tile=(32, 32),
              compressionargs={"level": 90, "outcolorspace": "rgb"})
    ref = tmp_path / "ic.tif"
    path = tmp_path / "oc.tif"
    tifffile.imwrite(ref, rgb, **kw)
    with patch.patched():
        tifffile.imwrite(path, rgb, **kw)
        ours = tifffile.imread(path)
    assert path.read_bytes() == ref.read_bytes()
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        assert page.photometric == tifffile.PHOTOMETRIC.RGB
        fh = page.parent.filehandle
        fh.seek(page.dataoffsets[0])
        stream = fh.read(page.databytecounts[0])
    assert [c[1:] for c in _sof(stream)[2]] == [(1, 1)] * 3
    np.testing.assert_array_equal(ours, tifffile.imread(ref))


@pytest.mark.parametrize("extra", [{}, {"subsampling": (1, 1)},
                                   {"subsampling": (2, 1)}])
def test_tifffile_patch_writes_lossless_rgb_like_imagecodecs(tmp_path,
                                                             extra):
    # tifffile adds subsampling=(2, 2), colorspace "RGB" and
    # outcolorspace "YCBCR" to every contiguous RGB JPEG it writes, also
    # a lossless one. imagecodecs applies neither there and writes an
    # unsubsampled RGB frame that decodes back exactly; with the patch
    # the file is the same, byte for byte, and reads back exactly with
    # and without the patch.
    tifffile = pytest.importorskip("tifffile")
    from opencodecs import tifffile_patch as patch

    rgb = _rgb((64, 64, 3), seed=12)
    kw = dict(compression="jpeg", tile=(32, 32),
              compressionargs={"lossless": True}, **extra)
    ref = tmp_path / "ic.tif"
    path = tmp_path / "oc.tif"
    tifffile.imwrite(ref, rgb, **kw)
    with patch.patched():
        tifffile.imwrite(path, rgb, **kw)
        ours = tifffile.imread(path)
    assert path.read_bytes() == ref.read_bytes()
    np.testing.assert_array_equal(ours, rgb)
    np.testing.assert_array_equal(tifffile.imread(path), rgb)


def test_tifffile_patch_writes_ycbcr_pixels(tmp_path):
    # photometric="ycbcr" hands the JPEG encoder YCbCr samples
    # (colorspace "YCBCR"), stored unconverted. The reference is the
    # file tifffile writes with imagecodecs: at quality 100 both read
    # back to the same RGB pixels, with and without the patch.
    tifffile = pytest.importorskip("tifffile")
    from opencodecs import tifffile_patch as patch

    ycc = _rgb((64, 64, 3), seed=13)
    kw = dict(photometric="ycbcr", compression="jpeg", tile=(32, 32),
              compressionargs={"level": 100})
    ref = tmp_path / "ic.tif"
    path = tmp_path / "oc.tif"
    tifffile.imwrite(ref, ycc, **kw)
    with patch.patched():
        tifffile.imwrite(path, ycc, **kw)
        ours = tifffile.imread(path)
    theirs = tifffile.imread(ref)
    np.testing.assert_array_equal(tifffile.imread(path), theirs)
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize("shape, dtype", [
    ((64, 64), np.uint16),
    ((64, 64, 3), np.uint16),
    ((64, 64, 4), np.uint8),
])
def test_tiff_writer_keeps_rejecting_what_its_tags_cannot_describe(
        tmp_path, shape, dtype):
    # The TIFF writers record BitsPerSample from the dtype and
    # PhotometricInterpretation from the sample count. A uint16 array
    # would be a 12-bit JPEG under BitsPerSample 16, and four samples a
    # CMYK JPEG under RGB, which libtiff and tifffile reject, so these
    # raise as they did in 0.4.0.
    data = _rng(4).integers(0, 256, shape).astype(dtype)
    with pytest.raises(J.JpegError, match="8-bit grayscale or RGB"):
        oc.write(None, data, format="tiff", compression="jpeg")
    with pytest.raises(J.JpegError, match="8-bit grayscale or RGB"):
        oc.write(tmp_path / "x.tif", data, format="tiff",
                 compression="jpeg", tile=(32, 32))
    from opencodecs.core.segment_compression import (
        bind_segment_encoder, encode_segment)
    with pytest.raises(J.JpegError, match="8-bit grayscale or RGB"):
        encode_segment(data, "jpeg")
    with pytest.raises(J.JpegError, match="8-bit grayscale or RGB"):
        bind_segment_encoder("jpeg")(data)


def test_tiff_writer_jpeg_reads_in_tifffile(tmp_path):
    # The reference reader is tifffile with imagecodecs, unpatched.
    tifffile = pytest.importorskip("tifffile")
    for shape in ((64, 48), (64, 48, 3)):
        data = _rng(5).integers(0, 256, shape, dtype=np.uint8)
        path = tmp_path / f"w{len(shape)}.tif"
        oc.write(path, data, format="tiff", compression="jpeg")
        np.testing.assert_array_equal(oc.read(path), tifffile.imread(path))


def test_tifffile_patch_reads_two_sample_lossless_jpeg(tmp_path):
    # tifffile writes a two-sample lossless JPEG TIFF (as DNG stores
    # some raw data) and reads it with imagecodecs. TurboJPEG has no
    # colorspace for two components, so jpeg raises for the stream;
    # the patch hands such a tile to imagecodecs instead of failing a
    # file that reads without it. Decoding stays exact.
    tifffile = pytest.importorskip("tifffile")
    from opencodecs import tifffile_patch as patch

    data = _rng(9).integers(0, 4096, (32, 40, 2), dtype=np.uint16)
    stream = ic.jpeg8_encode(data, lossless=True)
    np.testing.assert_array_equal(ic.jpeg_decode(stream), data)
    with pytest.raises((J.JpegError, NotImplementedError)):
        J.decode(stream)
    path = tmp_path / "two.tif"
    tifffile.imwrite(path, data, photometric="minisblack",
                     planarconfig="contig", compression="jpeg",
                     compressionargs={"lossless": True}, tile=(16, 16))
    with patch.patched():
        ours = tifffile.imread(path)
        # A stream without a frame still raises jpeg's error; only a
        # frame TurboJPEG has no colorspace for goes to imagecodecs.
        with pytest.raises((J.JpegError, NotImplementedError),
                           match="jpeg|JPEG"):
            patch.jpeg_decode(b"\xff\xd8\xff\xd9")
    np.testing.assert_array_equal(ours, data)
    np.testing.assert_array_equal(ours, tifffile.imread(path))


def test_tifffile_patch_reads_lossless_ycbcr_tiff(tmp_path):
    # tifffile writes photometric="ycbcr" with compressionargs
    # {"lossless": True} and asks the decoder for RGB. TurboJPEG's
    # lossless mode converts no colors and raises; imagecodecs then
    # reads the stored samples, so the file reads back exactly without
    # the patch. The patch hands such a tile to imagecodecs too, for
    # the file it writes and for the one imagecodecs writes.
    tifffile = pytest.importorskip("tifffile")
    from opencodecs import tifffile_patch as patch

    data = _rgb((64, 64, 3), seed=18)
    kw = dict(photometric="ycbcr", compression="jpeg", tile=(32, 32),
              compressionargs={"lossless": True})
    ref = tmp_path / "ic.tif"
    path = tmp_path / "oc.tif"
    tifffile.imwrite(ref, data, **kw)
    with patch.patched():
        tifffile.imwrite(path, data, **kw)
        ours = tifffile.imread(path)
        theirs = tifffile.imread(ref)
    np.testing.assert_array_equal(tifffile.imread(ref), data)
    np.testing.assert_array_equal(tifffile.imread(path), data)
    np.testing.assert_array_equal(ours, data)
    np.testing.assert_array_equal(theirs, data)


def test_cmyk_jpeg_tiff_reads(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    c4 = _rng(8).integers(0, 256, (32, 48, 4), dtype=np.uint8)
    path = tmp_path / "cmyk.tif"
    tifffile.imwrite(path, c4, photometric="separated", compression="jpeg")
    np.testing.assert_array_equal(oc.read(path), tifffile.imread(path))
