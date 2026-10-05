"""HTJ2K against independent references: imagecodecs, OpenJPEG, the spec.

Each test here pins a disagreement that used to exist, and checks it
against something other than our own encoder:

* the component transform (ISO/IEC 15444-1 Annex G.2/G.3, signaled in
  COD SGcod) on codestreams that imagecodecs, OpenJPH's ojph_compress,
  Kakadu and DICOM encoders write by default for RGB and RGBA;
* precisions above 16 bits and the NLT type 3 marker
  (ISO/IEC 15444-2) that carries float32;
* clamping, not wrapping, of lossy overshoot, judged against OpenJPEG;
* the meaning of ``level`` and the other imagecodecs keywords, judged
  by byte identity with ``imagecodecs.htj2k_encode``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

oj = pytest.importorskip("opencodecs.codecs._openjph")
imagecodecs = pytest.importorskip("imagecodecs")

from opencodecs import get_codec  # noqa: E402
from _ic_reference import skip_if_old_imagecodecs  # noqa: E402

pytestmark = skip_if_old_imagecodecs

try:
    imagecodecs.htj2k_encode(np.zeros((8, 8), np.uint8))
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"imagecodecs has no HTJ2K backend: {exc}",
                allow_module_level=True)

CONFORMANCE = (Path(__file__).resolve().parent.parent
               / ".test_data" / "htj2k" / "conformance")


def _image(shape, dtype, seed=0):
    """A smooth gradient plus noise across the dtype's whole range."""
    rng = np.random.default_rng(seed)
    info = np.iinfo(dtype)
    lo, hi = float(info.min), float(info.max)
    y, x = np.mgrid[: shape[0], : shape[1]]
    base = (3.0 * x + 5.0 * y)
    if len(shape) == 3:
        base = base[..., None] + 40.0 * np.arange(shape[2])
    a = lo + base / base.max() * (hi - lo) * 0.9
    a = a + rng.normal(0, (hi - lo) / 50, shape)
    return np.clip(a, lo, hi).astype(dtype)


def _cod_mct(codestream: bytes) -> int:
    """The SGcod multiple component transform byte (ISO/IEC 15444-1
    Table A.17): COD marker, Lcod (2), Scod (1), progression (1),
    layers (2), then MCT."""
    i = codestream.index(b"\xff\x52")
    return codestream[i + 8]


def _nlt_present(codestream: bytes) -> bool:
    """Whether the main header carries an NLT marker (0xFF76)."""
    end = codestream.index(b"\xff\x90")  # first SOT
    return b"\xff\x76" in codestream[:end]


# ---------------------------------------------------------------------------
# Component transform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int8, np.int16])
@pytest.mark.parametrize("shape", [(33, 47, 3), (40, 52, 4)])
def test_decodes_imagecodecs_rgb_with_component_transform(shape, dtype):
    """imagecodecs turns the RCT on for RGB/RGBA. Pulling such a
    codestream planar used to return the transformed samples (maxdiff
    252 on uint8, no error)."""
    a = _image(shape, dtype)
    enc = imagecodecs.htj2k_encode(a)
    assert _cod_mct(enc) == 1
    out = oj.decode(enc)
    assert out.dtype == a.dtype and out.shape == a.shape
    np.testing.assert_array_equal(out, a)
    np.testing.assert_array_equal(get_codec("htj2k").decode(enc), a)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_decodes_imagecodecs_lossy_rgb_like_imagecodecs(dtype):
    """The ICT path: same reconstruction as imagecodecs wherever
    imagecodecs does not wrap an overshoot (see the clamp test)."""
    a = _image((48, 64, 3), dtype, seed=4)
    enc = imagecodecs.htj2k_encode(a, 0.002)
    assert _cod_mct(enc) == 1
    ours = oj.decode(enc).astype(np.int64)
    ref = imagecodecs.htj2k_decode(enc).astype(np.int64)
    top = np.iinfo(dtype).max
    wrapped = (ours == top) | (ours == 0)
    np.testing.assert_array_equal(ours[~wrapped], ref[~wrapped])


@pytest.mark.parametrize("shape", [(33, 47, 3), (40, 52, 4)])
def test_encoder_writes_component_transform_like_imagecodecs(shape):
    """RGB/RGBA now get the RCT, as imagecodecs and ojph_compress give
    them, and the codestream is byte-identical to imagecodecs'.
    ``rgb=False`` keeps the old, untransformed codestream."""
    a = _image(shape, np.uint8, seed=1)
    ours = oj.encode(a)
    assert _cod_mct(ours) == 1
    assert ours == imagecodecs.htj2k_encode(a)
    # imagecodecs reads it back interleaved, no longer as (C, H, W).
    np.testing.assert_array_equal(imagecodecs.htj2k_decode(ours), a)

    plain = oj.encode(a, rgb=False)
    assert _cod_mct(plain) == 0
    assert plain == imagecodecs.htj2k_encode(a, rgb=False)
    # Without the transform, planar=None gives (C, H, W) as imagecodecs.
    np.testing.assert_array_equal(oj.decode(plain),
                                  imagecodecs.htj2k_decode(plain))
    np.testing.assert_array_equal(oj.decode(plain, planar=False), a)


def test_component_transform_with_reduce():
    a = _image((64, 80, 3), np.uint8, seed=2)
    enc = imagecodecs.htj2k_encode(a)
    for r in (1, 2):
        np.testing.assert_array_equal(
            oj.decode(enc, reduce=r), imagecodecs.htj2k_decode(enc, skipres=r))
    np.testing.assert_array_equal(
        oj.decode(enc, skipres=(2, 1)),
        imagecodecs.htj2k_decode(enc, skipres=(2, 1)))


def test_planar_output_and_input():
    a = _image((24, 30, 3), np.uint16, seed=3)
    enc = imagecodecs.htj2k_encode(a)
    np.testing.assert_array_equal(oj.decode(enc, planar=True),
                                  np.moveaxis(a, -1, 0))
    planar_in = np.ascontiguousarray(np.moveaxis(a, -1, 0))
    assert (oj.encode(planar_in, planar=True)
            == imagecodecs.htj2k_encode(planar_in, planar=True))


@pytest.mark.parametrize("components", [2, 5])
def test_any_component_count(components):
    a = _image((20, 23, components), np.uint8, seed=components)
    enc = oj.encode(a)
    assert enc == imagecodecs.htj2k_encode(a)
    np.testing.assert_array_equal(oj.decode(enc), np.moveaxis(a, -1, 0))
    np.testing.assert_array_equal(oj.decode(enc, planar=False), a)
    np.testing.assert_array_equal(
        oj.decode(imagecodecs.htj2k_encode(a), planar=False), a)


@pytest.mark.parametrize("name", ["ds0_ht_14_b11.j2k", "hifi_ht1_02.j2k",
                                  "ds1_ht_06_b11.j2k"])
def test_conformance_component_transform_matches_openjpeg(name):
    """JPEG committee conformance codestreams that use the component
    transform, decoded by OpenJPEG (an independent implementation)."""
    path = CONFORMANCE / name
    if not path.is_file():
        pytest.skip("conformance corpus missing")
    blob = path.read_bytes()
    assert _cod_mct(blob) == 1
    with warnings.catch_warnings():
        # ds1_ht_06 makes OpenJPH warn about an odd code-block; it still
        # decodes, and the comparison below is what matters.
        warnings.simplefilter("ignore", RuntimeWarning)
        ours = oj.decode(blob).astype(np.int64)
    ref = imagecodecs.jpeg2k_decode(blob).astype(np.int64)
    assert ours.shape == ref.shape
    assert np.abs(ours - ref).max() <= 1


# ---------------------------------------------------------------------------
# More than 16 bits, and NLT float32
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint32, np.int32])
@pytest.mark.parametrize("shape", [(33, 47), (21, 25, 3)])
def test_32bit_codestreams_decode_exactly(dtype, shape):
    """These used to come back as uint16/int16, clamped (maxdiff about
    one million, no error)."""
    a = _image(shape, dtype, seed=5)
    enc = imagecodecs.htj2k_encode(a)
    assert oj.decode_info(enc)["bit_depth"] == 32
    out = oj.decode(enc)
    assert out.dtype == dtype
    np.testing.assert_array_equal(out, a)
    assert oj.encode(a) == enc


def test_20bit_values_in_uint32():
    a = (np.arange(33 * 47, dtype=np.uint32).reshape(33, 47) * 641) % (1 << 20)
    enc = imagecodecs.htj2k_encode(a)
    np.testing.assert_array_equal(oj.decode(enc), a)


def test_float32_nlt_decodes_to_float32():
    """imagecodecs writes float32 as its bit patterns under NLT type 3
    (ISO/IEC 15444-2). These used to decode as int16 nonsense."""
    rng = np.random.default_rng(6)
    f = rng.normal(0, 100, (33, 47)).astype(np.float32)
    f[0, :5] = [-0.0, np.inf, -np.inf, 1e-40, -3.5]
    enc = imagecodecs.htj2k_encode(f, reversible=True)
    assert _nlt_present(enc)
    out = oj.decode(enc)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out.view(np.uint32), f.view(np.uint32))
    assert oj.encode(f, reversible=True) == enc
    assert get_codec("htj2k").decode(enc).dtype == np.float32


def test_decode_info_reports_decode_dtype():
    f = np.ones((8, 8), np.float32)
    assert oj.decode_info(imagecodecs.htj2k_encode(f))["dtype"] == np.float32
    u = np.ones((8, 8), np.uint32)
    assert oj.decode_info(imagecodecs.htj2k_encode(u))["dtype"] == np.uint32


def test_mixed_component_codestream_raises():
    """ds1_ht_07_b11 has components of different sizes; it must raise,
    not return planes sized after component 0."""
    path = CONFORMANCE / "ds1_ht_07_b11.j2k"
    if not path.is_file():
        pytest.skip("conformance corpus missing")
    with pytest.raises(oj.OpenJphError, match="components differ"):
        oj.decode(path.read_bytes())


# ---------------------------------------------------------------------------
# Lossy overshoot: clamp, as OpenJPEG and ojph_expand do
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.int8])
@pytest.mark.parametrize("level", [0.01, 0.1])
def test_lossy_overshoot_is_clamped_like_openjpeg(dtype, level):
    """ISO/IEC 15444-1 Annex G.1.2 names clipping to the nominal range
    as the usual treatment of quantization overshoot; OpenJPEG and
    OpenJPH's ojph_expand clip. imagecodecs.htj2k_decode wraps 256 to 0,
    so it is not the reference here."""
    info = np.iinfo(dtype)
    y, x = np.mgrid[:96, :128]
    s = np.sin(x / 7.0) * np.cos(y / 5.0) * 1.3
    rng = np.random.default_rng(7)
    a = np.clip(info.min + (s + 1) / 2 * (info.max - info.min)
                + rng.normal(0, 4, s.shape), info.min, info.max).astype(dtype)
    enc = oj.encode(a, level)
    assert enc == imagecodecs.htj2k_encode(a, level)
    ours = oj.decode(enc).astype(np.int64)
    openjpeg = imagecodecs.jpeg2k_decode(enc).astype(np.int64)
    assert np.abs(ours - openjpeg).max() <= 1
    assert ours.min() >= info.min and ours.max() <= info.max


# ---------------------------------------------------------------------------
# imagecodecs keywords
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [0.0, 1e-6, 0.01, 0.1, 0.5, 1, 30, 75,
                                   100, 250, 9.99e-6, 1e-5, 1.00000001e-5,
                                   1.0000001e-5, 1.1e-5])
def test_level_means_what_imagecodecs_means(level):
    """Below 1: quantization step. From 1: quality factor 1..100."""
    a = _image((64, 80, 3), np.uint8, seed=8)
    assert oj.encode(a, level) == imagecodecs.htj2k_encode(a, level)
    assert (get_codec("htj2k").encode(a, level=level)
            == imagecodecs.htj2k_encode(a, level))


@pytest.mark.parametrize("kw", [
    {"rgb": False}, {"resolutions": 2}, {"resolutions": 9},
    {"tlm": True}, {"tile": (32, 32), "tilepart": 3},
    {"block_size": (32, 32)}, {"prog_order": "RPCL"},
    {"reversible": False},
    {"tile": (32, 16)},
])
def test_encode_options_match_imagecodecs(kw):
    a = _image((64, 80, 3), np.uint8, seed=9)
    assert oj.encode(a, **kw) == imagecodecs.htj2k_encode(a, **kw)
    assert get_codec("htj2k").encode(a, **kw) == \
        imagecodecs.htj2k_encode(a, **kw)


def test_profile_is_passed_to_openjph():
    """Neither library exposes precincts, which both OpenJPH profiles
    restrict, so OpenJPH refuses either profile from both. What matters
    is that ours reaches OpenJPH and says why, not that it is dropped."""
    a = _image((64, 80, 3), np.uint8, seed=9)
    for profile in ("IMF", "BROADCAST"):
        with pytest.raises(Exception):
            imagecodecs.htj2k_encode(a, profile=profile, block_size=(32, 32))
        with pytest.raises(oj.OpenJphError, match="profile"):
            oj.encode(a, profile=profile, block_size=(32, 32))


def test_reversible_with_lossy_level_raises():
    """imagecodecs drops the level silently here; we refuse."""
    with pytest.raises(ValueError, match="reversible"):
        oj.encode(np.zeros((16, 16), np.uint8), 0.1, reversible=True)


def test_codec_rejects_unknown_options():
    codec = get_codec("htj2k")
    a = np.zeros((16, 16), np.uint8)
    with pytest.raises(TypeError, match="unsupported"):
        codec.encode(a, quality=50)
    # numthreads is an option since tile-parallel decode; this one is not.
    with pytest.raises(TypeError, match="unsupported"):
        codec.decode(codec.encode(a), workers=2)


# ---------------------------------------------------------------------------
# rgb=True is honored or refused, never dropped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.int16])
@pytest.mark.parametrize("components", [3, 4])
def test_rgb_true_applies_transform_to_planar_input(dtype, components):
    """imagecodecs drops rgb=True for planar input (MCT byte 0); here
    the transform is written, and OpenJPEG and imagecodecs both invert
    it to the original samples."""
    hwc = _image((29, 43, components), dtype, seed=21)
    chw = np.ascontiguousarray(np.moveaxis(hwc, -1, 0))
    enc = oj.encode(chw, planar=True, rgb=True)
    assert _cod_mct(enc) == 1
    assert _cod_mct(imagecodecs.htj2k_encode(chw, planar=True,
                                             rgb=True)) == 0
    np.testing.assert_array_equal(imagecodecs.jpeg2k_decode(enc), hwc)
    np.testing.assert_array_equal(imagecodecs.htj2k_decode(enc), hwc)
    np.testing.assert_array_equal(oj.decode(enc), hwc)
    np.testing.assert_array_equal(oj.decode(enc, planar=True), chw)
    # Planar input without rgb=True keeps imagecodecs' bytes.
    assert oj.encode(chw, planar=True) == imagecodecs.htj2k_encode(
        chw, planar=True)


def test_rgb_true_where_it_cannot_apply_raises():
    with pytest.raises(ValueError, match="float32"):
        oj.encode(np.zeros((16, 16, 3), np.float32), rgb=True)
    with pytest.raises(ValueError, match="3 or more"):
        oj.encode(np.zeros((16, 16), np.uint8), rgb=True)
    with pytest.raises(ValueError, match="3 or more"):
        oj.encode(np.zeros((16, 16, 2), np.uint16), rgb=True)
    # rgb=None and rgb=False on float32 still match imagecodecs.
    f = np.linspace(-2, 2, 16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)
    for rgb in (None, False):
        assert oj.encode(f, rgb=rgb) == imagecodecs.htj2k_encode(f, rgb=rgb)


# ---------------------------------------------------------------------------
# decode out=, as imagecodecs.htj2k_decode defines it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype,shape", [
    (np.uint8, (31, 37)), (np.int16, (31, 37, 3)), (np.uint32, (19, 23)),
])
def test_decode_into_out(dtype, shape):
    a = _image(shape, dtype, seed=22)
    enc = imagecodecs.htj2k_encode(a)
    ref = imagecodecs.htj2k_decode(enc)
    out = np.zeros(ref.shape, ref.dtype)
    got = oj.decode(enc, out=out)
    assert got is out
    np.testing.assert_array_equal(out, ref)
    np.testing.assert_array_equal(out, a)
    out2 = np.zeros(ref.shape, ref.dtype)
    assert get_codec("htj2k").decode(enc, out=out2) is out2
    np.testing.assert_array_equal(out2, a)
    # A writable buffer of the right size, as imagecodecs accepts.
    buf = bytearray(ref.nbytes)
    got = oj.decode(enc, out=buf)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    np.testing.assert_array_equal(np.frombuffer(buf, ref.dtype)
                                  .reshape(ref.shape), a)


def test_decode_into_float32_out():
    f = np.linspace(-3, 3, 24 * 20, dtype=np.float32).reshape(24, 20)
    enc = imagecodecs.htj2k_encode(f, reversible=True)
    out = np.empty_like(f)
    assert oj.decode(enc, out=out) is out
    np.testing.assert_array_equal(out.view(np.uint32), f.view(np.uint32))


def test_decode_out_mismatch_raises():
    enc = imagecodecs.htj2k_encode(_image((16, 20), np.uint8))
    with pytest.raises(ValueError, match="dtype"):
        oj.decode(enc, out=np.zeros((16, 20), np.uint16))
    with pytest.raises(ValueError, match="shape"):
        oj.decode(enc, out=np.zeros((16 * 20,), np.uint8))
    with pytest.raises(ValueError, match="bytes"):
        oj.decode(enc, out=bytearray(10))
    with pytest.raises(ValueError, match="writable"):
        oj.decode(enc, out=bytes(16 * 20))
    with pytest.raises(ValueError, match="C-contiguous"):
        oj.decode(enc, out=np.zeros((20, 16), np.uint8).T)


# ---------------------------------------------------------------------------
# planar=None: imagecodecs' shape rule
# ---------------------------------------------------------------------------


def _ic_encodes():
    """(components, rgb) pairs that imagecodecs can encode."""
    for components in (2, 3, 4, 5):
        for rgb in (None, False, True):
            if components == 2 and rgb:
                continue
            yield components, rgb


@pytest.mark.parametrize("components,rgb", list(_ic_encodes()))
@pytest.mark.parametrize("planar", [None, False, True])
def test_planar_default_follows_imagecodecs(components, rgb, planar):
    """imagecodecs.htj2k_decode interleaves a codestream that carries
    the component transform and returns any other multi-component
    codestream as (C, H, W). Every planar value gives its shape and
    samples, through the module and through the codec."""
    a = _image((20, 22, components), np.uint8, seed=components)
    enc = imagecodecs.htj2k_encode(a, rgb=rgb)
    ref = imagecodecs.htj2k_decode(enc, planar=planar)
    got = oj.decode(enc, planar=planar)
    assert got.shape == ref.shape
    np.testing.assert_array_equal(got, ref)
    np.testing.assert_array_equal(
        get_codec("htj2k").decode(enc, planar=planar), ref)
    out = np.empty_like(ref)
    assert oj.decode(enc, planar=planar, out=out) is out
    np.testing.assert_array_equal(out, ref)


def test_untransformed_rgb_still_decodes_correctly():
    """0.4.0 wrote RGB without the component transform (rgb=False
    writes the same codestream). Such a file now decodes in
    imagecodecs' layout, (C, H, W), with the right samples, and
    planar=False gives the old (H, W, C)."""
    a = _image((24, 30, 3), np.uint16, seed=21)
    enc = oj.encode(a, rgb=False)
    assert _cod_mct(enc) == 0
    np.testing.assert_array_equal(oj.decode(enc), np.moveaxis(a, -1, 0))
    np.testing.assert_array_equal(oj.decode(enc, planar=False), a)


@pytest.mark.parametrize("rgb", [None, False])
def test_pyramid_reader_levels_stay_interleaved(rgb):
    """The pyramid reader advertises (H, W, C) and must return it
    whether or not the codestream uses the component transform."""
    from opencodecs import Htj2kPyramidReader
    a = _image((64, 80, 3), np.uint8, seed=22)
    enc = imagecodecs.htj2k_encode(a, rgb=rgb)
    reader = Htj2kPyramidReader(enc)
    np.testing.assert_array_equal(reader.read_level(0), a)
    for i, level in enumerate(reader.levels):
        got = reader.read_level(i)
        assert got.shape == tuple(level.shape)
        # Reduced levels clamp overshoot where imagecodecs wraps it, so
        # the layout is checked against the planar decode, which the
        # tests above pin to imagecodecs.
        np.testing.assert_array_equal(
            got, np.moveaxis(oj.decode(enc, reduce=i, planar=True), 0, -1))


def test_dicomweb_frame_stays_interleaved():
    """DICOM color frames are (rows, columns, samples) whatever the
    codestream's transform; decode_frame keeps that layout."""
    from opencodecs import _dicomweb
    a = _image((24, 30, 3), np.uint8, seed=23)
    for rgb in (None, False):
        enc = imagecodecs.htj2k_encode(a, rgb=rgb)
        got = _dicomweb.decode_frame(enc, _dicomweb.TS_HTJ2K_LOSSLESS)
        np.testing.assert_array_equal(got, a)


def test_level_1e5_is_lossless_like_imagecodecs():
    """imagecodecs compares the step after holding it in a C float, so
    a level of exactly 1e-5 is lossless; 1.0000001e-5 is lossy."""
    a = (np.arange(40 * 52, dtype=np.uint16).reshape(40, 52) * 7)
    enc = oj.encode(a, 1e-5)
    assert enc == imagecodecs.htj2k_encode(a, 1e-5)
    assert enc == oj.encode(a)
    np.testing.assert_array_equal(imagecodecs.htj2k_decode(enc), a)
    assert oj.encode(a, 1.0000001e-5) != enc


def test_more_components_than_siz_allows_raises():
    """Csiz is 1 to 16384 (ISO/IEC 15444-1 Table A.9). imagecodecs and
    OpenJPH write 16385 components without an error; that codestream is
    outside the standard, so encode refuses it. 16384 still works."""
    ok = np.zeros((16384, 8, 8), np.uint8)
    ok[:, 2, 3] = np.arange(16384) % 251
    enc = oj.encode(ok, planar=True)
    assert int.from_bytes(enc[40:42], "big") == 16384  # Csiz
    np.testing.assert_array_equal(imagecodecs.htj2k_decode(enc), ok)
    with pytest.raises(oj.OpenJphError, match="16384"):
        oj.encode(np.zeros((16385, 8, 8), np.uint8), planar=True)


def test_encode_out_raises():
    """imagecodecs.htj2k_encode defines out=; opencodecs does not
    implement it for encode and refuses it rather than dropping it."""
    a = _image((16, 20), np.uint8)
    with pytest.raises(TypeError, match="out"):
        oj.encode(a, out=bytearray(1 << 16))
    with pytest.raises(TypeError, match="out"):
        get_codec("htj2k").encode(a, out=bytearray(1 << 16))
