"""Tests for the cms (color management) codec.

The codec is a thin ctypes wrapper around liblcms2; tests need an
actual ICC profile to be meaningful. We use lcms2's built-in
``cmsCreate_sRGBProfile`` (via ``cmsSaveProfileToMem``) so the tests
work on any platform with Little-CMS installed, without shipping a
real ICC binary blob in the repo.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

import opencodecs as oc


def _make_srgb_profile_bytes() -> bytes:
    """Return the bytes of lcms2's built-in sRGB profile."""
    from opencodecs._cms_codec import _load_lcms2
    lib = _load_lcms2()
    lib.cmsSaveProfileToMem.restype = ctypes.c_int
    lib.cmsSaveProfileToMem.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    hp = lib.cmsCreate_sRGBProfile()
    size = ctypes.c_uint32(0)
    lib.cmsSaveProfileToMem(hp, None, ctypes.byref(size))
    buf = (ctypes.c_ubyte * size.value)()
    lib.cmsSaveProfileToMem(hp, buf, ctypes.byref(size))
    lib.cmsCloseProfile(hp)
    return bytes(buf)


@pytest.fixture(scope="module")
def srgb_profile():
    try:
        return _make_srgb_profile_bytes()
    except ImportError:
        pytest.skip("liblcms2 not available on this platform")


def test_cms_codec_registered():
    assert oc.has_codec("cms")
    entry = next(c for c in oc.list_codecs() if c["name"] == "cms")
    assert entry["decode"] is True
    assert entry["encode"] is False


def test_cms_identity_srgb_to_srgb_rgb8(srgb_profile):
    """sRGB → sRGB with perceptual intent is the identity on every
    valid 8-bit RGB pixel."""
    c = oc.get_codec("cms")
    arr = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        [[128, 128, 128], [255, 255, 255], [0, 0, 0]],
    ], dtype=np.uint8)
    out = c.decode(arr, profile_in=srgb_profile, profile_out=srgb_profile)
    np.testing.assert_array_equal(out, arr)


def test_cms_identity_srgb_to_srgb_rgb16(srgb_profile):
    """Same as above but uint16 — exercises TYPE_RGB_16."""
    c = oc.get_codec("cms")
    arr = np.array([
        [[65535, 0, 0], [0, 65535, 0], [0, 0, 65535]],
        [[32768, 32768, 32768], [65535, 65535, 65535], [0, 0, 0]],
    ], dtype=np.uint16)
    out = c.decode(arr, profile_in=srgb_profile, profile_out=srgb_profile)
    np.testing.assert_array_equal(out, arr)


@pytest.mark.xfail(
    reason="lcms2 refuses cmsCreateTransform on RGBA-in / RGBA-out when "
           "both profiles are 3-channel sRGB (no per-pixel alpha to "
           "transform). The COPY_ALPHA flag is set but doesn't help "
           "with the built-in sRGB profile — a real RGBA workflow "
           "would supply a 4-channel destination profile. Documenting "
           "the limitation here rather than silently broadening the "
           "fallback.",
    strict=True,
)
def test_cms_identity_rgba8(srgb_profile):
    c = oc.get_codec("cms")
    arr = np.array([
        [[100, 50, 25, 200], [255, 0, 0, 128]],
    ], dtype=np.uint8)
    out = c.decode(arr, profile_in=srgb_profile, profile_out=srgb_profile)
    np.testing.assert_array_equal(out, arr)


def test_cms_default_target_is_srgb(srgb_profile):
    """profile_out=None should use the built-in sRGB profile."""
    c = oc.get_codec("cms")
    arr = np.random.default_rng(0).integers(0, 256, (4, 4, 3),
                                             dtype=np.uint8)
    out = c.decode(arr, profile_in=srgb_profile, profile_out=None)
    np.testing.assert_array_equal(out, arr)


def test_cms_intent_aliases(srgb_profile):
    """All four standard ICC intents accepted by string."""
    c = oc.get_codec("cms")
    arr = np.full((2, 2, 3), 128, dtype=np.uint8)
    for intent in ("perceptual", "relative", "relative_colorimetric",
                   "saturation", "absolute", "absolute_colorimetric"):
        out = c.decode(arr, profile_in=srgb_profile, intent=intent)
        np.testing.assert_array_equal(out, arr)


def test_cms_out_kwarg_zero_alloc(srgb_profile):
    c = oc.get_codec("cms")
    arr = np.array([[[100, 50, 25]]], dtype=np.uint8)
    target = np.empty_like(arr)
    out = c.decode(arr, profile_in=srgb_profile, out=target)
    assert out is target
    np.testing.assert_array_equal(out, arr)


def test_cms_encode_raises():
    """cms is a transform, not a compressor."""
    c = oc.get_codec("cms")
    with pytest.raises(NotImplementedError):
        c.encode(np.zeros((4, 4, 3), dtype=np.uint8))


def test_cms_bad_profile_raises(srgb_profile):
    """An obviously-not-an-ICC blob should error cleanly."""
    c = oc.get_codec("cms")
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="cmsOpenProfileFromMem"):
        c.decode(arr, profile_in=b"NOT AN ICC PROFILE")


# ----- built-in profile factories + sRGB ↔ Display-P3 helpers -----


def test_builtin_profile_icc_has_valid_signature():
    """Both built-in profiles start with the ICC ``acsp`` magic at
    offset 36 — the universal ICC file signature."""
    from opencodecs._cms_codec import _builtin_profile_icc
    for name in ("srgb", "display-p3"):
        icc = _builtin_profile_icc(name)
        assert isinstance(icc, bytes) and len(icc) > 128
        assert icc[36:40] == b"acsp", f"{name}: missing ICC signature"


def test_builtin_profile_icc_unknown_name():
    from opencodecs._cms_codec import _builtin_profile_icc
    with pytest.raises(ValueError, match="unknown built-in profile"):
        _builtin_profile_icc("rec2020")


def test_builtin_profile_icc_cache_returns_same_object():
    """Repeated calls return the cached bytes — no rebuild per call."""
    from opencodecs._cms_codec import _builtin_profile_icc
    a = _builtin_profile_icc("display-p3")
    b = _builtin_profile_icc("display-p3")
    assert a is b


def test_srgb_to_display_p3_primaries():
    """sRGB primaries map to known Display-P3 values (perceptual intent).

    Reference values come from a hand-check against the canonical
    sRGB→P3 matrix; we allow ±2 codes of slack for lcms2 drift.
    """
    from opencodecs._cms_codec import srgb_to_display_p3_uint8
    primaries = np.array(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8
    )
    out = srgb_to_display_p3_uint8(primaries)
    expected = np.array(
        [[[234,  51,  35],
          [117, 251,  76],
          [  0,   0, 245]]],
        dtype=np.int16,
    )
    diff = np.abs(out.astype(np.int16) - expected)
    assert diff.max() <= 2, (
        f"sRGB→P3 primaries drifted too far: got {out}, expected {expected}"
    )


def test_srgb_to_display_p3_gray_is_identity():
    """Neutral gray is invariant: sRGB and Display-P3 share D65 white,
    so achromatic samples round-trip within ±1 LSB."""
    from opencodecs._cms_codec import srgb_to_display_p3_uint8
    arr = np.tile(np.arange(0, 256, 16, dtype=np.uint8)[:, None, None], (1, 1, 3))
    out = srgb_to_display_p3_uint8(arr)
    diff = np.abs(out.astype(np.int16) - arr.astype(np.int16))
    assert diff.max() <= 1, f"gray drift: max diff {diff.max()}"


def test_srgb_to_display_p3_preserves_alpha():
    """RGBA path: RGB plane is transformed, alpha passes verbatim."""
    from opencodecs._cms_codec import srgb_to_display_p3_uint8
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    alpha = rng.integers(0, 256, size=(8, 8, 1), dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=-1)
    out = srgb_to_display_p3_uint8(rgba)
    assert out.shape == rgba.shape
    np.testing.assert_array_equal(out[..., 3], rgba[..., 3])
    rgb_only_out = srgb_to_display_p3_uint8(rgb)
    np.testing.assert_array_equal(out[..., :3], rgb_only_out)


def test_srgb_to_display_p3_rejects_wrong_dtype():
    from opencodecs._cms_codec import srgb_to_display_p3_uint8
    arr = np.zeros((4, 4, 3), dtype=np.uint16)
    with pytest.raises(TypeError, match="uint8"):
        srgb_to_display_p3_uint8(arr)


def test_srgb_to_display_p3_rejects_wrong_shape():
    from opencodecs._cms_codec import srgb_to_display_p3_uint8
    with pytest.raises(ValueError, match=r"\(H, W, 3\|4\)"):
        srgb_to_display_p3_uint8(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match=r"\(H, W, 3\|4\)"):
        srgb_to_display_p3_uint8(np.zeros((4, 4, 5), dtype=np.uint8))


def test_srgb_to_display_p3_out_kwarg():
    """`out=` writes into a preallocated destination of the right
    shape + dtype."""
    from opencodecs._cms_codec import srgb_to_display_p3_uint8
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    out = np.empty_like(arr)
    ret = srgb_to_display_p3_uint8(arr, out=out)
    reference = srgb_to_display_p3_uint8(arr)
    np.testing.assert_array_equal(out, reference)
    np.testing.assert_array_equal(ret, reference)
