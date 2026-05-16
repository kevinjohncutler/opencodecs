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
