"""ICC profile pass-through tests for PNG / JPEG / HEIF / AVIF.

The codec's ``encode(iccprofile=...)`` must embed the bytes into the
container, and ``read_icc_profile`` must recover them byte-identically.
``decode()`` is unchanged — the ICC profile is metadata, not pixel data.

We use a synthetic ICC blob (not a real sRGB profile) because we're
testing the embed/extract plumbing, not the profile contents.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc


_TEST_ICC = b"FAKE-ICC-PROFILE-FOR-PHASE-5-TESTS-" * 100  # 3500 bytes


def _rgb_uint8():
    return np.random.default_rng(0).integers(0, 256, (64, 64, 3),
                                              dtype=np.uint8)


@pytest.mark.parametrize("codec_name", ["png", "jpeg", "heif", "avif"])
def test_image_codec_iccprofile_roundtrip(codec_name):
    """Every supported image codec embeds and recovers ICC byte-identically."""
    if not oc.has_codec(codec_name):
        pytest.skip(f"{codec_name} codec not built")
    codec = oc.get_codec(codec_name)

    arr = _rgb_uint8()
    encode_kwargs = {}
    if codec_name == "heif":
        # libheif refuses to encode non-multiple-of-2 dimensions on some
        # x265 builds; pad if needed (our 64×64 is fine, but be defensive).
        encode_kwargs = {"lossless": True}
    if codec_name == "avif":
        encode_kwargs = {"lossless": True}

    plain = codec.encode(arr, **encode_kwargs)
    assert codec.read_icc_profile(plain) is None, (
        f"{codec_name}: blob with no iccprofile= shouldn't carry one"
    )

    with_icc = codec.encode(arr, iccprofile=_TEST_ICC, **encode_kwargs)
    got = codec.read_icc_profile(with_icc)
    assert got == _TEST_ICC, (
        f"{codec_name}: ICC roundtrip mismatch "
        f"(in={len(_TEST_ICC)} out={len(got) if got else None})"
    )


@pytest.mark.parametrize("codec_name", ["png", "jpeg", "heif", "avif"])
def test_image_codec_decode_still_works_with_icc(codec_name):
    """Embedding an ICC profile mustn't perturb the decoded pixels."""
    if not oc.has_codec(codec_name):
        pytest.skip(f"{codec_name} codec not built")
    codec = oc.get_codec(codec_name)

    arr = _rgb_uint8()
    extra = {"lossless": True} if codec_name in ("heif", "avif") else {}
    blob = codec.encode(arr, iccprofile=_TEST_ICC, **extra)
    back = codec.decode(blob)
    assert back.shape == arr.shape
    assert back.dtype == np.uint8
    # For lossy codecs (jpeg) skip pixel-equality; just check geometry.
    if codec_name in ("png", "heif", "avif"):
        np.testing.assert_array_equal(back, arr)
