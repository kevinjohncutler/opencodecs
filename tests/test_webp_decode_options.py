"""WebP decode options ``index`` and ``hasalpha``, as imagecodecs defines them.

``WebpCodec.decode`` accepted any keyword and dropped it, so
``hasalpha=False`` still returned RGBA and ``index=`` still returned the
whole animation. The reference is ``imagecodecs.webp_decode``, which
links its own libwebp; fixtures are written by imagecodecs, not by the
code under test.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("opencodecs.codecs._webp")
imagecodecs = pytest.importorskip("imagecodecs")

from opencodecs._webp_codec import WebpCodec  # noqa: E402
from opencodecs.codecs import _webp  # noqa: E402
from _ic_reference import skip_if_old_imagecodecs  # noqa: E402

pytestmark = skip_if_old_imagecodecs


def _rgb(seed=0, shape=(12, 17)):
    return np.random.default_rng(seed).integers(0, 256, shape + (3,), dtype=np.uint8)


def _rgba(seed=1, shape=(12, 17)):
    image = np.random.default_rng(seed).integers(0, 256, shape + (4,), dtype=np.uint8)
    image[0, 0, 3] = 0
    return image


@pytest.fixture(scope="module")
def animation():
    frames = np.random.default_rng(5).integers(0, 256, (3, 10, 14, 3), dtype=np.uint8)
    try:
        blob = imagecodecs.webp_encode(frames, lossless=True)
    except (ValueError, TypeError) as exc:
        pytest.skip(f"this imagecodecs cannot encode a WebP animation: {exc}")
    if np.asarray(imagecodecs.webp_decode(blob, index=None)).ndim != 4:
        pytest.skip("this imagecodecs did not write a multi-frame WebP")
    return blob, frames


@pytest.mark.parametrize("hasalpha", [None, True, False])
@pytest.mark.parametrize("make", [_rgb, _rgba], ids=["rgb", "rgba"])
def test_still_hasalpha_matches_imagecodecs(make, hasalpha):
    image = make()
    blob = imagecodecs.webp_encode(image, lossless=True)
    expected = imagecodecs.webp_decode(blob, hasalpha=hasalpha)
    for got in (WebpCodec().decode(blob, hasalpha=hasalpha),
                _webp.decode(blob, hasalpha=hasalpha)):
        assert got.shape == expected.shape
        np.testing.assert_array_equal(got, expected)
    channels = expected.shape[-1]
    np.testing.assert_array_equal(expected[..., :3], image[..., :3])
    if channels == 4 and image.shape[-1] == 3:
        assert (expected[..., 3] == 255).all()


def test_still_hasalpha_into_out():
    image = _rgba()
    blob = imagecodecs.webp_encode(image, lossless=True)
    out = np.empty(image.shape[:2] + (3,), np.uint8)
    assert WebpCodec().decode(blob, hasalpha=False, out=out) is out
    np.testing.assert_array_equal(out, image[..., :3])
    with pytest.raises(ValueError):
        WebpCodec().decode(blob, out=out)      # the file has alpha: 4 channels


@pytest.mark.parametrize("index", [0, 1, 2, -1, -3])
def test_animation_index_matches_imagecodecs(animation, index):
    """An animation without alpha decodes to RGB, as in imagecodecs.

    This returned RGBA with a constant 255 alpha. The fixture's VP8X
    header has the animation flag and not the alpha flag, which the
    test reads itself rather than trusting the code under test.
    """
    blob, frames = animation
    assert blob[12:16] == b"VP8X" and blob[20] & 0x02 and not blob[20] & 0x10
    expected = imagecodecs.webp_decode(blob, index=index)
    got = WebpCodec().decode(blob, index=index)
    assert got.shape == expected.shape == frames.shape[1:3] + (3,)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(got, frames[index])
    for hasalpha in (False, True):
        np.testing.assert_array_equal(
            WebpCodec().decode(blob, index=index, hasalpha=hasalpha),
            imagecodecs.webp_decode(blob, index=index, hasalpha=hasalpha))
    out = np.empty_like(got)
    assert WebpCodec().decode(blob, index=index, out=out) is out
    np.testing.assert_array_equal(out, expected)


def test_animation_stack_and_hasalpha(animation):
    blob, frames = animation
    whole = WebpCodec().decode(blob)
    expected = imagecodecs.webp_decode(blob)
    assert whole.shape == expected.shape == frames.shape
    np.testing.assert_array_equal(whole, expected)
    np.testing.assert_array_equal(whole, frames)
    rgb = WebpCodec().decode(blob, hasalpha=False, numthreads=1)
    np.testing.assert_array_equal(rgb, imagecodecs.webp_decode(blob, hasalpha=False))
    rgba = WebpCodec().decode(blob, hasalpha=True)
    assert rgba.shape == frames.shape[:3] + (4,)
    np.testing.assert_array_equal(rgba, imagecodecs.webp_decode(blob, hasalpha=True))
    with pytest.raises(ValueError):
        WebpCodec().decode(blob, out=np.empty_like(whole))


def _moving_square(n=4, channels=3, shape=(40, 60)):
    """Similar consecutive frames, the common animation: libwebp's
    animation encoder stores them as blended sub-frames and sets the
    VP8X alpha flag even though every input pixel is opaque."""
    frames = np.full((n,) + shape + (channels,), 20, np.uint8)
    if channels == 4:
        frames[..., 3] = 255
    for i in range(n):
        frames[i, 5:15, 5 + 8 * i:15 + 8 * i, 0] = 255
    return frames


def _opaque_and_transparent():
    rgb = np.random.default_rng(7).integers(0, 256, (12, 16, 3), dtype=np.uint8)
    opaque = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    transparent = opaque.copy()
    transparent[:4, :5, 3] = 0
    return opaque, transparent


def _animations():
    opaque, transparent = _opaque_and_transparent()
    yield "moving_rgb", _moving_square()
    yield "moving_rgba_opaque", _moving_square(channels=4)
    yield "opaque_then_transparent", np.stack([opaque, transparent])
    yield "transparent_then_opaque", np.stack([transparent, opaque])
    yield "opaque_opaque_transparent", np.stack(
        [opaque, opaque[::-1].copy(), transparent])


@pytest.mark.parametrize("lossless", [True, False], ids=["lossless", "lossy"])
@pytest.mark.parametrize("name,frames", list(_animations()),
                         ids=[n for n, _ in _animations()])
def test_animation_channels_match_imagecodecs(name, frames, lossless):
    """With hasalpha=None an animation keeps alpha only when a returned
    canvas has a pixel that is not fully opaque, as imagecodecs decides:
    every frame for the stack, the selected frame for index=.

    The expected channel count comes from the input frames, an
    independent reference: opaque input gives RGB however libwebp
    flagged the file. The moving-square fixture is the case that broke
    a VP8X-flag rule: libwebp flags it (0x10) although nothing is
    transparent, and imagecodecs still returns RGB.
    """
    try:
        blob = imagecodecs.webp_encode(frames, lossless=lossless)
    except (ValueError, TypeError) as exc:
        pytest.skip(f"this imagecodecs cannot encode a WebP animation: {exc}")
    if name.startswith("moving") and lossless:
        assert blob[12:16] == b"VP8X" and blob[20] & 0x10
    opaque = (frames[..., 3] == 255).reshape(len(frames), -1).all(1) \
        if frames.shape[-1] == 4 else np.ones(len(frames), bool)
    whole = WebpCodec().decode(blob)
    expected = imagecodecs.webp_decode(blob)
    assert whole.shape == expected.shape
    assert whole.shape[-1] == (3 if opaque.all() else 4)
    np.testing.assert_array_equal(whole, expected)
    for index in range(-len(frames), len(frames)):
        got = WebpCodec().decode(blob, index=index)
        expected = imagecodecs.webp_decode(blob, index=index)
        assert got.shape == expected.shape
        assert got.shape[-1] == (3 if opaque[index] else 4)
        np.testing.assert_array_equal(got, expected)
        out = np.empty_like(got)
        assert WebpCodec().decode(blob, index=index, out=out) is out
        np.testing.assert_array_equal(out, expected)
    if lossless and opaque.all():
        np.testing.assert_array_equal(whole, frames[..., :3])


@pytest.mark.parametrize("index", [3, -4, 10])
def test_animation_index_out_of_range(animation, index):
    blob, _ = animation
    with pytest.raises(IndexError):
        imagecodecs.webp_decode(blob, index=index)
    with pytest.raises(IndexError):
        WebpCodec().decode(blob, index=index)


def test_still_has_one_frame():
    image = _rgb()
    blob = imagecodecs.webp_encode(image, lossless=True)
    for index in (0, -1, None):
        np.testing.assert_array_equal(WebpCodec().decode(blob, index=index), image)
    for index in (1, -2):
        with pytest.raises(IndexError):
            imagecodecs.webp_decode(blob, index=index)
        with pytest.raises(IndexError):
            WebpCodec().decode(blob, index=index)


def test_unknown_decode_option_raises():
    blob = imagecodecs.webp_encode(_rgb(), lossless=True)
    with pytest.raises(TypeError):
        WebpCodec().decode(blob, has_alpha=False)
