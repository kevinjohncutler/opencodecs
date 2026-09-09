"""Animated WebP, which this codec did not read at all.

An animation decoded as its first frame with nothing saying the rest
existed. Reading it needs libwebpdemux, a separate library from libwebp
but not an optional one: libwebp's CMakeLists adds the webpdemux target
and installs it unconditionally.

imagecodecs writes the fixtures and supplies the expected values. It
links its own libwebp and is the reference this project is measured
against elsewhere, so bit-identical agreement is a real constraint,
where a round trip through our own encoder would pass on a symmetric
bug.

WebP frames are sub-rectangles with disposal and blending rules, so
frame N genuinely requires the ones before it and chunked stays False.
The tests check that the flag and the reader agree about that rather
than pretending otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc

imagecodecs = pytest.importorskip("imagecodecs")
pytestmark = pytest.mark.skipif(
    not oc.has_codec("webp"), reason="libwebp not built here")

N = 5


@pytest.fixture(scope="module")
def codec():
    return oc.get_codec("webp")


@pytest.fixture(scope="module")
def frames():
    return np.random.default_rng(0).integers(
        0, 255, (N, 32, 48, 3)).astype("u1")


@pytest.fixture(scope="module")
def animation(frames):
    blob = imagecodecs.webp_encode(frames, lossless=True)
    ref = np.asarray(imagecodecs.webp_decode(blob, index=None))
    assert ref.shape[0] == N, "imagecodecs did not write an animation"
    return blob, ref


def _as_rgba(a):
    """Our reader always returns RGBA; imagecodecs may return RGB."""
    if a.shape[-1] == 4:
        return a
    alpha = np.full(a.shape[:-1] + (1,), 255, dtype=a.dtype)
    return np.concatenate([a, alpha], axis=-1)


def test_frame_count_sees_every_frame(codec, animation):
    blob, ref = animation
    assert codec.frame_count(blob) == len(ref)


def test_a_still_reports_one_frame(codec, frames):
    assert codec.frame_count(codec.encode(frames[0], lossless=True)) == 1


def test_frames_are_bit_identical_to_imagecodecs(codec, animation):
    blob, ref = animation
    with codec.open(blob) as r:
        assert r.n_frames == len(ref)
        assert np.array_equal(np.stack(list(r.iter_frames())), _as_rgba(ref))


def test_indexing_matches_iteration(codec, animation):
    blob, ref = animation
    with codec.open(blob) as r:
        walked = list(r.iter_frames())
        for i in range(r.n_frames):
            assert np.array_equal(r[i], walked[i])


def test_negative_and_out_of_range(codec, animation):
    blob, ref = animation
    n = len(ref)
    with codec.open(blob) as r:
        assert np.array_equal(r[-1], r[n - 1])
        assert np.array_equal(r[-n], r[0])
        for bad in (n, n + 2, -n - 1):
            with pytest.raises(IndexError):
                r[bad]


def test_timestamps_increase(codec, animation):
    blob, _ = animation
    with codec.open(blob) as r:
        assert len(r.timestamps) == r.n_frames
        assert all(b > a for a, b in zip(r.timestamps, r.timestamps[1:]))


def test_read_stacks_an_animation_and_not_a_still(codec, animation, frames):
    blob, ref = animation
    with codec.open(blob) as r:
        assert r.read().shape == _as_rgba(ref).shape
    still = codec.encode(frames[0], lossless=True)
    with codec.open(still) as r:
        assert r.read().shape == frames[0].shape
        assert np.array_equal(r.read(), codec.decode(still))


def test_decode_returns_the_first_frame_of_an_animation(codec, animation):
    """This used to raise "WebP decode failed".

    The plain decoder cannot read an animation container, so the old
    behavior was an unexplained error rather than a wrong answer.
    Returning the first frame matches what the AVIF and HEIF codecs do
    for a multi-image file.
    """
    blob, ref = animation
    got = codec.decode(blob)
    assert got.shape[:2] == ref.shape[1:3]
    with codec.open(blob) as r:
        assert np.array_equal(got, r[0])


def test_decode_of_a_still_is_unchanged(codec, frames):
    still = codec.encode(frames[0], lossless=True)
    assert np.array_equal(codec.decode(still), frames[0])


def test_a_corrupt_still_still_raises(codec, frames):
    """The animation fallback must not swallow a real decode failure."""
    still = bytearray(codec.encode(frames[0], lossless=True))
    still[20:60] = b"\x00" * 40
    with pytest.raises(Exception):
        codec.decode(bytes(still))


def test_chunked_is_false_and_the_reader_says_so(codec, animation):
    """WebP frames carry disposal state, so random access cannot be
    cheap. The flag and the reader must agree rather than one of them
    advertising something the other does not do."""
    blob, _ = animation
    assert codec.chunked is False
    with codec.open(blob) as r:
        assert r.is_chunked is False


def test_an_alpha_animation_survives(codec):
    rgba = np.random.default_rng(4).integers(
        0, 255, (3, 24, 32, 4)).astype("u1")
    blob = imagecodecs.webp_encode(rgba, lossless=True)
    ref = np.asarray(imagecodecs.webp_decode(blob, index=None))
    with codec.open(blob) as r:
        assert r.shape[-1] == 4
        assert np.array_equal(np.stack(list(r.iter_frames())), _as_rgba(ref))
