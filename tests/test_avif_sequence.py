"""AVIF holds image sequences, and we used to return the first frame.

AVIF carries sequences on the same machinery as AV1 video, indexed
through avifDecoderNthImage. decode() only ever produced the primary
image, so an animated AVIF read back as one frame with nothing saying
the rest existed -- a silent wrong answer rather than an error.

imagecodecs writes the fixtures and supplies the expected values.
Round-tripping through our own encoder would let a symmetric bug pass:
if the writer and reader agreed on a wrong layout the test would be
green. imagecodecs links its own libavif and is the reference this
project is measured against elsewhere, so agreement with it is a real
constraint.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc

imagecodecs = pytest.importorskip("imagecodecs")
pytestmark = pytest.mark.skipif(
    not oc.has_codec("avif"), reason="libavif not built here")

N_FRAMES = 6


@pytest.fixture(scope="module")
def frames():
    return np.random.default_rng(0).integers(
        0, 255, (N_FRAMES, 48, 72, 3)).astype("u1")


@pytest.fixture(scope="module")
def sequence(frames):
    blob = imagecodecs.avif_encode(frames, numthreads=1)
    expected = np.asarray(imagecodecs.avif_decode(blob, index=None))
    assert expected.shape == frames.shape, "imagecodecs did not write a sequence"
    return blob, expected


@pytest.fixture(scope="module")
def codec():
    return oc.get_codec("avif")


def test_frame_count_sees_every_frame(codec, sequence):
    blob, expected = sequence
    assert codec.frame_count(blob) == len(expected)


def test_a_still_reports_one_frame(codec, frames):
    assert codec.frame_count(codec.encode(frames[0])) == 1


def test_every_frame_matches_imagecodecs(codec, sequence):
    blob, expected = sequence
    with codec.open(blob) as r:
        assert r.n_frames == len(expected)
        for i in range(r.n_frames):
            assert np.array_equal(r[i], expected[i]), f"frame {i}"


def test_iteration_matches_indexing(codec, sequence):
    blob, expected = sequence
    with codec.open(blob) as r:
        assert np.array_equal(np.stack(list(r.iter_frames())), expected)


def test_out_of_order_access_is_the_same(codec, sequence):
    """Random access must not depend on what was decoded before it.

    A decoder that only stepped forward would pass an in-order test and
    return the wrong frame here.
    """
    blob, expected = sequence
    with codec.open(blob) as r:
        for i in (4, 0, 5, 1, 3, 2, 5, 0):
            assert np.array_equal(r[i], expected[i]), f"frame {i} out of order"


def test_negative_and_out_of_range(codec, sequence):
    blob, expected = sequence
    n = len(expected)
    with codec.open(blob) as r:
        assert np.array_equal(r[-1], expected[n - 1])
        assert np.array_equal(r[-n], expected[0])
        for bad in (n, n + 3, -n - 1):
            with pytest.raises(IndexError):
                r[bad]


def test_read_stacks_a_sequence_and_not_a_still(codec, sequence, frames):
    """A still keeps returning (H, W, C), which is what decode() gives
    for the same file; only a real sequence gains an axis."""
    blob, expected = sequence
    with codec.open(blob) as r:
        assert r.read().shape == expected.shape
    still = codec.encode(frames[0])
    with codec.open(still) as r:
        assert r.read().shape == frames[0].shape
        assert np.array_equal(r.read(), codec.decode(still))


def test_shape_and_dtype_without_decoding_a_frame(codec, sequence):
    blob, expected = sequence
    with codec.open(blob) as r:
        assert r.shape == expected.shape[1:]
        assert r.dtype == expected.dtype


def test_duration_is_reported(codec, sequence):
    blob, _ = sequence
    with codec.open(blob) as r:
        assert r.duration > 0


def test_alpha_survives_a_sequence(codec):
    rgba = np.random.default_rng(3).integers(
        0, 255, (4, 32, 40, 4)).astype("u1")
    blob = imagecodecs.avif_encode(rgba, numthreads=1)
    expected = np.asarray(imagecodecs.avif_decode(blob, index=None))
    if expected.shape[-1] != 4:
        pytest.skip("imagecodecs dropped alpha writing this sequence")
    with codec.open(blob) as r:
        assert r.shape[-1] == 4
        assert np.array_equal(np.stack(list(r.iter_frames())), expected)


def test_decode_still_returns_the_primary_image(codec, sequence):
    """The old entry point must be unchanged for existing callers."""
    blob, expected = sequence
    assert np.array_equal(codec.decode(blob), expected[0])
