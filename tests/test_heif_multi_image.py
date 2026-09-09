"""A HEIF holds a set of top-level images; we read only the primary.

A burst, a Live Photo's stills, a depth-plus-color capture: all put
more than one top-level image in the file. decode() returned the
primary and said nothing about the rest, which is a silent wrong
answer rather than an error.

The fixture is written by libheif itself through ctypes, not by this
package's encoder. Nothing else here can write one -- imagecodecs
ships no heif_encode and our encoder takes a single image -- and more
to the point, a fixture produced and consumed by the same code would
pass even if both agreed on something wrong.

HEVC is lossy, so the content assertions are on values that survive it
unambiguously: flat frames of known brightness, checked by mean. That
tests ordering and identity, which is what the multi-image path is
actually responsible for; pixel fidelity is the single-image decode's
job and is covered against the corpus file elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc

from _libheif_writer import write_multi_image_heif, count_top_level_images

pytestmark = pytest.mark.skipif(
    not oc.has_codec("heif"), reason="libheif not built here")

LEVELS = (20, 80, 140, 200, 250)


@pytest.fixture(scope="module")
def multi(tmp_path_factory):
    """Five flat frames of increasing brightness, written by libheif."""
    frames = [np.full((32, 48, 3), v, dtype="u1") for v in LEVELS]
    p = tmp_path_factory.mktemp("heif") / "multi.heic"
    n = write_multi_image_heif(p, frames)
    if n is None:
        pytest.skip("no libheif with an HEVC encoder to write the fixture")
    assert n == len(LEVELS), f"libheif wrote {n} images, expected {len(LEVELS)}"
    return p.read_bytes()


@pytest.fixture(scope="module")
def codec():
    return oc.get_codec("heif")


def test_our_count_agrees_with_libheif(multi, tmp_path, codec):
    p = tmp_path / "c.heic"
    p.write_bytes(multi)
    assert codec.frame_count(multi) == count_top_level_images(p)
    assert codec.frame_count(multi) == len(LEVELS)


def test_a_single_image_heif_reports_one(codec):
    still = codec.encode(np.full((32, 40, 3), 90, dtype="u1"))
    assert codec.frame_count(still) == 1


def test_every_image_comes_back_in_order(codec, multi):
    """Ordering is the whole claim: image i must be the i-th written.

    A reader that returned the primary for every index would pass a
    shape test and fail this one.
    """
    with codec.open(multi) as r:
        assert r.n_frames == len(LEVELS)
        for i, level in enumerate(LEVELS):
            got = float(r[i].mean())
            assert abs(got - level) < 12, (
                f"image {i} has mean {got:.1f}, expected about {level}")


def test_indexing_matches_iteration(codec, multi):
    with codec.open(multi) as r:
        walked = [f.mean() for f in r.iter_frames()]
        indexed = [r[i].mean() for i in range(r.n_frames)]
    assert walked == pytest.approx(indexed)


def test_out_of_order_access(codec, multi):
    with codec.open(multi) as r:
        for i in (3, 0, 4, 1, 2, 4):
            assert abs(float(r[i].mean()) - LEVELS[i]) < 12


def test_negative_and_out_of_range(codec, multi):
    n = len(LEVELS)
    with codec.open(multi) as r:
        assert abs(float(r[-1].mean()) - LEVELS[-1]) < 12
        assert abs(float(r[-n].mean()) - LEVELS[0]) < 12
        for bad in (n, n + 2, -n - 1):
            with pytest.raises(IndexError):
                r[bad]


def test_decode_index_matches_the_reader(codec, multi):
    with codec.open(multi) as r:
        for i in range(r.n_frames):
            assert np.array_equal(codec.decode(multi, index=i), r[i])


def test_decode_without_index_is_unchanged(codec, multi):
    """The existing entry point must still return the primary image."""
    primary = codec.decode(multi)
    assert primary.shape == (32, 48, 3)


def test_read_stacks_only_when_shapes_agree(codec, tmp_path):
    """Images in one HEIF need not share a shape -- a depth map is
    smaller than its color image -- so read() returns a list rather
    than raising a broadcast error."""
    frames = [np.full((32, 48, 3), 60, dtype="u1"),
              np.full((16, 24, 3), 200, dtype="u1")]
    p = tmp_path / "ragged.heic"
    if write_multi_image_heif(p, frames) is None:
        pytest.skip("no libheif with an HEVC encoder")
    with codec.open(p.read_bytes()) as r:
        out = r.read()
    assert isinstance(out, list) and len(out) == 2
    assert out[0].shape[:2] == (32, 48)
    assert out[1].shape[:2] == (16, 24)


def test_the_corpus_file_still_reads(codec):
    import pathlib
    p = (pathlib.Path(__file__).resolve().parent.parent
         / ".test_data" / "heif" / "C001.heic")
    if not p.is_file():
        pytest.skip("fetch the heif corpus entry first")
    data = p.read_bytes()
    assert codec.frame_count(data) == 1
    with codec.open(data) as r:
        assert np.array_equal(r[0], codec.decode(data))
