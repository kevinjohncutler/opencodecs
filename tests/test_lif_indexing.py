"""Reaching LIF image N does not cost images 0 through N-1.

A LIF's XML header carries every image's memory-block offset and
length, so an image is a seek and a read; array_for_image was already
doing exactly that. What was missing was a __getitem__ to reach it, so
indexing fell through to Reader's default, which walks iter_frames().

The second thing these pin is that indexing does not MOVE the reader.
image() deliberately does, and iter_frames() used to go through it,
which left the reader pointing at the last image after iterating.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import opencodecs as oc

DATA = pathlib.Path(__file__).resolve().parent.parent / ".test_data" / "lif"
# Skip AppleDouble sidecars: the corpus lives on an SMB share, so
# ._<name>.lif files sit beside the real ones and glob returns them.
# One is 368 bytes of resource fork and fails with "bad magic".
LIFS = ([p for p in sorted(DATA.glob("*.lif")) if not p.name.startswith("._")]
        if DATA.is_dir() else [])
needs_corpus = pytest.mark.skipif(
    not LIFS, reason="fetch a LIF corpus entry first")


@pytest.fixture(params=["native", "auto"])
def reader(request):
    codec = oc.get_codec("lif")
    kw = {"backend": "native"} if request.param == "native" else {}
    try:
        r = codec.open(str(LIFS[0]), **kw)
    except TypeError:
        r = codec.open(str(LIFS[0]))
    with r:
        yield r


@needs_corpus
def test_indexing_agrees_with_iteration(reader):
    for i, frame in enumerate(reader.iter_frames()):
        assert np.array_equal(frame, reader[i]), f"image {i}"


@needs_corpus
def test_indexing_does_not_move_the_reader(reader):
    """r[i] must not change what read() returns next."""
    before = reader.read()
    if reader.n_images > 1:
        reader[reader.n_images - 1]
    after = reader.read()
    assert np.array_equal(before, after)


@needs_corpus
def test_iterating_leaves_the_reader_where_it_started(reader):
    before = reader.read()
    for _ in reader.iter_frames():
        pass
    assert np.array_equal(reader.read(), before)


@needs_corpus
def test_negative_and_out_of_range(reader):
    n = reader.n_images
    assert np.array_equal(reader[-1], reader[n - 1])
    for bad in (n, n + 3, -n - 1):
        with pytest.raises(IndexError):
            reader[bad]


@needs_corpus
def test_the_flag_and_the_method_agree():
    """chunked=True over the inherited walking __getitem__ is the
    contradiction test_chunked_means_cheap.py exists for."""
    codec = oc.get_codec("lif")
    assert codec.chunked is True
    from opencodecs._lif_native import LifNativeReader
    assert "__getitem__" in LifNativeReader.__dict__
