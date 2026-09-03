"""Gatan Digital Micrograph reader.

DM has no magic string and no header describing the image: everything is
found by walking a tag tree, so the tests that matter are the ones that
prove the walk stays in step. Two real files from the reference Python
implementation's test data, cross-checked against it.
"""

from __future__ import annotations

import pathlib
import struct

import numpy as np
import pytest

import opencodecs as oc
from opencodecs._dm import DmError, DmFile

DATA = pathlib.Path(__file__).resolve().parent.parent / ".test_data" / "dm"
STACK = DATA / "test_stackbuilder_imagestack.dm3"
HAADF = DATA / "Fei_HAADF-DE_location.dm3"
needs_corpus = pytest.mark.skipif(
    not STACK.is_file(),
    reason="fetch the gatan_dm_samples corpus entry first")


@needs_corpus
def test_tag_tree_walk_reaches_the_image():
    """The whole format is offsets, so getting there at all is the test.

    This parser first failed by returning from an array-of-struct entry
    without stepping over its bytes, which desynchronized everything
    after it: the tree still parsed, produced 11 plausible tags, and
    never found ImageList.
    """
    with DmFile(str(STACK)) as f:
        assert f.version == 3
        assert len(f.tags) > 200
        assert any(k.endswith("/ImageData/Data") for k in f.tags)


@needs_corpus
def test_thumbnail_is_identified_and_skipped():
    """The file says which entry is the preview; believe it.

    Guessing "index 0" is only conventionally right, and getting it
    wrong hands back a 24-bit RGBA rendering in place of the
    acquisition, which has the wrong dtype and the wrong values but a
    perfectly plausible shape.
    """
    with DmFile(str(STACK)) as f:
        assert f.thumbnail_indices == (0,)
        assert f.n_images == 1
        assert f.n_images_including_thumbnails == 2
        real = f.asarray()
        thumb = f.asarray(0, include_thumbnails=True)
    assert real.shape == (3, 2, 16)
    assert thumb.shape != real.shape
    assert thumb.dtype == np.dtype("<u4")          # packed RGBA


@needs_corpus
@pytest.mark.parametrize("path,shape,dtype", [
    (STACK, (3, 2, 16), "u4"),
    (HAADF, (4, 16), "u2"),
])
def test_real_images_decode(path, shape, dtype):
    with DmFile(str(path)) as f:
        a = f.asarray()
    assert a.shape == shape
    assert a.dtype == np.dtype("<" + dtype)


@needs_corpus
@pytest.mark.parametrize("path", [STACK, HAADF], ids=lambda p: p.stem[:20])
def test_matches_rosettasciio(path):
    """Agreement with the reference reader, which also skips thumbnails."""
    rsciio = pytest.importorskip("rsciio.digitalmicrograph")
    ref = rsciio.file_reader(str(path))
    with DmFile(str(path)) as f:
        ours = [f.asarray(i) for i in range(f.n_images)]
    assert len(ours) == len(ref)
    for got, entry in zip(ours, ref):
        expected = entry["data"]
        assert got.shape == expected.shape
        assert np.array_equal(got, expected)


@needs_corpus
def test_three_dimensional_stack_keeps_its_axes():
    """Dimensions are listed fastest-first, so the shape is reversed."""
    with DmFile(str(STACK)) as f:
        a = f.asarray()
    assert a.ndim == 3 and a.shape == (3, 2, 16)


@needs_corpus
def test_out_of_range_image_raises():
    with DmFile(str(STACK)) as f:
        with pytest.raises(IndexError):
            f.asarray(5)


@needs_corpus
def test_registered_and_dispatches_by_extension():
    assert oc.has_codec("dm")
    a = oc.read(str(STACK), format="dm")
    assert a.shape == (3, 2, 16)


# --------------------------------------------------------------------
# refusals and signature
# --------------------------------------------------------------------

def test_bad_version_raises():
    blob = struct.pack(">iii", 7, 100, 1) + b"\x00" * 32
    with pytest.raises(DmError, match="not a\\s+Digital Micrograph"):
        DmFile(blob)


def test_too_short_raises():
    with pytest.raises(DmError, match="too short"):
        DmFile(b"\x00\x00\x00\x03")


def test_signature_does_not_claim_arbitrary_files():
    """No magic string, so the check leans on structure.

    A file whose first four bytes happen to encode 3 must not be taken
    for a DM file just because of that.
    """
    codec = oc.get_codec("dm")
    assert not codec.signature(b"\x00\x00\x00\x03" + b"\xff" * 32)  # order != 0/1
    assert not codec.signature(b"\x00\x00\x00\x03" + b"\x00" * 32)  # root == 0
    assert codec.signature(struct.pack(">iii", 3, 1024, 1) + b"\x00" * 16)
