"""Every CZI opencodecs writes must open in the OTHER readers, not just ours.

WHY THIS FILE EXISTS
--------------------
On 2026-10-04 czicompress refused a file this writer had just produced:

    An unrecoverable runtime error occurred: Root-node "ImageDocument" not found.

The writer's default metadata was a bare ``<Metadata/>``. opencodecs' own
reader accepts that, so every round-trip test in the suite passed while the
files were unopenable by libCZI, czicompress and ZEN -- which is to say, by
everything a user would actually hand the file to. The bug was not in any
code path; it was in the shape of the testing. A suite that only reads its
own output cannot find it.

So these tests never use opencodecs to read. They write with opencodecs and
open with:

    pylibCZIrw     the libCZI binding, i.e. what ZEN and czicompress use
    czifile        Christoph Gohlke's independent pure-Python reader
    aicspylibczi   the Allen Institute's libCZI binding

Any reader that is not installed is skipped rather than silently passed over,
so a thinned environment shows up as skips instead of false confidence.

WHAT TO DO WHEN ONE OF THESE FAILS
----------------------------------
Assume the writer is wrong, not the reader. All three of these open files from
ZEN; we are the new participant in a format we did not define. A disagreement
between our reader and theirs means our reader is probably lenient in the same
way the writer is wrong, which is exactly how the metadata bug survived.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from opencodecs import CziWriter, czi_recompress  # noqa: E402

CORPUS = pathlib.Path(__file__).resolve().parents[1] / ".test_data" / "czi"


def _write(path, *, n=3, side=64, compression="none", **kw):
    """A small multi-channel CZI, written by opencodecs."""
    rs = np.random.RandomState(5)
    with CziWriter(str(path), compression=compression, **kw) as w:
        for c in range(n):
            w.write_frame(rs.randint(0, 4096, (side, side)).astype("uint16"),
                          dims=[("X", 0, side, side), ("Y", 0, side, side),
                                ("C", c, 1, 1)])
    return path


# --------------------------------------------------------------- pylibCZIrw
# The one that matters most: it is libCZI, so it is what ZEN and czicompress
# see. A file this reader rejects is a file our users cannot open.

@pytest.mark.parametrize("compression", ["none", "zstd", "zstdhdr"])
def test_pylibczirw_opens_what_we_write(tmp_path, compression):
    czirw = pytest.importorskip("pylibCZIrw.czi", reason="pylibCZIrw missing")
    src = _write(tmp_path / f"{compression}.czi", compression=compression)
    with czirw.open_czi(str(src)) as r:
        plane = r.read(plane={"C": 0})
        assert plane is not None and plane.size > 0
        assert r.total_bounding_rectangle[2:] == (64, 64)


def test_pylibczirw_reads_our_pixels_back_exactly(tmp_path):
    """Lossless must mean lossless to a FOREIGN decoder, not only to ours."""
    czirw = pytest.importorskip("pylibCZIrw.czi", reason="pylibCZIrw missing")
    rs = np.random.RandomState(9)
    frame = rs.randint(0, 4096, (64, 64)).astype("uint16")
    out = tmp_path / "exact.czi"
    with CziWriter(str(out), compression="zstdhdr") as w:
        w.write_frame(frame, dims=[("X", 0, 64, 64), ("Y", 0, 64, 64),
                                   ("C", 0, 1, 1)])
    with czirw.open_czi(str(out)) as r:
        got = np.squeeze(r.read(plane={"C": 0}))
    assert np.array_equal(got, frame), "a foreign reader sees different pixels"


def test_default_metadata_is_openable_by_libczi(tmp_path):
    """The exact bug this file was created for.

    The old default produced files libCZI refused outright. Asserting through
    our own reader could never have caught it.
    """
    czirw = pytest.importorskip("pylibCZIrw.czi", reason="pylibCZIrw missing")
    src = _write(tmp_path / "default_meta.czi")      # no metadata_xml passed
    with czirw.open_czi(str(src)) as r:
        assert r.total_bounding_rectangle is not None


def test_rootless_metadata_still_opens_because_it_is_wrapped(tmp_path):
    """Handing in the old broken value must not produce an unopenable file."""
    czirw = pytest.importorskip("pylibCZIrw.czi", reason="pylibCZIrw missing")
    src = _write(tmp_path / "rootless.czi", metadata_xml=b"<Metadata/>")
    with czirw.open_czi(str(src)) as r:
        assert r.total_bounding_rectangle is not None


# ------------------------------------------------------------------ czifile
# An independent implementation, not libCZI. It disagrees with libCZI in
# places, which is the point of having both.

def test_czifile_opens_what_we_write(tmp_path):
    czifile = pytest.importorskip("czifile", reason="czifile missing")
    src = _write(tmp_path / "cf.czi", compression="zstdhdr")
    with czifile.CziFile(str(src)) as c:
        # czifile exposes the directory and asarray(), not a .shape attribute.
        assert len(c.filtered_subblock_directory) == 3, "sub-blocks not seen"
        data = c.asarray()
        assert data.size > 0 and data.dtype == np.uint16


# ------------------------------------------------------------- aicspylibczi

def test_aicspylibczi_opens_what_we_write(tmp_path):
    aics = pytest.importorskip("aicspylibczi", reason="aicspylibczi missing")
    src = _write(tmp_path / "aics.czi", compression="zstdhdr")
    czi = aics.CziFile(str(src))
    assert czi.get_dims_shape() is not None
    img, _shape = czi.read_image(C=0)
    assert img.size > 0


# ------------------------------------------- recompressed real files
# czi_recompress output is what the ingest pipeline actually produces, so it
# is the output that most needs to open elsewhere.

@pytest.mark.skipif(not (CORPUS / "idr0011_plate1_scene1.czi").is_file(),
                    reason="corpus CZI not present")
def test_recompressed_real_file_opens_in_libczi(tmp_path):
    czirw = pytest.importorskip("pylibCZIrw.czi", reason="pylibCZIrw missing")
    dst = tmp_path / "re.czi"
    czi_recompress(CORPUS / "idr0011_plate1_scene1.czi", dst,
                   compression="zstdhdr")
    with czirw.open_czi(str(dst)) as r:
        assert r.total_bounding_rectangle is not None


@pytest.mark.skipif(not (CORPUS / "ome_axioscan_pyramid.czi").is_file(),
                    reason="corpus slide scan not present")
def test_carried_jpegxr_file_opens_in_libczi(tmp_path):
    """Carrying lossy sub-blocks across must leave a file libCZI still reads.

    This is the strongest interop case available: 481 JPEG XR sub-blocks
    copied verbatim into a container our writer built. If the container is
    wrong, libCZI will say so even though the payloads are untouched.
    """
    czirw = pytest.importorskip("pylibCZIrw.czi", reason="pylibCZIrw missing")
    dst = tmp_path / "carried.czi"
    info = czi_recompress(CORPUS / "ome_axioscan_pyramid.czi", dst,
                          compression="zstdhdr",
                          recompress_when=lambda e: e.compression == 0)
    assert info["carried"] == info["subblocks"]
    with czirw.open_czi(str(dst)) as r:
        assert r.total_bounding_rectangle is not None
