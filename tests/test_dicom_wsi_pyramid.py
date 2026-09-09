"""DICOM whole-slide images are pyramids spread across files.

A VL Whole Slide Microscopy image is a SERIES of instances describing
the same physical area at different pixel extents, sharing a
SeriesInstanceUID and an Imaged Volume and distinguished by Total Pixel
Matrix Columns / Rows. Nothing inside any one instance says it is level
2 of 5, so the pyramid only exists across the set -- which is why this
is a reader of its own rather than a switch on DicomFile.

pydicom writes the fixtures. It is the reference implementation of the
format, so the tags are spelled the way it spells them rather than the
way this reader happens to expect them.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc
from opencodecs._dicom import DicomError

from _dicom_wsi_writer import write_wsi_series

pytest.importorskip("pydicom")
pytestmark = pytest.mark.skipif(
    not oc.has_codec("dicom"), reason="dicom codec not available")

BASE = (np.arange(128 * 128, dtype="u1") % 251).reshape(128, 128)
LEVELS = [BASE, BASE[::2, ::2], BASE[::4, ::4], BASE[::8, ::8]]


@pytest.fixture
def slide(tmp_path):
    paths = write_wsi_series(tmp_path, LEVELS)
    if paths is None:
        pytest.skip("pydicom cannot write the fixture here")
    return tmp_path


def _open(src, **kw):
    from opencodecs._dicom_pyramid import DicomWsiPyramid
    return DicomWsiPyramid(src, **kw)


def test_levels_are_ordered_by_extent_not_filename(slide):
    """The fixture names files in the REVERSE of resolution order.

    The standard does not require instances to be named or ordered
    usefully and real scanners do not oblige, so a reader that trusted
    the directory listing would come out upside down and pass a
    "reads 4 levels" test.
    """
    with _open(slide) as pyr:
        assert pyr.n_levels == len(LEVELS)
        assert pyr.shapes == tuple(l.shape for l in LEVELS)


def test_downscale_factors(slide):
    with _open(slide) as pyr:
        assert pyr.downscale_factors == ((1, 1), (2, 2), (4, 4), (8, 8))


def test_each_level_decodes_to_its_own_pixels(slide):
    with _open(slide) as pyr:
        for i, expected in enumerate(LEVELS):
            got = pyr.level(i).reader.asarray()
            assert got.shape == expected.shape, i
            assert np.array_equal(got, expected), i


def test_best_level_for_picks_by_envelope(slide):
    with _open(slide) as pyr:
        assert pyr.best_level_for(max_pixels_y=200, max_pixels_x=200) == 0
        assert pyr.best_level_for(max_pixels_y=40, max_pixels_x=40) == 2
        assert pyr.best_level_for() == 0


def test_dtype_and_imaged_volume(slide):
    with _open(slide) as pyr:
        assert pyr.dtype == np.dtype("u1")
        assert pyr.imaged_volume == (1.0, 1.0)


def test_a_list_of_paths_works_too(slide):
    paths = sorted(slide.glob("*.dcm"))
    with _open(paths) as pyr:
        assert pyr.shapes == tuple(l.shape for l in LEVELS)


def test_two_series_in_one_directory_are_refused(tmp_path):
    """Silently mixing them would build a pyramid out of two slides."""
    write_wsi_series(tmp_path, LEVELS[:2])
    write_wsi_series(tmp_path / "b", LEVELS[2:]) if False else None
    (tmp_path / "b").mkdir(exist_ok=True)
    second = write_wsi_series(tmp_path / "b", LEVELS[2:])
    for p in second:
        p.rename(tmp_path / ("other_" + p.name))
    with pytest.raises(DicomError, match="series"):
        _open(tmp_path)


def test_choosing_a_series_by_uid(tmp_path):
    from pydicom.uid import generate_uid
    uid_a, uid_b = generate_uid(), generate_uid()
    write_wsi_series(tmp_path, LEVELS[:2], series_uid=uid_a)
    (tmp_path / "b").mkdir()
    for p in write_wsi_series(tmp_path / "b", LEVELS[2:], series_uid=uid_b):
        p.rename(tmp_path / ("other_" + p.name))
    with _open(tmp_path, series=uid_b) as pyr:
        assert pyr.n_levels == 2
        assert pyr.shapes == tuple(l.shape for l in LEVELS[2:])
    with pytest.raises(DicomError, match="no series"):
        _open(tmp_path, series="1.2.3.4.5")


def test_a_non_whole_slide_dicom_is_refused(tmp_path):
    """A CT is a DICOM and is not a pyramid."""
    from test_dicom_encodings import build_dicom, explicit_le
    a = (np.arange(16 * 20, dtype="u2") % 4000).reshape(16, 20)
    (tmp_path / "ct.dcm").write_bytes(build_dicom(explicit_le(), a))
    with pytest.raises(DicomError, match="Whole Slide"):
        _open(tmp_path)


def test_stray_files_in_the_directory_are_skipped(slide):
    """Slide directories routinely carry a DICOMDIR and junk."""
    (slide / "DICOMDIR").write_bytes(b"not a dicom file at all")
    (slide / "notes.txt").write_text("hello")
    with _open(slide) as pyr:
        assert pyr.n_levels == len(LEVELS)


def test_rgb_slides(tmp_path):
    rgb = [np.stack([l, l[::-1], l], axis=-1) for l in LEVELS[:2]]
    if write_wsi_series(tmp_path, rgb, samples_per_pixel=3) is None:
        pytest.skip("pydicom unavailable")
    with _open(tmp_path) as pyr:
        assert pyr.shapes[0] == rgb[0].shape
        assert np.array_equal(pyr.level(0).reader.asarray(), rgb[0])


def test_open_pyramid_dispatches_to_it(slide):
    """A directory has to be asked for by format=, because no
    extension implies a series of files."""
    with oc.open_pyramid(str(slide), format="dicom") as pyr:
        assert pyr.n_levels == len(LEVELS)
        assert pyr.shapes == tuple(l.shape for l in LEVELS)


def test_read_region_goes_through_the_chosen_level(slide):
    with oc.open_pyramid(str(slide), format="dicom") as pyr:
        lvl = pyr.best_level_for(max_pixels_y=20, max_pixels_x=20)
        got = pyr.read_region(lvl, y=(0, 8), x=(0, 8))
        assert np.array_equal(got, LEVELS[lvl][:8, :8])


def test_a_single_dcm_path_is_a_one_level_pyramid(slide):
    """Degenerate but not an error: one instance is one level."""
    one = sorted(slide.glob("*.dcm"))[0]
    with oc.open_pyramid(str(one)) as pyr:
        assert pyr.n_levels == 1
