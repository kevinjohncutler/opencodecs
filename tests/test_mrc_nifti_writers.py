"""MRC and NIfTI writers.

Round-tripping through our own reader proves self-consistency and little
else, so the tests that matter here read the output back with the
reference implementations, mrcfile and nibabel. A writer that agrees
with itself and with nobody else is the failure mode worth catching.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc
from opencodecs._mrc import MrcError, MrcStream
from opencodecs._mrc_writer import encode_mrc, write_mrc
from opencodecs._nifti import NiftiError, NiftiStream
from opencodecs._nifti_writer import encode_nifti, write_nifti


# --------------------------------------------------------------------
# MRC
# --------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["i1", "i2", "u2", "f2", "f4"])
def test_mrc_roundtrip_dtypes(dtype):
    a = (np.arange(3 * 4 * 5) % 29).astype(dtype).reshape(3, 4, 5)
    with MrcStream(encode_mrc(a)) as r:
        assert r.shape == (3, 4, 5)
        assert np.array_equal(np.asarray(r.asarray(), dtype=dtype), a)


def test_mrc_two_dimensional_is_written_as_one_section():
    a = np.arange(12, dtype="f4").reshape(3, 4)
    with MrcStream(encode_mrc(a)) as r:
        assert r.header["nz"] == 1
        assert r.shape == (3, 4)
        assert np.array_equal(r.asarray(), a)


def test_mrc_uint8_is_widened_not_reinterpreted():
    """MODE 0 is signed, so 200 must not become -56.

    Reinterpreting would round-trip through our own reader and be wrong
    in every other tool, which is the worst kind of correct.
    """
    a = np.array([[0, 127, 200, 255]], dtype="u1")
    with MrcStream(encode_mrc(a)) as r:
        out = r.asarray()
    assert out.dtype == np.dtype("<i2")
    assert list(out.ravel()) == [0, 127, 200, 255]


def test_mrc_header_statistics_are_filled():
    """Chimera and IMOD set contrast from these; zeros display as blank."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2, 8, 8)).astype("f4")
    with MrcStream(encode_mrc(a)) as r:
        h = r.header
    assert h["dmin"] == pytest.approx(float(a.min()), rel=1e-5)
    assert h["dmax"] == pytest.approx(float(a.max()), rel=1e-5)
    assert h["dmean"] == pytest.approx(float(a.mean()), rel=1e-4)
    assert h["rms"] == pytest.approx(float(a.std()), rel=1e-4)


def test_mrc_voxel_size_survives():
    a = np.zeros((2, 4, 6), dtype="f4")
    with MrcStream(encode_mrc(a, voxel_size=(1.5, 2.0, 3.0))) as r:
        assert r.voxel_size == pytest.approx((1.5, 2.0, 3.0), rel=1e-5)


def test_mrc_unsupported_dtype_is_refused():
    with pytest.raises(MrcError, match="cannot write dtype"):
        encode_mrc(np.zeros((2, 2), dtype="f8"))


def test_mrc_write_to_path(tmp_path):
    a = np.arange(24, dtype="i2").reshape(2, 3, 4)
    p = tmp_path / "out.mrc"
    write_mrc(p, a, voxel_size=1.25)
    assert np.array_equal(oc.read(str(p), format="mrc"), a)


def test_mrc_matches_mrcfile(tmp_path):
    """The check that matters: another implementation agrees."""
    mrcfile = pytest.importorskip("mrcfile")
    rng = np.random.default_rng(1)
    a = rng.integers(-1000, 1000, size=(3, 5, 7)).astype("i2")
    p = tmp_path / "out.mrc"
    write_mrc(p, a, voxel_size=(2.0, 2.0, 4.0))
    with mrcfile.open(str(p)) as m:
        assert np.array_equal(np.asarray(m.data), a)
        vs = m.voxel_size
        assert (float(vs.x), float(vs.y), float(vs.z)) == pytest.approx(
            (2.0, 2.0, 4.0), rel=1e-4)


# --------------------------------------------------------------------
# NIfTI
# --------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["u1", "i2", "u2", "i4", "f4", "f8"])
def test_nifti_roundtrip_dtypes(dtype):
    a = (np.arange(2 * 3 * 4) % 31).astype(dtype).reshape(2, 3, 4)
    with NiftiStream(encode_nifti(a)) as r:
        assert r.shape == (2, 3, 4)
        assert np.array_equal(np.asarray(r.asarray(), dtype=dtype), a)


def test_nifti_axis_order_survives_the_round_trip():
    """Fortran on the way out, Fortran on the way in.

    With distinct extents a C-order write would read back transposed
    while keeping the right shape.
    """
    a = np.arange(2 * 3 * 4, dtype="i2").reshape(2, 3, 4)
    with NiftiStream(encode_nifti(a)) as r:
        assert np.array_equal(np.asarray(r.asarray(), dtype="i2"), a)


def test_nifti_four_dimensional():
    a = np.arange(2 * 3 * 4 * 5, dtype="i2").reshape(2, 3, 4, 5)
    with NiftiStream(encode_nifti(a)) as r:
        assert r.shape == (2, 3, 4, 5)
        assert np.array_equal(np.asarray(r.asarray(), dtype="i2"), a)


def test_nifti_gzip_round_trip():
    a = np.arange(24, dtype="i2").reshape(2, 3, 4)
    blob = encode_nifti(a, compress=True)
    assert blob[:2] == b"\x1f\x8b"
    with NiftiStream(blob) as r:
        assert np.array_equal(np.asarray(r.asarray(), dtype="i2"), a)


def test_nifti_write_gzips_by_extension(tmp_path):
    a = np.arange(24, dtype="i2").reshape(2, 3, 4)
    p = tmp_path / "out.nii.gz"
    write_nifti(p, a)
    assert p.read_bytes()[:2] == b"\x1f\x8b"
    assert np.array_equal(
        np.asarray(oc.read(str(p), format="nifti"), dtype="i2"), a)


def test_nifti_dimension_over_the_int16_limit_is_refused():
    """NIfTI-1's dim is int16; silently truncating would corrupt the file."""
    with pytest.raises(NiftiError, match="32767"):
        # Build the header alone; allocating the array would be pointless.
        from opencodecs._nifti_writer import nifti1_header
        nifti1_header((40000, 2), np.dtype("u1"))


def test_nifti_matches_nibabel(tmp_path):
    """Reference implementation agreement, including the affine."""
    nib = pytest.importorskip("nibabel")
    rng = np.random.default_rng(2)
    a = rng.integers(0, 4000, size=(4, 5, 6)).astype("i2")
    p = tmp_path / "out.nii"
    write_nifti(p, a, voxel_size=(1.5, 2.0, 2.5))
    img = nib.load(str(p))
    assert np.array_equal(np.asanyarray(img.dataobj), a)
    assert img.shape == (4, 5, 6)
    zooms = img.header.get_zooms()[:3]
    assert tuple(float(z) for z in zooms) == pytest.approx((1.5, 2.0, 2.5), rel=1e-4)
    # An sform that says nothing has been reoriented.
    assert np.allclose(np.diag(img.affine)[:3], (1.5, 2.0, 2.5))


def test_nifti_gz_matches_nibabel(tmp_path):
    nib = pytest.importorskip("nibabel")
    a = np.arange(60, dtype="f4").reshape(3, 4, 5)
    p = tmp_path / "out.nii.gz"
    write_nifti(p, a)
    assert np.allclose(np.asanyarray(nib.load(str(p)).dataobj), a)
