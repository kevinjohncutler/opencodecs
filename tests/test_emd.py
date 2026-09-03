"""EMD reader (Berkeley/NCEM and Velox schemas).

Two conventions share the .emd extension, so the interesting tests are
about detecting which one a file uses and about axis order, where this
reader deliberately differs from hyperspy and needs to say so.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from opencodecs._emd import EmdError, EmdFile  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent.parent / ".test_data" / "emd"
FLAT = DATA / "Si100_2x1x1_3D.emd"
DEPTH = DATA / "Si100_1x1x3.emd"
needs_corpus = pytest.mark.skipif(
    not FLAT.is_file(), reason="fetch the velox_emd_samples corpus entry first")


def build_berkeley(path, arrays, *, array_name="data"):
    """Write a minimal Berkeley-schema EMD."""
    with h5py.File(path, "w") as f:
        root = f.create_group("experiment")
        root.attrs["emd_group_type"] = 2
        for i, a in enumerate(arrays):
            g = root.create_group(f"grp{i}")
            g.attrs["emd_group_type"] = 1
            g.create_dataset(array_name, data=a)
            for axis, n in enumerate(a.shape, start=1):
                g.create_dataset(f"dim{axis}", data=np.arange(n, dtype="f4"))
    return path


def build_velox(path, arrays):
    """Write a minimal Velox-schema EMD."""
    with h5py.File(path, "w") as f:
        img = f.create_group("Data").create_group("Image")
        for i, a in enumerate(arrays):
            g = img.create_group(f"{i:032x}")
            g.create_dataset("Data", data=a)
            meta = b'{"Detector": {"name": "HAADF"}}'
            g.create_dataset("Metadata",
                             data=np.frombuffer(meta, dtype="u1"))
    return path


# --------------------------------------------------------------------
# schema detection
# --------------------------------------------------------------------

def test_berkeley_schema_detected(tmp_path):
    a = np.arange(24, dtype="f4").reshape(4, 6)
    with EmdFile(str(build_berkeley(tmp_path / "b.emd", [a]))) as f:
        assert f.schema == "berkeley"
        assert f.n_datasets == 1
        assert np.array_equal(f.asarray(), a)


def test_velox_schema_detected(tmp_path):
    a = np.arange(24, dtype="u2").reshape(4, 6)
    with EmdFile(str(build_velox(tmp_path / "v.emd", [a]))) as f:
        assert f.schema == "velox"
        assert np.array_equal(f.asarray(), a)
        assert f.metadata()["Detector"]["name"] == "HAADF"


def test_plain_hdf5_is_refused(tmp_path):
    p = tmp_path / "plain.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("x", data=np.zeros((2, 2)))
    with pytest.raises(EmdError, match="not an EMD one"):
        EmdFile(str(p))


def test_data_array_name_is_not_assumed(tmp_path):
    """The Berkeley schema does not fix the array's name.

    The canonical name is `data`, but prismatic's 4D-STEM output calls it
    `realslice`, so the array is whichever dataset is not a dim vector.
    """
    a = np.arange(12, dtype="f4").reshape(3, 4)
    p = build_berkeley(tmp_path / "r.emd", [a], array_name="realslice")
    with EmdFile(str(p)) as f:
        assert np.array_equal(f.asarray(), a)


def test_multiple_datasets_stay_separate(tmp_path):
    a = np.arange(6, dtype="f4").reshape(2, 3)
    b = a + 100
    with EmdFile(str(build_berkeley(tmp_path / "m.emd", [a, b]))) as f:
        assert f.n_datasets == 2
        assert np.array_equal(f.asarray(0), a)
        assert np.array_equal(f.asarray(1), b)
        with pytest.raises(IndexError):
            f.asarray(2)


def test_axes_align_with_the_stored_shape(tmp_path):
    a = np.zeros((5, 3, 7), dtype="f4")
    with EmdFile(str(build_berkeley(tmp_path / "a.emd", [a]))) as f:
        lengths = [v.shape[0] for v in f.axes()]
    assert lengths == [5, 3, 7]


# --------------------------------------------------------------------
# real files
# --------------------------------------------------------------------

@needs_corpus
def test_real_file_matches_the_stored_dataset():
    """We return the array as HDF5 holds it, with dim vectors in step."""
    with EmdFile(str(FLAT)) as f:
        assert f.schema == "berkeley"
        got = f.asarray()
        lengths = [v.shape[0] for v in f.axes()]
    with h5py.File(FLAT, "r") as h:
        raw = h["4DSTEM_simulation/data/realslices/"
                "annular_detector_depth0000/realslice"][...]
    assert np.array_equal(got, raw)
    assert lengths == list(got.shape)


@needs_corpus
def test_axis_convention_differs_from_hyperspy_by_a_transpose():
    """A deliberate difference, pinned so it cannot drift into a bug.

    hyperspy orders navigation axes first, so RosettaSciIO returns the
    transpose of what the file stores. We return the stored layout, the
    same choice made for MRC and N5, and the two agree exactly once that
    is accounted for.
    """
    rsciio = pytest.importorskip("rsciio.emd")
    ref = rsciio.file_reader(str(FLAT))[0]["data"]
    with EmdFile(str(FLAT)) as f:
        got = f.asarray()
    assert got.shape == ref.shape[::-1]
    assert np.array_equal(got, ref.T)


@needs_corpus
def test_separate_groups_are_not_merged():
    """RosettaSciIO stacks the depth groups; the file keeps them apart.

    Exposing what the file contains leaves the choice to the caller,
    who can stack them and cannot unstack them.
    """
    with EmdFile(str(DEPTH)) as f:
        assert f.n_datasets == 2
        shapes = [f.shape(i) for i in range(f.n_datasets)]
    assert shapes == [(7, 7, 208), (7, 7, 208)]
