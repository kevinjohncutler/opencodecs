"""Readers for formats that are a header followed by a contiguous buffer.

.npy and uncompressed .nii are the same shape as MRC: a short header,
then the voxels. Both used to be read whole -- np.load on the entire
file, and NiftiStream holding the bytes in memory -- which is correct
and costs the whole file to look at one plane.

What these tests check is BYTES MOVED, because that is the only thing
that separates a real offset read from a whole-file read followed by a
slice. Both return identical pixels; only the transfer says which
happened.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

import opencodecs as oc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


@pytest.fixture(scope="module")
def volume():
    return (np.arange(64 * 96 * 32, dtype="i2") % 3000).reshape(64, 96, 32)


def _server():
    try:
        from _range_http_server import range_http_server
    except ImportError:
        pytest.skip("range test server helper unavailable")
    return range_http_server


# ---- .npy -----


def test_npy_plane_is_the_right_pixels(tmp_path, volume):
    p = tmp_path / "v.npy"
    np.save(p, volume)
    with oc.get_codec("numpy").open(str(p)) as r:
        assert r.shape == volume.shape
        assert r.dtype == volume.dtype
        assert r.n_frames == volume.shape[0]
        for i in (0, 31, 63, -1):
            np.testing.assert_array_equal(np.asarray(r.plane(i)), volume[i])
        np.testing.assert_array_equal(np.asarray(r.read()), volume)


def test_npy_plane_out_of_range(tmp_path, volume):
    p = tmp_path / "v.npy"
    np.save(p, volume)
    with oc.get_codec("numpy").open(str(p)) as r:
        with pytest.raises(IndexError):
            r.plane(len(volume))


def test_npy_fortran_order_refuses_a_plane_rather_than_lying(tmp_path,
                                                             volume):
    """A leading-axis plane of a column-major array is not contiguous.

    Reading "the plane's bytes" there would return a stride of the
    whole buffer -- plausible-looking numbers from the wrong places.
    Refusing and reading whole is the honest answer.
    """
    p = tmp_path / "f.npy"
    np.save(p, np.asfortranarray(volume))
    with oc.get_codec("numpy").open(str(p)) as r:
        assert r.is_chunked is False
        np.testing.assert_array_equal(np.asarray(r.read()), volume)
        with pytest.raises(TypeError, match="Fortran"):
            r.plane(1)


def test_npy_rejects_something_that_is_not_a_npy(tmp_path):
    p = tmp_path / "not.npy"
    p.write_bytes(b"definitely not a numpy file, but long enough")
    with pytest.raises(ValueError, match="NUMPY"):
        oc.get_codec("numpy").open(str(p))


def test_npy_over_http_moves_one_plane(tmp_path, volume):
    server = _server()
    p = tmp_path / "v.npy"
    np.save(p, volume)
    size = p.stat().st_size
    with server(tmp_path) as s:
        base, tracker = s if isinstance(s, tuple) else (s, None)
        with oc.get_codec("numpy").open(f"{base}/v.npy") as r:
            assert r.shape == volume.shape
            before = tracker.bytes_served if tracker else 0
            got = np.asarray(r.plane(40))
        np.testing.assert_array_equal(got, volume[40])
        if tracker is not None:
            moved = tracker.bytes_served - before
            assert moved < size / 3, (
                f"one plane moved {moved} of {size} bytes")


# ---- .nii -----


def test_nifti_slice_is_along_the_last_axis(tmp_path, volume):
    """NIfTI stores the first dimension fastest.

    So the contiguous run is a slice of the LAST axis, the opposite of
    MRC. Getting this backwards would return correctly-shaped nonsense.
    """
    p = tmp_path / "v.nii"
    oc.write_nifti(str(p), volume)
    with oc.get_codec("nifti").open(str(p)) as r:
        full = np.asarray(r.read())
        for i in (0, full.shape[-1] // 2, full.shape[-1] - 1):
            np.testing.assert_array_equal(
                np.asarray(r.slice_at(i)), full[..., i])


def test_uncompressed_nifti_is_not_held_in_memory(tmp_path, volume):
    p = tmp_path / "v.nii"
    oc.write_nifti(str(p), volume)
    with oc.get_codec("nifti").open(str(p)) as r:
        assert r.is_memory_resident is False


def test_gzipped_nifti_is_held_in_memory_and_says_so(tmp_path, volume):
    """A gzip member has no random access, so this one really must be
    read whole. The reader reporting which path it took is what keeps
    a caller from wondering why one file is slower than another."""
    p = tmp_path / "v.nii.gz"
    oc.write_nifti(str(p), volume)
    with oc.get_codec("nifti").open(str(p)) as r:
        assert r.is_memory_resident is True
        full = np.asarray(r.read())
        np.testing.assert_array_equal(
            np.asarray(r.slice_at(3)), full[..., 3])


def test_nifti_over_http_moves_one_slice(tmp_path, volume):
    server = _server()
    p = tmp_path / "v.nii"
    oc.write_nifti(str(p), volume)
    size = p.stat().st_size
    with oc.get_codec("nifti").open(str(p)) as r:
        want = np.asarray(r.slice_at(20))
    with server(tmp_path) as s:
        base, tracker = s if isinstance(s, tuple) else (s, None)
        with oc.get_codec("nifti").open(f"{base}/v.nii") as r:
            before = tracker.bytes_served if tracker else 0
            got = np.asarray(r.slice_at(20))
        np.testing.assert_array_equal(got, want)
        if tracker is not None:
            moved = tracker.bytes_served - before
            assert moved < size / 3, (
                f"one slice moved {moved} of {size} bytes")


def test_nifti_slice_out_of_range(tmp_path, volume):
    p = tmp_path / "v.nii"
    oc.write_nifti(str(p), volume)
    with oc.get_codec("nifti").open(str(p)) as r:
        with pytest.raises(IndexError):
            r.slice_at(volume.shape[-1] + 5)
