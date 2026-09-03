"""NIfTI-1 / NIfTI-2 reader.

Synthetic headers cover the version, byte order, datatype and scaling
matrix. The two real volumes cover what a hand-built header cannot: a
4-D scan written by a scanner's own converter, in both header versions,
cross-checked against nibabel.
"""

from __future__ import annotations

import gzip
import pathlib
import struct

import numpy as np
import pytest

import opencodecs as oc
from opencodecs._nifti import (NIFTI1_HEADER_SIZE, NIFTI2_HEADER_SIZE,
                               NiftiError, NiftiStream)

DATA = pathlib.Path(__file__).resolve().parent.parent / ".test_data" / "nifti"
NII1 = DATA / "example4d.nii.gz"
NII2 = DATA / "example_nifti2.nii.gz"


def build_nifti1(arr, *, order="<", datatype=4, scl=(0.0, 0.0), magic=b"n+1\x00"):
    h = bytearray(NIFTI1_HEADER_SIZE)
    struct.pack_into(order + "i", h, 0, NIFTI1_HEADER_SIZE)
    dim = [arr.ndim] + list(arr.shape) + [1] * (7 - arr.ndim)
    struct.pack_into(order + "8h", h, 40, *dim)
    struct.pack_into(order + "2h", h, 70, datatype, arr.dtype.itemsize * 8)
    struct.pack_into(order + "8f", h, 76, 1.0, *([2.0] * 7))
    struct.pack_into(order + "f", h, 108, float(NIFTI1_HEADER_SIZE + 4))
    struct.pack_into(order + "2f", h, 112, *scl)
    h[344:348] = magic
    return bytes(h) + b"\x00" * 4 + arr.astype(order + arr.dtype.str[1:]).tobytes(order="F")


def build_nifti2(arr, *, order="<", datatype=4, scl=(0.0, 0.0), magic=b"n+2\x00"):
    h = bytearray(NIFTI2_HEADER_SIZE)
    struct.pack_into(order + "i", h, 0, NIFTI2_HEADER_SIZE)
    h[4:8] = magic
    struct.pack_into(order + "2h", h, 12, datatype, arr.dtype.itemsize * 8)
    dim = [arr.ndim] + list(arr.shape) + [1] * (7 - arr.ndim)
    struct.pack_into(order + "8q", h, 16, *dim)
    struct.pack_into(order + "8d", h, 104, 1.0, *([2.0] * 7))
    struct.pack_into(order + "q", h, 168, NIFTI2_HEADER_SIZE + 4)
    struct.pack_into(order + "2d", h, 176, *scl)
    return bytes(h) + b"\x00" * 4 + arr.astype(order + arr.dtype.str[1:]).tobytes(order="F")


# --------------------------------------------------------------------
# both versions, both byte orders
# --------------------------------------------------------------------

@pytest.mark.parametrize("build", [build_nifti1, build_nifti2])
@pytest.mark.parametrize("order", ["<", ">"])
def test_version_and_byteorder(build, order):
    a = np.arange(4 * 3 * 2, dtype="i2").reshape(4, 3, 2)
    got = oc.read(build(a, order=order), format="nifti")
    assert got.shape == (4, 3, 2)
    assert np.array_equal(np.asarray(got, dtype="i2"), a)


@pytest.mark.parametrize("code,dtype", [
    (2, "u1"), (4, "i2"), (8, "i4"), (16, "f4"), (64, "f8"),
    (256, "i1"), (512, "u2"), (768, "u4"), (1024, "i8"),
])
def test_datatypes(code, dtype):
    a = (np.arange(2 * 3 * 4) % 50).astype(dtype).reshape(2, 3, 4)
    got = oc.read(build_nifti1(a, datatype=code), format="nifti")
    assert np.array_equal(np.asarray(got, dtype=dtype), a)


def test_axis_order_is_fortran_not_c():
    """The subtle one: a C-order reshape gives the right shape, wrong data.

    NIfTI stores the first dimension fastest. With distinct extents a
    C-order read produces an array of the same shape whose voxels are
    transposed, which no shape assertion catches.
    """
    a = np.arange(2 * 3 * 4, dtype="i2").reshape(2, 3, 4)
    got = oc.read(build_nifti1(a), format="nifti")
    assert got.shape == a.shape
    assert np.array_equal(np.asarray(got, dtype="i2"), a)
    # And prove the distinction is real for this array.
    c_order = np.frombuffer(a.tobytes(order="F"), dtype="i2").reshape(a.shape)
    assert not np.array_equal(c_order, a)


def test_four_dimensional():
    a = np.arange(3 * 4 * 2 * 5, dtype="i2").reshape(3, 4, 2, 5)
    got = oc.read(build_nifti1(a), format="nifti")
    assert got.shape == (3, 4, 2, 5)
    assert np.array_equal(np.asarray(got, dtype="i2"), a)


# --------------------------------------------------------------------
# gzip and scaling
# --------------------------------------------------------------------

def test_gzip_is_transparent():
    """Nearly every NIfTI in the wild is .nii.gz."""
    a = np.arange(24, dtype="i2").reshape(2, 3, 4)
    plain = build_nifti1(a)
    assert np.array_equal(
        np.asarray(oc.read(gzip.compress(plain), format="nifti"), dtype="i2"), a)


def test_scaling_is_applied_when_it_changes_something():
    a = np.arange(24, dtype="i2").reshape(2, 3, 4)
    blob = build_nifti1(a, scl=(2.0, 10.0))
    with NiftiStream(blob) as r:
        assert r.has_scaling
        scaled = r.asarray()
        raw = r.asarray(scaled=False)
    assert np.allclose(scaled, a * 2.0 + 10.0)
    assert np.array_equal(np.asarray(raw, dtype="i2"), a)


@pytest.mark.parametrize("scl", [(0.0, 0.0), (1.0, 0.0)])
def test_identity_scaling_leaves_dtype_alone(scl):
    """Slope 0 means "none" per the spec, and 1/0 is the identity.

    Widening an integer volume to float for either would cost memory and
    change the dtype a caller sees, for no change in value.
    """
    a = np.arange(24, dtype="i2").reshape(2, 3, 4)
    with NiftiStream(build_nifti1(a, scl=scl)) as r:
        assert not r.has_scaling
        out = r.asarray()
    assert out.dtype == np.dtype("<i2")
    assert np.array_equal(out, a)


# --------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------

def test_not_nifti_raises():
    with pytest.raises(NiftiError, match="sizeof_hdr"):
        NiftiStream(b"\x00" * 400)


def test_bad_magic_raises():
    a = np.zeros((2, 2, 2), dtype="i2")
    with pytest.raises(NiftiError, match="bad magic"):
        NiftiStream(build_nifti1(a, magic=b"xxxx"))


def test_truncated_data_raises():
    a = np.arange(24, dtype="i2").reshape(2, 3, 4)
    with pytest.raises(NiftiError, match="truncated"):
        oc.read(build_nifti1(a)[:-10], format="nifti")


def test_unsupported_datatype_raises():
    """RGB24 is stored 3 bytes per voxel, not as a scalar type."""
    a = np.zeros((2, 2, 2), dtype="u1")
    with pytest.raises(NiftiError, match="RGB24"):
        NiftiStream(build_nifti1(a, datatype=128))


def test_hdr_img_pair_is_refused_with_an_explanation():
    """ni1 means the voxels are in a separate .img file."""
    a = np.zeros((2, 2, 2), dtype="i2")
    with pytest.raises(NiftiError, match=r"\.img"):
        oc.read(build_nifti1(a, magic=b"ni1\x00"), format="nifti")


# --------------------------------------------------------------------
# registry
# --------------------------------------------------------------------

def test_registered_and_sniffable():
    assert oc.has_codec("nifti")
    a = np.zeros((2, 3, 4), dtype="i2")
    assert oc.read(build_nifti1(a)).shape == (2, 3, 4)
    assert oc.read(build_nifti2(a)).shape == (2, 3, 4)


def test_codec_encode_round_trips():
    """Encoding landed after this reader; the codec surface must expose it."""
    a = np.arange(2 * 3 * 4, dtype="i2").reshape(2, 3, 4)
    codec = oc.get_codec("nifti")
    assert codec.can_encode
    blob = codec.encode(a, voxel_size=(1.0, 1.0, 2.0))
    assert np.array_equal(
        np.asarray(oc.read(blob, format="nifti"), dtype="i2"), a)


# --------------------------------------------------------------------
# real volumes
# --------------------------------------------------------------------

@pytest.mark.skipif(not NII1.is_file(),
                    reason="run `python corpus/corpus.py fetch nifti_reference_pair`")
def test_real_nifti1_scan():
    with NiftiStream(str(NII1)) as r:
        h = r.header
        assert h["version"] == 1 and h["magic"] == "n+1"
        assert r.shape == (128, 96, 24, 2)
        a = r.asarray()
    assert a.shape == (128, 96, 24, 2) and a.dtype == np.dtype("<i2")


@pytest.mark.skipif(not NII2.is_file(),
                    reason="run `python corpus/corpus.py fetch nifti_reference_pair`")
def test_real_nifti2_scan():
    """NIfTI-2 moves the magic and widens half the header; check we follow."""
    with NiftiStream(str(NII2)) as r:
        h = r.header
        assert h["version"] == 2 and h["magic"] == "n+2"
        assert r.shape == (32, 20, 12, 2)
        assert r.asarray().shape == (32, 20, 12, 2)


@pytest.mark.skipif(not (NII1.is_file() and NII2.is_file()),
                    reason="run `python corpus/corpus.py fetch nifti_reference_pair`")
@pytest.mark.parametrize("path", [NII1, NII2], ids=["nifti1", "nifti2"])
def test_real_volumes_agree_with_nibabel(path):
    nib = pytest.importorskip("nibabel")
    ref = np.asanyarray(nib.load(str(path)).dataobj)
    got = oc.read(str(path), format="nifti")
    assert got.shape == ref.shape
    assert np.allclose(np.asarray(got, dtype="f8"),
                       np.asarray(ref, dtype="f8"))
