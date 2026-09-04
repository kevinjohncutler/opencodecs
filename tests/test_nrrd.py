"""NRRD reader.

The corpus holds one 30x30x30 ball in raw, gzip and bzip2 plus a
detached header pair, so the picture is constant and any difference
between decodes is the parser. Synthetic files cover the dtype and
endianness matrix, which no single real volume can.
"""

from __future__ import annotations

import gzip
import pathlib

import numpy as np
import pytest

import opencodecs as oc
from opencodecs._nrrd import NrrdError, NrrdFile

DATA = pathlib.Path(__file__).resolve().parent.parent / ".test_data" / "nrrd"
RAW, GZ, BZ2 = DATA / "ball_raw.nrrd", DATA / "ball_gz.nrrd", DATA / "ball_bz2.nrrd"
NHDR = DATA / "BallBinary30x30x30.nhdr"
needs_corpus = pytest.mark.skipif(
    not RAW.is_file(),
    reason="fetch the nrrd_pynrrd_variants corpus entry first")

_TYPE_NAME = {"u1": "uchar", "i1": "signed char", "i2": "short",
              "u2": "ushort", "i4": "int", "f4": "float", "f8": "double"}


def build_nrrd(a, *, encoding="raw", endian="little"):
    # ``sizes`` lists the fastest-varying axis first, so it is the
    # reverse of the numpy shape and the buffer is plain C order: the
    # last numpy axis, which is the first listed size, varies fastest.
    # Writing Fortran bytes here instead reverses the axes a second
    # time, and the reader then agrees with the fixture while both
    # disagree with every other NRRD implementation.
    sizes = " ".join(str(s) for s in a.shape[::-1])
    order = "<" if endian == "little" else ">"
    body = a.astype(order + a.dtype.str[1:]).tobytes()
    if encoding == "gzip":
        body = gzip.compress(body)
    elif encoding == "bzip2":
        import bz2
        body = bz2.compress(body)
    head = (f"NRRD0004\ntype: {_TYPE_NAME[a.dtype.str[1:]]}\n"
            f"dimension: {a.ndim}\nsizes: {sizes}\n"
            f"encoding: {encoding}\nendian: {endian}\n\n").encode()
    return head + body


@pytest.mark.parametrize("dtype", ["u1", "i2", "u2", "i4", "f4", "f8"])
@pytest.mark.parametrize("endian", ["little", "big"])
def test_dtypes_and_endianness(dtype, endian):
    a = (np.arange(2 * 3 * 4) % 37).astype(dtype).reshape(2, 3, 4)
    got = oc.read(build_nrrd(a, endian=endian), format="nrrd")
    assert np.array_equal(np.asarray(got, dtype=dtype), a)


@pytest.mark.parametrize("encoding", ["raw", "gzip", "bzip2"])
def test_encodings(encoding):
    a = np.arange(2 * 3 * 4, dtype="i2").reshape(2, 3, 4)
    got = oc.read(build_nrrd(a, encoding=encoding), format="nrrd")
    assert np.array_equal(np.asarray(got, dtype="i2"), a)


def test_axis_order_is_reversed_from_sizes():
    """`sizes` lists the fastest axis first, so numpy's shape is reversed.

    With distinct extents, ignoring that returns a transposed volume of
    the right rank, which a shape check alone would pass.
    """
    a = np.arange(2 * 3 * 4, dtype="i2").reshape(2, 3, 4)
    blob = build_nrrd(a)
    assert b"sizes: 4 3 2" in blob
    got = oc.read(blob, format="nrrd")
    assert got.shape == (2, 3, 4)
    assert np.array_equal(np.asarray(got, dtype="i2"), a)


def test_comments_and_key_value_pairs_are_ignored_safely():
    a = np.arange(6, dtype="u1").reshape(2, 3)
    blob = build_nrrd(a)
    head, _, body = blob.partition(b"\n\n")
    blob = head + b"\n# a comment\nfoo:=bar\n\n" + body
    assert np.array_equal(np.asarray(oc.read(blob, format="nrrd"), dtype="u1"), a)


def test_not_nrrd_raises():
    with pytest.raises(NrrdError, match="not a NRRD file"):
        NrrdFile(b"NOPE\nnothing")


def test_truncated_data_raises():
    a = np.arange(24, dtype="i2").reshape(2, 3, 4)
    with pytest.raises(NrrdError, match="truncated"):
        oc.read(build_nrrd(a)[:-10], format="nrrd")


def test_dimension_mismatch_raises():
    a = np.arange(6, dtype="u1").reshape(2, 3)
    blob = build_nrrd(a).replace(b"dimension: 2", b"dimension: 3")
    with pytest.raises(NrrdError, match="dimension says"):
        oc.read(blob, format="nrrd")


def test_detached_from_bytes_explains_itself():
    """A detached header read from bytes cannot resolve its data file."""
    blob = (b"NRRD0004\ntype: short\ndimension: 2\nsizes: 2 3\n"
            b"encoding: raw\nendian: little\ndata file: elsewhere.raw\n\n")
    with pytest.raises(NrrdError, match=r"read from\s+bytes"):
        NrrdFile(blob).asarray()


def test_registered_and_sniffable():
    assert oc.has_codec("nrrd")
    a = np.zeros((2, 3), dtype="u1")
    assert oc.read(build_nrrd(a)).shape == (2, 3)


# --------------------------------------------------------------------
# real files
# --------------------------------------------------------------------

@needs_corpus
def test_all_encodings_of_the_same_volume_agree():
    with NrrdFile(str(RAW)) as f:
        expected = f.asarray()
    for path in (GZ, BZ2, NHDR):
        with NrrdFile(str(path)) as f:
            assert np.array_equal(f.asarray(), expected), path.name


@needs_corpus
def test_detached_header_reads_its_sibling_raw_file():
    """The .nhdr names a separate .raw; a reader without that has nothing."""
    with NrrdFile(str(NHDR)) as f:
        assert f.detached_data_file is not None
        a = f.asarray()
    assert a.shape == (30, 30, 30)


@needs_corpus
@pytest.mark.parametrize("path", [RAW, GZ, BZ2, NHDR], ids=lambda p: p.stem)
def test_matches_pynrrd(path):
    """Against pynrrd's C ordering, which is the convention we report.

    pynrrd defaults to ``index_order="F"``, which hands back the array
    in ``sizes`` order. Ours is the reverse of that, so the comparison
    has to name the ordering or it is not comparing anything.
    """
    pynrrd = pytest.importorskip("nrrd")
    ref, _ = pynrrd.read(str(path), index_order="C")
    with NrrdFile(str(path)) as f:
        got = f.asarray()
    assert got.shape == ref.shape and np.array_equal(got, ref)


def test_axis_order_on_a_volume_that_is_not_a_cube(tmp_path):
    """The corpus volumes are 30x30x30, where a transposed read passes.

    Reversing the axes twice is invisible on a cube: the shape is
    unchanged, and reshape(sizes).T equals reshape(sizes, order="F")
    exactly when sizes is its own reverse. It took a volume with three
    different extents, written by pynrrd itself, to show the bug.
    """
    pynrrd = pytest.importorskip("nrrd")
    a = np.arange(2 * 3 * 5, dtype="i2").reshape(2, 3, 5)
    p = tmp_path / "brick.nrrd"
    pynrrd.write(str(p), a, index_order="C")
    assert list(pynrrd.read_header(str(p))["sizes"]) == [5, 3, 2]
    with NrrdFile(str(p)) as f:
        got, shape = f.asarray(), f.shape
    assert shape == a.shape
    assert np.array_equal(got, a), f"got\n{got}\nwant\n{a}"


def test_our_files_read_back_by_pynrrd(tmp_path):
    """The fixture builder agrees with pynrrd, not only with us."""
    pynrrd = pytest.importorskip("nrrd")
    a = (np.arange(2 * 3 * 5) % 29).astype("i2").reshape(2, 3, 5)
    p = tmp_path / "ours.nrrd"
    p.write_bytes(build_nrrd(a))
    ref, _ = pynrrd.read(str(p), index_order="C")
    assert ref.shape == a.shape and np.array_equal(ref, a)
