"""A raw NRRD slice is a seek, and iterating one does not materialize it.

``sizes`` lists the fastest-varying axis first and ``shape`` is its
reverse, so ``shape`` is C-ordered and the slowest axis is axis 0: one
slice is one contiguous run of bytes, exactly as in MRC. Defining
``_frame`` is what ArrayReader routes ``iter_frames`` and ``__getitem__``
through, so without it iterating a volume read the whole thing first.

The gz, bzip2, text and hex encodings have no byte position for slice N.
They fall back to reading the volume and slicing it, which is the same
answer at the same cost as before, and the tests check that the answer
is the same rather than that the path is.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc

nrrd = pytest.importorskip("nrrd")

from _range_http_server import range_http_server

VOL = (np.arange(12 * 20 * 24, dtype="<u2") % 4093).reshape(12, 20, 24)


@pytest.fixture
def raw_volume(tmp_path):
    p = tmp_path / "raw.nrrd"
    nrrd.write(str(p), np.asfortranarray(VOL.T), {}, index_order="F")
    return p


def test_shape_and_bulk_read_are_unchanged(raw_volume):
    with oc.get_codec("nrrd").open(str(raw_volume)) as r:
        assert r.shape == VOL.shape
        assert np.array_equal(r.read(), VOL)


def test_every_slice_matches_the_volume(raw_volume):
    with oc.get_codec("nrrd").open(str(raw_volume)) as r:
        full = r.read()
        for i in range(VOL.shape[0]):
            assert np.array_equal(r[i], full[i]), f"slice {i}"


def test_iteration_yields_the_same_slices(raw_volume):
    with oc.get_codec("nrrd").open(str(raw_volume)) as r:
        got = list(r.iter_frames())
    assert len(got) == VOL.shape[0]
    assert np.array_equal(np.stack(got), VOL)


def test_negative_and_out_of_range(raw_volume):
    n = VOL.shape[0]
    with oc.get_codec("nrrd").open(str(raw_volume)) as r:
        assert np.array_equal(r[-1], VOL[n - 1])
        assert np.array_equal(r[-n], VOL[0])
        for bad in (n, n + 2, -n - 1):
            with pytest.raises(IndexError):
                r[bad]


@pytest.mark.parametrize("encoding", ["gzip", "bzip2"])
def test_compressed_encodings_give_the_same_answer(tmp_path, encoding):
    """No random access in a single compressed stream, so the fallback
    reads the volume. The answer must be identical anyway."""
    p = tmp_path / f"c_{encoding}.nrrd"
    nrrd.write(str(p), np.asfortranarray(VOL.T), {}, index_order="F",
               compression_level=1)
    with oc.get_codec("nrrd").open(str(p)) as r:
        if r.encoding in ("raw", ""):
            pytest.skip("pynrrd wrote this one raw")
        assert r.is_slice_addressable is False
        assert np.array_equal(r[3], VOL[3])
        assert np.array_equal(np.stack(list(r.iter_frames())), VOL)


def test_a_2d_nrrd_has_one_frame(tmp_path):
    a = (np.arange(6 * 7, dtype="<u2")).reshape(6, 7)
    p = tmp_path / "flat.nrrd"
    nrrd.write(str(p), np.asfortranarray(a.T), {}, index_order="F")
    with oc.get_codec("nrrd").open(str(p)) as r:
        assert np.array_equal(r[0], a)
        with pytest.raises(IndexError):
            r[1]


def test_reading_one_slice_over_http_moves_one_slice(tmp_path):
    """The point of the offset read, stated in bytes."""
    big = (np.arange(64 * 128 * 128, dtype="<u2") % 4093).reshape(64, 128, 128)
    p = tmp_path / "big.nrrd"
    nrrd.write(str(p), np.asfortranarray(big.T), {}, index_order="F")
    slice_bytes = big[0].nbytes
    with range_http_server(str(tmp_path)) as (base, t):
        with oc.get_codec("nrrd").open(f"{base}/big.nrrd") as r:
            got = r[40]
        served = t.bytes_served
    assert np.array_equal(got, big[40])
    assert served < slice_bytes * 3, (
        f"moved {served} bytes for a {slice_bytes}-byte slice of a "
        f"{big.nbytes}-byte volume")
