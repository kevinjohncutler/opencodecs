"""``Codec.open`` is documented to return a ``Reader``. Check that it does.

The container readers were each wired to a codec adapter but kept their
own shapes: ``open()`` handed back an ``MrcStream`` or an ``NrrdFile``
with no ``read()`` and no ``iter_frames()``, so the one interface the
package advertises for multi-frame data was not actually there for the
formats most likely to need it. This is the test that stops that
drifting back.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc
from opencodecs.core.codec import Reader

from test_dicom_encodings import build_dicom, explicit_le
from test_dm_synthetic import build_dm


def write_mrc(path, a):
    from opencodecs._mrc_writer import encode_mrc
    path.write_bytes(encode_mrc(a))
    return path


def write_nifti(path, a):
    from opencodecs._nifti_writer import encode_nifti
    path.write_bytes(encode_nifti(a))
    return path


def write_nrrd(path, a):
    sizes = " ".join(str(s) for s in a.shape[::-1])
    head = (f"NRRD0004\ntype: short\ndimension: {a.ndim}\nsizes: {sizes}\n"
            f"encoding: raw\nendian: little\n\n").encode()
    path.write_bytes(head + a.astype("<i2").tobytes())
    return path


def write_dicom(path, a):
    path.write_bytes(build_dicom(explicit_le(), a))
    return path


VOL = np.arange(3 * 4 * 6, dtype="i2").reshape(3, 4, 6)

CASES = [
    ("mrc", "v.mrc", write_mrc, VOL),
    ("nifti", "v.nii", write_nifti, VOL),
    ("nrrd", "v.nrrd", write_nrrd, VOL),
    ("dicom", "v.dcm", write_dicom, VOL.astype("<u2")),
]


@pytest.fixture(params=CASES, ids=[c[0] for c in CASES])
def opened(request, tmp_path):
    name, filename, writer, data = request.param
    path = writer(tmp_path / filename, data)
    with oc.open(str(path), format=name) as reader:
        yield name, reader, data


def test_open_returns_a_reader(opened):
    _, reader, _ = opened
    assert isinstance(reader, Reader)


def test_shape_and_dtype_are_available_without_reading(opened):
    _, reader, data = opened
    assert reader.shape == data.shape
    assert reader.dtype.kind == data.dtype.kind
    assert reader.dtype.itemsize == data.dtype.itemsize


def test_read_returns_the_whole_volume(opened):
    _, reader, data = opened
    assert np.array_equal(reader.read(), data)


def test_iter_frames_yields_one_plane_each(opened):
    _, reader, data = opened
    frames = list(reader.iter_frames())
    assert len(frames) == data.shape[0] == reader.n_frames
    for i, frame in enumerate(frames):
        assert np.array_equal(frame, data[i])


def test_iterating_the_reader_is_iter_frames(opened):
    _, reader, data = opened
    assert np.array_equal(np.stack(list(reader)), data)


def test_random_access_matches_iteration(opened):
    _, reader, data = opened
    if not reader.is_chunked:
        pytest.skip("format has no per-frame access")
    for i in range(reader.n_frames):
        assert np.array_equal(reader[i], data[i])
    assert np.array_equal(reader[-1], data[-1])
    with pytest.raises(IndexError):
        reader[reader.n_frames]


# --------------------------------------------------------------------
# 2D formats collapse to a one-element iterator, per the Reader docstring
# --------------------------------------------------------------------

def test_single_image_is_a_one_element_iterator(tmp_path):
    a = np.arange(4 * 6, dtype="i2").reshape(4, 6)
    path = write_nrrd(tmp_path / "flat.nrrd", a)
    with oc.open(str(path), format="nrrd") as r:
        frames = list(r.iter_frames())
        assert r.n_frames == 1 and len(frames) == 1
    assert np.array_equal(frames[0], a)


# --------------------------------------------------------------------
# the multi-image containers, where a "frame" is a whole image
# --------------------------------------------------------------------

def test_dm_exposes_each_image_as_a_frame(tmp_path):
    a = np.arange(4 * 6, dtype="u2").reshape(4, 6)
    path = tmp_path / "i.dm3"
    path.write_bytes(build_dm(3, a))
    with oc.open(str(path), format="dm") as r:
        assert isinstance(r, Reader)
        assert r.shape == a.shape and r.dtype == np.dtype("<u2")
        assert r.n_frames == 1
        assert np.array_equal(r.read(), a)
        assert np.array_equal(r[0], a)


def test_emd_exposes_each_dataset_as_a_frame(tmp_path):
    h5py = pytest.importorskip("h5py")
    from opencodecs._emd import EmdFile
    a = np.arange(12, dtype="f4").reshape(3, 4)
    b = a + 100
    p = tmp_path / "m.emd"
    with h5py.File(p, "w") as f:
        root = f.create_group("experiment")
        root.attrs["emd_group_type"] = 2
        for i, arr in enumerate((a, b)):
            g = root.create_group(f"grp{i}")
            g.attrs["emd_group_type"] = 1
            g.create_dataset("data", data=arr)
    with EmdFile(str(p)) as r:
        assert isinstance(r, Reader)
        assert r.shape == a.shape and r.n_frames == 2
        assert np.array_equal(r.read(), a)
        frames = list(r.iter_frames())
        assert np.array_equal(frames[0], a) and np.array_equal(frames[1], b)
        assert r.shape_at(1) == b.shape
        assert r.dtype_at(1) == b.dtype


# --------------------------------------------------------------------
# dispatch: every one of these answers to oc.read by extension
# --------------------------------------------------------------------

@pytest.mark.parametrize("name,filename,writer,data", CASES,
                         ids=[c[0] for c in CASES])
def test_read_dispatches_by_extension(tmp_path, name, filename, writer, data):
    path = writer(tmp_path / filename, data)
    assert np.array_equal(oc.read(str(path)), data)


def test_dicom_is_registered():
    """It was reachable only as DicomFile, which oc.read could not find."""
    assert oc.has_codec("dicom")
    codec = oc.get_codec("dicom")
    assert ".dcm" in codec.file_extensions
    assert codec.signature(b"\x00" * 128 + b"DICM" + b"\x00" * 8)
    assert not codec.signature(b"\x89PNG\r\n\x1a\n" + b"\x00" * 128)
