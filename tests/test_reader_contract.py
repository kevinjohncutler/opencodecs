"""``Codec.open`` is documented to return a ``Reader``. Check that it does.

The container readers were each wired to a codec adapter but kept their
own shapes: ``open()`` handed back an ``MrcStream`` or an ``NrrdFile``
with no ``read()`` and no ``iter_frames()``, so the one interface the
package advertises for multi-frame data was not actually there for the
formats most likely to need it. This is the test that stops that
drifting back.

The first half covers those six in detail, against data written here so
the values can be checked as well as the shape of the API. The second
half runs the same contract across *every* registered codec with an
``open()``, which is what caught the three older readers that had the
same problem and no test: fits, oib and gif.
"""

from __future__ import annotations

import pathlib

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


# --------------------------------------------------------------------
# Every format that has an open(), not just the ones added recently
# --------------------------------------------------------------------
#
# The six scientific readers above were fixed after a review pointed out
# that Codec.open did not return a Reader. Checking the rest afterwards
# found three more, none of which any test touched:
#
#   * fits  -- open() imported opencodecs._fits_reader, a module that
#              has never existed, so it raised ModuleNotFoundError for
#              every caller. decode() goes through open(), so that was
#              broken too.
#   * oib   -- n_frames came off a t > z > c priority list, which need
#              not name axis 0. On an XYCZ file the array is
#              (channels=2, z=6, y, x), n_frames said 6, and
#              iter_frames walked off the end of a 2-long axis.
#   * gif   -- the reader is a Cython extension type, which cannot
#              inherit a Python ABC, so isinstance(r, Reader) was False
#              for GIF alone.
#
# So this runs the contract over every registered codec that has an
# open(), against a real file, and it is the reason a fourth cannot
# quietly join them.

CORPUS = pathlib.Path(__file__).resolve().parent.parent / ".test_data"


def _corpus_file(subdir, pattern):
    """A real file, minus the AppleDouble siblings an SMB share leaves."""
    d = CORPUS / subdir
    if not d.is_dir():
        return None
    files = sorted(p for p in d.glob(pattern) if not p.name.startswith("._"))
    return files[0] if files else None


def _make_tiff(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    p = tmp_path / "v.tif"
    tifffile.imwrite(p, np.arange(4 * 8 * 6, dtype="u1").reshape(4, 8, 6))
    return p


def _make_jxl(tmp_path):
    a = np.arange(32 * 40 * 3, dtype="u1").reshape(32, 40, 3)
    p = tmp_path / "v.jxl"
    p.write_bytes(oc.write(None, a, format="jxl", lossless=True))
    return p


def _make_fits(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    p = tmp_path / "v.fits"
    fits.writeto(p, np.arange(3 * 5 * 7, dtype="i2").reshape(3, 5, 7),
                 overwrite=True)
    return p


ALL_FORMATS = [
    ("tiff", _make_tiff),
    ("jxl", _make_jxl),
    ("fits", _make_fits),
    ("gif", lambda _: _corpus_file("gif", "*.gif")),
    ("czi", lambda _: _corpus_file("czi", "*.czi")),
    ("lif", lambda _: _corpus_file("lif", "*.lif")),
    ("nd2", lambda _: _corpus_file("nd2", "*.nd2")),
    ("oib", lambda _: _corpus_file("oib", "*.oib")),
    ("oir", lambda _: _corpus_file("oir", "*.oir")),
    ("eer", lambda _: _corpus_file("eer", "*.eer")),
    ("vsi", lambda _: _corpus_file("vsi", "*.vsi")),
]


@pytest.fixture(params=ALL_FORMATS, ids=[f[0] for f in ALL_FORMATS])
def any_format(request, tmp_path):
    name, make = request.param
    if not oc.has_codec(name):
        pytest.skip(f"{name} codec not built in this environment")
    path = make(tmp_path)
    if path is None or not pathlib.Path(path).exists():
        pytest.skip(f"no {name} fixture available")
    return name, str(path)


def test_every_open_returns_a_reader(any_format):
    name, path = any_format
    with oc.open(path, format=name) as r:
        assert isinstance(r, Reader), (
            f"{name}: open() returned {type(r).__name__}, not a Reader")


def test_every_reader_reports_shape_and_dtype(any_format):
    """And dtype is an np.dtype, which is what the contract annotates.

    Two readers wrap Cython extensions that hand back the scalar type
    (numpy.uint8) instead. They compare equal, so it looks harmless
    until a caller reaches for .kind or .byteorder.
    """
    name, path = any_format
    with oc.open(path, format=name) as r:
        assert isinstance(r.shape, tuple) and r.shape, f"{name}: bad shape"
        assert isinstance(r.dtype, np.dtype), (
            f"{name}: dtype is {type(r.dtype).__name__}, not np.dtype")


def test_every_reader_can_read(any_format):
    name, path = any_format
    with oc.open(path, format=name) as r:
        arr = r.read()
    assert isinstance(arr, np.ndarray) and arr.size


def test_every_reader_can_iterate(any_format):
    """A fresh reader: several of these stream, so read() first would
    legitimately leave nothing to iterate."""
    name, path = any_format
    with oc.open(path, format=name) as r:
        n = 0
        for frame in r.iter_frames():
            assert isinstance(frame, np.ndarray)
            n += 1
            if n >= 8:            # enough to prove it yields; some are large
                break
    assert n >= 1, f"{name}: iter_frames yielded nothing"


def test_n_frames_agrees_with_what_iteration_yields(any_format):
    """The OIB bug in one assertion.

    n_frames is what iter_frames will produce. Deriving it from
    somewhere else lets the two disagree, and iteration then either
    stops early or indexes past the end.
    """
    name, path = any_format
    with oc.open(path, format=name) as r:
        declared = r.n_frames
        if declared is None:
            pytest.skip(f"{name} does not know its frame count in advance")
        if declared > 64:
            pytest.skip(f"{name} has {declared} frames; too slow to count")
        actual = sum(1 for _ in r.iter_frames())
    assert actual == declared, (
        f"{name}: n_frames says {declared}, iter_frames yielded {actual}")


def test_fits_open_is_reachable(tmp_path):
    """It imported a module that does not exist, so this never worked."""
    fits = pytest.importorskip("astropy.io.fits")
    a = np.arange(3 * 5 * 7, dtype="i2").reshape(3, 5, 7)
    p = tmp_path / "v.fits"
    fits.writeto(p, a, overwrite=True)
    with oc.open(str(p), format="fits") as r:
        assert isinstance(r, Reader)
        assert np.array_equal(r.read(), a)
    # decode() routes through open(), so it was broken by the same line
    assert np.array_equal(oc.read(str(p), format="fits"), a)
