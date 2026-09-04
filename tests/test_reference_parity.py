"""Every container reader against its reference implementation.

The reader bugs found on 2026-09-03 were all the same shape: a fixture
written by the same assumption as the reader, on a volume symmetric
enough that a transposition was invisible. Six defects, none of which
any test noticed, because nothing in the suite held an independent
opinion about what the bytes meant.

So this file has the reference library hold the pen. Where one exists it
writes the fixture and we read it back; where the format has no writer
we read a real corpus file and compare pixel for pixel. Shapes are
asymmetric on every axis on purpose -- ``(3, 5, 7)`` rather than
``(4, 4, 4)`` -- because a transposition bug on a cube passes every
check you can write.

The reference packages are all optional. tensorstore and ndtiff are not
in the ``test`` extra, so those skip unless installed:

    pip install tensorstore ndtiff

Formats covered elsewhere, and why not here: FITS already has astropy
write its fixtures (``test_fits.py``), EER is compared against
imagecodecs on real Falcon4 data (``test_eer.py``), MRC and NIfTI
against mrcfile and nibabel (``test_mrc_nifti_writers.py``), NRRD
against pynrrd (``test_nrrd.py``), DM and EMD against RosettaSciIO
(``test_dm.py``, ``test_emd.py``), and CZI, ND2 and LIF against their
vendor libraries (``test_corpus_vendors.py``).
"""

from __future__ import annotations

import pathlib
import shutil

import numpy as np
import pytest

DATA = pathlib.Path(__file__).resolve().parent.parent / ".test_data"

# Asymmetric on every axis: a cube hides exactly the bug this file exists
# to catch.
SHAPES = [(3, 5, 7), (5, 7, 9)]


def _real_files(directory, pattern):
    """Corpus files, minus the AppleDouble siblings an SMB share leaves.

    ``._name`` files are resource forks, not data. Some reference
    readers glob a directory and try to parse them, which fails in a way
    that looks like a corrupt dataset.
    """
    d = DATA / directory
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob(pattern) if not p.name.startswith("._"))


# ====================================================================
# N5 vs tensorstore
# ====================================================================

def _ts_open(root, path, **kw):
    import tensorstore as ts
    spec = {"driver": "n5",
            "kvstore": {"driver": "file", "path": str(root)},
            "path": path}
    spec.update(kw)
    return ts.open(spec).result()


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("compression", ["raw", "gzip"])
def test_n5_tensorstore_writes_we_read(tmp_path, shape, compression):
    """tensorstore keeps N5's own order; we present its reverse.

    N5 dimension 0 varies fastest, so a numpy reader that wants the
    fastest axis last -- which is what C order means, and what pynrrd,
    FITS and z5py all do -- reports the reverse of what the file lists.
    tensorstore instead exposes the file's order unchanged. Both are
    self-consistent, and the pair differs by exactly a transpose. That
    is what this pins: not our convention against itself, but the bytes.
    """
    pytest.importorskip("tensorstore")
    from opencodecs._n5 import N5Array

    a = (np.arange(int(np.prod(shape))) % 251).astype("uint16").reshape(shape)
    root = tmp_path / "store"
    store = _ts_open(root, "vol", create=True, delete_existing=True,
                     metadata={"dataType": "uint16",
                               "dimensions": list(a.shape),
                               "blockSize": [2, 3, 4],
                               "compression": {"type": compression}})
    store[...] = a

    z = N5Array(str(root), "vol")
    assert z.shape == a.shape[::-1]
    got = np.asarray(z.asarray(), dtype="uint16")
    assert np.array_equal(got, a.T), "not a pure transpose of tensorstore"


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_n5_we_write_tensorstore_reads(tmp_path, shape):
    """The other direction, so the fixture builder is checked too."""
    pytest.importorskip("tensorstore")
    from test_n5 import write_n5

    a = (np.arange(int(np.prod(shape))) % 199).astype("<u2").reshape(shape)
    write_n5(tmp_path, a, (2, 3, 4), path="vol")
    ref = np.asarray(_ts_open(tmp_path, "vol").read().result())
    assert ref.shape == a.shape[::-1]
    assert np.array_equal(ref, a.T)


def test_n5_real_corpus_blocks_match_tensorstore():
    """The corpus blocks were only ever checked for shape and range.

    ``min() < max()`` passes on a transposed block, so the one real N5
    volume in the suite was not testing the layout at all.
    """
    pytest.importorskip("tensorstore")
    from opencodecs._n5 import N5Array

    root = DATA / "n5" / "jrc_hela-2.n5"
    if not root.is_dir():
        pytest.skip("fetch the n5 corpus entry first")
    array = "em/fibsem-uint16/s4"

    z = N5Array(str(root), array)
    store = _ts_open(root, array)
    assert tuple(store.shape) == z.shape[::-1]

    checked = 0
    for idx in np.ndindex(*z.chunk_grid):
        block = z.read_block(idx)
        if block is None:
            continue
        extent = [min(c, s - i * c)
                  for i, c, s in zip(idx, z.chunks, z.shape)]
        got = np.asarray(block)[tuple(slice(0, e) for e in extent)]
        window = tuple(slice(i * c, i * c + e)
                       for i, c, e in zip(idx, z.chunks, extent))
        ref = np.asarray(store[tuple(reversed(window))].read().result())
        assert np.array_equal(got, ref.T), f"block {idx} differs"
        checked += 1
    assert checked, "no blocks present in the corpus"


# ====================================================================
# OME-Zarr vs zarr
# ====================================================================

@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_omezarr_matches_zarr(tmp_path, zarr_format, shape):
    """zarr writes, we read. Both storage versions, ragged chunk grids.

    ``(5, 7, 9)`` with ``(2, 3, 4)`` chunks leaves a partial chunk on
    every axis, which is where an off-by-one in the trimming shows up.
    """
    zarr = pytest.importorskip("zarr")
    if zarr_format == 3 and not zarr.__version__.startswith("3"):
        pytest.skip("zarr v3 store needs zarr-python 3")
    from opencodecs._omezarr import OmeZarrArray

    a = (np.arange(int(np.prod(shape))) % 211).astype("u2").reshape(shape)
    path = tmp_path / "a.zarr"
    z = zarr.create_array(store=str(path), shape=a.shape, chunks=(2, 3, 4),
                          dtype="u2", zarr_format=zarr_format)
    z[...] = a

    ours = OmeZarrArray(str(path))
    assert ours.shape == a.shape
    assert np.array_equal(np.asarray(ours.read()), a)

    inner = tuple(slice(1, s - 1) for s in shape)
    assert np.array_equal(np.asarray(ours.read_region(inner)), a[inner]), \
        "sub-region straddling chunk boundaries differs"


# ====================================================================
# Imaris vs the raw HDF5 datasets
# ====================================================================

def _imaris_file():
    files = _real_files("imaris", "*.ims")
    if not files:
        pytest.skip("fetch the imaris corpus entry first")
    return files[0]


def test_imaris_levels_match_the_stored_datasets():
    """Imaris has no second Python reader, so h5py is the reference.

    That is still an independent opinion: it reads the raw dataset, and
    what is under test is everything this reader does on top -- level
    discovery, the declared-versus-stored extent, and the trim.
    """
    h5py = pytest.importorskip("h5py")
    from opencodecs._imaris import ImarisReader

    path = _imaris_file()
    reader = ImarisReader(str(path))
    with h5py.File(path, "r") as h:
        for index, level in enumerate(reader.levels):
            node = h[f"/DataSet/ResolutionLevel {index}"
                     f"/TimePoint 0/Channel 0/Data"]
            ref = node[0, :level.shape[0], :level.shape[1]]
            got = np.asarray(reader.read_region(index))
            assert got.shape == ref.shape, f"level {index}"
            assert np.array_equal(got, ref), f"level {index} values differ"


def test_imaris_stored_array_is_padded_and_trimmed():
    """The stored array is larger than the image; the attributes rule.

    Returning the stored extent instead would hand back a plausible
    array with a border of whatever the padding happens to contain.
    """
    h5py = pytest.importorskip("h5py")
    from opencodecs._imaris import ImarisReader

    path = _imaris_file()
    reader = ImarisReader(str(path))
    with h5py.File(path, "r") as h:
        stored = h["/DataSet/ResolutionLevel 0"
                   "/TimePoint 0/Channel 0/Data"].shape
    declared = reader.levels[0].shape
    assert declared[0] <= stored[1] and declared[1] <= stored[2]
    assert declared != stored[1:], (
        "this fixture is supposed to be padded; if upstream rewrote it "
        "unpadded, this documents the change rather than a bug")


def test_imaris_edge_region_is_not_padding():
    """A bbox at the far corner is the place padding leaks through."""
    h5py = pytest.importorskip("h5py")
    from opencodecs._imaris import ImarisReader

    path = _imaris_file()
    reader = ImarisReader(str(path))
    height, width = reader.levels[0].shape
    y0, x0 = max(0, height - 40), max(0, width - 40)
    got = np.asarray(reader.read_region(0, y=(y0, height), x=(x0, width)))
    with h5py.File(path, "r") as h:
        ref = h["/DataSet/ResolutionLevel 0"
                "/TimePoint 0/Channel 0/Data"][0, y0:height, x0:width]
    assert got.shape == ref.shape and np.array_equal(got, ref)


# ====================================================================
# NDTiff vs the ndtiff package
# ====================================================================

def test_ndtiff_frames_match_the_reference_reader(tmp_path):
    """Every frame, not a sample: the index is what could go wrong."""
    ndtiff = pytest.importorskip("ndtiff")
    from opencodecs._ndtiff import NDTiffDataset

    source = DATA / "ndtiff"
    if not (source / "NDTiff.index").is_file():
        pytest.skip("fetch the ndtiff corpus entry first")

    # The reference reader globs the directory and chokes on the
    # AppleDouble files an SMB share leaves behind, so give it a clean
    # copy. Ours reads NDTiff.index and is unaffected either way.
    clean = tmp_path / "dataset"
    clean.mkdir()
    for f in source.iterdir():
        if f.is_file() and not f.name.startswith("._"):
            shutil.copy2(f, clean / f.name)

    reference = ndtiff.Dataset(str(clean))
    ours = NDTiffDataset(str(clean))
    coordinates = list(reference.get_image_coordinates_list())
    assert coordinates, "reference reader found no images"
    assert len(ours) == len(coordinates)
    for key in coordinates:
        theirs = np.asarray(reference.read_image(**key))
        mine = np.asarray(ours.read_frame(**key))
        assert mine.shape == theirs.shape, key
        assert np.array_equal(mine, theirs), f"frame {key} differs"


# ====================================================================
# DICOM: the real corpus, against pydicom
# ====================================================================

DICOM_FILES = _real_files("dicom", "*.dcm")


@pytest.mark.parametrize(
    "path", DICOM_FILES or [pytest.param(None, marks=pytest.mark.skip(
        reason="fetch the dicom corpus entry first"))],
    ids=lambda p: p.name if p is not None else "no-corpus")
def test_dicom_corpus_matches_pydicom(path):
    """Six files spanning explicit, implicit, RLE and JPEG 2000.

    The synthetic coverage in ``test_dicom_encodings.py`` is about the
    encodings; this is about real files written by real tools, and it
    is the only check on the encapsulated decode path against something
    that is not us.
    """
    pydicom = pytest.importorskip("pydicom")
    from opencodecs._dicom import DicomFile

    dataset = pydicom.dcmread(str(path))
    try:
        ref = dataset.pixel_array
    except Exception as exc:                            # noqa: BLE001
        pytest.skip(f"pydicom cannot decode this one: {exc}")

    with DicomFile(str(path)) as d:
        got = d.asarray()
    assert got.shape == ref.shape
    assert got.dtype.kind == ref.dtype.kind
    assert np.array_equal(got, ref), (
        f"differs from pydicom; transfer syntax "
        f"{dataset.file_meta.TransferSyntaxUID}")


# ====================================================================
# the skips themselves
# ====================================================================

# Reference readers that are pure Python or have wheels everywhere we
# test. If one of these is missing, the parity check it backs is not
# running, and a silent skip is exactly how six reader bugs survived a
# green suite. The rest (tensorstore, ndtiff, czifile, nd2, readlif,
# oiffile) are allowed to be absent because they are not available on
# every platform and Python version we build for.
CORE_REFERENCES = {
    "pydicom": "DICOM",
    "nrrd": "NRRD",
    "nibabel": "NIfTI",
    "mrcfile": "MRC",
    "astropy": "FITS",
}

OPTIONAL_REFERENCES = {
    "rsciio": "DM and EMD",
    "tensorstore": "N5",
    "ndtiff": "NDTiff",
    "czifile": "CZI",
    "nd2": "ND2",
    "readlif": "LIF",
    "oiffile": "OIB",
}


def _present(module):
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):                    # pragma: no cover
        return False


@pytest.mark.skipif(
    not __import__("os").environ.get("OPENCODECS_REQUIRE_REFERENCE"),
    reason="set OPENCODECS_REQUIRE_REFERENCE=1 to enforce (CI does)")
def test_core_reference_implementations_are_installed():
    """Turn a silent skip into a failure on the job that promises them.

    Every cross-check in this suite is an ``importorskip``, so without
    the reference readers installed the whole independent-verification
    layer reports success while testing nothing. That was the state of
    CI until 2026-09-03: ``.[test]`` installed none of them.
    """
    missing = {m: fmt for m, fmt in CORE_REFERENCES.items() if not _present(m)}
    assert not missing, (
        "reference readers missing, so these parity checks are inert: "
        + ", ".join(f"{m} ({fmt})" for m, fmt in sorted(missing.items()))
        + ". Install with: pip install -e .[reference]")


def test_report_reference_coverage(capsys):
    """Always print what is and is not being cross-checked.

    Not an assertion: on a developer machine some of these are
    reasonably absent. The point is that the output says so, rather
    than the skips scrolling past unread.
    """
    everything = {**CORE_REFERENCES, **OPTIONAL_REFERENCES}
    have = {m: f for m, f in everything.items() if _present(m)}
    missing = {m: f for m, f in everything.items() if m not in have}
    with capsys.disabled():
        print(f"\n  cross-checked ({len(have)}): "
              f"{', '.join(sorted(have.values())) or 'nothing'}")
        if missing:
            print(f"  NOT cross-checked ({len(missing)}): "
                  f"{', '.join(sorted(missing.values()))}")
    assert everything
