"""``Codec.chunked`` promises cheap indexing, not merely indexing.

Two flags in this package answer questions that sound identical:

    Reader.is_chunked   can you index this reader at all
    Codec.chunked       is indexing CHEAP -- frame N without
                        decoding 0..N-1

Nothing said so, and the gap between them is where a false claim
lives. EER set both: indexing worked, but through the base Reader's
default ``__getitem__``, which walks ``iter_frames()`` to get there --
6.71 ms for frame 0 and 1218 ms for frame 720 of 721. The capability
manifest reads the second flag, so it promised random access that
took longer than reading the file.

GIF is the honest version of the same shape: ``is_chunked = True`` and
``chunked = False``, with a comment saying it replays from frame 0
because disposal state forbids seeking.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import opencodecs as oc
from opencodecs.core.codec import Codec, Reader


def _readers():
    out = []
    for info in sorted(oc.list_codecs(), key=lambda c: c["name"]):
        cls = type(oc.get_codec(info["name"]))
        for k in cls.__mro__:
            if "open" in k.__dict__:
                if k is not Codec:
                    out.append(info["name"])
                break
    return out


def test_a_chunked_codec_does_not_use_the_walking_default():
    """``chunked = True`` with the inherited ``__getitem__`` is a
    contradiction.

    The default walks ``iter_frames()`` and picks, so a reader using it
    cannot be fetching frame N without decoding the ones before. This
    is the structural half of the claim; the timing half needs a
    multi-frame file and lives in the per-format tests.
    """
    offenders = []
    for name in _readers():
        codec = oc.get_codec(name)
        if not codec.chunked:
            continue
        # Find the class that would answer r[i].
        impl = None
        for k in type(codec).__mro__:
            if "open" in k.__dict__:
                impl = k
                break
        # The reader class is not known without opening a file, so this
        # checks the codec's declared reader where it exposes one.
        reader_cls = getattr(codec, "reader_class", None)
        if reader_cls is None:
            continue
        if not any("__getitem__" in k.__dict__
                   for k in reader_cls.__mro__ if k is not Reader):
            offenders.append(name)
    assert not offenders, (
        f"these declare chunked=True but inherit the walking "
        f"__getitem__: {offenders}")


def test_eer_random_access_is_cheap_and_says_so():
    """The claim this file exists because of, now the other way round.

    EER frames are independent and sit at their own IFD offsets, so
    cheap random access was always buildable; it simply was not built,
    and chunked said otherwise. EerReader indexes at the offset now, so
    the flag is True and the timing test in test_eer_parallel.py is
    what keeps it honest.
    """
    if not oc.has_codec("eer"):
        pytest.skip("eer not built here")
    codec = oc.get_codec("eer")
    assert codec.chunked is True
    from opencodecs._eer_reader import EerReader
    assert "__getitem__" in EerReader.__dict__, (
        "chunked=True with the inherited walking __getitem__ is the "
        "exact contradiction this file was written for")


def test_gif_offers_indexing_without_claiming_it_is_cheap():
    """The two flags disagreeing is not automatically a bug.

    GIF frames carry disposal state, so frame N genuinely requires the
    ones before it. Indexing is still offered, and saying so through
    is_chunked while chunked stays False is exactly right.
    """
    if not oc.has_codec("gif"):
        # giflib is not available on every CI runner, and a codec that
        # did not build is not a codec making a wrong claim.
        pytest.skip("gif not built here")
    assert oc.get_codec("gif").chunked is False
    # Asked of an instance: the reader that answers r[i] for GIF is a
    # Cython type, and the flag lives on the wrapper beside it.
    img = (np.arange(32 * 32, dtype="u1") % 251).reshape(32, 32)
    blob = oc.get_codec("gif").encode(img)
    with oc.get_codec("gif").open(blob) as r:
        assert getattr(r, "is_chunked", False) is True, (
            "GIF offers indexing; only its cost is the caveat")


@pytest.mark.parametrize("name", ["mrc", "oir", "vsi"])
def test_indexing_is_flat_across_frames(name, tmp_path):
    """The last frame must cost about what the first does.

    Timing in a test is usually a bad idea, so the threshold is loose
    on purpose: this is not measuring speed, it is separating O(1)
    from walking the whole file, which on these samples differ by two
    orders of magnitude.
    """
    import glob
    import pathlib

    if not oc.has_codec(name):
        pytest.skip(f"{name} not built here")
    corpus = pathlib.Path(__file__).resolve().parent.parent / ".test_data"
    codec = oc.get_codec(name)

    if name == "mrc":
        vol = (np.arange(24 * 64 * 64, dtype="i2") % 3000).reshape(24, 64, 64)
        path = tmp_path / "v.mrc"
        oc.write_mrc(str(path), vol)
    else:
        hits = [h for e in (codec.file_extensions or ())
                for h in sorted(glob.glob(f"{corpus}/**/*{e}", recursive=True))
                if "/._" not in h]
        if not hits:
            pytest.skip(f"no {name} sample in the corpus")
        path = pathlib.Path(hits[0])

    with codec.open(str(path)) as r:
        n = r.n_frames or 1
        if n < 8:
            pytest.skip(f"{name} sample has only {n} frames")

    def best(idx):
        codec.open(str(path))[idx]
        b = 1e9
        for _ in range(3):
            s = time.perf_counter()
            codec.open(str(path))[idx]
            b = min(b, time.perf_counter() - s)
        return b

    first, last = best(0), best(n - 1)
    assert last < max(first * 8, first + 0.05), (
        f"{name}: frame {n - 1} took {last * 1e3:.1f} ms against "
        f"{first * 1e3:.1f} ms for frame 0, which is walking, not "
        f"random access")

    # Flat is necessary and not sufficient. A reader that decodes the
    # whole file for EVERY frame is perfectly flat, and the first
    # version of this test would have passed it. FITS caught that: all
    # three timings came out equal, which reads as O(1) and is the
    # signature of the opposite. Bytes moved is the check that cannot
    # be argued with, and it lives in the per-format tests where a
    # range server is set up.


@pytest.mark.parametrize("name", ["dicom", "fits"])
def test_one_frame_over_http_moves_one_frame(name, tmp_path):
    """The measurement that settles it: bytes off the wire.

    Both of these declared chunked = False by inheriting it, while
    their readers had per-frame offsets and used them. Locally that is
    hard to see -- a reader that loads everything and slices returns
    correct data quickly enough to look fine, and for FITS ``read()``
    returns a single HDU so it is not even a useful baseline. Over a
    range server, fetching the last frame either moves a frame's worth
    or it moves the file.
    """
    import pathlib as _p
    import sys as _sys

    _sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
    try:
        from _range_http_server import range_http_server
    except ImportError:
        pytest.skip("range test server helper unavailable")
    if not oc.has_codec(name):
        pytest.skip(f"{name} not built here")

    side, n = 256, 16
    if name == "fits":
        afits = pytest.importorskip("astropy.io.fits")
        hdus = [afits.PrimaryHDU(np.zeros((side, side), np.int16))]
        for i in range(n - 1):
            hdus.append(afits.ImageHDU(
                np.full((side, side), i, np.int16), name=f"IM{i}"))
        path = tmp_path / "many.fits"
        afits.HDUList(hdus).writeto(path, overwrite=True)
    else:
        pydicom = pytest.importorskip("pydicom")
        from pydicom.dataset import Dataset, FileMetaDataset

        frames = np.stack(
            [np.full((side, side), i, np.uint16) for i in range(n)])
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        ds.SOPClassUID = pydicom.uid.MRImageStorage
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.Rows, ds.Columns = side, side
        ds.NumberOfFrames = n
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = frames.tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        path = tmp_path / "many.dcm"
        try:
            ds.save_as(path, enforce_file_format=True)
        except TypeError:                                     # older pydicom
            ds.save_as(path, write_like_original=False)

    size = path.stat().st_size
    codec = oc.get_codec(name)
    with codec.open(str(path)) as r:
        total = r.n_frames or 1
        local = np.asarray(r[total - 1])

    with range_http_server(tmp_path) as s:
        base, tracker = s if isinstance(s, tuple) else (s, None)
        before = tracker.bytes_served if tracker else 0
        with codec.open(f"{base}/{path.name}") as r:
            got = np.asarray(r[total - 1])
        assert np.array_equal(got, local), "remote frame differs from local"
        if tracker is not None:
            moved = tracker.bytes_served - before
            assert moved < size / 3, (
                f"{name}: fetching frame {total - 1} moved {moved} of "
                f"{size} bytes; that is the file, not a frame")
