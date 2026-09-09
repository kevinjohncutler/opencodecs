"""The HDF5-backed readers over HTTP, through one shared helper.

hdf5, emd and imaris all reach h5py through ``_h5_common.h5_source``,
and _hdf5_http has had a range-reading file-like since before emd and
imaris existed. They simply never met: each opened h5py itself, so
only the hdf5 codec had a remote path and it was a separate public
function. Teaching the one helper about URLs gave all three range
reads at once, which is what the manifest meant by calling emd's gaps
a wiring problem rather than a format limit.

The point of these tests is the BYTES MOVED. A reader that downloads
the whole file and then slices it returns identical data and looks
perfectly correct; only the transfer size tells the two apart.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

import opencodecs as oc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
h5py = pytest.importorskip("h5py")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """A directory with one big HDF5 and one big EMD, served by range."""
    from _range_http_server import range_http_server

    d = tmp_path_factory.mktemp("h5remote")
    small = np.arange(64 * 64, dtype="f4").reshape(64, 64)
    big = np.zeros((256, 256, 64), "f4")

    with h5py.File(d / "plain.h5", "w") as f:
        f.create_dataset("small", data=small, chunks=(16, 16))
        f.create_dataset("zbig", data=big, chunks=(32, 32, 8))
    with h5py.File(d / "s.emd", "w") as f:
        g = f.create_group("d/aaa")
        g.attrs["emd_group_type"] = 1
        g.create_dataset("data", data=small)
        g2 = f.create_group("d/zzz")
        g2.attrs["emd_group_type"] = 1
        g2.create_dataset("data", data=big, chunks=(32, 32, 8))

    with range_http_server(d) as s:
        base, tracker = s if isinstance(s, tuple) else (s, None)
        yield d, base, tracker, small


def test_hdf5_reads_a_dataset_over_http_without_the_file(served):
    d, base, tracker, small = served
    size = (d / "plain.h5").stat().st_size
    before = tracker.bytes_served if tracker else 0
    with oc.get_codec("hdf5").open(f"{base}/plain.h5", dataset="small") as r:
        got = np.asarray(r.read())
    assert np.array_equal(got, small)
    if tracker is not None:
        moved = tracker.bytes_served - before
        assert moved < size / 10, (
            f"moved {moved} of {size} bytes; that is a download, not a "
            f"range read")


def test_emd_reads_a_dataset_over_http_without_the_file(served):
    d, base, tracker, small = served
    size = (d / "s.emd").stat().st_size
    before = tracker.bytes_served if tracker else 0
    with oc.get_codec("emd").open(f"{base}/s.emd") as r:
        got = np.asarray(r[0])
    assert np.array_equal(got, small)
    if tracker is not None:
        moved = tracker.bytes_served - before
        assert moved < size / 10, (
            f"moved {moved} of {size} bytes; that is a download, not a "
            f"range read")


def test_remote_and_local_agree(served):
    d, base, tracker, small = served
    local = np.asarray(oc.get_codec("emd").open(str(d / "s.emd"))[0])
    with oc.get_codec("emd").open(f"{base}/s.emd") as r:
        assert np.array_equal(np.asarray(r[0]), local)


def test_h5_source_maps_each_kind_to_what_h5py_takes(tmp_path):
    """The helper's whole job, checked directly."""
    import io

    from opencodecs._h5_common import h5_source

    p = tmp_path / "x.h5"
    with h5py.File(p, "w") as f:
        f["d"] = np.arange(8)
    raw = p.read_bytes()

    assert h5_source(str(p)) == str(p), "a path must pass through"
    assert isinstance(h5_source(raw), io.BytesIO), "bytes must be wrapped"
    assert type(h5_source("https://example.com/x.h5")).__name__ \
        == "_HTTPFileLike"
    fh = io.BytesIO(raw)
    assert h5_source(fh) is fh, "a file-like must pass through untouched"


def test_emd_codec_flags_match_what_the_reader_does():
    """The flags said no to things the reader had always done.

    EmdFile has advertised is_chunked = True since it was written,
    while the codec flag beside it said chunked = False. The manifest
    reads the codec flag, so it recorded a capability as missing that
    was there the whole time.
    """
    codec = oc.get_codec("emd")
    assert codec.chunked is True
    assert codec.streaming_decode is True
