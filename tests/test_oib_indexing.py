"""An OIB index reads only the streams behind it.

An OIB is an OLE2 compound file whose directory gives every stream's
sector chain, and every frame is its own TIFF stream. So a plane has
always been addressable; the reader assembled the whole experiment and
sliced the result, and said so honestly with is_chunked = False.

Axis 0 is not necessarily one frame. FluoView order is (T?, C?, Z?, H,
W) with singleton axes collapsed, so on the corpus file -- 2 channels
of 6 z-slices -- index 0 is six streams, and reading it must read
exactly those six.
"""

from __future__ import annotations

import pathlib
import shutil

import numpy as np
import pytest

import opencodecs as oc

from _range_http_server import range_http_server

OIB = (pathlib.Path(__file__).resolve().parent.parent
       / ".test_data" / "oib" / "imagesc_71616_60x.oib")
needs_corpus = pytest.mark.skipif(
    not OIB.is_file(), reason="fetch the oib corpus entry first")


@pytest.fixture(scope="module")
def reader():
    with oc.get_codec("oib").open(str(OIB)) as r:
        yield r


@pytest.fixture(scope="module")
def full(reader):
    return reader.read()


@needs_corpus
def test_every_index_matches_the_bulk_read(reader, full):
    assert reader.n_frames == full.shape[0]
    for i in range(reader.n_frames):
        assert np.array_equal(reader[i], full[i]), f"index {i}"


@needs_corpus
def test_iteration_matches_the_bulk_read(reader, full):
    assert np.array_equal(np.stack(list(reader.iter_frames())), full)


@needs_corpus
def test_negative_and_out_of_range(reader, full):
    n = full.shape[0]
    assert np.array_equal(reader[-1], full[n - 1])
    assert np.array_equal(reader[-n], full[0])
    for bad in (n, n + 2, -n - 1):
        with pytest.raises(IndexError):
            reader[bad]


@needs_corpus
def test_the_flag_and_the_reader_agree(reader):
    """chunked=True over the inherited walking __getitem__ is the
    contradiction test_chunked_means_cheap.py exists for."""
    assert oc.get_codec("oib").chunked is True
    assert reader.is_chunked is True
    from opencodecs._oib_native import OibNativeReader
    assert "__getitem__" in OibNativeReader.__dict__


@needs_corpus
def test_indexing_moves_less_than_a_full_read(tmp_path):
    """Stated in bytes, which is the part that cannot be argued with.

    The corpus file is 2 channels of 6 z-slices, so index 0 is half the
    streams and should move about half the file.
    """
    shutil.copy(OIB, tmp_path / "a.oib")
    codec = oc.get_codec("oib")

    def served(fn):
        with range_http_server(str(tmp_path)) as (base, t):
            with codec.open(f"{base}/a.oib") as r:
                fn(r)
            return t.bytes_served

    one = served(lambda r: r[0])
    whole = served(lambda r: r.read())
    assert one < whole * 0.75, (
        f"index 0 moved {one} bytes against {whole} for the whole file")


@needs_corpus
def test_opening_over_http_does_not_move_the_file(tmp_path):
    """Opening reads the directory and the metadata streams only."""
    shutil.copy(OIB, tmp_path / "b.oib")
    size = (tmp_path / "b.oib").stat().st_size
    with range_http_server(str(tmp_path)) as (base, t):
        with oc.get_codec("oib").open(f"{base}/b.oib") as r:
            shape = r.shape
        served = t.bytes_served
    assert shape[-2:] == (1024, 1024)
    assert served < size * 0.05, f"opening moved {served} of {size} bytes"
