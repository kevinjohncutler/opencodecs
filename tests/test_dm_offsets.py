"""A DM file is read at its offsets, not swallowed.

The tag tree carries an absolute offset and length for every image, so
nothing about the format requires holding the file -- the reader simply
used to. These tests pin the consequence rather than the implementation:
opening a file must not move its image data, and reading one image must
not move the others.

The corpus fixtures are 32 KB and 50 KB, both smaller than the reader's
window, so they cannot tell a windowed read from a whole-file one. That
is why these build their own.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc
from opencodecs._dm import DmFile

from _range_http_server import range_http_server
from test_dm_synthetic import build_dm

# Comfortably past the 128 KB window, so a whole-file read is
# unmistakable in the byte counts.
BIG = (np.arange(1024 * 1024, dtype="u2") % 4093).reshape(1024, 1024)


@pytest.fixture(scope="module")
def dm4(tmp_path_factory):
    p = tmp_path_factory.mktemp("dm") / "big.dm4"
    p.write_bytes(build_dm(4, BIG))
    return p


def test_opening_does_not_move_the_image(dm4, tmp_path):
    """Metadata is at the front; the samples are 2 MB of it that a
    caller asking for the shape never needs."""
    with range_http_server(str(dm4.parent)) as (base, t):
        with oc.get_codec("dm").open(f"{base}/{dm4.name}") as f:
            assert f.shape_at(0) == BIG.shape
            assert f.n_images == 1
        opened = t.bytes_served
    assert opened < BIG.nbytes // 4, (
        f"opening moved {opened} bytes of a {dm4.stat().st_size}-byte file")


def test_reading_one_image_moves_about_one_image(dm4):
    with range_http_server(str(dm4.parent)) as (base, t):
        with oc.get_codec("dm").open(f"{base}/{dm4.name}") as f:
            arr = f.asarray(0)
        served = t.bytes_served
    assert np.array_equal(arr, BIG)
    # The image itself, plus the windows the tag walk needed. Anything
    # near twice the image means it was fetched and then fetched again.
    assert served < BIG.nbytes * 1.5, f"{served} bytes for a {BIG.nbytes} image"


def test_http_matches_local_bytes(dm4):
    local = oc.get_codec("dm").decode(str(dm4))
    with range_http_server(str(dm4.parent)) as (base, _):
        remote = oc.get_codec("dm").decode(f"{base}/{dm4.name}")
    assert np.array_equal(local, remote)
    assert local.dtype == remote.dtype


@pytest.mark.parametrize("version", [3, 4])
def test_window_boundary_does_not_split_a_value(version, tmp_path):
    """A tag straddling the window edge must still read.

    The window refetches from the requested offset when a read falls
    outside it, so a value split across the boundary is the case that
    would break if it refetched from the window start instead. These
    sizes put the tags that FOLLOW the image -- Dimensions and DataType
    -- right at the 128 KB edge, which is the only place a value can
    land across it. Two rows, because a dimension is a uint16.
    """
    for n in (65528, 65536, 65544):
        img = (np.arange(n, dtype="u2") % 251).reshape(2, n // 2)
        p = tmp_path / f"w{version}_{n}.dm{version}"
        p.write_bytes(build_dm(version, img))
        with DmFile(str(p)) as f:
            assert np.array_equal(f.asarray(0), img)


def test_a_closed_reader_releases_its_source(dm4):
    f = DmFile(str(dm4))
    f.asarray(0)
    f.close()
    with pytest.raises(Exception):
        f.asarray(0)
