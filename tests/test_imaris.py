"""Imaris (.ims) reader.

Synthetic files are built here to the same layout Imaris writes, which
lets us pin the two conventions that make the format more than "HDF5
with arrays in it": the padded storage array and the character-array
attributes. The real Convallaria stack confirms those conventions are
what Bitplane's software actually emits rather than what we assume.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from opencodecs._imaris import ImarisError, ImarisReader  # noqa: E402

REAL = (pathlib.Path(__file__).resolve().parent.parent / ".test_data"
        / "imaris" / "Convallaria_3C_1T_2x2grid_confocal.ims")


def _char_attr(group, key, value):
    """Write an attribute the way Imaris does: an array of single chars.

    Each element must be a one-byte bytes object. Passing the ints from
    ``list(b"8")`` instead makes numpy stringify 56 and truncate it to
    b"5", which silently writes the wrong number.
    """
    chars = [bytes([c]) for c in str(value).encode("ascii")]
    group.attrs[key] = np.array(chars, dtype="|S1")


def build_ims(path, levels, *, channels=1, timepoints=1, dtype="<u2", pad=0):
    """Write a minimal Imaris-shaped HDF5 file.

    ``levels`` is a list of (z, y, x) real extents. Each stored array is
    allocated ``pad`` larger in y and x, with the extra zeroed, which is
    what Imaris does to land on chunk boundaries.
    """
    with h5py.File(path, "w") as f:
        ds = f.create_group("DataSet")
        for li, (z, y, x) in enumerate(levels):
            lg = ds.create_group(f"ResolutionLevel {li}")
            for t in range(timepoints):
                tg = lg.create_group(f"TimePoint {t}")
                for c in range(channels):
                    cg = tg.create_group(f"Channel {c}")
                    stored = np.zeros((z, y + pad, x + pad), dtype=dtype)
                    real = (np.arange(z * y * x, dtype=dtype).reshape(z, y, x)
                            + (c * 1000) + (t * 100))
                    stored[:z, :y, :x] = real
                    cg.create_dataset("Data", data=stored)
                    _char_attr(cg, "ImageSizeX", x)
                    _char_attr(cg, "ImageSizeY", y)
                    _char_attr(cg, "ImageSizeZ", z)
        info = f.create_group("DataSetInfo")
        img = info.create_group("Image")
        z0, y0, x0 = levels[0]
        for k, v in (("X", x0), ("Y", y0), ("Z", z0)):
            _char_attr(img, k, v)
        for i, ext in enumerate((10.0, 20.0, 3.0)):
            _char_attr(img, f"ExtMin{i}", 0.0)
            _char_attr(img, f"ExtMax{i}", ext)
        for c in range(channels):
            cg = info.create_group(f"Channel {c}")
            _char_attr(cg, "Name", f"chan{c}")
    return path


# --------------------------------------------------------------------
# the two Imaris conventions
# --------------------------------------------------------------------

def test_stored_padding_is_cropped_away(tmp_path):
    """Imaris allocates on chunk bounds; the extra must not be returned.

    Without the crop a caller gets a border of fabricated zeros that
    looks like real background.
    """
    p = build_ims(tmp_path / "a.ims", [(1, 10, 12)], pad=6)
    with h5py.File(p, "r") as f:
        stored = f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"]["Channel 0"]["Data"]
        assert stored.shape == (1, 16, 18)
    with ImarisReader(str(p)) as r:
        assert r.level_shape(0) == (1, 10, 12)
        a = r.read(level=0)
    assert a.shape == (1, 10, 12)
    assert int(a.sum()) == int(np.arange(1 * 10 * 12).sum())


def test_character_array_attributes_are_decoded(tmp_path):
    """Imaris stores attributes as arrays of single characters."""
    p = build_ims(tmp_path / "a.ims", [(1, 4, 4)])
    with h5py.File(p, "r") as f:
        raw = f["DataSetInfo"]["Image"].attrs["X"]
        assert not isinstance(raw, (str, bytes))   # it really is an array
    with ImarisReader(str(p)) as r:
        assert r.info["X"] == "4"
        assert r.channel_info(0)["Name"] == "chan0"


def test_each_level_is_cropped_by_its_own_extent(tmp_path):
    """The global Image extent describes level 0 only.

    Using it for every level would over-crop the coarser ones to the
    full-resolution size, or worse, index out of range.
    """
    p = build_ims(tmp_path / "a.ims", [(1, 16, 16), (1, 8, 8), (1, 4, 4)], pad=4)
    with ImarisReader(str(p)) as r:
        assert [r.level_shape(i) for i in range(3)] == [
            (1, 16, 16), (1, 8, 8), (1, 4, 4)]
        assert r.read(level=2).shape == (1, 4, 4)


# --------------------------------------------------------------------
# pyramid contract
# --------------------------------------------------------------------

def test_pyramid_levels_are_the_yx_plane(tmp_path):
    """PyramidLevel.shape is (y, x); the accessor keeps the full (z, y, x).

    The shared region API measures bounding boxes against the level
    shape, so handing it (z, y, x) would make a single-plane stack
    report a height of 1.
    """
    p = build_ims(tmp_path / "a.ims", [(3, 16, 20), (3, 8, 10)])
    with ImarisReader(str(p)) as r:
        assert r.n_levels == 2
        assert [lv.shape for lv in r.levels] == [(16, 20), (8, 10)]
        assert r.levels[0].reader.shape == (3, 16, 20)
        assert r.levels[1].downscale == (2, 2)


def test_read_region_returns_the_requested_box(tmp_path):
    p = build_ims(tmp_path / "a.ims", [(1, 16, 20)])
    with ImarisReader(str(p)) as r:
        reg = r.read_region(0, y=(2, 10), x=(3, 15))
        assert reg.shape == (8, 12)
        full = r.read(level=0)[0]
        assert np.array_equal(reg, full[2:10, 3:15])


def test_read_region_on_a_stack_keeps_z(tmp_path):
    p = build_ims(tmp_path / "a.ims", [(4, 16, 20)])
    with ImarisReader(str(p)) as r:
        assert r.read_region(0, y=(0, 8), x=(0, 8)).shape == (4, 8, 8)


def test_best_level_for_uses_the_plane_height(tmp_path):
    p = build_ims(tmp_path / "a.ims", [(1, 64, 64), (1, 32, 32), (1, 16, 16)])
    with ImarisReader(str(p)) as r:
        assert r.best_level_for(max_pixels_y=20) == 2


# --------------------------------------------------------------------
# channels, timepoints, failure modes
# --------------------------------------------------------------------

def test_channels_and_timepoints_are_addressable(tmp_path):
    p = build_ims(tmp_path / "a.ims", [(1, 4, 4)], channels=3, timepoints=2)
    with ImarisReader(str(p)) as r:
        assert (r.n_channels, r.n_timepoints) == (3, 2)
        c0 = r.read(channel=0)
        c2 = r.read(channel=2)
        assert not np.array_equal(c0, c2)
        assert not np.array_equal(r.read(timepoint=0), r.read(timepoint=1))


def test_level_ordering_is_numeric_not_lexical(tmp_path):
    """"ResolutionLevel 10" must sort after "ResolutionLevel 9"."""
    p = build_ims(tmp_path / "a.ims", [(1, 2 ** (11 - i), 2 ** (11 - i))
                                       for i in range(11)])
    with ImarisReader(str(p)) as r:
        heights = [lv.shape[0] for lv in r.levels]
        assert heights == sorted(heights, reverse=True), heights


def test_plain_hdf5_is_refused(tmp_path):
    p = tmp_path / "plain.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("data", data=np.zeros((4, 4)))
    with pytest.raises(ImarisError, match="not an\\s+Imaris one"):
        ImarisReader(str(p))


def test_out_of_range_index_raises(tmp_path):
    p = build_ims(tmp_path / "a.ims", [(1, 4, 4)])
    with ImarisReader(str(p)) as r:
        with pytest.raises(IndexError):
            r.read(channel=5)


# --------------------------------------------------------------------
# real Imaris output
# --------------------------------------------------------------------

@pytest.mark.skipif(not REAL.is_file(),
                    reason="run `python corpus/corpus.py fetch imaris_convallaria`")
def test_real_file_structure():
    with ImarisReader(str(REAL)) as r:
        assert (r.n_levels, r.n_timepoints, r.n_channels) == (3, 1, 3)
        assert r.dtype == np.dtype("<u2")
        assert [lv.shape for lv in r.levels] == [(1949, 1949), (974, 974), (487, 487)]
        assert [lv.downscale for lv in r.levels] == [(1, 1), (2, 2), (4, 4)]


@pytest.mark.skipif(not REAL.is_file(),
                    reason="run `python corpus/corpus.py fetch imaris_convallaria`")
def test_real_file_is_padded_on_disk():
    """The padding is not hypothetical: this file stores 2048 for 1949."""
    with ImarisReader(str(REAL)) as r:
        stored = r._channel_group(0, 0, 0)["Data"].shape
        assert stored == (1, 2048, 2048)
        assert r.level_shape(0) == (1, 1949, 1949)
        a = r.read(level=0)
    assert a.shape == (1, 1949, 1949)
    assert int(a.max()) > 0


@pytest.mark.skipif(not REAL.is_file(),
                    reason="run `python corpus/corpus.py fetch imaris_convallaria`")
def test_real_file_channels_differ():
    with ImarisReader(str(REAL)) as r:
        assert not np.array_equal(r.read(level=2, channel=0),
                                  r.read(level=2, channel=1))
        assert r.channel_info(0)["Color"].startswith("1.000")
