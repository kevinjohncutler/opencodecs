"""MRC2014 / CCP4 map reader.

Synthetic headers cover the mode and endianness matrix, because we can
construct every combination and no real corpus has all of them. The real
EMDB deposit covers what a synthetic header cannot: a non-zero extended
header, a permuted axis order, and stats we can check against the values
the depositor's own software wrote into the header.
"""

from __future__ import annotations

import pathlib
import struct

import numpy as np
import pytest

import opencodecs as oc
from opencodecs._mrc import HEADER_SIZE, MrcError, MrcStream

REAL_MAP = (pathlib.Path(__file__).resolve().parent.parent
            / ".test_data" / "mrc" / "emd_3001.map")


def build_mrc(nx, ny, nz, mode, data, *, byteorder="<", nsymbt=0, ext=b"",
              mapc=1, mapr=2, maps=3, cella=None, with_magic=True):
    """Assemble a minimal but valid MRC file."""
    h = bytearray(HEADER_SIZE)
    e = byteorder
    struct.pack_into(e + "iiii", h, 0, nx, ny, nz, mode)
    struct.pack_into(e + "iii", h, 28, nx, ny, nz)              # MX MY MZ
    cella = cella or (nx * 1.0, ny * 1.0, nz * 1.0)
    struct.pack_into(e + "fff", h, 40, *cella)
    struct.pack_into(e + "iii", h, 64, mapc, mapr, maps)
    struct.pack_into(e + "i", h, 92, nsymbt)
    if with_magic:
        h[208:212] = b"MAP "
    h[212:214] = b"\x44\x44" if byteorder == "<" else b"\x11\x11"
    return bytes(h) + ext + data.tobytes()


# --------------------------------------------------------------------
# modes and byte order
# --------------------------------------------------------------------

@pytest.mark.parametrize("mode,dtype", [
    (0, "i1"), (1, "i2"), (2, "f4"), (6, "u2"), (12, "f2"),
])
@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_modes_and_byteorder_roundtrip(mode, dtype, byteorder):
    a = (np.arange(3 * 5 * 4) % 97).astype(byteorder + dtype).reshape(3, 5, 4)
    blob = build_mrc(4, 5, 3, mode, a, byteorder=byteorder)
    got = oc.read(blob, format="mrc")
    assert got.shape == (3, 5, 4)
    assert np.array_equal(np.asarray(got, dtype=dtype), np.asarray(a, dtype=dtype))


def test_single_section_is_two_dimensional():
    """nz == 1 is how MRC spells 'this is an image', so return 2-D."""
    a = np.arange(20, dtype="f4").reshape(4, 5)
    got = oc.read(build_mrc(5, 4, 1, 2, a), format="mrc")
    assert got.shape == (4, 5)


def test_extended_header_is_skipped_and_readable():
    """Data starts after NSYMBT bytes, and the bytes are retrievable."""
    ext = bytes(range(256)) * 2                     # 512 bytes
    a = np.arange(3 * 5 * 4, dtype="f4").reshape(3, 5, 4)
    blob = build_mrc(4, 5, 3, 2, a, nsymbt=len(ext), ext=ext)
    with MrcStream(blob) as r:
        assert r.data_offset == HEADER_SIZE + len(ext)
        assert r.extended_header() == ext
        assert np.array_equal(r.asarray(), a)


def test_plane_reads_one_section():
    a = np.arange(3 * 5 * 4, dtype="f4").reshape(3, 5, 4)
    with MrcStream(build_mrc(4, 5, 3, 2, a)) as r:
        assert r.n_planes == 3
        for i in range(3):
            assert np.array_equal(r.plane(i), a[i])
        with pytest.raises(IndexError):
            r.plane(3)


# --------------------------------------------------------------------
# things that should be refused
# --------------------------------------------------------------------

def test_truncated_header_raises():
    with pytest.raises(MrcError, match="shorter than"):
        MrcStream(b"\x00" * 100)


def test_truncated_data_raises():
    """A header promising more voxels than the file holds is an error.

    Reading it as whatever bytes happen to be present is how a truncated
    download turns into a plausible-looking but wrong volume.
    """
    a = np.arange(3 * 5 * 4, dtype="f4").reshape(3, 5, 4)
    blob = build_mrc(4, 5, 3, 2, a)[:-40]
    with pytest.raises(MrcError, match="truncated"):
        oc.read(blob, format="mrc")


def test_unsupported_mode_raises():
    """MODE 101 is the 4-bit packed mode; we do not unpack it."""
    blob = build_mrc(4, 5, 1, 101, np.zeros(20, dtype="u1"))
    with pytest.raises(MrcError, match="MODE 101"):
        oc.read(blob, format="mrc")


def test_negative_dimension_raises():
    blob = build_mrc(-4, 5, 1, 2, np.zeros(20, dtype="f4"))
    with pytest.raises(MrcError, match="negative dimension"):
        oc.read(blob, format="mrc")


# --------------------------------------------------------------------
# axis order
# --------------------------------------------------------------------

def test_axis_order_reported_and_canonical_off_by_default():
    """A permuted map comes back in stored order unless asked otherwise.

    Returning the file's own layout is the honest default and matches
    every other MRC reader, but a caller who assumes (z, y, x) would get
    a silently transposed volume, so the permutation has to be visible.
    """
    a = np.arange(2 * 3 * 4, dtype="f4").reshape(2, 3, 4)
    blob = build_mrc(4, 3, 2, 2, a, mapc=3, mapr=1, maps=2)
    with MrcStream(blob) as r:
        assert r.axis_order == (2, 1, 3)
        assert not r.is_canonical
        assert np.array_equal(r.asarray(), a)          # stored order
        canon = r.asarray(canonical=True)
    # stored axes are (sections=y, rows=x, cols=z); (z, y, x) is (2, 0, 1)
    assert np.array_equal(canon, np.transpose(a, (2, 0, 1)))


def test_canonical_is_a_noop_when_already_zyx():
    a = np.arange(2 * 3 * 4, dtype="f4").reshape(2, 3, 4)
    with MrcStream(build_mrc(4, 3, 2, 2, a)) as r:
        assert r.is_canonical
        assert np.array_equal(r.asarray(canonical=True), a)


def test_inconsistent_axis_header_is_left_alone():
    """A header whose MAPC/MAPR/MAPS is not a permutation of 1,2,3.

    Refusing to transpose is right here: we cannot know what was meant,
    and inventing an order would be worse than returning the bytes.
    """
    a = np.arange(2 * 3 * 4, dtype="f4").reshape(2, 3, 4)
    blob = build_mrc(4, 3, 2, 2, a, mapc=1, mapr=1, maps=1)
    with MrcStream(blob) as r:
        assert np.array_equal(r.asarray(canonical=True), a)


# --------------------------------------------------------------------
# registry wiring
# --------------------------------------------------------------------

def test_codec_registered_and_sniffable():
    assert oc.has_codec("mrc")
    a = np.zeros((4, 5), dtype="f4")
    # No format= and no filename: dispatch must come from the MAP
    # identifier at byte 208, which is why the sniff window is 256.
    assert oc.read(build_mrc(5, 4, 1, 2, a)).shape == (4, 5)


def test_no_magic_still_opens_when_asked_explicitly():
    """Pre-2014 files omit the identifier; format="mrc" must still work."""
    a = np.zeros((4, 5), dtype="f4")
    blob = build_mrc(5, 4, 1, 2, a, with_magic=False)
    assert oc.read(blob, format="mrc").shape == (4, 5)


def test_codec_encode_round_trips():
    """Encoding landed after this reader; the codec surface must expose it."""
    a = (np.arange(2 * 3 * 4, dtype="f4") - 5).reshape(2, 3, 4)
    codec = oc.get_codec("mrc")
    assert codec.can_encode
    blob = codec.encode(a, voxel_size=(1.0, 2.0, 3.0))
    assert np.array_equal(oc.read(blob, format="mrc"), a)


# --------------------------------------------------------------------
# real EMDB deposit
# --------------------------------------------------------------------

@pytest.mark.skipif(not REAL_MAP.is_file(),
                    reason="run `python corpus/corpus.py fetch emdb_map_3001`")
def test_real_emdb_map_matches_its_own_header_statistics():
    """The strongest check available without a second implementation.

    The depositor's software wrote DMIN, DMAX and DMEAN into the header
    from the data it was writing. If our decode reproduces all three, we
    read the same voxels in the same order they were written, which no
    shape assertion can establish.
    """
    with MrcStream(str(REAL_MAP)) as r:
        h = r.header
        a = r.asarray()
        assert h["mode"] == 2 and a.dtype == np.float32
        assert a.shape == (h["nz"], h["ny"], h["nx"])
        assert float(a.min()) == pytest.approx(h["dmin"], rel=1e-5)
        assert float(a.max()) == pytest.approx(h["dmax"], rel=1e-5)
        assert float(a.mean()) == pytest.approx(h["dmean"], rel=1e-3)


@pytest.mark.skipif(not REAL_MAP.is_file(),
                    reason="run `python corpus/corpus.py fetch emdb_map_3001`")
def test_real_emdb_map_has_a_permuted_axis_order():
    """This deposit is why canonical= exists rather than being theoretical."""
    with MrcStream(str(REAL_MAP)) as r:
        assert r.axis_order == (2, 1, 3), (
            "EMD-3001 is expected to store MAPC/MAPR/MAPS = 3/1/2; if "
            "upstream re-deposited the map, update this test")
        assert not r.is_canonical
        stored = r.asarray()
        canon = r.asarray(canonical=True)
        assert canon.shape == (stored.shape[2], stored.shape[0], stored.shape[1])
        assert float(canon.sum()) == pytest.approx(float(stored.sum()), rel=1e-5)


@pytest.mark.skipif(not REAL_MAP.is_file(),
                    reason="run `python corpus/corpus.py fetch emdb_map_3001`")
def test_real_emdb_map_agrees_with_mrcfile():
    """Cross-validate against the reference implementation where present."""
    mrcfile = pytest.importorskip("mrcfile")
    with mrcfile.open(str(REAL_MAP)) as m:
        ref = np.asarray(m.data)
    got = oc.read(str(REAL_MAP), format="mrc")
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert np.array_equal(got, ref)


@pytest.mark.skipif(not REAL_MAP.is_file(),
                    reason="run `python corpus/corpus.py fetch emdb_map_3001`")
def test_real_emdb_map_plane_matches_full_read():
    with MrcStream(str(REAL_MAP)) as r:
        full = r.asarray()
        for i in (0, r.n_planes // 2, r.n_planes - 1):
            assert np.array_equal(r.plane(i), full[i])
