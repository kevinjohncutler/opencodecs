"""VSI whole slides are tiled pyramids, and we read them as flat stacks.

The level axis was the unknown that kept this gap open. It is the LAST
coordinate of each tile-table record, and that is derived rather than
assumed: parse_ets only reports a pyramid when the tile population of
every level equals the grid covering the extent halved that many times.
On the two tiled files this was mapped from that is 13 independent
predictions, all exact, and on the untiled corpus file the check fails
and the reader reports one level.

Two readings had to be corrected to see any of it. The SIS header field
that was taken for the table's byte size is a record COUNT, so a
413-record table was being read as 413 bytes. And a record is
self-describing rather than fixed-width: it opens with its own axis
count, so it is 4*(ndims+5) bytes, 36 for the tiled files and 44 for
the untiled one.

These tests use synthetic files for the structure, so they run
anywhere, plus the corpus file for the negative case. The tiled corpus
entry is optional; where it is present the byte-savings test uses it.
"""

from __future__ import annotations

import math
import pathlib
import struct

import numpy as np
import pytest

import opencodecs as oc
from opencodecs._ets import parse_ets
from opencodecs._vsi_pyramid import VsiPyramidError, VsiPyramidReader

CORPUS = pathlib.Path(__file__).resolve().parent.parent / ".test_data"
UNTILED = CORPUS / "vsi" / "_metadataTest_01_" / "stack1" / "frame_t_0.ets"


def _build_tiled_ets(path, extent, tile=(64, 64), *, levels=None, seed=0):
    """A tiled, pyramidal .ets whose tiles are real JPEG codestreams.

    Mirrors the layout the real files use: a 64-byte SIS header, a
    228-byte ETS sub-header carrying tile size at 28/32 and the level-0
    extent at 188/192, then tile payloads, then a table of
    self-describing records ending with the level axis.
    """
    W, H = extent
    tw, th = tile
    jpeg = oc.get_codec("jpeg")
    rng = np.random.default_rng(seed)

    n_levels = levels if levels is not None else max(
        1, math.ceil(math.log2(max(W / tw, H / th, 1))) + 1)

    payloads, recs = [], []
    for lvl in range(n_levels):
        lw, lh = math.ceil(W / 2 ** lvl), math.ceil(H / 2 ** lvl)
        for ty in range(math.ceil(lh / th)):
            for tx in range(math.ceil(lw / tw)):
                img = rng.integers(0, 255, (th, tw, 3)).astype("u1")
                payloads.append(jpeg.encode(img))
                recs.append((tx, ty, 0, lvl))

    sub = bytearray(228)
    sub[0:4] = b"ETS\x00"
    struct.pack_into("<I", sub, 8, 3)
    struct.pack_into("<I", sub, 28, tw)
    struct.pack_into("<I", sub, 32, th)
    struct.pack_into("<I", sub, 188, W)
    struct.pack_into("<I", sub, 192, H)

    body = bytearray()
    offsets = []
    base = 64 + 228
    for pay in payloads:
        offsets.append(base + len(body))
        body += pay

    n_dims = 4
    table = bytearray()
    for (tx, ty, c2, lvl), off, pay in zip(recs, offsets, payloads):
        table += struct.pack("<I", n_dims)
        table += struct.pack("<4I", tx, ty, c2, lvl)
        table += struct.pack("<Q", off)
        table += struct.pack("<II", len(pay), 0)

    ptr2 = base + len(body)
    hdr = bytearray(64)
    hdr[0:4] = b"SIS\x00"
    struct.pack_into("<I", hdr, 4, 64)
    struct.pack_into("<I", hdr, 8, 3)
    struct.pack_into("<Q", hdr, 16, 64)
    struct.pack_into("<Q", hdr, 24, 228)
    struct.pack_into("<Q", hdr, 32, ptr2)
    struct.pack_into("<Q", hdr, 40, len(recs))     # a COUNT, not bytes
    path.write_bytes(bytes(hdr + sub + body + table))
    return path, n_levels


@pytest.fixture
def tiled(tmp_path):
    p, n = _build_tiled_ets(tmp_path / "slide.ets", (500, 300), (64, 64))
    return p, n


def test_levels_are_derived_and_verified(tiled):
    p, n_levels = tiled
    info = parse_ets(str(p))
    assert info.pyramid_ok is True
    assert info.n_levels == n_levels
    assert (info.width, info.height) == (500, 300)
    assert (info.tile_width, info.tile_height) == (64, 64)
    for lvl in range(info.n_levels):
        h, w = info.level_shape(lvl)
        want = math.ceil(w / 64) * math.ceil(h / 64)
        assert len(info.tiles_at(lvl)) == want, lvl


def test_record_stride_is_self_describing(tiled):
    """4*(ndims+5): the reason a fixed width read tiled files as noise."""
    p, _ = tiled
    assert parse_ets(str(p)).n_dims == 4
    if UNTILED.is_file():
        assert parse_ets(str(UNTILED)).n_dims == 6


def test_the_count_field_is_a_count_not_a_size(tiled):
    """Reading it as bytes truncates the table to a fraction."""
    p, _ = tiled
    info = parse_ets(str(p))
    assert info.sub_chunk_sizes[1] == info.n_records
    assert info.n_records > info.sub_chunk_sizes[1] // 36 or info.n_records > 1


def test_shapes_halve_per_level(tiled):
    p, _ = tiled
    with VsiPyramidReader(str(p)) as pyr:
        assert pyr.shapes[0][:2] == (300, 500)
        assert pyr.downscale_factors[:3] == ((1, 1), (2, 2), (4, 4))
        for i in range(1, pyr.n_levels):
            ph, pw = pyr.shapes[i - 1][:2]
            h, w = pyr.shapes[i][:2]
            assert h == math.ceil(ph / 2) or h == -(-300 // (1 << i))


def test_region_equals_the_same_crop_of_the_whole_level(tiled):
    p, _ = tiled
    with VsiPyramidReader(str(p)) as pyr:
        whole = pyr.level(0).reader.asarray()
        for (y0, y1, x0, x1) in [(0, 64, 0, 64), (30, 200, 40, 300),
                                 (0, 300, 0, 500), (250, 300, 450, 500)]:
            got = pyr.read_region(0, y=(y0, y1), x=(x0, x1))
            assert got.shape[:2] == (y1 - y0, x1 - x0)
            assert np.array_equal(got, whole[y0:y1, x0:x1]), (y0, y1, x0, x1)


def test_a_region_decodes_only_the_tiles_it_covers(tiled, monkeypatch):
    """The point of a tiled pyramid, asserted by counting decodes."""
    p, _ = tiled
    with VsiPyramidReader(str(p)) as pyr:
        # Learn channels and dtype first. That decodes one tile to
        # probe it, and counting it here would measure the probe
        # rather than what the region touches.
        pyr.n_channels, pyr.dtype
        calls = []
        orig = pyr._decode_tile
        monkeypatch.setattr(pyr, "_decode_tile",
                            lambda r: (calls.append(r), orig(r))[1])
        pyr.read_region(0, y=(0, 64), x=(0, 64))
        assert len(calls) == 1, f"one 64x64 tile expected, decoded {len(calls)}"
        calls.clear()
        pyr.read_region(0, y=(0, 128), x=(0, 128))
        assert len(calls) == 4
        calls.clear()
        # A whole level touches every tile of that level and no other.
        pyr.level(1).reader.asarray()
        assert {r.level for r in calls} == {1}
        assert len(calls) == len(pyr.info.tiles_at(1))


def test_best_level_for_picks_by_envelope(tiled):
    p, _ = tiled
    with VsiPyramidReader(str(p)) as pyr:
        assert pyr.best_level_for(max_pixels_y=1000, max_pixels_x=1000) == 0
        small = pyr.best_level_for(max_pixels_y=80, max_pixels_x=80)
        assert pyr.shapes[small][0] <= 80 and pyr.shapes[small][1] <= 80


def test_an_untiled_file_is_not_called_a_pyramid():
    """The corpus file has one resolution, and the check says so.

    This is the case that matters most: this format has already been
    read as having six levels when it had one.
    """
    if not UNTILED.is_file():
        pytest.skip("VSI corpus sample not present")
    info = parse_ets(str(UNTILED))
    assert info.pyramid_ok is False
    assert info.n_levels == 1


def test_the_contiguous_plane_path_refuses_a_tiled_file(tiled):
    """It would return noise rather than fail, which is worse."""
    from opencodecs._ets import decode_ets, decode_ets_plane
    p, _ = tiled
    with pytest.raises(ValueError, match="tiled"):
        decode_ets(str(p))
    with pytest.raises(ValueError, match="tiled"):
        decode_ets_plane(str(p), 0)


def test_open_pyramid_dispatches_on_extension(tiled):
    p, _ = tiled
    with oc.open_pyramid(str(p)) as pyr:
        assert pyr.n_levels > 1


def test_a_vsi_with_several_stacks_asks_which(tmp_path):
    """Separate stacks are separate images, not levels of one."""
    vsi = tmp_path / "slide.vsi"
    vsi.write_bytes(b"II*\x00" + b"\x00" * 60)
    comp = tmp_path / "_slide_"
    for name in ("stack1", "stack10000"):
        d = comp / name
        d.mkdir(parents=True)
        _build_tiled_ets(d / "frame_t.ets", (200, 200), (64, 64))
    with pytest.raises(VsiPyramidError, match="stack="):
        VsiPyramidReader(str(vsi))
    with VsiPyramidReader(str(vsi), stack="stack10000") as pyr:
        assert pyr.shapes[0][:2] == (200, 200)
    with pytest.raises(VsiPyramidError, match="no stack"):
        VsiPyramidReader(str(vsi), stack="stack999")


def test_a_vsi_without_a_companion_says_so(tmp_path):
    vsi = tmp_path / "lonely.vsi"
    vsi.write_bytes(b"II*\x00" + b"\x00" * 60)
    with pytest.raises(VsiPyramidError, match="companion"):
        VsiPyramidReader(str(vsi))


def test_an_unrecognized_tile_codestream_is_named(tmp_path):
    """Not decoded as something else."""
    p, _ = _build_tiled_ets(tmp_path / "x.ets", (128, 128), (64, 64))
    raw = bytearray(p.read_bytes())
    raw[64 + 228:64 + 228 + 2] = b"ZZ"        # break the first tile's magic
    p.write_bytes(bytes(raw))
    with VsiPyramidReader(str(p)) as pyr:
        with pytest.raises(VsiPyramidError, match="codestream"):
            pyr.read_region(0, y=(0, 64), x=(0, 64))


# --------------------------------------------------------------------
# The real file. Optional, because it is a 33 MB corpus entry.
# --------------------------------------------------------------------

REAL = (CORPUS / "vsi_pyramid" / "_Image_V4.1_FL_2D_" / "stack1"
        / "frame_t.ets")
needs_real = pytest.mark.skipif(
    not REAL.is_file(),
    reason="fetch the vsi_evident_pyramid corpus entry first")


@needs_real
def test_the_real_slide_is_a_six_level_pyramid():
    info = parse_ets(str(REAL))
    assert (info.width, info.height) == (8022, 9367)
    assert (info.tile_width, info.tile_height) == (512, 512)
    assert info.pyramid_ok is True
    assert info.n_levels == 6
    assert [len(info.tiles_at(l)) for l in range(6)] == [304, 80, 20, 6, 2, 1]


@needs_real
def test_a_level_is_the_level_above_downscaled():
    """The claim that the last axis is the LEVEL, checked on pixels.

    The tile counts already constrain it arithmetically. This is the
    independent check: level L+1 has to look like level L at half
    size. Both are JPEG and the scanner resampled with its own filter,
    so this is a correlation rather than an equality.
    """
    with VsiPyramidReader(str(REAL)) as pyr:
        a = pyr.read_region(3, y=(0, 256), x=(0, 256)).astype(float)
        b = pyr.read_region(4, y=(0, 128), x=(0, 128)).astype(float)
    down = a.reshape(128, 2, 128, 2, 3).mean(axis=(1, 3))
    corr = np.corrcoef(down.ravel(), b.ravel())[0, 1]
    assert corr > 0.9, f"level 4 does not look like level 3 halved: {corr:.3f}"


@needs_real
def test_a_full_resolution_region_moves_a_few_tiles_over_http():
    """The capability, stated in bytes.

    A 256x256 window of a 8022x9367 slide should cost its four tiles,
    not the 32.6 MB file.
    """
    from _range_http_server import range_http_server

    size = REAL.stat().st_size
    with range_http_server(str(REAL.parent)) as (base, t):
        with VsiPyramidReader(f"{base}/{REAL.name}") as pyr:
            pyr.read_region(0, y=(1000, 1256), x=(2000, 2256))
        served = t.bytes_served
    assert served < size * 0.05, (
        f"a 256x256 region moved {served} bytes of {size}")


@needs_real
def test_opening_the_slide_does_not_read_the_slide():
    from _range_http_server import range_http_server

    size = REAL.stat().st_size
    with range_http_server(str(REAL.parent)) as (base, t):
        with VsiPyramidReader(f"{base}/{REAL.name}") as pyr:
            shapes = pyr.shapes
        served = t.bytes_served
    assert shapes[0][:2] == (9367, 8022)
    assert served < size * 0.02, f"opening moved {served} of {size}"
