"""Pyramid level geometry and directory parsing on whole-slide-shaped files.

Zen slide scans store sub-block starts in full-resolution slide
coordinates: far from zero, often negative, and not divided by a level's
scale. The pyramid reader used to take those starts as level pixels, so a
real Axioscan slide reported every level as (N, 0). These tests build
slides whose levels are exact decimations of the full-resolution image at
an offset, negative origin, so a level read has one right answer.
"""
from __future__ import annotations

import pathlib
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import _czi_fixture as fx  # noqa: E402

from opencodecs._czi_reader import (  # noqa: E402
    CziPyramidReader, CziReader, _parse_directory_entries,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
AXIOSCAN = ROOT / ".test_data" / "czi" / "ome_axioscan_pyramid.czi"


def _slide_bytes(image, tile, levels, origin, *, extra=True, dtype_samples=1):
    """Serialize ``image`` as a tiled pyramid whose starts are slide coordinates.

    Level k holds ``image[::2**k, ::2**k]`` cut into ``tile``-sized stored
    tiles (edge tiles smaller), each placed at ``origin + position * 2**k``
    with logical extent ``stored * 2**k``, the way Zen writes pyramids.
    """
    oy, ox = origin
    pixel_type = fx._pixel_type_for(image, dtype_samples)
    header_size = (32 + 88 + 31) // 32 * 32
    metadata = fx._build_metadata_segment(b"<Metadata/>")
    cur = header_size + len(metadata)
    segs, meta = [], []
    m_index = 0
    for k in range(levels):
        f = 1 << k
        level = np.ascontiguousarray(image[::f, ::f])
        for y in range(0, level.shape[0], tile):
            for x in range(0, level.shape[1], tile):
                t = np.ascontiguousarray(level[y:y + tile, x:x + tile])
                h, w = t.shape[:2]
                dims = ([(b"C", 0), (b"Z", 0), (b"M", m_index)] if extra else [])
                m_index += 1
                seg, _ = fx._build_subblock(
                    t, pixel_type, 0, False, cur, logical_shape=(h * f, w * f),
                    location=(oy + y * f, ox + x * f),
                    pyramid_type=0 if k == 0 else 2, extra_dims=dims)
                segs.append(seg)
                meta.append({"file_position": cur, "pixel_type": pixel_type,
                             "compression": 0, "stored_w": w, "stored_h": h,
                             "logical_w": w * f, "logical_h": h * f,
                             "start_x": ox + x * f, "start_y": oy + y * f,
                             "pyramid_type": 0 if k == 0 else 2, "extra_dims": dims})
                cur += len(seg)
    directory_position = cur
    directory = fx._build_directory_segment(meta)
    cur += len(directory)
    out = bytearray(fx._build_file_header(directory_position=directory_position,
                                          metadata_position=header_size, file_size=cur))
    out += b"\x00" * (header_size - len(out))
    out += metadata
    for s in segs:
        out += s
    out += directory
    return bytes(out)


@pytest.fixture(scope="module")
def slide():
    rs = np.random.RandomState(20260924)
    # Asymmetric and not a multiple of the tile or of 8, so edge tiles and
    # floor division both matter.
    image = rs.randint(0, 65535, (357, 509)).astype(np.uint16)
    return image, _slide_bytes(image, 64, 4, origin=(20086, -104227))


def test_levels_are_the_slide_shrunk_by_their_scale(slide):
    image, data = slide
    with CziPyramidReader(CziReader(buffer=data)) as p:
        assert p.origin == (20086, -104227)
        shapes = [lv.shape for lv in p.levels]
        assert [lv.downscale for lv in p.levels] == [(1, 1), (2, 2), (4, 4), (8, 8)]
    assert shapes == [image[::f, ::f].shape for f in (1, 2, 4, 8)]


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_whole_level_read_equals_the_decimated_image(slide, level):
    image, data = slide
    f = 1 << level
    with CziPyramidReader(CziReader(buffer=data)) as p:
        np.testing.assert_array_equal(p.read_region(level), image[::f, ::f])


def test_region_reads_are_in_level_pixels_from_the_origin(slide):
    image, data = slide
    rs = np.random.RandomState(7)
    with CziPyramidReader(CziReader(buffer=data)) as p:
        for level in range(4):
            ref = image[::1 << level, ::1 << level]
            h, w = ref.shape
            for _ in range(12):
                y0 = int(rs.randint(0, h - 1))
                x0 = int(rs.randint(0, w - 1))
                y1 = int(rs.randint(y0 + 1, h + 1))
                x1 = int(rs.randint(x0 + 1, w + 1))
                got = p.read_region(level, y=(y0, y1), x=(x0, x1))
                np.testing.assert_array_equal(got, ref[y0:y1, x0:x1])


def test_origin_at_zero_keeps_the_old_coordinates():
    """Files whose slide starts at zero read exactly as before."""
    rs = np.random.RandomState(3)
    image = rs.randint(0, 255, (100, 140)).astype(np.uint8)
    data = _slide_bytes(image, 32, 2, origin=(0, 0), extra=False)
    with CziPyramidReader(CziReader(buffer=data)) as p:
        assert p.origin == (0, 0)
        np.testing.assert_array_equal(p.read_region(0, y=(10, 90), x=(5, 133)),
                                      image[10:90, 5:133])
        np.testing.assert_array_equal(p.read_region(1), image[::2, ::2])


def test_fast_directory_parse_matches_the_entry_parser(slide):
    """The batched parser and the one-entry parser agree field for field."""
    _, data = slide
    r = CziReader(buffer=data)
    try:
        directory_position = struct.unpack_from("<q", data, 32 + 4 * 4 + 32 + 4)[0]
        count = struct.unpack_from("<I", data, directory_position + 32)[0]
        start = directory_position + 32 + 128
        fast = _parse_directory_entries(data, start, count)
        slow, off = [], start
        for _ in range(count):
            entry, advance = r._parse_directory_entry(off)
            slow.append(entry)
            off += advance
        assert fast == slow == r.entries
        assert {e.mosaic_index for e in fast} == set(range(count))
        # Storage order is X Y C Z M; entries list it reversed, drop M into
        # mosaic_index and append the implicit samples axis.
        assert fast[0].dims == ("Z", "C", "Y", "X", "S")
    finally:
        r.close()


def test_range_source_parses_the_same_directory(slide):
    """Remote sources keep the per-entry path; the result is the same."""
    _, data = slide

    class Source:
        def read_at(self, offset, length):
            return data[offset:offset + length]

    local = CziReader(buffer=data)
    remote = CziReader(data_source=Source(), size=len(data))
    try:
        assert remote.entries == local.entries
    finally:
        local.close()
        remote.close()


@pytest.mark.skipif(not AXIOSCAN.exists(), reason="corpus slide not downloaded")
def test_real_axioscan_slide_matches_czifile_geometry():
    """czifile parses the directory independently; both must agree.

    Level 0 must equal czifile's image origin and shape. For every level,
    each sub-block's start and stored size must match czifile's entry for
    the same file position, and the level's extent must be what those
    independent entries give: the furthest floor((start - origin) / scale)
    + stored size. Coarser levels need not be exactly the slide divided by
    their scale, because Zen's coarse tiles stop short of the full-resolution
    edge.
    """
    czifile = pytest.importorskip("czifile")
    with czifile.CziFile(str(AXIOSCAN)) as ref:
        image = ref.scenes[0] if not callable(ref.scenes) else ref.scenes()
        ref_shape = tuple(image.shape[-2:])
        ref_start = tuple(image.start[-2:])
        theirs = {}
        for d in ref.subblock_directory:
            yi, xi = d.dims.index("Y"), d.dims.index("X")
            theirs[d.file_position] = (d.start[yi], d.start[xi], d.shape[yi], d.shape[xi],
                                       d.stored_shape[yi], d.stored_shape[xi])
    with CziPyramidReader(CziReader(AXIOSCAN)) as p:
        assert p.origin == ref_start
        assert p.levels[0].shape == ref_shape
        oy, ox = p.origin
        for lv in p.levels:
            sy, sx = lv.downscale
            ext_y = ext_x = 0
            for e in lv.reader:
                y_i, x_i = e.dims.index("Y"), e.dims.index("X")
                ours = (e.start[y_i], e.start[x_i], e.shape[y_i], e.shape[x_i],
                        e.stored_shape[y_i], e.stored_shape[x_i])
                assert ours == theirs[e.file_position]
                ty0, tx0, _, _, th, tw = theirs[e.file_position]
                ext_y = max(ext_y, (ty0 - oy) // sy + th)
                ext_x = max(ext_x, (tx0 - ox) // sx + tw)
            assert lv.shape == (ext_y, ext_x)
