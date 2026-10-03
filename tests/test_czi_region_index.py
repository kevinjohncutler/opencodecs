"""Spatial index and bounded regional reads (priority 4).

``CziPyramidReader._read_region`` used to scan every entry in a level and
decode intersecting tiles serially. Each level now carries an immutable grid
index, intersecting tiles decode through the bounded pipeline when there is
enough work to share, and composition stays in directory order on the
calling thread.

The reference used here is the old scan, spelled out in full, so agreement
is measured against the previous behavior rather than against the new code
twice. The cases that matter: many small crops on a regular mosaic, tiles
that overlap (later wins), irregular sizes with gaps (zero fill), clipped
edges, color samples, a request that spans two planes (refused), serial and
parallel paths agreeing, and the narrow stub contract the older pyramid tests
depend on.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc

pytestmark = pytest.mark.skipif(
    not oc.has_codec("czi"),
    reason="czi codec requires native zstd + bytetools extensions",
)

from _czi_fixture import mosaic_czi_bytes  # noqa: E402

from opencodecs._czi_reader import (  # noqa: E402
    CziError, CziPyramidReader, CziSubBlockEntry, _CziLevelIndex,
    _level_placement,
)


# ---------------------------------------------------------------------------
# Reference: the scan this replaced
# ---------------------------------------------------------------------------


def reference_region(czi, level, y0, y1, x0, x1):
    entries = level.reader
    ref = entries[0]
    ref_dims = ref.dims
    y_i = ref_dims.index("Y") if "Y" in ref_dims else len(ref_dims) - 2
    x_i = ref_dims.index("X") if "X" in ref_dims else len(ref_dims) - 1
    out_extra = (ref.samples,) if ref.samples > 1 else ()
    out = np.zeros((y1 - y0, x1 - x0) + out_extra, dtype=level.dtype)
    hits = 0
    for e in entries:
        ty, tx = e.start[y_i], e.start[x_i]
        th, tw = e.stored_shape[y_i], e.stored_shape[x_i]
        iy0, iy1 = max(y0, ty), min(y1, ty + th)
        ix0, ix1 = max(x0, tx), min(x1, tx + tw)
        if iy1 <= iy0 or ix1 <= ix0:
            continue
        hits += 1
        tile = np.moveaxis(czi._decode_one(e), (y_i, x_i), (0, 1))
        if e.samples > 1:
            tile = tile.reshape(tile.shape[0], tile.shape[1], -1)[:, :, :e.samples]
        else:
            tile = tile.reshape(tile.shape[0], tile.shape[1])
        out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = \
            tile[iy0 - ty:iy1 - ty, ix0 - tx:ix1 - tx]
    return out, hits


def _tile(shape, seed, dtype=np.uint16):
    rs = np.random.RandomState(seed)
    if dtype == np.uint8:
        return rs.randint(0, 256, shape).astype(dtype)
    return rs.randint(0, 65535, shape).astype(dtype)


def _canvas(tiles, dtype):
    """Ground truth: paste tiles in directory order onto a zero canvas."""
    h = max(sy + t.shape[0] for t, (sy, sx), *_ in tiles)
    w = max(sx + t.shape[1] for t, (sy, sx), *_ in tiles)
    extra = tiles[0][0].shape[2:]
    canvas = np.zeros((h, w) + extra, dtype=dtype)
    for t, (sy, sx), *_ in tiles:
        canvas[sy:sy + t.shape[0], sx:sx + t.shape[1]] = t
    return canvas


def _open(data, **kw):
    r = oc.get_codec("czi").open(data)
    return r, CziPyramidReader(r, **kw)


def _random_boxes(rs, h, w, n, max_h, max_w):
    for _ in range(n):
        bh = int(rs.randint(1, max_h + 1))
        bw = int(rs.randint(1, max_w + 1))
        y0 = int(rs.randint(0, max(1, h - bh + 1)))
        x0 = int(rs.randint(0, max(1, w - bw + 1)))
        yield y0, min(h, y0 + bh), x0, min(w, x0 + bw)


# ---------------------------------------------------------------------------
# Regular mosaic
# ---------------------------------------------------------------------------


def test_many_small_crops_on_a_regular_mosaic_match_the_scan():
    th, tw, rows, cols = 32, 32, 6, 5
    tiles = [(_tile((th, tw), 100 + r * cols + c), (r * th, c * tw))
              for r in range(rows) for c in range(cols)]
    canvas = _canvas(tiles, np.uint16)
    data = mosaic_czi_bytes(tiles, compression=6, hilo=True)
    r, p = _open(data)
    try:
        level = p.level(0)
        assert level.shape == canvas.shape
        rs = np.random.RandomState(1)
        for y0, y1, x0, x1 in _random_boxes(rs, *canvas.shape[:2], 50, 70, 70):
            got = p.read_region(0, y=(y0, y1), x=(x0, x1))
            expected, hits = reference_region(r, level, y0, y1, x0, x1)
            np.testing.assert_array_equal(got, expected)
            np.testing.assert_array_equal(got, canvas[y0:y1, x0:x1])
            st = p.region_stats
            assert st["tiles_decoded"] == st["tiles_intersecting"] == hits
            assert st["candidates_examined"] <= st["level_entries"]
    finally:
        p.close()


def test_a_crop_inside_one_tile_examines_few_candidates():
    tiles = [(_tile((32, 32), 200 + i), ((i // 8) * 32, (i % 8) * 32))
             for i in range(64)]
    data = mosaic_czi_bytes(tiles, compression=6, hilo=True)
    r, p = _open(data)
    try:
        canvas = _canvas(tiles, np.uint16)
        # The first query scans the bounds once instead of building the grid,
        # so a viewer's first crop does not pay for the whole index.
        got = p.read_region(0, y=(70, 90), x=(40, 60))  # strictly inside tile (2,1)
        np.testing.assert_array_equal(got, canvas[70:90, 40:60])
        st = p.region_stats
        assert st["level_entries"] == 64
        assert st["tiles_decoded"] == 1
        assert st["candidates_examined"] == 64 and st["index_cells"] == 0
        # From the second query on, the grid narrows the candidates.
        got = p.read_region(0, y=(70, 90), x=(40, 60))
        np.testing.assert_array_equal(got, canvas[70:90, 40:60])
        st = p.region_stats
        assert st["tiles_decoded"] == 1
        assert st["index_cells"] >= 1
        assert st["candidates_examined"] <= 4
    finally:
        p.close()


def test_full_extent_read_matches_canvas():
    tiles = [(_tile((16, 24), 300 + i), ((i // 3) * 16, (i % 3) * 24))
             for i in range(9)]
    canvas = _canvas(tiles, np.uint16)
    r, p = _open(mosaic_czi_bytes(tiles, compression=5))
    try:
        np.testing.assert_array_equal(p.read_region(0), canvas)
        assert p.region_stats["tiles_decoded"] == 9
    finally:
        p.close()


# ---------------------------------------------------------------------------
# Overlap, irregular layouts, gaps, edges, color
# ---------------------------------------------------------------------------


def test_overlapping_tiles_keep_directory_order_last_wins():
    a = _tile((32, 32), 1)
    b = _tile((32, 32), 2)
    tiles = [(a, (0, 0)), (b, (16, 16))]
    canvas = _canvas(tiles, np.uint16)
    r, p = _open(mosaic_czi_bytes(tiles, compression=6, hilo=True))
    try:
        level = p.level(0)
        for box in [(0, 48, 0, 48), (16, 32, 16, 32), (10, 40, 10, 40), (20, 30, 0, 20)]:
            got = p.read_region(0, y=box[:2], x=box[2:])
            expected, _ = reference_region(r, level, *box)
            np.testing.assert_array_equal(got, expected)
            np.testing.assert_array_equal(got, canvas[box[0]:box[1], box[2]:box[3]])
        # The overlap square is b's pixels, not a's.
        np.testing.assert_array_equal(p.read_region(0, y=(16, 32), x=(16, 32)),
                                      b[:16, :16])
    finally:
        p.close()


def test_irregular_sizes_with_gaps_fill_zero_and_match_the_scan():
    tiles = [
        (_tile((32, 32), 11), (0, 0)),
        (_tile((48, 16), 12), (0, 40)),      # gap of 8 columns before it
        (_tile((16, 64), 13), (60, 8)),      # gap rows 48..60, offset x
        (_tile((8, 8), 14), (100, 100)),     # far corner, tiny
    ]
    canvas = _canvas(tiles, np.uint16)
    r, p = _open(mosaic_czi_bytes(tiles, compression=6, hilo=True))
    try:
        level = p.level(0)
        assert level.shape == canvas.shape == (108, 108)
        rs = np.random.RandomState(3)
        boxes = list(_random_boxes(rs, 108, 108, 40, 60, 60))
        boxes += [(30, 70, 20, 60), (48, 60, 0, 108), (0, 108, 0, 108)]
        for y0, y1, x0, x1 in boxes:
            got = p.read_region(0, y=(y0, y1), x=(x0, x1))
            expected, hits = reference_region(r, level, y0, y1, x0, x1)
            np.testing.assert_array_equal(got, expected)
            np.testing.assert_array_equal(got, canvas[y0:y1, x0:x1])
            assert p.region_stats["tiles_decoded"] == hits
        # A box entirely inside a gap decodes nothing and is all zero.
        got = p.read_region(0, y=(50, 58), x=(0, 8))
        assert not got.any()
        assert p.region_stats["tiles_decoded"] == 0
    finally:
        p.close()


def test_clipped_edges_and_single_pixel_boxes():
    tiles = [(_tile((20, 30), 21 + i), ((i // 2) * 20, (i % 2) * 30))
             for i in range(4)]
    canvas = _canvas(tiles, np.uint16)
    r, p = _open(mosaic_czi_bytes(tiles, compression=6, hilo=True))
    try:
        h, w = canvas.shape
        for box in [(h - 1, h, w - 1, w), (0, 1, 0, 1), (19, 21, 29, 31),
                    (0, h, w - 1, w), (h - 1, h, 0, w)]:
            got = p.read_region(0, y=box[:2], x=box[2:])
            np.testing.assert_array_equal(got, canvas[box[0]:box[1], box[2]:box[3]])
    finally:
        p.close()


def test_color_mosaic_keeps_the_sample_axis():
    tiles = [(_tile((16, 16, 3), 31 + i, np.uint8), ((i // 2) * 16, (i % 2) * 16))
             for i in range(4)]
    canvas = _canvas(tiles, np.uint8)
    r, p = _open(mosaic_czi_bytes(tiles, compression=6, hilo=True))
    try:
        level = p.level(0)
        got = p.read_region(0, y=(8, 24), x=(8, 24))
        assert got.shape == (16, 16, 3)
        expected, _ = reference_region(r, level, 8, 24, 8, 24)
        np.testing.assert_array_equal(got, expected)
        np.testing.assert_array_equal(got, canvas[8:24, 8:24])
    finally:
        p.close()


# ---------------------------------------------------------------------------
# Planes
# ---------------------------------------------------------------------------


def test_request_spanning_two_planes_is_refused():
    a = _tile((32, 32), 41)
    b = _tile((32, 32), 42)
    tiles = [(a, (0, 0), [(b"C", 0)]), (b, (0, 0), [(b"C", 1)])]
    r, p = _open(mosaic_czi_bytes(tiles, compression=6, hilo=True))
    try:
        with pytest.raises(CziError, match="distinct planes"):
            p.read_region(0, y=(0, 32), x=(0, 32))
    finally:
        p.close()


def test_planes_at_different_places_are_fine_until_a_request_spans_both():
    a = _tile((32, 32), 43)
    b = _tile((32, 32), 44)
    tiles = [(a, (0, 0), [(b"C", 0)]), (b, (0, 64), [(b"C", 1)])]
    r, p = _open(mosaic_czi_bytes(tiles, compression=6, hilo=True))
    try:
        np.testing.assert_array_equal(p.read_region(0, y=(0, 32), x=(0, 32)), a)
        np.testing.assert_array_equal(p.read_region(0, y=(0, 32), x=(64, 96)), b)
        with pytest.raises(CziError, match="distinct planes"):
            p.read_region(0, y=(0, 32), x=(0, 96))
    finally:
        p.close()


# ---------------------------------------------------------------------------
# Serial vs parallel, stub contract, small directory
# ---------------------------------------------------------------------------


def test_serial_and_parallel_paths_agree():
    # 512x512 uint16 tiles are 512 KB each, so a full read spans several
    # 2 MiB decode batches and the parallel path actually runs; 8 KB tiles
    # would all fit one batch and the reader would correctly stay serial.
    tiles = [(_tile((512, 512), 500 + i), ((i // 4) * 512, (i % 4) * 512))
             for i in range(16)]
    canvas = _canvas(tiles, np.uint16)
    data = mosaic_czi_bytes(tiles, compression=6, hilo=True)
    r1, serial = _open(data, decode_workers=1)
    r2, parallel = _open(data, decode_workers=4, max_pending_bytes=1 << 20)
    try:
        boxes = [(0, 2048, 0, 2048), (100, 1500, 50, 1900), (7, 9, 7, 9),
                 (0, 2048, 1020, 1030)]
        saw_parallel = False
        for y0, y1, x0, x1 in boxes:
            a = serial.read_region(0, y=(y0, y1), x=(x0, x1))
            b = parallel.read_region(0, y=(y0, y1), x=(x0, x1))
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(a, canvas[y0:y1, x0:x1])
            assert serial.region_stats["workers"] == 1
            if parallel.region_stats["batches"] >= 2:
                assert parallel.region_stats["workers"] > 1
                saw_parallel = True
        assert saw_parallel, "no box exercised the parallel path"
    finally:
        serial.close()
        parallel.close()


def _mk_entry(shape, stored_shape, *, dims=("Y", "X", "S"), start=None, tag=0):
    if start is None:
        start = tuple(0 for _ in shape)
    return CziSubBlockEntry(
        file_position=tag, pixel_type=0, compression=0,
        dimensions_count=len(dims), dims=tuple(dims), shape=tuple(shape),
        stored_shape=tuple(stored_shape), start=tuple(start),
        mosaic_index=-1, scene_index=-1, storage_size=0, pyramid_type=0,
    )


class _Stub:
    """The narrow contract test_czi_pyramid.py relies on: canned entries and
    a positional-only _decode_one returning 2-D pixels; no source at all."""

    def __init__(self, entries):
        self.entries = list(entries)

    def scale_factors_per_level(self, axes=("Y", "X")):
        from opencodecs._czi_reader import CziReader
        return CziReader.scale_factors_per_level(self, axes=axes)

    def entries_at_level(self, level=0, *, axes=("Y", "X")):
        from opencodecs._czi_reader import CziReader
        return CziReader.entries_at_level(self, level, axes=axes)

    def close(self):
        pass

    def _decode_one(self, entry):
        h, w = entry.stored_shape[0], entry.stored_shape[1]
        return np.full((h, w), entry.file_position, dtype=np.uint8)


def test_stub_reader_without_a_source_still_works():
    entries = [_mk_entry((16, 16, 1), (16, 16, 1), start=(0, 0, 0), tag=7),
               _mk_entry((16, 16, 1), (16, 16, 1), start=(0, 16, 0), tag=9)]
    p = CziPyramidReader(_Stub(entries))
    try:
        out = p.read_region(0, y=(4, 12), x=(8, 24))
        assert out.shape == (8, 16)
        assert np.all(out[:, :8] == 7) and np.all(out[:, 8:] == 9)
        assert "source_requests" not in p.region_stats
        assert p.region_stats["tiles_decoded"] == 2
    finally:
        p.close()


def test_small_directory_control():
    tiles = [(_tile((32, 32), 61), (0, 0)), (_tile((32, 32), 62), (0, 32))]
    canvas = _canvas(tiles, np.uint16)
    r, p = _open(mosaic_czi_bytes(tiles, compression=6, hilo=True))
    try:
        got = p.read_region(0, y=(5, 20), x=(20, 50))
        np.testing.assert_array_equal(got, canvas[5:20, 20:50])
        st = p.region_stats
        assert st["level_entries"] == 2
        assert st["candidates_examined"] <= 2
        assert st["tiles_decoded"] == 2
        # A level this small is always scanned; it never builds a grid.
        assert st["elapsed_s"] >= 0 and st["index_cells"] == 0
    finally:
        p.close()


# ---------------------------------------------------------------------------
# The index itself
# ---------------------------------------------------------------------------


def test_index_query_matches_brute_force_on_irregular_tiles():
    rs = np.random.RandomState(5)
    entries = []
    for i in range(80):
        h, w = int(rs.randint(1, 50)), int(rs.randint(1, 50))
        y, x = int(rs.randint(0, 300)), int(rs.randint(0, 300))
        entries.append(_mk_entry((h, w, 1), (h, w, 1), start=(y, x, 0), tag=i))
    index = _CziLevelIndex(entries, _level_placement(entries, (0, 0), (1.0, 1.0)))
    assert index.n == 80 and index.n_cells >= 1
    for _ in range(200):
        y0, x0 = int(rs.randint(-10, 320)), int(rs.randint(-10, 320))
        y1, x1 = y0 + int(rs.randint(0, 120)), x0 + int(rs.randint(0, 120))
        hits, examined = index.query(y0, y1, x0, x1)
        # A zero-area box intersects nothing, which is also what the
        # region assembly does (it skips empty intersections), even though
        # a naive strict-inequality test would say a straddling tile hits.
        if y1 <= y0 or x1 <= x0:
            expected = []
        else:
            expected = [i for i, e in enumerate(entries)
                        if e.start[0] < y1 and e.start[0] + e.stored_shape[0] > y0
                        and e.start[1] < x1 and e.start[1] + e.stored_shape[1] > x0]
        assert hits == expected
        assert examined >= len(hits)
    assert index.query(10, 10, 0, 50) == ([], 0)
    assert index.query(1000, 1100, 1000, 1100) == ([], 0)


def test_range_buffer_counts_physical_requests():
    from opencodecs._czi_reader import _RangeBuffer

    class Source:
        def __init__(self, blob):
            self.blob = blob

        def read_at(self, offset, length):
            return self.blob[offset:offset + length]

    buf = _RangeBuffer(Source(bytes(range(256)) * 4), 1024)
    assert buf.requests == 0                                # nothing until the header is wanted
    buf.fetch_header()
    buf.fetch_header()
    assert buf.requests == 1 and buf.bytes_read == 128     # header prefetch, once
    assert bytes(buf[10:20]) == bytes(range(10, 20))        # served from cache
    assert buf.requests == 1
    assert bytes(buf[500:504]) == bytes(range(244, 248))    # physical read
    assert buf.requests == 2 and buf.bytes_read == 132
